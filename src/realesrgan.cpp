// realesrgan implemented with ncnn library
//
// 기존 구현과의 차이점 (성능/안정성 개선):
//
//  1. 타일 crop / BGR<->RGB 변환 / 결과를 cv::Mat으로 되돌리는 작업을 전부
//     GPU vulkan compute shader(realesrgan_preproc.comp, realesrgan_postproc.comp)
//     로 옮겼습니다. 예전 코드는 이 작업들을 CPU에서, 그것도 결과를 되돌릴 때는
//     픽셀 하나하나를 순회하는 이중 for문(to_ocv)으로 처리해서 큰 이미지일수록
//     크게 느려졌습니다.
//  2. GPU가 fp16 storage + int8 storage를 지원하면, 입력 픽셀 버퍼를 변환 없이
//     그대로 업로드해서 CPU 색공간 변환 자체를 생략합니다(realesrgan-ncnn-vulkan
//     공식 배포판과 동일한 fast path).
//  3. 타일마다 vulkan 커맨드를 새로 submit하는 대신 Blob/Staging 할당자를
//     프레임(가로 밴드) 단위로 재사용해서 GPU 메모리 재할당 오버헤드를 줄였습니다.
//
// GFPGAN 파이프라인은 항상 3채널 BGR(cv::imread(..., 1))만 다루므로, 공식
// realesrgan-ncnn-vulkan에 있던 알파 채널/TTA 처리는 뺐습니다 (필요해지면
// realesrgan/src/realesrgan.cpp 쪽 구현을 참고해서 다시 붙이면 됩니다).

#include "realesrgan.h"

#include <algorithm>
#include <vector>

static const uint32_t realesrgan_preproc_spv_data[] = {
    #include "realesrgan_preproc.spv.hex.h"
};
static const uint32_t realesrgan_preproc_fp16s_spv_data[] = {
    #include "realesrgan_preproc_fp16s.spv.hex.h"
};
static const uint32_t realesrgan_preproc_int8s_spv_data[] = {
    #include "realesrgan_preproc_int8s.spv.hex.h"
};
static const uint32_t realesrgan_postproc_spv_data[] = {
    #include "realesrgan_postproc.spv.hex.h"
};
static const uint32_t realesrgan_postproc_fp16s_spv_data[] = {
    #include "realesrgan_postproc_fp16s.spv.hex.h"
};
static const uint32_t realesrgan_postproc_int8s_spv_data[] = {
    #include "realesrgan_postproc_int8s.spv.hex.h"
};

RealESRGAN::RealESRGAN(int gpuid, bool _tta_mode)
{
    net.opt.use_vulkan_compute = true;
    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_int8_storage = true;
    net.opt.use_int8_arithmetic = false;

    net.set_vulkan_device(gpuid);

    realesrgan_preproc = 0;
    realesrgan_postproc = 0;
    tta_mode = _tta_mode;

    // 예전 기본값과 동일하게 유지 (호출부에서 다시 덮어써도 무방)
    scale = 4;
    tile_size = 400;
    tile_pad = 10;
}

RealESRGAN::~RealESRGAN()
{
    delete realesrgan_preproc;
    delete realesrgan_postproc;
}

int RealESRGAN::load(const std::string& parampath, const std::string& modelpath)
{
    int ret = net.load_param(parampath.c_str());
    if (ret < 0)
    {
        fprintf(stderr, "open param file %s failed\n", parampath.c_str());
        return -1;
    }
    ret = net.load_model(modelpath.c_str());
    if (ret < 0)
    {
        fprintf(stderr, "open bin file %s failed\n", modelpath.c_str());
        return -1;
    }

    // preprocess / postprocess 파이프라인 초기화
    std::vector<ncnn::vk_specialization_type> specializations(1);
    specializations[0].i = 0; // bgr = 0 : cv::Mat은 이미 BGR 순서이므로 셰이더가 그대로 읽고 그대로 씀

    realesrgan_preproc = new ncnn::Pipeline(net.vulkan_device());
    realesrgan_preproc->set_optimal_local_size_xyz(32, 32, 3);

    realesrgan_postproc = new ncnn::Pipeline(net.vulkan_device());
    realesrgan_postproc->set_optimal_local_size_xyz(32, 32, 3);

    if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
        realesrgan_preproc->create(realesrgan_preproc_int8s_spv_data, sizeof(realesrgan_preproc_int8s_spv_data), specializations);
    else if (net.opt.use_fp16_storage)
        realesrgan_preproc->create(realesrgan_preproc_fp16s_spv_data, sizeof(realesrgan_preproc_fp16s_spv_data), specializations);
    else
        realesrgan_preproc->create(realesrgan_preproc_spv_data, sizeof(realesrgan_preproc_spv_data), specializations);

    if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
        realesrgan_postproc->create(realesrgan_postproc_int8s_spv_data, sizeof(realesrgan_postproc_int8s_spv_data), specializations);
    else if (net.opt.use_fp16_storage)
        realesrgan_postproc->create(realesrgan_postproc_fp16s_spv_data, sizeof(realesrgan_postproc_fp16s_spv_data), specializations);
    else
        realesrgan_postproc->create(realesrgan_postproc_spv_data, sizeof(realesrgan_postproc_spv_data), specializations);

    return 0;
}

int RealESRGAN::tile_process(const cv::Mat& inimage, cv::Mat& outimage)
{
    if (inimage.empty())
        return -1;

    // 이 파이프라인은 8UC3 BGR을 기대합니다. 그 외 타입이 들어오면 변환합니다.
    cv::Mat img3;
    if (inimage.type() != CV_8UC3)
        inimage.convertTo(img3, CV_8UC3);
    else
        img3 = inimage.isContinuous() ? inimage : inimage.clone();

    const int w = img3.cols;
    const int h = img3.rows;
    const int channels = 3;

    outimage.create(h * scale, w * scale, CV_8UC3);

    const unsigned char* pixeldata = img3.data;
    const int prepadding = tile_pad;
    const int TILE_SIZE_X = tile_size;
    const int TILE_SIZE_Y = tile_size;

    ncnn::VkAllocator* blob_vkallocator = net.vulkan_device()->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = net.vulkan_device()->acquire_staging_allocator();

    ncnn::Option opt = net.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    for (int yi = 0; yi < ytiles; yi++)
    {
        const int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        const int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding, h);

        // 가로 밴드(row band) 하나를 통째로 업로드합니다. img3가 연속 버퍼이므로
        // (isContinuous 보장, 위에서 clone 처리) w*channels 스트라이드로 안전하게
        // 포인터 산술이 가능합니다.
        ncnn::Mat in;
        if (opt.use_fp16_storage && opt.use_int8_storage)
        {
            // fast path: CPU 색공간 변환 없이 원본 BGR 바이트를 그대로 GPU로 올림
            in = ncnn::Mat(w, (in_tile_y1 - in_tile_y0), (void*)(pixeldata + (size_t)in_tile_y0 * w * channels), (size_t)channels, 1);
        }
        else
        {
            // 구형 GPU (fp16/int8 storage 미지원) 폴백 경로
            in = ncnn::Mat::from_pixels(pixeldata + (size_t)in_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGR, w, (in_tile_y1 - in_tile_y0));
        }

        ncnn::VkCompute cmd(net.vulkan_device());

        ncnn::VkMat in_gpu;
        {
            cmd.record_clone(in, in_gpu, opt);
            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        const int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
        const int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h);

        ncnn::VkMat out_gpu;
        if (opt.use_fp16_storage && opt.use_int8_storage)
            out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, (size_t)channels, 1, blob_vkallocator);
        else
            out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, channels, (size_t)4u, 1, blob_vkallocator);

        for (int xi = 0; xi < xtiles; xi++)
        {
            // ---- preproc: crop + (필요시) 색공간 변환을 GPU 셰이더에서 처리 ----
            ncnn::VkMat in_tile_gpu;
            ncnn::VkMat dummy_alpha_gpu; // 알파 채널 없음 (셰이더 바인딩 슬롯 채우기용, 실제로는 안 쓰임)
            {
                const int tile_x0 = xi * TILE_SIZE_X - prepadding;
                const int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding;
                const int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                const int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding;

                in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);

                std::vector<ncnn::VkMat> bindings(3);
                bindings[0] = in_gpu;
                bindings[1] = in_tile_gpu;
                bindings[2] = dummy_alpha_gpu;

                std::vector<ncnn::vk_constant_type> constants(13);
                constants[0].i = in_gpu.w;
                constants[1].i = in_gpu.h;
                constants[2].i = in_gpu.cstep;
                constants[3].i = in_tile_gpu.w;
                constants[4].i = in_tile_gpu.h;
                constants[5].i = in_tile_gpu.cstep;
                constants[6].i = prepadding;
                constants[7].i = prepadding;
                constants[8].i = xi * TILE_SIZE_X;
                constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                constants[10].i = channels;
                constants[11].i = 0; // alphaw
                constants[12].i = 0; // alphah

                ncnn::VkMat dispatcher;
                dispatcher.w = in_tile_gpu.w;
                dispatcher.h = in_tile_gpu.h;
                dispatcher.c = channels;

                cmd.record_pipeline(realesrgan_preproc, bindings, constants, dispatcher);
            }

            // ---- 신경망 추론 ----
            ncnn::VkMat out_tile_gpu;
            {
                ncnn::Extractor ex = net.create_extractor();

                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("data", in_tile_gpu);
                ex.extract("output", out_tile_gpu, cmd);
            }

            // ---- postproc: 결과 타일을 큰 out_gpu 캔버스에 GPU에서 바로 합성 ----
            {
                std::vector<ncnn::VkMat> bindings(3);
                bindings[0] = out_tile_gpu;
                bindings[1] = dummy_alpha_gpu;
                bindings[2] = out_gpu;

                std::vector<ncnn::vk_constant_type> constants(13);
                constants[0].i = out_tile_gpu.w;
                constants[1].i = out_tile_gpu.h;
                constants[2].i = out_tile_gpu.cstep;
                constants[3].i = out_gpu.w;
                constants[4].i = out_gpu.h;
                constants[5].i = out_gpu.cstep;
                constants[6].i = xi * TILE_SIZE_X * scale;
                constants[7].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                constants[8].i = prepadding * scale;
                constants[9].i = prepadding * scale;
                constants[10].i = channels;
                constants[11].i = 0;
                constants[12].i = 0;

                ncnn::VkMat dispatcher;
                dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                dispatcher.h = out_gpu.h;
                dispatcher.c = channels;

                cmd.record_pipeline(realesrgan_postproc, bindings, constants, dispatcher);
            }

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        // ---- 밴드 하나를 최종 cv::Mat 출력 버퍼로 다운로드 ----
        {
            ncnn::Mat out;
            if (opt.use_fp16_storage && opt.use_int8_storage)
            {
                out = ncnn::Mat(out_gpu.w, out_gpu.h,
                                 outimage.data + (size_t)yi * scale * TILE_SIZE_Y * outimage.step,
                                 (size_t)channels, 1);
            }

            cmd.record_clone(out_gpu, out, opt);
            cmd.submit_and_wait();

            if (!(opt.use_fp16_storage && opt.use_int8_storage))
            {
                out.to_pixels(outimage.data + (size_t)yi * scale * TILE_SIZE_Y * outimage.step, ncnn::Mat::PIXEL_BGR);
            }
        }
    }

    net.vulkan_device()->reclaim_blob_allocator(blob_vkallocator);
    net.vulkan_device()->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}
