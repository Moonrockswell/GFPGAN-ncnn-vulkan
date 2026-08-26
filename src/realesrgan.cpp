// realesrgan implemented with ncnn library
//
// GPU 셰이더 기반 타일 파이프라인 버전 (재구성본).
//
// 이 파일은 원본이 실수로 덮어써져서, 다음 근거를 바탕으로 xinntao의 공식
// realesrgan-ncnn-vulkan 소스(https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)의
// 타일 처리 파이프라인을 참고해 이 프로젝트의 실제 헤더/셰이더/CMakeLists에
// 맞춰 다시 작성했습니다:
//   - realesrgan_preproc.comp / realesrgan_postproc.comp 가 공식 배포판과
//     100% 동일 (diff 결과 완전 일치)
//   - CMakeLists.txt가 이 두 셰이더만 fp32/fp16s/int8s 세 변형으로 컴파일
//     (tta 변형은 컴파일하지 않음 -> tta 미지원)
//   - realesrgan.h가 공식 헤더에서 bicubic_2x/3x/4x(알파 채널용), wstring
//     오버로드를 제거하고, process()->tile_process(cv::Mat 기반)로,
//     tilesize/prepadding -> tile_size/tile_pad 로 이름을 맞춘 축소판
//   - 이전에 실제로 겪었던 에러 로그(find_blob_index_by_name input failed,
//     "data"를 써보라는 힌트)로 볼 때, 원본은 공식 소스의 "data" 대신
//     "input" 블롭 이름을 썼던 것으로 확인됨 -> 아래에서 "input"/"output" 사용
//
// 알파 채널(RGBA) 입력은 지원하지 않습니다 (헤더에 bicubic_2x/3x/4x 멤버가
// 없어서 알파 채널 업스케일 경로 자체가 빠져 있음). gfpgan-ncnn-vulkan-demo.cpp
// 쪽에서 cv::imread(..., 1)로 항상 3채널 BGR로 읽어오므로 문제 없습니다.
// TTA(8-way test-time augmentation)도 셰이더가 컴파일되어 있지 않아 미지원이며,
// tta_mode 멤버는 헤더와의 호환을 위해서만 남겨두고 실제로는 사용하지 않습니다.

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

RealESRGAN::RealESRGAN(int gpuid, bool _tta_mode, bool force_fp32)
{
    net.opt.use_vulkan_compute = true;
    // GTX 660 등 구형(Kepler) GPU는 fp16/int8 storage buffer 확장(VK_KHR_16bit_storage/
    // VK_KHR_8bit_storage)을 드라이버가 불완전하게 지원해서, 겉으론 에러 없이 실행되지만
    // 타일 결과가 조용히 깨지는(반복/뒤섞임) 사례가 있습니다. -Fp32Only 1로 이 경로를
    // 강제로 끄고 검증할 수 있게 옵션을 추가했습니다.
    net.opt.use_fp16_packed = !force_fp32;
    net.opt.use_fp16_storage = !force_fp32;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_int8_storage = !force_fp32;
    net.opt.use_int8_arithmetic = false;

    net.set_vulkan_device(gpuid);

    realesrgan_preproc = 0;
    realesrgan_postproc = 0;
    tta_mode = _tta_mode; // 셰이더 미컴파일로 현재 미사용, 헤더 호환용 보관

    // 아래 세 값은 -rn 으로 어떤 모델이 로드되는지에 따라
    // gfpgan-ncnn-vulkan-demo.cpp의 main()에서 다시 설정됩니다.
    scale = 4;
    tile_size = 400;
    tile_pad = 10;
}

RealESRGAN::~RealESRGAN()
{
    delete realesrgan_preproc;
    delete realesrgan_postproc;

    net.clear();
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

    // preprocess / postprocess GPU 파이프라인 초기화
    {
        std::vector<ncnn::vk_specialization_type> specializations(1);
        // OpenCV(cv::imread)는 항상 BGR로 읽어오므로, 플랫폼(_WIN32) 여부와
        // 무관하게 항상 bgr=1로 고정합니다 (공식 소스는 stb_image/webp 기준으로
        // 플랫폼별 분기가 있었지만, 이 프로젝트는 항상 OpenCV를 씁니다).
        specializations[0].i = 1;

        realesrgan_preproc = new ncnn::Pipeline(net.vulkan_device());
        realesrgan_preproc->set_optimal_local_size_xyz(32, 32, 3);

        realesrgan_postproc = new ncnn::Pipeline(net.vulkan_device());
        realesrgan_postproc->set_optimal_local_size_xyz(32, 32, 3);

        if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
        {
            realesrgan_preproc->create(realesrgan_preproc_int8s_spv_data, sizeof(realesrgan_preproc_int8s_spv_data), specializations);
            realesrgan_postproc->create(realesrgan_postproc_int8s_spv_data, sizeof(realesrgan_postproc_int8s_spv_data), specializations);
        }
        else if (net.opt.use_fp16_storage)
        {
            realesrgan_preproc->create(realesrgan_preproc_fp16s_spv_data, sizeof(realesrgan_preproc_fp16s_spv_data), specializations);
            realesrgan_postproc->create(realesrgan_postproc_fp16s_spv_data, sizeof(realesrgan_postproc_fp16s_spv_data), specializations);
        }
        else
        {
            realesrgan_preproc->create(realesrgan_preproc_spv_data, sizeof(realesrgan_preproc_spv_data), specializations);
            realesrgan_postproc->create(realesrgan_postproc_spv_data, sizeof(realesrgan_postproc_spv_data), specializations);
        }
    }

    return 0;
}

int RealESRGAN::tile_process(const cv::Mat& inimage, cv::Mat& outimage)
{
    // 이 프로젝트는 항상 3채널 BGR(cv::imread(..., 1))만 다룹니다.
    const int channels = 3;
    const int w = inimage.cols;
    const int h = inimage.rows;

    outimage.create(h * scale, w * scale, CV_8UC3);

    const unsigned char* pixeldata = inimage.isContinuous() ? inimage.data : nullptr;
    cv::Mat inimage_cont;
    if (!pixeldata)
    {
        // ROI 등으로 메모리가 연속이 아닌 경우를 대비한 안전장치
        inimage_cont = inimage.clone();
        pixeldata = inimage_cont.data;
    }

    // tile_size <= 0 은 "타일 안 씀(이미지 전체를 타일 1개로 처리)"을 의미합니다.
    // 예전에는 이 값을 그대로 나눗셈에 써서 0으로 나누기 크래시가 났습니다.
    const int TILE_SIZE_X = tile_size > 0 ? tile_size : w;
    const int TILE_SIZE_Y = tile_size > 0 ? tile_size : h;
    const int prepadding = tile_pad;

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
        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding, h);

        ncnn::Mat in;
        if (opt.use_fp16_storage && opt.use_int8_storage)
        {
            in = ncnn::Mat(w, (in_tile_y1 - in_tile_y0), (void*)(pixeldata + (size_t)in_tile_y0 * w * channels), (size_t)channels, 1);
        }
        else
        {
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

        int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
        int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h);

        ncnn::VkMat out_gpu;
        if (opt.use_fp16_storage && opt.use_int8_storage)
        {
            out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, (size_t)channels, 1, blob_vkallocator);
        }
        else
        {
            out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, channels, (size_t)4u, 1, blob_vkallocator);
        }

        for (int xi = 0; xi < xtiles; xi++)
        {
            // preproc: 타일 crop + BGR->RGB (GPU 셰이더 내부에서 처리)
            ncnn::VkMat in_tile_gpu;
            {
                int tile_x0 = xi * TILE_SIZE_X - prepadding;
                int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding;
                int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding;

                in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);

                ncnn::VkMat dummy_alpha; // 알파 채널 미지원 - 항상 빈 버퍼

                std::vector<ncnn::VkMat> bindings(3);
                bindings[0] = in_gpu;
                bindings[1] = in_tile_gpu;
                bindings[2] = dummy_alpha;

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
                constants[11].i = 0;
                constants[12].i = 0;

                ncnn::VkMat dispatcher;
                dispatcher.w = in_tile_gpu.w;
                dispatcher.h = in_tile_gpu.h;
                dispatcher.c = channels;

                cmd.record_pipeline(realesrgan_preproc, bindings, constants, dispatcher);
            }

            // realesrgan 네트워크 추론
            ncnn::VkMat out_tile_gpu;
            {
                ncnn::Extractor ex = net.create_extractor();

                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("input", in_tile_gpu);
                ex.extract("output", out_tile_gpu, cmd);
            }

            // postproc: 타일을 out_gpu의 제 위치에 합성 (GPU 셰이더 내부에서 RGB->BGR)
            {
                ncnn::VkMat dummy_alpha;

                std::vector<ncnn::VkMat> bindings(3);
                bindings[0] = out_tile_gpu;
                bindings[1] = dummy_alpha;
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

            fprintf(stderr, "background upscale %.2f%%\n", (float)(yi * xtiles + xi) / (ytiles * xtiles) * 100);
        }

        // 다운로드: out_gpu -> outimage의 해당 행 구간
        {
            unsigned char* out_row_ptr = outimage.data + (size_t)yi * scale * TILE_SIZE_Y * w * scale * channels;

            ncnn::Mat out;
            if (opt.use_fp16_storage && opt.use_int8_storage)
            {
                out = ncnn::Mat(out_gpu.w, out_gpu.h, (void*)out_row_ptr, (size_t)channels, 1);
                cmd.record_clone(out_gpu, out, opt);
                cmd.submit_and_wait();
            }
            else
            {
                cmd.record_clone(out_gpu, out, opt);
                cmd.submit_and_wait();
                out.to_pixels(out_row_ptr, ncnn::Mat::PIXEL_BGR);
            }
        }
    }

    net.vulkan_device()->reclaim_blob_allocator(blob_vkallocator);
    net.vulkan_device()->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}
