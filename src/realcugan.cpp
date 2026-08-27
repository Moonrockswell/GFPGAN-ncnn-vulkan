// realcugan implemented with ncnn library
//
// nihui/realcugan-ncnn-vulkan(공식 소스, 20220728 태그)의 GPU 셰이더 타일
// 파이프라인을 이 프로젝트의 realesrgan.cpp와 동일한 구조/컨벤션으로 이식한
// 버전입니다. syncgap/tta/알파채널/noise 레벨은 헤더 주석에 적은 이유로
// 의도적으로 뺐습니다.
//
// realcugan_preproc.comp는 realesrgan_preproc.comp와 100% 동일해서(diff
// 완전 일치) 셰이더 자체는 그대로 재사용 가능하지만, 클래스 경계를 명확히
// 하기 위해 이 클래스 전용으로 별도 컴파일합니다.
//
// postproc은 scale에 따라 두 갈래로 나뉩니다(공식 구현과 동일한 분기):
//   - scale == 4: realcugan_4x_postproc 사용. 이 모델은 "원본 이미지의
//     4배 최근접 업샘플 + 신경망이 예측한 잔차"를 더하는 방식이라, 원본
//     타일(in_gpu)을 postproc 셰이더에도 같이 넘겨야 합니다.
//   - scale == 2 또는 3: realcugan_postproc 사용. 신경망 출력을 그대로 씀
//     (잔차 없음).

#include "realcugan.h"

#include <algorithm>
#include <vector>

static const uint32_t realcugan_preproc_spv_data[] = {
    #include "realcugan_preproc.spv.hex.h"
};
static const uint32_t realcugan_preproc_fp16s_spv_data[] = {
    #include "realcugan_preproc_fp16s.spv.hex.h"
};
static const uint32_t realcugan_preproc_int8s_spv_data[] = {
    #include "realcugan_preproc_int8s.spv.hex.h"
};
static const uint32_t realcugan_postproc_spv_data[] = {
    #include "realcugan_postproc.spv.hex.h"
};
static const uint32_t realcugan_postproc_fp16s_spv_data[] = {
    #include "realcugan_postproc_fp16s.spv.hex.h"
};
static const uint32_t realcugan_postproc_int8s_spv_data[] = {
    #include "realcugan_postproc_int8s.spv.hex.h"
};
static const uint32_t realcugan_4x_postproc_spv_data[] = {
    #include "realcugan_4x_postproc.spv.hex.h"
};
static const uint32_t realcugan_4x_postproc_fp16s_spv_data[] = {
    #include "realcugan_4x_postproc_fp16s.spv.hex.h"
};
static const uint32_t realcugan_4x_postproc_int8s_spv_data[] = {
    #include "realcugan_4x_postproc_int8s.spv.hex.h"
};

RealCUGAN::RealCUGAN(int gpuid, bool force_fp32)
{
    net.opt.use_vulkan_compute = true;
    // RealESRGAN 쪽과 동일한 이유로 fp32 강제 옵션을 유지합니다
    // (구형 GPU의 fp16/int8 storage 확장 호환성 문제 대응용, -Fp32Only 참고).
    net.opt.use_fp16_packed = !force_fp32;
    net.opt.use_fp16_storage = !force_fp32;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_int8_storage = !force_fp32;
    net.opt.use_int8_arithmetic = false;

    net.set_vulkan_device(gpuid);

    realcugan_preproc = 0;
    realcugan_postproc = 0;
    realcugan_4x_postproc = 0;

    // 아래 값들은 -rn 으로 어떤 RealCUGAN 모델(up2x/up3x/up4x-*)이
    // 로드되는지에 따라 gfpgan-ncnn-vulkan-demo.cpp의 main()에서
    // 다시 설정됩니다.
    scale = 2;
    tile_size = 400;
    tile_pad = 10;
}

RealCUGAN::~RealCUGAN()
{
    delete realcugan_preproc;
    delete realcugan_postproc;
    delete realcugan_4x_postproc;

    net.clear();
}

int RealCUGAN::load(const std::string& parampath, const std::string& modelpath)
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

    {
        std::vector<ncnn::vk_specialization_type> specializations(1);
        // cv::imread는 항상 BGR로 읽어오므로 bgr=1로 고정합니다.
        specializations[0].i = 1;

        realcugan_preproc = new ncnn::Pipeline(net.vulkan_device());
        realcugan_preproc->set_optimal_local_size_xyz(32, 32, 3);

        realcugan_postproc = new ncnn::Pipeline(net.vulkan_device());
        realcugan_postproc->set_optimal_local_size_xyz(32, 32, 3);

        realcugan_4x_postproc = new ncnn::Pipeline(net.vulkan_device());
        realcugan_4x_postproc->set_optimal_local_size_xyz(32, 32, 3);

        if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
        {
            realcugan_preproc->create(realcugan_preproc_int8s_spv_data, sizeof(realcugan_preproc_int8s_spv_data), specializations);
            realcugan_postproc->create(realcugan_postproc_int8s_spv_data, sizeof(realcugan_postproc_int8s_spv_data), specializations);
            realcugan_4x_postproc->create(realcugan_4x_postproc_int8s_spv_data, sizeof(realcugan_4x_postproc_int8s_spv_data), specializations);
        }
        else if (net.opt.use_fp16_storage)
        {
            realcugan_preproc->create(realcugan_preproc_fp16s_spv_data, sizeof(realcugan_preproc_fp16s_spv_data), specializations);
            realcugan_postproc->create(realcugan_postproc_fp16s_spv_data, sizeof(realcugan_postproc_fp16s_spv_data), specializations);
            realcugan_4x_postproc->create(realcugan_4x_postproc_fp16s_spv_data, sizeof(realcugan_4x_postproc_fp16s_spv_data), specializations);
        }
        else
        {
            realcugan_preproc->create(realcugan_preproc_spv_data, sizeof(realcugan_preproc_spv_data), specializations);
            realcugan_postproc->create(realcugan_postproc_spv_data, sizeof(realcugan_postproc_spv_data), specializations);
            realcugan_4x_postproc->create(realcugan_4x_postproc_spv_data, sizeof(realcugan_4x_postproc_spv_data), specializations);
        }
    }

    return 0;
}

int RealCUGAN::tile_process(const cv::Mat& inimage, cv::Mat& outimage)
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
        inimage_cont = inimage.clone();
        pixeldata = inimage_cont.data;
    }

    // tile_size <= 0 은 "타일 안 씀"을 의미합니다 (RealESRGAN과 동일).
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
        // RealCUGAN 신경망 내부에 stride-2 다운샘플/업샘플이 있어서, 타일
        // 높이(패딩 포함)가 배율에 따라 2 또는 4의 배수로 맞아야 합니다.
        // (공식 구현과 동일한 보정 - 안 맞으면 Convolution/Deconvolution
        // 단계에서 텐서 크기 불일치로 실패합니다.)
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;
        int prepadding_bottom = prepadding;
        if (scale == 3)
        {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4)
        {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

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
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;
            int prepadding_right = prepadding;
            if (scale == 3)
            {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4)
            {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            // preproc: 타일 crop + BGR->RGB (realesrgan_preproc와 100% 동일한 셰이더)
            ncnn::VkMat in_tile_gpu;
            {
                int tile_x0 = xi * TILE_SIZE_X - prepadding;
                int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding_right;
                int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding_bottom;

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

                cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
            }

            // realcugan 네트워크 추론 (입력 블롭 "in0", 출력 블롭 "out0" - 공식 param 파일 기준)
            ncnn::VkMat out_tile_gpu;
            {
                ncnn::Extractor ex = net.create_extractor();

                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("in0", in_tile_gpu);
                ex.extract("out0", out_tile_gpu, cmd);
            }

            // postproc: scale==4는 원본 4배 최근접 업샘플 + 잔차, scale==2/3은 신경망 출력 그대로
            if (scale == 4)
            {
                ncnn::VkMat dummy_alpha;

                std::vector<ncnn::VkMat> bindings(4);
                bindings[0] = in_gpu;
                bindings[1] = out_tile_gpu;
                bindings[2] = dummy_alpha;
                bindings[3] = out_gpu;

                std::vector<ncnn::vk_constant_type> constants(16);
                constants[0].i = in_gpu.w;
                constants[1].i = in_gpu.h;
                constants[2].i = in_gpu.cstep;
                constants[3].i = out_tile_gpu.w;
                constants[4].i = out_tile_gpu.h;
                constants[5].i = out_tile_gpu.cstep;
                constants[6].i = out_gpu.w;
                constants[7].i = out_gpu.h;
                constants[8].i = out_gpu.cstep;
                constants[9].i = xi * TILE_SIZE_X;
                constants[10].i = std::min(yi * TILE_SIZE_Y, prepadding);
                constants[11].i = xi * TILE_SIZE_X * scale;
                constants[12].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                constants[13].i = channels;
                constants[14].i = 0;
                constants[15].i = 0;

                ncnn::VkMat dispatcher;
                dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                dispatcher.h = out_gpu.h;
                dispatcher.c = channels;

                cmd.record_pipeline(realcugan_4x_postproc, bindings, constants, dispatcher);
            }
            else
            {
                ncnn::VkMat dummy_alpha;

                std::vector<ncnn::VkMat> bindings(3);
                bindings[0] = out_tile_gpu;
                bindings[1] = dummy_alpha;
                bindings[2] = out_gpu;

                std::vector<ncnn::vk_constant_type> constants(11);
                constants[0].i = out_tile_gpu.w;
                constants[1].i = out_tile_gpu.h;
                constants[2].i = out_tile_gpu.cstep;
                constants[3].i = out_gpu.w;
                constants[4].i = out_gpu.h;
                constants[5].i = out_gpu.cstep;
                constants[6].i = xi * TILE_SIZE_X * scale;
                constants[7].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                constants[8].i = channels;
                constants[9].i = 0;
                constants[10].i = 0;

                ncnn::VkMat dispatcher;
                dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                dispatcher.h = out_gpu.h;
                dispatcher.c = channels;

                cmd.record_pipeline(realcugan_postproc, bindings, constants, dispatcher);
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
