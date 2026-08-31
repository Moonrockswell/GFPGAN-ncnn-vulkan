// waifu2x cunet noise-only(scale 안 바뀜) 모델을 GPU(Vulkan) 타일
// 파이프라인으로 돌리는 구현.
//
// realesrgan.cpp와 거의 동일한 구조입니다(같은 방식으로 타일 crop/색공간
// 변환/GPU<->CPU 전송을 GPU 셰이더 안에서 처리). 차이점은 헤더 주석 및
// 아래 각 지점의 주석 참고.
//
// waifu2x_preproc.comp / waifu2x_postproc.comp 는 nihui/waifu2x-ncnn-vulkan
// 공식 배포판(https://github.com/nihui/waifu2x-ncnn-vulkan)의 셰이더를
// 이 프로젝트의 realesrgan_preproc.comp/realesrgan_postproc.comp 형식(로컬
// sfp 매크로 정의 등)에 맞춰 옮긴 것입니다. postproc은 순수 비정규화+타일
// 복사 로직이라 realesrgan_postproc.comp와 100% 동일해서 그대로 재사용
// (파일명만 waifu2x_postproc.comp로 복사)했고, preproc은 타일 경계 바깥을
// 채우는 방식만 다릅니다(waifu2x 공식 배포판 기준 클램프 패딩; RealESRGAN은
// 반사 패딩).

#include "waifu2x_denoise.h"

#include <algorithm>
#include <vector>

static const uint32_t waifu2x_preproc_spv_data[] = {
    #include "waifu2x_preproc.spv.hex.h"
};
static const uint32_t waifu2x_preproc_fp16s_spv_data[] = {
    #include "waifu2x_preproc_fp16s.spv.hex.h"
};
static const uint32_t waifu2x_preproc_int8s_spv_data[] = {
    #include "waifu2x_preproc_int8s.spv.hex.h"
};
static const uint32_t waifu2x_postproc_spv_data[] = {
    #include "waifu2x_postproc.spv.hex.h"
};
static const uint32_t waifu2x_postproc_fp16s_spv_data[] = {
    #include "waifu2x_postproc_fp16s.spv.hex.h"
};
static const uint32_t waifu2x_postproc_int8s_spv_data[] = {
    #include "waifu2x_postproc_int8s.spv.hex.h"
};

Waifu2xDenoise::Waifu2xDenoise(int gpuid, bool force_fp32)
{
    net.opt.use_vulkan_compute = true;
    net.opt.use_fp16_packed = !force_fp32;
    net.opt.use_fp16_storage = !force_fp32;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_int8_storage = !force_fp32;
    net.opt.use_int8_arithmetic = false;

    net.set_vulkan_device(gpuid);

    waifu2x_preproc = 0;
    waifu2x_postproc = 0;

    // -nn CLI 파싱 쪽(gfpgan-ncnn-vulkan-demo.cpp)에서 모델 로드 직후
    // 다시 설정하지만, 안전한 기본값을 둡니다.
    tile_size = 400;
    prepadding = 28; // cunet, scale=1(노이즈 전용) 케이스의 공식 기본값
}

Waifu2xDenoise::~Waifu2xDenoise()
{
    delete waifu2x_preproc;
    delete waifu2x_postproc;

    net.clear();
}

int Waifu2xDenoise::load(const std::string& parampath, const std::string& modelpath)
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

    // preprocess / postprocess GPU 파이프라인 초기화 (realesrgan.cpp와 동일 패턴)
    {
        std::vector<ncnn::vk_specialization_type> specializations(1);
        specializations[0].i = 1; // 항상 cv::imread(BGR) 입력이므로 bgr=1 고정

        waifu2x_preproc = new ncnn::Pipeline(net.vulkan_device());
        waifu2x_preproc->set_optimal_local_size_xyz(32, 32, 3);

        waifu2x_postproc = new ncnn::Pipeline(net.vulkan_device());
        waifu2x_postproc->set_optimal_local_size_xyz(32, 32, 3);

        if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
        {
            waifu2x_preproc->create(waifu2x_preproc_int8s_spv_data, sizeof(waifu2x_preproc_int8s_spv_data), specializations);
            waifu2x_postproc->create(waifu2x_postproc_int8s_spv_data, sizeof(waifu2x_postproc_int8s_spv_data), specializations);
        }
        else if (net.opt.use_fp16_storage)
        {
            waifu2x_preproc->create(waifu2x_preproc_fp16s_spv_data, sizeof(waifu2x_preproc_fp16s_spv_data), specializations);
            waifu2x_postproc->create(waifu2x_postproc_fp16s_spv_data, sizeof(waifu2x_postproc_fp16s_spv_data), specializations);
        }
        else
        {
            waifu2x_preproc->create(waifu2x_preproc_spv_data, sizeof(waifu2x_preproc_spv_data), specializations);
            waifu2x_postproc->create(waifu2x_postproc_spv_data, sizeof(waifu2x_postproc_spv_data), specializations);
        }
    }

    return 0;
}

int Waifu2xDenoise::tile_process(const cv::Mat& inimage, cv::Mat& outimage)
{
    bool gpu_submit_failed = false;
    // RealESRGAN::tile_process와 거의 동일하되, scale이 항상 1이라 출력
    // 크기가 입력과 같고(outimage.create(h, w, ...)), prepadding 값과
    // 셰이더 파이프라인/블롭 이름만 다릅니다.
    const int channels = 3;
    const int w = inimage.cols;
    const int h = inimage.rows;
    const int scale = 1;

    outimage.create(h, w, CV_8UC3);

    const unsigned char* pixeldata = inimage.isContinuous() ? inimage.data : nullptr;
    cv::Mat inimage_cont;
    if (!pixeldata)
    {
        inimage_cont = inimage.clone();
        pixeldata = inimage_cont.data;
    }

    // tile_size <= 0 은 "타일 안 씀(이미지 전체를 타일 1개로 처리)"을 의미합니다.
    const int TILE_SIZE_X = tile_size > 0 ? tile_size : w;
    const int TILE_SIZE_Y = tile_size > 0 ? tile_size : h;
    const int pad = prepadding;

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
        // cunet은 내부적으로 2번의 다운샘플(1/4)+업샘플 구조라, 네트워크에
        // 들어가는 타일의 가로/세로가 4의 배수가 아니면 출력 크기가 입력과
        // 어긋나서 타일 이음매가 깨집니다. 공식 waifu2x-ncnn-vulkan(waifu2x.cpp)
        // 과 동일하게, 아래(bottom)쪽에 4의 배수를 맞추기 위한 여분 패딩을
        // 추가로 더합니다. (위/왼쪽 prepadding은 그대로 pad 고정)
        int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;
        int pad_bottom = pad + ((tile_h_nopad + 3) / 4 * 4 - tile_h_nopad);

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - pad, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + pad_bottom, h);

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
                if (cmd.submit_and_wait() != 0) { fprintf(stderr, "Error: GPU command submission failed - tile result may be corrupted or blank\n"); gpu_submit_failed = true; }
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
            // preproc: 타일 crop(경계 바깥은 클램프 패딩) + BGR->RGB
            ncnn::VkMat in_tile_gpu;
            {
                // y축과 동일한 이유로, 우측(right)에도 4의 배수 정렬용 여분
                // 패딩을 추가합니다. tile_x0/좌측 기준 pad는 그대로 둡니다
                // (아래 constants[7] pad_left도 원래 pad 값 그대로 사용 -
                //  우측으로 늘어난 여분은 in_tile_gpu.w가 커지는 것만으로
                //  자연스럽게 반영되고, 크롭 시작 좌표 계산에는 영향 없음).
                int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;
                int pad_right = pad + ((tile_w_nopad + 3) / 4 * 4 - tile_w_nopad);

                int tile_x0 = xi * TILE_SIZE_X - pad;
                int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + pad_right;
                int tile_y0 = yi * TILE_SIZE_Y - pad;
                int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + pad_bottom;

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
                constants[6].i = pad;
                constants[7].i = pad;
                constants[8].i = xi * TILE_SIZE_X;
                constants[9].i = std::min(yi * TILE_SIZE_Y, pad);
                constants[10].i = channels;
                constants[11].i = 0;
                constants[12].i = 0;

                ncnn::VkMat dispatcher;
                dispatcher.w = in_tile_gpu.w;
                dispatcher.h = in_tile_gpu.h;
                dispatcher.c = channels;

                cmd.record_pipeline(waifu2x_preproc, bindings, constants, dispatcher);
            }

            // waifu2x(cunet noise-only) 네트워크 추론
            // 공식 배포판과 동일하게 블롭 이름은 "Input1"/"Eltwise4"
            // (realesrgan.cpp는 "input"/"output")
            ncnn::VkMat out_tile_gpu;
            {
                ncnn::Extractor ex = net.create_extractor();

                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("Input1", in_tile_gpu);
                ex.extract("Eltwise4", out_tile_gpu, cmd);
            }

            // postproc: 타일을 out_gpu의 제 위치에 합성 (GPU 셰이더 내부에서 RGB->BGR)
            // RealESRGAN과 달리 crop_x/crop_y가 없음 - out_tile_gpu가 이미
            // valid conv를 거쳐 프리패딩이 제거된 정확한 크기로 나오기 때문에
            // 그대로 offset_x만큼 이동해서 복사하면 됨 (waifu2x_postproc.comp 참고)
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
                constants[6].i = xi * TILE_SIZE_X * scale;                                              // offset_x
                constants[7].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);   // gx_max
                constants[8].i = channels;
                constants[9].i = 0;  // alphaw (알파 미지원)
                constants[10].i = 0; // alphah

                ncnn::VkMat dispatcher;
                dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                dispatcher.h = out_gpu.h;
                dispatcher.c = channels;

                cmd.record_pipeline(waifu2x_postproc, bindings, constants, dispatcher);
            }

            if (xtiles > 1)
            {
                if (cmd.submit_and_wait() != 0) { fprintf(stderr, "Error: GPU command submission failed - tile result may be corrupted or blank\n"); gpu_submit_failed = true; }
                cmd.reset();
            }

            fprintf(stderr, "ai denoise %.2f%%\n", (float)(yi * xtiles + xi) / (ytiles * xtiles) * 100);
        }

        // 다운로드: out_gpu -> outimage의 해당 행 구간
        {
            unsigned char* out_row_ptr = outimage.data + (size_t)yi * scale * TILE_SIZE_Y * w * scale * channels;

            ncnn::Mat out;
            if (opt.use_fp16_storage && opt.use_int8_storage)
            {
                out = ncnn::Mat(out_gpu.w, out_gpu.h, (void*)out_row_ptr, (size_t)channels, 1);
                cmd.record_clone(out_gpu, out, opt);
                if (cmd.submit_and_wait() != 0) { fprintf(stderr, "Error: GPU command submission failed - tile result may be corrupted or blank\n"); gpu_submit_failed = true; }
            }
            else
            {
                cmd.record_clone(out_gpu, out, opt);
                if (cmd.submit_and_wait() != 0) { fprintf(stderr, "Error: GPU command submission failed - tile result may be corrupted or blank\n"); gpu_submit_failed = true; }
                out.to_pixels(out_row_ptr, ncnn::Mat::PIXEL_BGR);
            }
        }
    }

    net.vulkan_device()->reclaim_blob_allocator(blob_vkallocator);
    net.vulkan_device()->reclaim_staging_allocator(staging_vkallocator);

    return gpu_submit_failed ? -1 : 0;
}
