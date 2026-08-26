#ifndef REALESRGAN_H
#define REALESRGAN_H

#include <string>
#include <opencv2/opencv.hpp>

// ncnn
#include "net.h"
#include "gpu.h"
#include "layer.h"

// GPU 셰이더 기반 타일 파이프라인으로 교체한 버전.
//
// 예전 구현은 타일 crop / 색공간 변환(BGR<->RGB) / GPU 결과를 다시 cv::Mat으로
// 옮기는 작업을 전부 CPU에서(그것도 픽셀 단위 이중 for문으로) 처리했습니다.
// 이 버전은 그 세 단계를 realesrgan-ncnn-vulkan 공식 배포판과 동일한 Vulkan
// compute shader(realesrgan_preproc.comp / realesrgan_postproc.comp)로 옮겨서
// GPU 안에서 끝내고, CPU<->GPU 전송은 타일당 업로드 1번 / 다운로드 1번으로
// 줄였습니다.
//
// 공개 인터페이스(scale, tile_size, tile_pad, load(), tile_process())는
// gfpgan-ncnn-vulkan-demo.cpp 쪽 호출부를 그대로 유지할 수 있도록 예전과
// 동일하게 맞췄습니다.
class RealESRGAN
{
public:
    RealESRGAN(int gpuid = 0, bool tta_mode = false, bool force_fp32 = false);
    ~RealESRGAN();

    int load(const std::string& parampath, const std::string& modelpath);

    // inimage: cv::imread(..., 1)로 읽은 8UC3 BGR 이미지
    // outimage: scale배 업스케일된 8UC3 BGR 이미지
    int tile_process(const cv::Mat& inimage, cv::Mat& outimage);

public:
    // realesrgan parameters (기존 코드와 이름/의미 동일하게 유지)
    int scale;
    int tile_size;   // 예전 이름 유지 (내부적으로 GPU 타일 한 변의 크기로 사용)
    int tile_pad;    // 예전 이름 유지 (내부적으로 prepadding으로 사용)

private:
    ncnn::Net net;
    ncnn::Pipeline* realesrgan_preproc;
    ncnn::Pipeline* realesrgan_postproc;
    bool tta_mode;
};

#endif // REALESRGAN_H
