#ifndef WAIFU2X_DENOISE_H
#define WAIFU2X_DENOISE_H

#include <string>
#include <opencv2/opencv.hpp>

// ncnn
#include "net.h"
#include "gpu.h"
#include "layer.h"

// waifu2x cunet 계열의 "노이즈 제거 전용"(noise-only, 배율 안 바뀜) 모델을
// 돌리기 위한 클래스. -nn 옵션용. RealESRGAN 클래스(realesrgan.h/.cpp)와
// 뼈대는 동일하되(GPU 셰이더 기반 타일 파이프라인, cv::Mat 인터페이스),
// waifu2x-ncnn-vulkan 공식 배포판(https://github.com/nihui/waifu2x-ncnn-vulkan)
// 과 동일하게 다음 두 가지가 다릅니다:
//   - 추론 블롭 이름이 "Input1" / "Eltwise4" (RealESRGAN은 "input"/"output")
//   - 전용 프리프로세스 셰이더 waifu2x_preproc.comp 사용. RealESRGAN의
//     realesrgan_preproc.comp가 타일 경계 바깥을 반사(mirror) 패딩하는 것과
//     달리, waifu2x 공식 배포판은 클램프(가장자리 픽셀 반복) 패딩을 씁니다.
//     postproc은 순수 비정규화+복사 로직이라 realesrgan_postproc.comp를
//     그대로 재사용합니다(waifu2x_postproc.comp로 이름만 복사).
//
// scale은 항상 1로 고정됩니다(모델 자체가 배율을 바꾸지 않는 noise-only
// 버전 - noise1_model.bin처럼 "scale2.0x" 접미사가 없는 모델 파일 사용).
// 알파 채널/TTA는 RealESRGAN 클래스와 같은 이유로 미지원입니다(이 프로젝트는
// 항상 cv::imread(..., 1)로 3채널 BGR만 다룸).
class Waifu2xDenoise
{
public:
    Waifu2xDenoise(int gpuid = 0, bool force_fp32 = false);
    ~Waifu2xDenoise();

    int load(const std::string& parampath, const std::string& modelpath);

    // inimage: cv::imread(..., 1)로 읽은 8UC3 BGR 이미지
    // outimage: 같은 크기(scale=1), 노이즈만 제거된 8UC3 BGR 이미지
    int tile_process(const cv::Mat& inimage, cv::Mat& outimage);

public:
    int tile_size;   // RealESRGAN::tile_size와 동일한 의미(GPU 타일 한 변의 크기)
    int prepadding;  // RealESRGAN::tile_pad와 동일한 역할.
                      // waifu2x-ncnn-vulkan 공식 배포판의 cunet, scale=1(노이즈 전용)
                      // 케이스 공식 기본값 = 28 (main.cpp의 prepadding 결정 로직 참고)

private:
    ncnn::Net net;
    ncnn::Pipeline* waifu2x_preproc;
    ncnn::Pipeline* waifu2x_postproc;
};

#endif // WAIFU2X_DENOISE_H
