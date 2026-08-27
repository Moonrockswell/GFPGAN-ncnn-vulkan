#ifndef REALCUGAN_H
#define REALCUGAN_H

#include <string>
#include <opencv2/opencv.hpp>

// ncnn
#include "net.h"
#include "gpu.h"
#include "layer.h"

// RealCUGAN(nihui/realcugan-ncnn-vulkan)을 이 프로젝트의 GPU 셰이더 타일
// 파이프라인 컨벤션(RealESRGAN/Waifu2xDenoise와 동일한 구조)에 맞춰 이식한
// 축소판입니다. 원본과 달리 다음은 지원하지 않습니다(이 프로젝트가 다루는
// 용도에 불필요하다고 판단해 의도적으로 뺌):
//   - syncgap(타일 이음매 동기화, SE 레이어 전용 기법): RealESRGAN 쪽에
//     대응되는 훅이 없어서 애초에 이식 대상에서 제외하기로 함(설계 논의 참고)
//   - tta_mode(8-way test-time augmentation): 셰이더 미컴파일, 프로젝트
//     전체적으로 RealESRGAN도 동일하게 tta 미지원
//   - 알파 채널(RGBA): gfpgan-ncnn-vulkan-demo.cpp가 항상 3채널 BGR로
//     읽어오므로 불필요
//   - noise 레벨(-n): 이 프로젝트는 waifu2x(cunet)의 -AiDenoise가 별도
//     디노이즈 단계를 담당하므로, RealCUGAN 쪽은 no-denoise/conservative
//     가중치 파일만 채택함(denoise3x류는 제외) - 어떤 가중치를 쓸지는
//     load()에 넘기는 모델 파일 경로로 결정되고, 클래스 자체는 noise
//     레벨 개념을 모름
//
// scale은 2/3/4를 모두 지원합니다(모델 파일에 따라 결정). scale==4일 때만
// realcugan_4x_postproc(원본 이미지의 4배 최근접 업샘플에 잔차를 더하는
// 방식)을 쓰고, 2/3은 realcugan_postproc(잔차 없이 직접 출력)을 씁니다 -
// 공식 구현과 동일한 분기입니다.
class RealCUGAN
{
public:
    RealCUGAN(int gpuid = 0, bool force_fp32 = false);
    ~RealCUGAN();

    int load(const std::string& parampath, const std::string& modelpath);

    // inimage: cv::imread(..., 1)로 읽은 8UC3 BGR 이미지
    // outimage: scale배 업스케일된 8UC3 BGR 이미지
    int tile_process(const cv::Mat& inimage, cv::Mat& outimage);

public:
    int scale;       // 2, 3, 4 중 하나 (로드한 모델 파일에 맞게 호출부에서 설정)
    int tile_size;   // RealESRGAN과 동일한 이름 컨벤션
    int tile_pad;    // RealESRGAN과 동일한 이름 컨벤션 (내부적으로 prepadding으로 사용)

private:
    ncnn::Net net;
    ncnn::Pipeline* realcugan_preproc;
    ncnn::Pipeline* realcugan_postproc;
    ncnn::Pipeline* realcugan_4x_postproc;
};

#endif // REALCUGAN_H
