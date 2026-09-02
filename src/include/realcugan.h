#ifndef REALCUGAN_H
#define REALCUGAN_H

#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

// ncnn
#include "net.h"
#include "gpu.h"
#include "layer.h"

// RealCUGAN(nihui/realcugan-ncnn-vulkan)을 이 프로젝트의 GPU 셰이더 타일
// 파이프라인 컨벤션(RealESRGAN/Waifu2xDenoise와 동일한 구조)에 맞춰 이식한
// 축소판입니다. 원본과 달리 다음은 지원하지 않습니다(이 프로젝트가 다루는
// 용도에 불필요하다고 판단해 의도적으로 뺌):
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
// syncgap(타일 이음매 동기화)는 지원합니다 - 처음엔 "RealESRGAN 쪽에
// 대응되는 훅이 없어서 이식 불가"로 판단해 뺐었는데, 그건 RealESRGAN 얘기고
// RealCUGAN 자신은 SE(Squeeze-Excitation, gap0~gap3) 레이어가 실제로 있어서
// 여기서는 그대로 적용 가능합니다(syncgap 멤버 변수 참고).
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
    int tile_delay_ms;  // RealESRGAN과 동일한 의미(-tiledelay 옵션)
    // 0(기본)=끔, 1=켬("rough" 버전). 타일 경계에서 SE(Squeeze-Excitation)
    // 레이어가 타일마다 따로 계산되면서 생기는 은은한 격자 얼룩을 없애줍니다.
    // 공식 구현의 4단계 순차 동기화(syncgap=1, 최고 품질) 대신, 한 번에
    // gap0~gap3를 다 뽑아서 평균 내는 "rough"(syncgap=2) 버전으로 구현했습니다
    // - 품질 차이는 거의 없지만 신경망을 2번만 통과하면 되어 구현이 훨씬
    // 간단합니다. 단, 타일마다 신경망을 사실상 2번 돌리는 셈이라 켜면 처리
    // 시간이 거의 2배로 늘고 GPU 부담도 커집니다.
    int syncgap;

private:
    ncnn::Net net;
    ncnn::Pipeline* realcugan_preproc;
    ncnn::Pipeline* realcugan_postproc;
    ncnn::Pipeline* realcugan_4x_postproc;

    // syncgap용 타일별 중간 특징(gap0~gap3) 캐시. nihui/realcugan-ncnn-vulkan의
    // FeatureCache를 그대로 이식(이 프로젝트는 tta_mode가 없어서 그 축만 뺌).
    class FeatureCache
    {
    public:
        void clear() { cache.clear(); }

        std::string make_key(int yi, int xi, const std::string& name) const
        {
            return std::to_string(yi) + "-" + std::to_string(xi) + "-" + name;
        }

        void load(int yi, int xi, const std::string& name, ncnn::VkMat& feat)
        {
            feat = cache[make_key(yi, xi, name)];
        }

        void save(int yi, int xi, const std::string& name, ncnn::VkMat& feat)
        {
            cache[make_key(yi, xi, name)] = feat;
        }

    public:
        std::map<std::string, ncnn::VkMat> cache;
    };

    int tile_process_plain(const cv::Mat& inimage, cv::Mat& outimage);
    int tile_process_syncgap(const cv::Mat& inimage, cv::Mat& outimage);

    // syncgap 1단계: 타일마다 신경망을 gap 블롭까지만 돌려서 캐시에 저장
    int syncgap_stage0(const cv::Mat& inimage, const std::vector<std::string>& outnames,
                        const ncnn::Option& opt, ncnn::VkAllocator* blob_vkallocator,
                        ncnn::VkAllocator* staging_vkallocator, FeatureCache& cache);
    // syncgap 2단계: 캐시에 모인 타일별 gap 값을 전체 평균 내서 모든 타일에 재주입
    int syncgap_sync(const std::vector<std::string>& names, const ncnn::Option& opt,
                      int xtiles, int ytiles, FeatureCache& cache);
    // syncgap 3단계: 동기화된 gap 값을 넣어 신경망을 마저 돌려 최종 출력 생성
    int syncgap_stage2(const cv::Mat& inimage, const std::vector<std::string>& names,
                        cv::Mat& outimage, const ncnn::Option& opt,
                        ncnn::VkAllocator* blob_vkallocator, ncnn::VkAllocator* staging_vkallocator,
                        FeatureCache& cache);
};

#endif // REALCUGAN_H
