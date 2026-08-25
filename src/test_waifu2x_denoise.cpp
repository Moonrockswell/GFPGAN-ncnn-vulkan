// Waifu2xDenoise 클래스만 단독으로 돌려보는 검증용 도구.
// GFPGAN/RealESRGAN 모델 없이도 -nn 셰이더/파이프라인 로직만 확인 가능.
//
// 사용법: ./test_waifu2x_denoise <noise_level 1~3> <model_dir> <out.png>
// 합성 노이즈가 낀 테스트 이미지를 만들어 처리한 뒤, 처리 전/후 표준편차와
// 원본(무노이즈) 대비 오차를 출력합니다.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <opencv2/opencv.hpp>

#include "waifu2x_denoise.h"

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "usage: %s <noise_level 1-3> <model_dir> <out.png>\n", argv[0]);
        return 1;
    }

    int noiseLevel = atoi(argv[1]);
    std::string modelDir = argv[2];
    std::string outPath = argv[3];

    // 256x256 그라디언트+도형 합성 이미지(= "원본, 노이즈 없음")를 만들고,
    // 여기에 가우시안 노이즈를 얹어 "노이즈 낀 입력"을 만든다.
    const int W = 256, H = 256;
    cv::Mat clean(H, W, CV_8UC3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            clean.at<cv::Vec3b>(y, x) = cv::Vec3b((uchar)(x), (uchar)(y), 128);
    cv::circle(clean, cv::Point(128, 128), 60, cv::Scalar(20, 200, 60), -1);
    cv::rectangle(clean, cv::Point(20, 180), cv::Point(100, 230), cv::Scalar(220, 60, 20), -1);

    cv::Mat noise(H, W, CV_32FC3);
    cv::randn(noise, 0, 25.0); // 표준편차 25짜리 가우시안 노이즈
    cv::Mat clean32f, noisy32f, noisy;
    clean.convertTo(clean32f, CV_32FC3);
    noisy32f = clean32f + noise;
    noisy32f.convertTo(noisy, CV_8UC3);

    cv::imwrite("test_clean.png", clean);
    cv::imwrite("test_noisy.png", noisy);

    ncnn::create_gpu_instance();

    cv::Mat denoised;
    int ret = 0;
    {
        // Waifu2xDenoise(net 포함, GPU 디바이스 자원을 들고 있음)가
        // destroy_gpu_instance()보다 먼저 소멸되도록 스코프로 감싼다.
        // (순서를 반대로 하면 net.clear()가 이미 파괴된 vulkan device를
        // 참조해서 세그폴트 - 클래스 자체 버그가 아니라 호출 순서 버그)
        Waifu2xDenoise denoiser(ncnn::get_default_gpu_index());
        denoiser.tile_size = 150;
        denoiser.prepadding = 28;

        std::string paramPath = modelDir + "/noise" + std::to_string(noiseLevel) + "_model.param";
        std::string binPath = modelDir + "/noise" + std::to_string(noiseLevel) + "_model.bin";
        fprintf(stderr, "loading %s\n", paramPath.c_str());

        ret = denoiser.load(paramPath, binPath);
        if (ret == 0)
        {
            denoiser.tile_process(noisy, denoised);
        }
        else
        {
            fprintf(stderr, "load failed\n");
        }
    }

    ncnn::destroy_gpu_instance();

    if (ret != 0)
        return 1;

    cv::imwrite(outPath, denoised);

    if (denoised.empty() || denoised.cols != W || denoised.rows != H)
    {
        fprintf(stderr, "FAIL: output size mismatch (%dx%d)\n", denoised.cols, denoised.rows);
        return 1;
    }

    // 정량 비교: (a) 노이즈 낀 입력 vs 원본 오차, (b) 디노이즈 결과 vs 원본 오차.
    // (b)가 (a)보다 확실히 작아야 "진짜로 노이즈를 줄였다"고 볼 수 있음.
    cv::Mat diffNoisy, diffDenoised;
    cv::absdiff(noisy, clean, diffNoisy);
    cv::absdiff(denoised, clean, diffDenoised);

    cv::Scalar meanAbsErrNoisy = cv::mean(diffNoisy);
    cv::Scalar meanAbsErrDenoised = cv::mean(diffDenoised);

    double maeNoisy = (meanAbsErrNoisy[0] + meanAbsErrNoisy[1] + meanAbsErrNoisy[2]) / 3.0;
    double maeDenoised = (meanAbsErrDenoised[0] + meanAbsErrDenoised[1] + meanAbsErrDenoised[2]) / 3.0;

    printf("mean abs error (noisy input   vs clean): %.3f\n", maeNoisy);
    printf("mean abs error (denoised output vs clean): %.3f\n", maeDenoised);

    if (maeDenoised < maeNoisy)
        printf("RESULT: PASS - denoised output is closer to the clean image than the noisy input\n");
    else
        printf("RESULT: FAIL - denoised output did NOT improve over the noisy input\n");

    return 0;
}
