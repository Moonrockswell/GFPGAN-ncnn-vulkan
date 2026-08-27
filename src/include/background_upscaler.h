#ifndef BACKGROUND_UPSCALER_H
#define BACKGROUND_UPSCALER_H

#include "realesrgan.h"
#include "realcugan.h"

// RealESRGAN(사진)과 RealCUGAN(애니메이션/일러스트, realesrgan-x4plus-anime를
// 대체)은 서로 다른 클래스라 restore_one_image() 등 호출부가 어느 쪽이든
// 신경 안 쓰고 tile_process()를 부를 수 있도록 감싸는 얇은 래퍼입니다.
// 상속 대신 포인터 전환 방식을 쓴 이유: 두 클래스 모두 ncnn::Net을 멤버로
// 들고 있어 가상함수 하나 위해 인터페이스를 새로 파는 것보다, 이렇게 얇게
// 감싸는 편이 기존 RealESRGAN/RealCUGAN 코드를 안 건드려도 돼서 더 안전합니다.
class BackgroundUpscaler
{
public:
    BackgroundUpscaler() : esrgan(nullptr), cugan(nullptr), scale(4), tile_size(400) {}

    // main()에서 -rn 값에 따라 둘 중 하나만 만들어서 넘겨줍니다.
    // 소유권은 계속 main() 쪽 unique_ptr/스택 변수가 가지고, 여기는 그냥
    // 참조만 보관합니다 (main()의 지역 변수 수명 > 이 래퍼 수명이 보장됨).
    void set_backend(RealESRGAN* e) { esrgan = e; cugan = nullptr; }
    void set_backend(RealCUGAN* c) { cugan = c; esrgan = nullptr; }

    // scale/tile_size는 실제 백엔드에도 반영해야 하므로 setter를 통해서만 바꿉니다.
    void set_scale(int s) {
        scale = s;
        if (esrgan) esrgan->scale = s;
        if (cugan) cugan->scale = s;
    }
    void set_tile_size(int t) {
        tile_size = t;
        if (esrgan) esrgan->tile_size = t;
        if (cugan) cugan->tile_size = t;
    }

    int tile_process(const cv::Mat& inimage, cv::Mat& outimage) const {
        if (esrgan) return esrgan->tile_process(inimage, outimage);
        if (cugan) return cugan->tile_process(inimage, outimage);
        fprintf(stderr, "BackgroundUpscaler: no backend set\n");
        return -1;
    }

    int scale;      // set_scale()로만 갱신 - real_esrgan.scale / real_cugan.scale과 항상 동기화됨
    int tile_size;  // set_tile_size()로만 갱신

private:
    RealESRGAN* esrgan;
    RealCUGAN* cugan;
};

#endif // BACKGROUND_UPSCALER_H
