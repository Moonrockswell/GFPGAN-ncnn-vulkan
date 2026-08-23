#include <cstdio>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <net.h>
#include "gfpgan.h"
#include "face.h"
#include "realesrgan.h"

#define RESTORE_WHOLE_IMAGE 1   //0-only restore face, 1-restore whole image
#define RESTORE_IMAGE_COLOR 0   //0-no color image, 1-coloring grayscale images

// 자동 화이트밸런스(-wb)는 opencv_contrib의 xphoto 모듈이 필요합니다.
// opencv_contrib 없이 빌드된 환경(xphoto.hpp가 없는 경우)이라면 이 값을
// 0으로 바꿔서 컴파일하세요. 그러면 -wb 옵션은 그대로 받아들이되 아무
// 효과도 적용하지 않는 no-op으로 동작합니다 (빌드 자체는 항상 성공).
#ifndef HAVE_XPHOTO
#define HAVE_XPHOTO 1
#endif

#if HAVE_XPHOTO
#include <opencv2/xphoto.hpp>
#endif

// cv::detailEnhance()는 OpenCV 기본(main) 모듈의 photo.hpp에 포함되어
// 있어 opencv_contrib 없이도 항상 사용 가능합니다 (xphoto와 달리 별도
// 확인/매크로가 필요 없음).
#include <opencv2/photo.hpp>

namespace fs = std::filesystem;

// Default folder (relative to the executable / current working directory)
// where the .param / .bin model files are expected to be found.
static const char *DEFAULT_MODEL_DIR = "./gfpgan-models";

static void print_usage(const char *progname) {
    fprintf(stderr,
        "Usage: %s -i infile -o outfile [options]\n"
        "\n"
        "  -h                       show this help\n"
        "  -i input-path          input image path (jpg/png/webp) or folder\n"
        "  -o output-path       output image path (jpg/png/webp) or folder\n"
        "                            If -o is omitted\n"
        "                            1) single file input - saved next to the input as <name>-output.<ext>\n"
        "                            2) saved into a new '<foldername>-output' subfolder\n"
        "                               inside the input folder, original files kept\n"
        "                            3) . is recognized as the current folder\n"
        "  \n"
        "  -m model-path       folder path to the pre-trained models (default=%s)\n"
        "  -f output format     output image format (jpg/png/webp, default=png)\n"
        "  -t tile-size             background upscale tile-size, must be > 0 (default = 300) \n"
        "                            smaller values reduce GPU memory load per step\n"
        "                            useful to avoid GPU timeouts (Vulkan device lost) on older GPUs\n"
        "  -s scale                 1=off (keep original resolution), 2=on (default, 2x upscale)\n"
        "  -mf max-faces          max face candidates processed per image, 1-20 (default = 5)\n"
        "                            raise this if legitimate photos have more than 5 faces;\n"
        "                            excess candidates beyond this cap are dropped by confidence score\n"
        "  -wb white-balance   0=off (default), 1=on - auto white balance (Gray-World)\n"
        "                            corrects color casts (fluorescent green/yellow tint, tungsten\n"
        "                            orange tint, shade blue tint); applied before denoise/CLAHE/sharpen\n"
        "                            (requires opencv_contrib's xphoto module; no-op if not built with it)\n"
        "  -de detail-enhance  0-100, default = 0 (off) - edge-preserving detail enhancement\n"
        "                            more natural alternative/addition to -sp: enhances fine texture\n"
        "                            while preserving major edges, so it avoids the haloing that -sp\n"
        "                            can produce at high strength; applied after CLAHE, before -sp\n"
        "\n"
        "*Unmodifiable Options*\n"
        "\n"
        "  -n model name       GFPGANCleanv1-NoCE-C2 supports only one type of model\n",
        progname, DEFAULT_MODEL_DIR);
}

static std::string to_lower(std::string s) {
    for (char &c : s) c = (char) tolower((unsigned char) c);
    return s;
}

// Returns true if fmt is one of the supported output formats.
static bool is_supported_format(const std::string &fmt) {
    std::string f = to_lower(fmt);
    return f == "jpg" || f == "jpeg" || f == "png" || f == "webp";
}

// Replaces (or appends) the extension of path with the given format,
// e.g. apply_format("result.png", "jpg") -> "result.jpg"
static std::string apply_format(const std::string &path, const std::string &fmt) {
    size_t slash = path.find_last_of("/\\");
    size_t dot = path.find_last_of('.');
    std::string base = (dot != std::string::npos && (slash == std::string::npos || dot > slash))
                        ? path.substr(0, dot)
                        : path;
    return base + "." + fmt;
}

// Default output path for a single input file when -o is not given:
// same folder as the input, "<name>-output.<ext>" (no space before "-output").
// Extension follows -f when given, otherwise always png.
static std::string default_single_output(const std::string &inputPath, const std::string &format) {
    fs::path p(inputPath);
    std::string stem = p.stem().string();
    std::string ext = !format.empty() ? to_lower(format) : "png";
    fs::path outPath = p.parent_path() / (stem + "-output." + ext);
    return outPath.string();
}

// Default output folder for a directory input when -o is not given:
// "<foldername>-output" created as a subfolder inside the input folder,
// e.g. ./image -> ./image/image-output
// Special case: "." (or a bare trailing slash) has no folder name of its
// own, so instead of producing an odd ".-output" folder we just use "output".
static std::string default_output_folder(const std::string &inputDir) {
    fs::path p(inputDir);
    std::string dirname = p.filename().string();
    if (dirname.empty() || dirname == "." || dirname == "..") {
        fs::path outDir = p / "output";
        return outDir.string();
    }
    fs::path outDir = p / (dirname + "-output");
    return outDir.string();
}

static bool is_image_file(const fs::path &p) {
    if (!fs::is_regular_file(p)) return false;
    std::string ext = p.extension().string();
    if (!ext.empty() && ext[0] == '.') ext = ext.substr(1);
    return is_supported_format(ext);
}

static void to_ocv(const ncnn::Mat &result, cv::Mat &out) {
    cv::Mat cv_result_32F = cv::Mat::zeros(cv::Size(512, 512), CV_32FC3);
    for (int i = 0; i < result.h; i++) {
        for (int j = 0; j < result.w; j++) {
            cv_result_32F.at<cv::Vec3f>(i, j)[2] = (result.channel(0)[i * result.w + j] + 1) / 2;
            cv_result_32F.at<cv::Vec3f>(i, j)[1] = (result.channel(1)[i * result.w + j] + 1) / 2;
            cv_result_32F.at<cv::Vec3f>(i, j)[0] = (result.channel(2)[i * result.w + j] + 1) / 2;
        }
    }

    cv::Mat cv_result_8U;
    cv_result_32F.convertTo(cv_result_8U, CV_8UC3, 255.0, 0);

    cv_result_8U.copyTo(out);

}

#if RESTORE_WHOLE_IMAGE

static void paste_faces_to_input_image(const cv::Mat &restored_face, cv::Mat &trans_matrix_inv, cv::Mat &bg_upsample) {
    trans_matrix_inv.at<float>(0, 2) += 1.0;
    trans_matrix_inv.at<float>(1, 2) += 1.0;

    cv::Mat inv_restored;
    cv::warpAffine(restored_face, inv_restored, trans_matrix_inv, bg_upsample.size(), 1, 0);
    cv::Mat mask = cv::Mat::ones(cv::Size(512, 512), CV_8UC1) * 255;

    cv::Mat inv_mask;
    cv::warpAffine(mask, inv_mask, trans_matrix_inv, bg_upsample.size(), 1, 0);

    cv::Mat inv_mask_erosion;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4, 4));
    cv::erode(inv_mask, inv_mask_erosion, kernel);

    cv::Mat pasted_face;
    cv::bitwise_and(inv_restored, inv_restored, pasted_face, inv_mask_erosion);

    int total_face_area = cv::countNonZero(inv_mask_erosion);
    if (total_face_area <= 0) {
        // The aligned face fell almost entirely outside the photo
        // (e.g. a face cropped at the image edge). Nothing valid to
        // paste back, so skip this face instead of crashing on a
        // zero-size erosion kernel below.
        return;
    }

    int w_edge = int(std::sqrt(total_face_area) / 20);
    // cv::getStructuringElement/cv::erode require a kernel of at least 1x1;
    // a tiny detected face area can make w_edge (and therefore the kernel
    // size) collapse to 0, which crashes with an OpenCV assertion. Clamp
    // to a sane minimum instead.
    int erosion_radius = std::max(1, w_edge * 2);
    cv::Mat inv_mask_center;

    kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(erosion_radius, erosion_radius));
    cv::erode(inv_mask_erosion, inv_mask_center, kernel);

    // cv::GaussianBlur requires an odd kernel size. blur_size is used below as
    // (blur_size + 1), so blur_size itself must be even for the final kernel
    // to be odd. w_edge * 2 is always even, EXCEPT when w_edge is 0, in which
    // case std::max(1, 0) previously collapsed this to 1 (odd) -> kernel size
    // 2 (even) -> "Assertion failed: ksize.width % 2 == 1" crash. Clamp to a
    // minimum of 2 instead of 1 so the kernel size stays odd in all cases.
    int blur_size = std::max(2, w_edge * 2);
    cv::Mat inv_soft_mask;
    cv::GaussianBlur(inv_mask_center, inv_soft_mask, cv::Size(blur_size + 1, blur_size + 1), 0, 0, 4);

    for (int h = 0; h < bg_upsample.rows; h++) {
        for (int w = 0; w < bg_upsample.cols; w++) {
            float alpha = inv_soft_mask.at<uchar>(h, w) / 255.0;
            bg_upsample.at<cv::Vec3b>(h, w)[0] =
                    pasted_face.at<cv::Vec3b>(h, w)[0] * alpha + (1 - alpha) * bg_upsample.at<cv::Vec3b>(h, w)[0];
            bg_upsample.at<cv::Vec3b>(h, w)[1] =
                    pasted_face.at<cv::Vec3b>(h, w)[1] * alpha + (1 - alpha) * bg_upsample.at<cv::Vec3b>(h, w)[1];
            bg_upsample.at<cv::Vec3b>(h, w)[2] =
                    pasted_face.at<cv::Vec3b>(h, w)[2] * alpha + (1 - alpha) * bg_upsample.at<cv::Vec3b>(h, w)[2];
        }
    }
}

#endif

// ---------- 후처리: 디노이즈 / 샤프닝 (1배 스케일, 최종 합성본에만 적용) ----------
// 얼굴 보정 단계와 섞으면 얼굴 디테일이 뭉개지거나 부자연스러워질 수 있어서,
// 배경 업스케일 + 얼굴 합성이 전부 끝난 최종 이미지에 대해서만, 크기 변경 없이
// (스케일 1배) 마지막 후처리로 적용합니다.

// ---------- 자동 화이트밸런스 (Gray-World 알고리즘) ----------
// enabled: 0(끔, 기본) / 1(켬). 이미지 전체의 평균 색이 회색(무채색)에
// 가까워야 한다는 가정 하에, R/G/B 채널의 평균 밝기 차이를 계산해서
// 색 편향을 자동으로 상쇄합니다. 형광등의 초록/노랑끼, 백열등/텅스텐
// 조명의 붉은끼, 그늘에서 찍힌 사진의 푸른끼 등을 보정하는 데 효과적
// 입니다.
// 다른 후처리(디노이즈/CLAHE/샤프닝)보다 먼저 적용해서, 색이 바로잡힌
// 이미지를 기준으로 나머지 대비/선명도 보정이 이뤄지도록 합니다.
// 주의: 인물 사진처럼 피부색(주로 붉은 계열) 하나가 화면 넓은 비중을
// 차지하면 "회색 가정"이 깨져서 과보정(색이 부자연스럽게 틀어짐)될 수
// 있습니다. 그래서 기본값은 꺼짐이며, 색이 눈에 띄게 편향된 사진에서만
// 켜서 쓰는 것을 권장합니다.
#if HAVE_XPHOTO
static void apply_white_balance(cv::Mat &img, int enabled) {
    if (enabled <= 0) return;
    cv::Ptr<cv::xphoto::GrayworldWB> wb = cv::xphoto::createGrayworldWB();
    // saturationThreshold: 이 값보다 채도가 높은 픽셀(피부, 원색 옷 등
    // 이미 색이 뚜렷한 영역)은 회색 가정 계산에서 제외해 과보정을 줄입니다.
    // 기본값(0.9)보다 살짝 올려서 인물 사진에서 조금 더 안전하게 동작하도록 함.
    wb->setSaturationThreshold(0.95f);
    cv::Mat out;
    wb->balanceWhite(img, out);
    img = out;
}
#else
// opencv_contrib(xphoto)가 빌드에 포함되지 않은 환경: 옵션은 받아들이되
// 아무 효과도 적용하지 않는 no-op. (아래에서 -wb 값이 1이어도 조용히
// 무시되며 별도 에러는 내지 않습니다.)
static void apply_white_balance(cv::Mat &, int) {}
#endif

// strength: 0(끔) ~ 100. cv::fastNlMeansDenoisingColored의 h(밝기)/hColor(색상)
// 강도 파라미터로 매핑합니다 (대략 1~15 범위, OpenCV 권장 기본값이 h=3 부근).
static void apply_denoise(cv::Mat &img, int strength) {
    if (strength <= 0) return;
    if (strength > 100) strength = 100;
    float h = 1.0f + (strength / 100.0f) * 14.0f;
    cv::Mat out;
    cv::fastNlMeansDenoisingColored(img, out, h, h, 7, 21);
    img = out;
}

// strength: 0(끔) ~ 100. CLAHE(Contrast Limited Adaptive Histogram Equalization)로
// 저노출/뿌연 사진의 대비를 지역적으로 끌어올립니다. RGB 채널에 직접 걸면
// 색이 틀어지므로, Lab 색공간의 밝기(L) 채널에만 적용하고 색상(a/b) 채널은
// 그대로 둡니다. clipLimit이 클수록 대비 향상이 강해지지만 노이즈도 같이
// 증폭될 수 있어 1.0~4.0 범위로 제한합니다. 타일 크기는 8x8 고정(표준값).
static void apply_clahe(cv::Mat &img, int strength) {
    if (strength <= 0) return;
    if (strength > 100) strength = 100;
    double clipLimit = 1.0 + (strength / 100.0) * 3.0;

    cv::Mat lab;
    cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, cv::Size(8, 8));
    clahe->apply(channels[0], channels[0]);

    cv::merge(channels, lab);
    cv::cvtColor(lab, img, cv::COLOR_Lab2BGR);
}

// strength: 0(끔) ~ 100. 가우시안 블러 버전과의 차이를 더해주는 언샵 마스크 방식.
// amount가 클수록 윤곽선 대비가 강해집니다 (대략 0~2.0x 범위로 매핑).
static void apply_sharpen(cv::Mat &img, int strength) {
    if (strength <= 0) return;
    if (strength > 100) strength = 100;
    double amount = (strength / 100.0) * 2.0;
    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, cv::Size(0, 0), 3.0);
    cv::Mat sharpened;
    cv::addWeighted(img, 1.0 + amount, blurred, -amount, 0, sharpened);
    img = sharpened;
}

// strength: 0(끔) ~ 100. cv::detailEnhance()를 이용한 디테일 향상.
// 기존 -sp(언샵 마스크) 샤프닝은 밝기 차이가 큰 윤곽선 주변을 단순
// 대비 증폭시키는 방식이라, 강하게 걸면 윤곽선을 따라 밝고 어두운
// "테두리"(헤일로)가 생기고 화면 전체가 부자연스럽게 딱딱해 보일 수
// 있습니다.
// 반면 detailEnhance는 edge-preserving 필터(도메인 변환 필터) 기반이라
// 큰 윤곽선(예: 얼굴 윤곽, 배경 경계)은 그대로 보존하면서, 그 안쪽의
// 미세한 텍스처(머리카락 올, 피부 결, 옷감 무늬 같은 작은 디테일)만
// 국소적으로 또렷하게 살려줍니다. 그 결과 언샵 마스크보다 눈에 덜
// 띄고 자연스러운 "화질이 좋아진" 느낌을 줍니다.
// -sp와 별도 옵션이며, 둘 다 켜도 되고(디테일 향상 후 마지막에 약한
// 샤프닝만 살짝 얹는 조합도 가능) -sp 대신 이것만 켜도 됩니다.
static void apply_detail_enhance(cv::Mat &img, int strength) {
    if (strength <= 0) return;
    if (strength > 100) strength = 100;
    // sigma_s(공간 표준편차, 필터가 참고하는 주변 반경): 클수록 더 넓은
    // 영역까지 함께 보고 판단해서 효과가 굵고 진해짐. 10~70 범위로 매핑.
    // sigma_r(색상/밝기 표준편차, 엣지로 인식하는 민감도): 작을수록 약한
    // 경계도 엣지로 보존해 디테일을 더 세밀하게 살림. 0.15~0.40 범위로 매핑.
    // (OpenCV 기본값: sigma_s=10, sigma_r=0.15, 즉 strength 0에 해당하는 값)
    float sigma_s = 10.0f + (strength / 100.0f) * 60.0f;
    float sigma_r = 0.15f + (strength / 100.0f) * 0.25f;
    cv::Mat out;
    cv::detailEnhance(img, out, sigma_s, sigma_r);
    img = out;
}

// Runs the full restoration pipeline on a single image and writes the result.
// Models are already loaded, so this can be called repeatedly for batch processing.
static bool restore_one_image(GFPGAN &gfpgan,
#if RESTORE_WHOLE_IMAGE
                               Face &face_detector, RealESRGAN &real_esrgan,
#endif
                               const std::string &inputPath, const std::string &outputPath,
                               int denoiseStrength, int sharpenStrength, int claheStrength, int scale,
                               size_t maxFacesPerImage, int whiteBalance, int detailEnhanceStrength) {
    cv::Mat img = cv::imread(inputPath, 1);
    if (img.empty()) {
        fprintf(stderr, "cv::imread %s failed\n", inputPath.c_str());
        return false;
    }

    // Wrap the whole restoration pipeline so that an unexpected OpenCV/ncnn
    // exception on one image (e.g. an assertion failure, or a Vulkan error
    // surfaced as an exception) doesn't take down the whole batch run.
    // We log the failure and skip just this image; the caller's loop moves
    // on to the next file.
    try {

#if RESTORE_WHOLE_IMAGE
    cv::Mat bg_upsample;
    real_esrgan.tile_process(img, bg_upsample);

    std::vector<Object> objects;
    // nms_threshold: 0.3 (원래 기본값) -> 0.45 로 상향.
    // 겹치는 검출 박스를 더 적극적으로 하나로 합쳐서, 만화/일러스트처럼
    // 실사 얼굴 검출기가 오작동하기 쉬운 이미지에서 같은 부위 주변에
    // 중복으로 잡히는 오탐지 박스 수를 줄입니다. prob_threshold(0.7)는 그대로.
    face_detector.detect(img, objects, 0.7f, 0.45f);

    // 만화/일러스트/스캔 이미지는 실사 얼굴 학습 기반 검출기 특성상
    // 눈/무늬 등을 얼굴로 오탐지해서 후보가 비정상적으로 많이 나올 수 있습니다.
    // 후보 하나당 GFPGAN 512x512 GPU 추론이 한 번씩 더 들어가므로, 오탐지가
    // 쌓이면 VRAM/처리시간이 누적되어 device lost(vkQueueSubmit failed)로
    // 이어질 수 있습니다. 점수(score) 상위 kMaxFacesPerImage개만 남겨서
    // 이런 오탐지 폭주가 GPU를 죽이는 것을 막습니다. 정상적인 인물 사진은
    // 보통 이 한도 안에 들어오므로 실질적인 영향이 없습니다.
    // 상한값은 -mf 옵션으로 조절 가능 (기본 5, 단체 사진처럼 얼굴이 실제로
    // 많은 경우 -mf 10 등으로 올려서 정상 얼굴이 잘리는 것을 방지).
    if (objects.size() > maxFacesPerImage) {
        std::sort(objects.begin(), objects.end(),
                   [](const Object &a, const Object &b) { return a.score > b.score; });
        fprintf(stderr, "Warning: %zu face candidates detected, keeping top %zu by confidence "
                         "(likely false positives on illustration/scan input)\n",
                objects.size(), maxFacesPerImage);
        objects.resize(maxFacesPerImage);
    }

    std::vector<cv::Mat> trans_img;
    std::vector<cv::Mat> trans_matrix_inv;
    face_detector.align_warp_face(img, objects, trans_matrix_inv, trans_img);

    for (size_t i = 0; i < objects.size(); i++) {
        ncnn::Mat gfpgan_result;
        gfpgan.process(trans_img[i], gfpgan_result);

        cv::Mat restored_face;
        to_ocv(gfpgan_result, restored_face);

        paste_faces_to_input_image(restored_face, trans_matrix_inv[i], bg_upsample);
    }

    // -s 1(off): RealESRGAN 배경 업스케일 + GFPGAN 얼굴 보정은 항상 내부적으로
    // 2배 해상도로 수행됩니다 (모델 자체가 2배 고정이라 이 과정 자체는 끌 수
    // 없음). 다만 "스케일 오프"를 원하면, 화질 향상 효과는 그대로 누리면서
    // 최종 결과물 크기만 원본과 같게 돌려주기 위해 합성이 끝난 직후 원본
    // 해상도로 다시 축소합니다. 후처리(디노이즈/샤프닝)는 이 축소가 끝난
    // "최종 배포 크기" 기준으로 적용되도록 그 다음에 수행합니다.
    if (scale == 1) {
        cv::resize(bg_upsample, bg_upsample, img.size(), 0, 0, cv::INTER_AREA);
    }

    // 얼굴 보정 + 합성이 전부 끝난 뒤, 크기 변경 없이(1배) 후처리 적용
    // 순서: 화이트밸런스(색 편향 보정) -> 디노이즈(잡티 제거) -> CLAHE(대비 향상)
    //       -> 디테일 향상(자연스러운 텍스처 보강) -> 샤프닝(마지막 선명도 마무리)
    apply_white_balance(bg_upsample, whiteBalance);
    apply_denoise(bg_upsample, denoiseStrength);
    apply_clahe(bg_upsample, claheStrength);
    apply_detail_enhance(bg_upsample, detailEnhanceStrength);
    apply_sharpen(bg_upsample, sharpenStrength);

    cv::imwrite(outputPath, bg_upsample);
#else
    ncnn::Mat gfpgan_result;
    gfpgan.process(img, gfpgan_result);

    cv::Mat restored_face;
    to_ocv(gfpgan_result, restored_face);

    apply_white_balance(restored_face, whiteBalance);
    apply_denoise(restored_face, denoiseStrength);
    apply_clahe(restored_face, claheStrength);
    apply_detail_enhance(restored_face, detailEnhanceStrength);
    apply_sharpen(restored_face, sharpenStrength);

    cv::imwrite(outputPath, restored_face);
#endif

    } catch (const cv::Exception &e) {
        fprintf(stderr, "OpenCV error while processing '%s': %s\n", inputPath.c_str(), e.what());
        return false;
    } catch (const std::exception &e) {
        fprintf(stderr, "Error while processing '%s': %s\n", inputPath.c_str(), e.what());
        return false;
    }

    return true;
}

int main(int argc, char **argv) {
    std::string imagepath;
    std::string outputpath;
    std::string modeldir = DEFAULT_MODEL_DIR;
    std::string format;
    int tilesize = 400;  // background upscale tile size, overridable via -t
    int denoiseStrength = 0;  // -dn, 0-100, default off
    int sharpenStrength = 0;  // -sp, 0-100, default off
    int claheStrength = 0;  // -cl, 0-100, default off (CLAHE 자동 대비 향상)
    int scale = 2;  // -s, 1=off(원본 해상도), 2=on(기본, 2배 업스케일)
    int maxFaces = 5;  // -mf, 이미지 한 장당 처리할 최대 얼굴 후보 수 (기본 5)
    int whiteBalance = 0;  // -wb, 0=off(기본)/1=on, 자동 화이트밸런스(Gray-World)
    int detailEnhanceStrength = 0;  // -de, 0-100, default off, edge-preserving 디테일 향상

    if (argc < 2) {
        print_usage(argv[0]);
        return -1;
    }

    // Support Windows-style "/?" as well as -h, wherever it appears.
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "/?") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) {
            imagepath = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            outputpath = argv[++i];
        } else if (arg == "-m" && i + 1 < argc) {
            modeldir = argv[++i];
        } else if (arg == "-f" && i + 1 < argc) {
            format = argv[++i];
        } else if (arg == "-t" && i + 1 < argc) {
            tilesize = std::atoi(argv[++i]);
        } else if (arg == "-dn" && i + 1 < argc) {
            denoiseStrength = std::atoi(argv[++i]);
        } else if (arg == "-sp" && i + 1 < argc) {
            sharpenStrength = std::atoi(argv[++i]);
        } else if (arg == "-cl" && i + 1 < argc) {
            claheStrength = std::atoi(argv[++i]);
        } else if (arg == "-s" && i + 1 < argc) {
            scale = std::atoi(argv[++i]);
        } else if (arg == "-mf" && i + 1 < argc) {
            maxFaces = std::atoi(argv[++i]);
        } else if (arg == "-wb" && i + 1 < argc) {
            whiteBalance = std::atoi(argv[++i]);
        } else if (arg == "-de" && i + 1 < argc) {
            detailEnhanceStrength = std::atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n\n", arg.c_str());
            print_usage(argv[0]);
            return -1;
        }
    }

    if (imagepath.empty()) {
        fprintf(stderr, "Error: -i input-path is required\n\n");
        print_usage(argv[0]);
        return -1;
    }

    if (!format.empty() && !is_supported_format(format)) {
        fprintf(stderr, "Error: unsupported -f format '%s' (use jpg/png/webp)\n\n", format.c_str());
        print_usage(argv[0]);
        return -1;
    }

    if (tilesize <= 0) {
        fprintf(stderr, "Error: -t tile-size must be a positive integer (got '%d')\n\n", tilesize);
        print_usage(argv[0]);
        return -1;
    }

    if (denoiseStrength < 0 || denoiseStrength > 100) {
        fprintf(stderr, "Error: -dn denoise-strength must be between 0 and 100 (got '%d')\n\n", denoiseStrength);
        print_usage(argv[0]);
        return -1;
    }

    if (sharpenStrength < 0 || sharpenStrength > 100) {
        fprintf(stderr, "Error: -sp sharpen-strength must be between 0 and 100 (got '%d')\n\n", sharpenStrength);
        print_usage(argv[0]);
        return -1;
    }

    if (claheStrength < 0 || claheStrength > 100) {
        fprintf(stderr, "Error: -cl clahe-strength must be between 0 and 100 (got '%d')\n\n", claheStrength);
        print_usage(argv[0]);
        return -1;
    }

    if (scale != 1 && scale != 2) {
        fprintf(stderr, "Error: -s scale must be 1 (off) or 2 (on, default) (got '%d')\n\n", scale);
        print_usage(argv[0]);
        return -1;
    }

    if (maxFaces < 1 || maxFaces > 20) {
        fprintf(stderr, "Error: -mf max-faces must be between 1 and 20 (got '%d')\n\n", maxFaces);
        print_usage(argv[0]);
        return -1;
    }

    if (whiteBalance != 0 && whiteBalance != 1) {
        fprintf(stderr, "Error: -wb white-balance must be 0 (off, default) or 1 (on) (got '%d')\n\n", whiteBalance);
        print_usage(argv[0]);
        return -1;
    }
#if !HAVE_XPHOTO
    if (whiteBalance == 1) {
        fprintf(stderr, "Warning: -wb 1 requested but this build has no opencv_contrib/xphoto support; "
                         "white balance will be skipped (no-op).\n");
    }
#endif

    if (detailEnhanceStrength < 0 || detailEnhanceStrength > 100) {
        fprintf(stderr, "Error: -de detail-enhance must be between 0 and 100 (got '%d')\n\n", detailEnhanceStrength);
        print_usage(argv[0]);
        return -1;
    }

    if (!fs::exists(imagepath)) {
        fprintf(stderr, "Error: input path '%s' does not exist\n", imagepath.c_str());
        return -1;
    }

    // Strip a single trailing slash/backslash so path concatenation below is clean.
    if (!modeldir.empty() && (modeldir.back() == '/' || modeldir.back() == '\\')) {
        modeldir.pop_back();
    }

    GFPGAN gfpgan;
    gfpgan.load(modeldir + "/encoder.param", modeldir + "/encoder.bin", modeldir + "/style.bin");

#if RESTORE_WHOLE_IMAGE
    Face face_detector;
    face_detector.load(modeldir + "/yolov5-blazeface.param", modeldir + "/yolov5-blazeface.bin");

    RealESRGAN real_esrgan;
    real_esrgan.load(modeldir + "/real_esrgan.param", modeldir + "/real_esrgan.bin");
    real_esrgan.tile_size = tilesize;   // -t 로 넘긴 값 적용 (기본 400)
#endif

    bool inputIsDir = fs::is_directory(imagepath);

    if (inputIsDir) {
        // Batch mode: process every jpg/png/webp file directly inside the folder
        // (not recursively) and write results into an output subfolder.
        std::string outDir = !outputpath.empty() ? outputpath : default_output_folder(imagepath);

        std::error_code ec;
        fs::create_directories(outDir, ec);
        if (ec) {
            fprintf(stderr, "Error: could not create output folder '%s'\n", outDir.c_str());
            return -1;
        }

        int processed = 0;
        for (const auto &entry : fs::directory_iterator(imagepath)) {
            if (!is_image_file(entry.path())) continue;

            std::string outExt = !format.empty() ? to_lower(format) : "png";
            std::string outName = entry.path().stem().string() + "." + outExt;
            std::string outPath = (fs::path(outDir) / outName).string();

            fprintf(stderr, "Processing %s -> %s\n", entry.path().string().c_str(), outPath.c_str());
#if RESTORE_WHOLE_IMAGE
            if (restore_one_image(gfpgan, face_detector, real_esrgan, entry.path().string(), outPath,
                                   denoiseStrength, sharpenStrength, claheStrength, scale, maxFaces, whiteBalance, detailEnhanceStrength)) {
#else
            if (restore_one_image(gfpgan, entry.path().string(), outPath,
                                   denoiseStrength, sharpenStrength, claheStrength, scale, maxFaces, whiteBalance, detailEnhanceStrength)) {
#endif
                processed++;
            }
        }

        if (processed == 0) {
            fprintf(stderr, "No jpg/png/webp images found in '%s'\n", imagepath.c_str());
            return -1;
        }
        fprintf(stderr, "Done. %d image(s) saved to %s\n", processed, outDir.c_str());
    } else {
        // Single file mode.
        std::string outPath;
        if (!outputpath.empty()) {
            outPath = format.empty() ? outputpath : apply_format(outputpath, format);
        } else {
            outPath = default_single_output(imagepath, format);
        }

#if RESTORE_WHOLE_IMAGE
        if (!restore_one_image(gfpgan, face_detector, real_esrgan, imagepath, outPath,
                                denoiseStrength, sharpenStrength, claheStrength, scale, maxFaces, whiteBalance, detailEnhanceStrength)) {
#else
        if (!restore_one_image(gfpgan, imagepath, outPath,
                                denoiseStrength, sharpenStrength, claheStrength, scale, maxFaces, whiteBalance, detailEnhanceStrength)) {
#endif
            return -1;
        }
        fprintf(stderr, "Saved %s\n", outPath.c_str());
    }

    return 0;
}
