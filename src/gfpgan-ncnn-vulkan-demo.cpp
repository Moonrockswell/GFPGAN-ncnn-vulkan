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
        "\n"
        "*Unmodifiable Options*\n"
        "\n"
        "  -s scale                 upscale ratio (default=2)\n"
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

// Runs the full restoration pipeline on a single image and writes the result.
// Models are already loaded, so this can be called repeatedly for batch processing.
static bool restore_one_image(GFPGAN &gfpgan,
#if RESTORE_WHOLE_IMAGE
                               Face &face_detector, RealESRGAN &real_esrgan,
#endif
                               const std::string &inputPath, const std::string &outputPath,
                               int denoiseStrength, int sharpenStrength) {
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
    const size_t kMaxFacesPerImage = 5;
    if (objects.size() > kMaxFacesPerImage) {
        std::sort(objects.begin(), objects.end(),
                   [](const Object &a, const Object &b) { return a.score > b.score; });
        fprintf(stderr, "Warning: %zu face candidates detected, keeping top %zu by confidence "
                         "(likely false positives on illustration/scan input)\n",
                objects.size(), kMaxFacesPerImage);
        objects.resize(kMaxFacesPerImage);
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

    // 얼굴 보정 + 합성이 전부 끝난 뒤, 크기 변경 없이(1배) 후처리 적용
    apply_denoise(bg_upsample, denoiseStrength);
    apply_sharpen(bg_upsample, sharpenStrength);

    cv::imwrite(outputPath, bg_upsample);
#else
    ncnn::Mat gfpgan_result;
    gfpgan.process(img, gfpgan_result);

    cv::Mat restored_face;
    to_ocv(gfpgan_result, restored_face);

    apply_denoise(restored_face, denoiseStrength);
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
                                   denoiseStrength, sharpenStrength)) {
#else
            if (restore_one_image(gfpgan, entry.path().string(), outPath,
                                   denoiseStrength, sharpenStrength)) {
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
                                denoiseStrength, sharpenStrength)) {
#else
        if (!restore_one_image(gfpgan, imagepath, outPath,
                                denoiseStrength, sharpenStrength)) {
#endif
            return -1;
        }
        fprintf(stderr, "Saved %s\n", outPath.c_str());
    }

    return 0;
}
