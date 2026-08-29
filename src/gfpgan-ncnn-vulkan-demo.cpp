#include <cstdio>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <thread>
#include <chrono>
#include <net.h>
#include "gfpgan.h"
#include "face.h"
#include "realesrgan.h"
#include "realcugan.h"
#include "background_upscaler.h"
#include "waifu2x_denoise.h"

// CMakeLists.txt가 구성 시점에 git 커밋 해시/빌드 날짜를 넣어줌. CMake를 안 거치고
// 이 파일 하나만 다른 방식으로 컴파일하는 경우를 위한 안전한 기본값(폴백).
#ifndef PIXELFORGE_GIT_HASH
#define PIXELFORGE_GIT_HASH "unknown"
#endif
#ifndef PIXELFORGE_BUILD_DATE
#define PIXELFORGE_BUILD_DATE __DATE__ " " __TIME__
#endif
// 제품 버전(사람이 직접 관리) - RealESRGAN+RealCUGAN+waifu2x+GFPGAN 4개 엔진이
// 전부 통합된 시점을 1.0으로 확정함. 이후 기능이 크게 늘거나 호환성이 깨지는
// 변경이 있을 때만 사람이 직접 올릴 것 (git 해시/빌드 날짜처럼 자동 갱신 아님).
#define PIXELFORGE_VERSION "1.0"
// 이 exe에 실제로 들어있는 기능 요약 - 옛날 빌드로 착각해서 헤매는 걸 막기 위한 것이라,
// 새 기능을 추가할 때마다 여기도 같이 갱신할 것.
#define PIXELFORGE_FEATURE_SUMMARY "RealESRGAN+RealCUGAN(6 models)+syncgap+waifu2x+fp32diag+TDR-safe-tiling+imread-retry"

#ifdef _WIN32
#include <windows.h>
#endif

#define RESTORE_WHOLE_IMAGE 1   //0-only restore face, 1-restore whole image
#define RESTORE_IMAGE_COLOR 0   //0-no color image, 1-coloring grayscale images

// 자동 화이트밸런스(-wb)는 opencv_contrib의 xphoto 모듈이 필요합니다.
// CMakeLists.txt가 빌드 시점에 opencv2/xphoto.hpp 존재 여부를 자동으로
// 감지해서 HAVE_XPHOTO를 정의해주므로 보통은 이 블록이 실행되지 않습니다.
// 혹시 CMake 없이 이 파일 하나만 단독 컴파일하는 경우를 대비한 안전장치로,
// 그런 경우의 기본값은 "없다고 가정(0)"으로 둡니다. contrib이 확실히 있는
// 환경에서 단독 컴파일한다면 이 값을 1로 바꾸세요.
#ifndef HAVE_XPHOTO
#define HAVE_XPHOTO 0
#endif

#if HAVE_XPHOTO
#include <opencv2/xphoto.hpp>
#endif

// cv::detailEnhance()는 OpenCV 기본(main) 모듈의 photo.hpp에 포함되어
// 있어 opencv_contrib 없이도 항상 사용 가능합니다 (xphoto와 달리 별도
// 확인/매크로가 필요 없음).
#include <opencv2/photo.hpp>
// cv::setNumThreads() (-threads 옵션용)
#include <opencv2/core.hpp>

namespace fs = std::filesystem;

// Default folder (relative to the executable / current working directory)
// where the .param / .bin model files are expected to be found.
static const char *DEFAULT_MODEL_DIR = "./gfpgan-models";
// RealESRGAN(배경 업스케일) 모델은 GFPGAN(얼굴 보정) 모델과 별도 폴더에 둡니다.
// 공식 realesrgan-ncnn-vulkan 배포본의 모델 파일명을 변형 없이 그대로 사용합니다.
static const char *DEFAULT_REALESRGAN_MODEL_DIR = "./realesrgan-models";
static const char *DEFAULT_REALCUGAN_MODEL_DIR = "./realcugan-models";
// waifu2x cunet 노이즈 제거(-nn) 모델도 별도 폴더. 공식 waifu2x-ncnn-vulkan
// models-cunet 배포본의 noise{1,2,3}_model.param/.bin 파일명을 그대로 사용
// (scale 접미사 없는 버전 - 배율은 안 바꾸고 노이즈만 제거).
static const char *DEFAULT_WAIFU2X_MODEL_DIR = "./waifu2x-models";

// ---------- ANSI 컬러 (gfpgan-auto-composite.bat 와 동일 팔레트) ----------
// -h 출력에 색을 입히기 위한 매크로들입니다. 실제로 터미널에 반영되려면
// Windows 콘솔에서 VT100 이스케이프 처리를 켜줘야 하므로, 아래
// enable_ansi_colors()를 main() 맨 앞에서 한 번 호출합니다(파이프로 리다이렉트
// 되거나 콘솔이 아닌 경우에도 SetConsoleMode 호출 자체는 안전하게 무시됩니다).
#define C_CYAN   "\x1b[96m"
#define C_YELLOW "\x1b[93m"
#define C_GRAY   "\x1b[90m"
#define C_WHITE  "\x1b[97m"
#define C_GREEN  "\x1b[92m"
#define C_RESET  "\x1b[0m"

static void enable_ansi_colors() {
#ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) return;
    DWORD mode = 0;
    if (!GetConsoleMode(hOut, &mode)) return;
    SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);

    // -h 도움말은 stderr로 나가므로 stderr 핸들도 같이 켜줍니다.
    HANDLE hErr = GetStdHandle(STD_ERROR_HANDLE);
    if (hErr != INVALID_HANDLE_VALUE && GetConsoleMode(hErr, &mode)) {
        SetConsoleMode(hErr, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
#endif
}

static void print_version() {
    fprintf(stderr, C_WHITE "  PixelForge " C_RESET C_YELLOW "v" PIXELFORGE_VERSION C_RESET "\n");
    fprintf(stderr, C_GRAY "  (formerly GFPGAN Auto Composite - face restore + RealESRGAN/RealCUGAN\n  background upscale + waifu2x denoise, all-in-one)\n" C_RESET);
    fprintf(stderr, C_GRAY "  build " C_RESET "%s" C_GRAY " (git " C_RESET "%s" C_GRAY ")" C_RESET "\n",
            PIXELFORGE_BUILD_DATE, PIXELFORGE_GIT_HASH);
    fprintf(stderr, C_GRAY "  features: " C_RESET "%s\n", PIXELFORGE_FEATURE_SUMMARY);
}

static void print_usage(const char *progname) {
    // main()이 -h 파싱 전에 이미 print_version()을 한 번 찍었으므로, 여기서 또
    // 부르면 배너가 두 번 중복 출력됨. 여기서는 부제목 한 줄만 추가로 보여준다.
    fprintf(stderr, C_CYAN "===========================================================\n" C_RESET);
    fprintf(stderr, "  " C_GRAY "Face restore (GFPGAN) + background upscale (RealESRGAN/RealCUGAN)\n" C_RESET);
    fprintf(stderr, C_CYAN "===========================================================\n" C_RESET);
    fprintf(stderr, "\n");
    fprintf(stderr, "  " C_WHITE "Usage:" C_RESET " %s -i infile -o outfile [options]\n\n", progname);

    // ---------------- I/O ----------------
    fprintf(stderr, C_CYAN "  [ I/O ]" C_RESET "\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  " C_WHITE "-h" C_RESET "                      show this help\n");
    fprintf(stderr, "  " C_WHITE "-i" C_RESET " input-path           input image path (jpg/png/webp) or folder\n");
    fprintf(stderr, "  " C_WHITE "-o" C_RESET " output-path          output image path (jpg/png/webp) or folder\n");
    fprintf(stderr, C_GRAY
        "                          If -o is omitted\n"
        "                          1) single file input - saved next to the input as <name>-output.<ext>\n"
        "                          2) saved into a new '<foldername>-output' subfolder\n"
        "                             inside the input folder, original files kept\n"
        "                          3) . is recognized as the current folder\n" C_RESET);
    fprintf(stderr, "\n");

    // ---------------- MODELS / FORMAT / TILE ----------------
    fprintf(stderr, C_CYAN "  [ MODELS / OUTPUT ]" C_RESET "\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  " C_WHITE "-m" C_RESET " model-path           folder path to the GFPGAN (face) models "
        C_GRAY "(default=%s)" C_RESET "\n", DEFAULT_MODEL_DIR);
    fprintf(stderr, "  " C_WHITE "-rm" C_RESET " model-path          folder path to the RealESRGAN (background) models "
        C_GRAY "(default=%s)" C_RESET "\n", DEFAULT_REALESRGAN_MODEL_DIR);
    fprintf(stderr, "  " C_WHITE "-rcm" C_RESET " model-path         folder path to the RealCUGAN (anime/illustration) models "
        C_GRAY "(default=%s)" C_RESET "\n", DEFAULT_REALCUGAN_MODEL_DIR);
    fprintf(stderr, "  " C_WHITE "-n" C_RESET " model name           GFPGANCleanv1-NoCE-C2 supports only one type of model "
        C_GRAY "(fixed)" C_RESET "\n");
    fprintf(stderr, "  " C_WHITE "-rn" C_RESET " model-name          RealESRGAN model to use; auto-picked from " C_WHITE "-s" C_RESET " if omitted:\n");
    fprintf(stderr, C_GRAY
        "                          -s 1/2 -> realesrgan-x2plus, -s 3/4 -> realesrgan-x4plus\n"
        "                          (picks whichever model's native scale is closest, avoiding both\n"
        "                          wasted GPU work and excessive upscaling)\n" C_RESET);
    fprintf(stderr, "      " C_GREEN "realesrgan-x2plus" C_RESET "       general photos, native 2x - about half\n");
    fprintf(stderr, C_GRAY "                          the GPU work of x4plus; auto default for -s 1/2\n" C_RESET);
    fprintf(stderr, "      " C_GREEN "realesrgan-x4plus" C_RESET "       general photos, native 4x - sharper/more\n");
    fprintf(stderr, C_GRAY "                          detail, roughly double the GPU work; auto default for -s 3/4\n" C_RESET);
    fprintf(stderr, "      " C_GREEN "realesr-animevideov3" C_RESET "    video frames (lightweight)\n");
    fprintf(stderr, C_GRAY
        "                          anime/illustration stills: use RealCUGAN instead (below), it beats\n"
        "                          realesrgan-x4plus-anime on that content - the -anime RealESRGAN model\n"
        "                          has been removed from this build\n" C_RESET);
    fprintf(stderr, "      " C_GREEN "realcugan-pro-2x" C_RESET "        anime/illustration, native 2x, highest quality\n");
    fprintf(stderr, "      " C_GREEN "realcugan-pro-2x-conservative" C_RESET " same, but preserves original line art\n");
    fprintf(stderr, C_GRAY "                          more (less aggressive re-drawing)\n" C_RESET);
    fprintf(stderr, "      " C_GREEN "realcugan-pro-3x" C_RESET "        anime/illustration, native 3x, highest quality\n");
    fprintf(stderr, "      " C_GREEN "realcugan-pro-3x-conservative" C_RESET " same, but preserves original line art more\n");
    fprintf(stderr, "      " C_GREEN "realcugan-se-4x" C_RESET "         anime/illustration, native 4x (pro has no 4x model)\n");
    fprintf(stderr, "      " C_GREEN "realcugan-se-4x-conservative" C_RESET " same, but preserves original line art more\n");
    fprintf(stderr, "      " C_GREEN "realcugan-se-4x-conservative" C_RESET "  same, but preserves original line art more\n");
    fprintf(stderr, C_GRAY
        "                          RealCUGAN models assume -AiDenoise already cleaned up noise/\n"
        "                          compression artifacts if needed, so only no-denoise/conservative\n"
        "                          weights are included (no separate denoise-strength model here)\n" C_RESET);
    fprintf(stderr, "  " C_WHITE "-f" C_RESET " output format        output image format (jpg/png/webp, "
        C_GREEN "default=png" C_RESET ")\n");
    fprintf(stderr, "  " C_WHITE "-t" C_RESET " tile-size            background upscale tile-size, 0=no tiling "
        C_GREEN "(default = 300)" C_RESET "\n");
    fprintf(stderr, C_GRAY
        "                          smaller values reduce GPU memory load per step\n"
        "                          useful to avoid GPU timeouts (Vulkan device lost) on older GPUs\n" C_RESET);
    fprintf(stderr, "  " C_WHITE "-fp32" C_RESET " 0/1                force fp16/int8 GPU optimizations off (1=off), "
        C_GREEN "(default=0)" C_RESET "\n");
    fprintf(stderr, C_GRAY
        "                          diagnostic option for older GPUs (e.g. Kepler-gen) where tile\n"
        "                          results come out corrupted\n" C_RESET);
    fprintf(stderr, "  " C_WHITE "-syncgap" C_RESET " 0/1             sync RealCUGAN's tile-boundary SE gate (1=on), "
        C_GREEN "(default=0)" C_RESET "\n");
    fprintf(stderr, C_GRAY
        "                          removes faint grid artifacts at tile borders, roughly doubles\n"
        "                          processing time; ignored when a RealESRGAN model is in use\n" C_RESET);
    fprintf(stderr, "  " C_WHITE "-s" C_RESET " scale                final output scale relative to original: 1/2/3/4 "
        C_GREEN "(default = 2)" C_RESET "\n");
    fprintf(stderr, C_GRAY
        "                          realesrgan-x4plus / -anime always run internally at their\n"
        "                          native 4x scale, then resize to whichever of 1/2/3/4 you pick;\n"
        "                          realesr-animevideov3 loads a separate model per scale (2/3/4)\n"
        "                          and outputs it directly, no resize needed (1 falls back to a\n"
        "                          post-resize since there is no native 1x model)\n" C_RESET);
    fprintf(stderr, "\n");

    // ---------------- NEURAL PASSES ----------------
    fprintf(stderr, C_CYAN "  [ NEURAL PASSES ]" C_RESET C_GRAY "  GPU (Vulkan) - runs on the full frame, faces\n"
        "  included, before the background pipeline below\n" C_RESET);
    fprintf(stderr, "\n");
    fprintf(stderr, "  " C_WHITE "-fr" C_RESET " face-restore        0=off, 1=on "
        C_GREEN "(default)" C_RESET " - face detection + GFPGAN face restoration\n");
    fprintf(stderr, C_GRAY
        "                          0 skips face detection/GFPGAN entirely and applies only the\n"
        "                          RealESRGAN background upscale to the whole image; useful for\n"
        "                          anime/illustration/3D-render input where a real-face detector\n"
        "                          should not run or produces false positives\n" C_RESET);
    fprintf(stderr, "  " C_WHITE "-mf" C_RESET " max-faces           max face candidates processed per image, 1-20 "
        C_GREEN "(default = 5)" C_RESET "\n");
    fprintf(stderr, C_GRAY
        "                          raise this if legitimate photos have more than 5 faces;\n"
        "                          excess candidates beyond this cap are dropped by confidence score\n" C_RESET);
    fprintf(stderr, "  " C_WHITE "-nn" C_RESET " ai-denoise-level    0=off "
        C_GREEN "(default)" C_RESET ", 1/2/3 - waifu2x cunet AI noise reduction\n");
    fprintf(stderr, C_GRAY
        "                          applied to the ORIGINAL image before RealESRGAN upscale (so\n"
        "                          the noise itself doesn't get magnified along with everything\n"
        "                          else); this is a separate tool from -dn (algorithmic, background-\n"
        "                          only, 0-100): -nn runs earlier, over the whole image (faces\n"
        "                          included), and can be combined with -dn (e.g. -nn to remove\n"
        "                          heavy noise first, -dn afterward for a light finishing touch)\n" C_RESET);
    fprintf(stderr, "  " C_WHITE "-wm" C_RESET " model-path          folder path to the waifu2x (-nn) models "
        C_GRAY "(default=%s)" C_RESET "\n", DEFAULT_WAIFU2X_MODEL_DIR);
    fprintf(stderr, "\n");

    // ---------------- BACKGROUND PIPELINE ----------------
    fprintf(stderr, C_CYAN "  [ BACKGROUND PIPELINE ]" C_RESET C_GRAY "  applied to background only, faces are\n"
        "  always restored separately by GFPGAN, in this fixed order:\n" C_RESET);
    fprintf(stderr, "  " C_GRAY "white-balance -> vignette-correct -> scratch-removal -> denoise -> clahe ->\n"
        "  detail-enhance -> sharpen\n" C_RESET);
    fprintf(stderr, "\n");

    fprintf(stderr, "  " C_WHITE "-wb" C_RESET " white-balance       0=off "
        C_GREEN "(default)" C_RESET ", 1=on - auto white balance (Gray-World)\n");
    fprintf(stderr, C_GRAY
        "                          corrects color casts (fluorescent green/yellow tint, tungsten\n"
        "                          orange tint, shade blue tint); applied first, before every\n"
        "                          other effect below\n"
        "                          (requires opencv_contrib's xphoto module; no-op if not built with it)\n" C_RESET);
    fprintf(stderr, "  " C_WHITE "-vg" C_RESET " vignette-correct    0-100, "
        C_GREEN "default = 0 (off)" C_RESET " - corrects dark corners/edges (vignetting)\n");
    fprintf(stderr, C_GRAY
        "                          brightens the image radially (more toward the corners, none at\n"
        "                          the center); an approximate correction, not a lens-specific one;\n"
        "                          applied right after -wb, before -sr\n" C_RESET);
    fprintf(stderr, "  " C_WHITE "-sr" C_RESET " scratch-removal     0-100, "
        C_GREEN "default = 0 (off)" C_RESET " - removes thin scratches/dust specks\n");
    fprintf(stderr, C_GRAY
        "                          typical of old print/film scans, via inpainting; background\n"
        "                          only (face areas are always left untouched); applied right\n"
        "                          after -vg, before -dn\n" C_RESET);
    fprintf(stderr, "  " C_WHITE "-dn" C_RESET " denoise             0-100, "
        C_GREEN "default = 0 (off)" C_RESET " - reduces background grain/noise\n");
    fprintf(stderr, C_GRAY
        "                          (e.g. sensor noise, JPEG blockiness, scan grain); higher values\n"
        "                          smooth more but can start to soften fine detail; applied right\n"
        "                          after -sr, before -cl\n" C_RESET);
    fprintf(stderr, "  " C_WHITE "-cl" C_RESET " clahe               0-100, "
        C_GREEN "default = 0 (off)" C_RESET " - CLAHE local contrast enhancement\n");
    fprintf(stderr, C_GRAY
        "                          (Contrast Limited Adaptive Histogram Equalization); brings out\n"
        "                          local detail in flat/washed-out areas without blowing out\n"
        "                          highlights elsewhere; applied right after -dn, before -de\n" C_RESET);
    fprintf(stderr, "  " C_WHITE "-de" C_RESET " detail-enhance      0-100, "
        C_GREEN "default = 0 (off)" C_RESET " - edge-preserving detail enhancement\n");
    fprintf(stderr, C_GRAY
        "                          more natural alternative/addition to -sp: enhances fine texture\n"
        "                          while preserving major edges, so it avoids the haloing that -sp\n"
        "                          can produce at high strength; applied after -cl, before -sp\n" C_RESET);
    fprintf(stderr, "  " C_WHITE "-sp" C_RESET " sharpen             0-100, "
        C_GREEN "default = 0 (off)" C_RESET " - unsharp-mask style sharpening\n");
    fprintf(stderr, C_GRAY
        "                          boosts edge contrast for a crisper look; can produce haloing\n"
        "                          (double-edge outlines) at high strength - try -de instead or\n"
        "                          alongside it at a lower value; applied last, after -de\n" C_RESET);
    fprintf(stderr, "\n");

    // ---------------- PERFORMANCE ----------------
    fprintf(stderr, C_CYAN "  [ PERFORMANCE ]" C_RESET "\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  " C_WHITE "-threads" C_RESET " N              limit CPU threads used by OpenCV post-processing (denoise,\n");
    fprintf(stderr, C_GRAY
        "                          CLAHE, scratch-removal, etc, default = -1 = unlimited); does NOT\n"
        "                          affect GPU (Vulkan) computation, only the CPU-side OpenCV steps\n" C_RESET);
    fprintf(stderr, "  " C_WHITE "-delay-ms" C_RESET " N             wait N milliseconds after each image finishes when batch-\n");
    fprintf(stderr, C_GRAY
        "                          processing a folder (default = 0 = no delay); helps avoid\n"
        "                          pinning the GPU at 100%% back-to-back with no breathing room\n" C_RESET);
    fprintf(stderr, "\n");
    fprintf(stderr, C_GRAY "-----------------------------------------------------------\n" C_RESET);
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

static void paste_faces_to_input_image(const cv::Mat &restored_face, cv::Mat &trans_matrix_inv, cv::Mat &bg_upsample, cv::Mat &faceRegionMask) {
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

    // 이 얼굴이 실제로 합성되는 영역을 배경-전용 마스크(faceRegionMask)에
    // 누적해둡니다. 스크래치 제거(-sr)가 이 마스크를 이용해 얼굴 영역은
    // 절대 건드리지 않고 배경에만 적용되도록 하기 위함입니다.
    cv::bitwise_or(faceRegionMask, inv_mask_erosion, faceRegionMask);

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
// backgroundMask: 이 값이 255인 픽셀에만 결과가 반영됩니다(그 외, 즉 얼굴
// 영역은 원본 그대로 둠). 전역 색/밝기 보정이라 이미지 전체를 대상으로
// 계산은 하되, 최종 반영은 배경 픽셀에만 copyTo로 제한합니다.
#if HAVE_XPHOTO
static void apply_white_balance(cv::Mat &img, int enabled, const cv::Mat &backgroundMask) {
    if (enabled <= 0) return;
    cv::Ptr<cv::xphoto::GrayworldWB> wb = cv::xphoto::createGrayworldWB();
    // saturationThreshold: 이 값보다 채도가 높은 픽셀(피부, 원색 옷 등
    // 이미 색이 뚜렷한 영역)은 회색 가정 계산에서 제외해 과보정을 줄입니다.
    // 기본값(0.9)보다 살짝 올려서 인물 사진에서 조금 더 안전하게 동작하도록 함.
    wb->setSaturationThreshold(0.95f);
    cv::Mat out;
    wb->balanceWhite(img, out);
    out.copyTo(img, backgroundMask);
}
#else
// opencv_contrib(xphoto)가 빌드에 포함되지 않은 환경: 옵션은 받아들이되
// 아무 효과도 적용하지 않는 no-op. (아래에서 -wb 값이 1이어도 조용히
// 무시되며 별도 에러는 내지 않습니다.)
static void apply_white_balance(cv::Mat &, int, const cv::Mat &) {}
#endif

// strength: 0(끔) ~ 100. 렌즈/스캔 특성상 사진 가장자리(특히 네 모서리)가
// 중심보다 어둡게 나오는 비네팅을 보정합니다. 이미지 중심에서의 거리에
// 비례해 밝기를 부드럽게 끌어올리는 방사형(radial) 게인을 곱하는 근사적
// 보정입니다. 실제 렌즈 광학 특성을 역산하는 정밀 보정은 아니지만, 중심
// 대비 모서리가 어두운 사진 전반에 두루 통합니다.
// 화이트밸런스와 같은 "전역 밝기/색 보정" 성격이라 그 바로 뒤, 디노이즈
// 보다 앞서 적용합니다(먼저 밝기를 고르게 맞춘 뒤 잡티 제거 -> 대비 ->
// 디테일 -> 샤프닝 순).
static void apply_vignette_correction(cv::Mat &img, int strength, const cv::Mat &backgroundMask) {
    if (strength <= 0) return;
    if (strength > 100) strength = 100;
    // maxBoost: 모서리(중심에서 가장 먼 지점)에서 밝기를 최대 몇 배까지
    // 끌어올릴지. 0.15(약하게)~0.75(강하게) 범위로 매핑. 중심은 항상
    // 1.0(변화 없음)에서 시작합니다.
    double maxBoost = 0.15 + (strength / 100.0) * 0.60;

    int rows = img.rows, cols = img.cols;
    if (rows <= 0 || cols <= 0) return;
    float cx = cols / 2.0f, cy = rows / 2.0f;
    float maxDist = std::sqrt(cx * cx + cy * cy);
    if (maxDist <= 0.0f) return;

    // 픽셀별 방사형 게인 맵을 미리 계산합니다 (0=중심 -> 1.0배, 1=모서리
    // -> 1.0+maxBoost배). 거리의 제곱에 비례시켜서(코사인 4승 비네팅
    // 감쇠 근사) 중심 근처는 거의 그대로 두고 모서리로 갈수록 완만한
    // 곡선을 그리며 강하게 밝아지도록 합니다.
    cv::Mat gain(rows, cols, CV_32FC1);
    for (int y = 0; y < rows; y++) {
        float *row = gain.ptr<float>(y);
        float dy = y - cy;
        for (int x = 0; x < cols; x++) {
            float dx = x - cx;
            float distNorm = std::sqrt(dx * dx + dy * dy) / maxDist;
            row[x] = 1.0f + static_cast<float>(maxBoost) * (distNorm * distNorm);
        }
    }

    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    for (auto &ch : channels) {
        cv::Mat ch32f;
        ch.convertTo(ch32f, CV_32FC1);
        ch32f = ch32f.mul(gain);
        ch32f.convertTo(ch, ch.type());  // 다시 원래 타입(8-bit)으로, 0~255는 자동 클리핑(saturate_cast)
    }
    cv::Mat out;
    cv::merge(channels, out);
    out.copyTo(img, backgroundMask);
}

// strength: 0(끔) ~ 100. 오래된 인화지/필름 스캔에 흔한, 가늘고 긴 "선"
// 형태의 스크래치(흠집)를 찾아서 cv::inpaint()로 주변 픽셀을 참고해
// 자연스럽게 메꿉니다.
// backgroundMask: 이 값이 255인 픽셀에만 적용됩니다(그 외는 원본 그대로
// 둠). 이 함수는 "가늘고 긴 선"을 찾는 방식이라 머리카락/눈썹/잔주름
// 같은 얼굴의 가는 선까지 스크래치로 오탐지하기 쉬우므로, 얼굴이 합성된
// 영역은 항상 제외한 배경 전용 마스크를 넘겨받아 그 영역에만 적용합니다.
static void apply_scratch_removal(cv::Mat &img, int strength, const cv::Mat &backgroundMask) {
    if (strength <= 0) return;
    if (strength > 100) strength = 100;

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // top-hat: 배경보다 밝은 가는 선(밝은 스크래치), black-hat: 배경보다
    // 어두운 가는 선(먼지/그을음/어두운 흠집)을 각각 검출합니다. 커널을
    // 작은 원형(3x3)으로 잡아서 폭이 좁고 배경과 뚜렷이 대비되는 결함만
    // 걸러내고, 넓은 면적의 자연스러운 명암(피부 톤, 배경 그라데이션)은
    // 건드리지 않습니다.
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

    cv::Mat topHat, blackHat;
    cv::morphologyEx(gray, topHat, cv::MORPH_TOPHAT, kernel);
    cv::morphologyEx(gray, blackHat, cv::MORPH_BLACKHAT, kernel);

    // detectThreshold: 강도가 높을수록 더 옅은(약한) 결함까지 잡아냄
    // (35 -> 8 범위, 값이 작을수록 민감).
    int detectThreshold = 35 - (strength * 27 / 100);
    if (detectThreshold < 8) detectThreshold = 8;

    cv::Mat brightMask, darkMask, scratchMask;
    cv::threshold(topHat, brightMask, detectThreshold, 255, cv::THRESH_BINARY);
    cv::threshold(blackHat, darkMask, detectThreshold, 255, cv::THRESH_BINARY);
    cv::bitwise_or(brightMask, darkMask, scratchMask);

    // 배경 영역(얼굴이 아닌 곳)에서 검출된 결함만 남깁니다.
    cv::bitwise_and(scratchMask, backgroundMask, scratchMask);

    if (cv::countNonZero(scratchMask) == 0) return;

    // 검출된 선을 1px 정도 살짝 넓혀서 경계까지 확실히 덮은 뒤 인페인팅합니다.
    cv::dilate(scratchMask, scratchMask, cv::Mat(), cv::Point(-1, -1), 1);

    // inpaintRadius: 결함 주변 몇 픽셀을 참고해서 채울지. 강도가 높을수록
    // 조금 더 넓게 참고(3~7px)해서 더 매끄럽게 메웁니다.
    double inpaintRadius = 3.0 + (strength / 100.0) * 4.0;
    cv::Mat result;
    cv::inpaint(img, scratchMask, result, inpaintRadius, cv::INPAINT_TELEA);
    img = result;
}

// strength: 0(끔) ~ 100. cv::fastNlMeansDenoisingColored의 h(밝기)/hColor(색상)
// 강도 파라미터로 매핑합니다 (대략 1~15 범위, OpenCV 권장 기본값이 h=3 부근).
static void apply_denoise(cv::Mat &img, int strength, const cv::Mat &backgroundMask) {
    if (strength <= 0) return;
    if (strength > 100) strength = 100;
    float h = 1.0f + (strength / 100.0f) * 14.0f;
    cv::Mat out;
    cv::fastNlMeansDenoisingColored(img, out, h, h, 7, 21);
    out.copyTo(img, backgroundMask);
}

// strength: 0(끔) ~ 100. CLAHE(Contrast Limited Adaptive Histogram Equalization)로
// 저노출/뿌연 사진의 대비를 지역적으로 끌어올립니다. RGB 채널에 직접 걸면
// 색이 틀어지므로, Lab 색공간의 밝기(L) 채널에만 적용하고 색상(a/b) 채널은
// 그대로 둡니다. clipLimit이 클수록 대비 향상이 강해지지만 노이즈도 같이
// 증폭될 수 있어 1.0~4.0 범위로 제한합니다. 타일 크기는 8x8 고정(표준값).
static void apply_clahe(cv::Mat &img, int strength, const cv::Mat &backgroundMask) {
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
    cv::Mat out;
    cv::cvtColor(lab, out, cv::COLOR_Lab2BGR);
    out.copyTo(img, backgroundMask);
}

// strength: 0(끔) ~ 100. 가우시안 블러 버전과의 차이를 더해주는 언샵 마스크 방식.
// amount가 클수록 윤곽선 대비가 강해집니다 (대략 0~2.0x 범위로 매핑).
static void apply_sharpen(cv::Mat &img, int strength, const cv::Mat &backgroundMask) {
    if (strength <= 0) return;
    if (strength > 100) strength = 100;
    double amount = (strength / 100.0) * 2.0;
    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, cv::Size(0, 0), 3.0);
    cv::Mat sharpened;
    cv::addWeighted(img, 1.0 + amount, blurred, -amount, 0, sharpened);
    sharpened.copyTo(img, backgroundMask);
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
static void apply_detail_enhance(cv::Mat &img, int strength, const cv::Mat &backgroundMask) {
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
    out.copyTo(img, backgroundMask);
}

// Runs the full restoration pipeline on a single image and writes the result.
// Models are already loaded, so this can be called repeatedly for batch processing.
static bool restore_one_image(GFPGAN &gfpgan,
#if RESTORE_WHOLE_IMAGE
                               Face &face_detector, BackgroundUpscaler &bg_upscaler,
#endif
                               Waifu2xDenoise *waifu2x_denoise,
                               const std::string &inputPath, const std::string &outputPath,
                               int denoiseStrength, int sharpenStrength, int claheStrength, int scale,
                               size_t maxFacesPerImage, int whiteBalance, int detailEnhanceStrength,
                               int vignetteStrength, int scratchStrength, int faceRestore) {
    // 방금 새로 생성/복사된 파일은 백신/실시간 보호 프로그램(Windows Defender,
    // Advanced SystemCare 등)이 짧게 스캔하느라 아주 잠깐 잠그는 경우가 있어서,
    // 그 타이밍에 걸리면 실제로는 멀쩡한 파일인데도 cv::imread가 실패합니다.
    // 최대 3번, 사이에 짧게 대기하며 재시도해서 이런 일시적 실패를 흡수합니다.
    cv::Mat img;
    const int kMaxImreadAttempts = 3;
    for (int attempt = 1; attempt <= kMaxImreadAttempts; attempt++) {
        img = cv::imread(inputPath, 1);
        if (!img.empty()) break;
        if (attempt < kMaxImreadAttempts) {
            fprintf(stderr, "  (retry %d/%d) cv::imread %s returned empty, retrying shortly - possibly locked by antivirus/real-time scan\n",
                    attempt, kMaxImreadAttempts - 1, inputPath.c_str());
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
    if (img.empty()) {
        fprintf(stderr, "cv::imread %s failed\n", inputPath.c_str());
        return false;
    }

    // -nn(AI 노이즈 제거): 파이프라인의 가장 앞단, RealESRGAN 업스케일보다도
    // 먼저 원본 이미지에 적용합니다. 노이즈가 있는 채로 업스케일하면 노이즈
    // 자체도 같이 확대되므로, 순서상 업스케일 이전이 이론적으로 맞습니다.
    // waifu2x_denoise가 nullptr이면(-nn 0, 기본값) 완전히 건너뜁니다.
    if (waifu2x_denoise != nullptr) {
        cv::Mat denoised;
        waifu2x_denoise->tile_process(img, denoised);
        img = denoised;
    }

    // Wrap the whole restoration pipeline so that an unexpected OpenCV/ncnn
    // exception on one image (e.g. an assertion failure, or a Vulkan error
    // surfaced as an exception) doesn't take down the whole batch run.
    // We log the failure and skip just this image; the caller's loop moves
    // on to the next file.
    try {

#if RESTORE_WHOLE_IMAGE
    cv::Mat bg_upsample;
    bg_upscaler.tile_process(img, bg_upsample);

    // 얼굴이 실제로 합성되는 영역을 누적할 마스크. 스크래치 제거(-sr)가
    // 이 영역은 절대 건드리지 않고 배경에만 적용되도록 하는 데 씁니다.
    // -fr 0(얼굴 복원 끔)이면 얼굴 영역이 없으므로 항상 빈 마스크로 남습니다.
    cv::Mat faceRegionMask = cv::Mat::zeros(bg_upsample.size(), CV_8UC1);

    if (faceRestore) {
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
    // bg_upscaler.scale: realesrgan-x4plus 모델의 네이티브 배율(4).
    // bg_upsample이 실제로 이 배율로 만들어지므로, 얼굴을 붙여넣을 좌표
    // 변환도 반드시 같은 배율을 써야 얼굴이 배경 위 정확한 위치에
    // 합성됩니다.
    face_detector.align_warp_face(img, objects, trans_matrix_inv, trans_img, bg_upscaler.scale);

    for (size_t i = 0; i < objects.size(); i++) {
        ncnn::Mat gfpgan_result;
        gfpgan.process(trans_img[i], gfpgan_result);

        cv::Mat restored_face;
        to_ocv(gfpgan_result, restored_face);

        paste_faces_to_input_image(restored_face, trans_matrix_inv[i], bg_upsample, faceRegionMask);
    }
    } // faceRestore

    // RealESRGAN 배경 업스케일 + GFPGAN 얼굴 보정은 항상 내부적으로 모델
    // 네이티브 배율(realesrgan-x4plus 기준 4배)로 수행됩니다 (모델 구조 자체가
    // 4배 고정이라 이 내부 처리 자체는 끌 수 없음). -s 옵션으로 고른 최종
    // 배율(1/2/3/4)이 이 네이티브 배율과 다르면, 화질 향상 효과는 그대로
    // 누리면서 최종 결과물 크기만 원하는 배율로 맞추기 위해 합성이 끝난
    // 직후 리사이즈합니다. 후처리(디노이즈/샤프닝)는 이 리사이즈가 끝난
    // "최종 배포 크기" 기준으로 적용되도록 그 다음에 수행합니다.
    if (scale != bg_upscaler.scale) {
        cv::Size targetSize(img.cols * scale, img.rows * scale);
        // 축소(다운샘플)는 INTER_AREA가, 확대(업샘플)는 INTER_LANCZOS4가
        // 더 선명하고 계단현상이 적어서 방향에 따라 보간 방식을 다르게 씁니다.
        int interp = (scale < bg_upscaler.scale) ? cv::INTER_AREA : cv::INTER_LANCZOS4;
        cv::resize(bg_upsample, bg_upsample, targetSize, 0, 0, interp);
        cv::resize(faceRegionMask, faceRegionMask, targetSize, 0, 0, cv::INTER_NEAREST);
    }

    // 얼굴 영역 경계 바로 바깥까지 안전하게 "얼굴"로 취급하도록 마스크를
    // 살짝 넓힌(팽창) 뒤, 나머지 전체를 "배경"으로 간주해 스크래치 제거의
    // 적용 범위를 제한합니다.
    cv::Mat faceRegionMaskSafe;
    cv::dilate(faceRegionMask, faceRegionMaskSafe, cv::Mat(), cv::Point(-1, -1), 3);
    cv::Mat backgroundMask;
    cv::bitwise_not(faceRegionMaskSafe, backgroundMask);

    // 얼굴 보정 + 합성이 전부 끝난 뒤, 크기 변경 없이(1배) 후처리 적용
    // 순서: 화이트밸런스(색 편향 보정) -> 비네팅 보정(가장자리 밝기 보정)
    //       -> 스크래치 제거(배경 전용) -> 디노이즈(잡티 제거) -> CLAHE(대비 향상)
    //       -> 디테일 향상(자연스러운 텍스처 보강) -> 샤프닝(마지막 선명도 마무리)
    apply_white_balance(bg_upsample, whiteBalance, backgroundMask);
    apply_vignette_correction(bg_upsample, vignetteStrength, backgroundMask);
    apply_scratch_removal(bg_upsample, scratchStrength, backgroundMask);
    apply_denoise(bg_upsample, denoiseStrength, backgroundMask);
    apply_clahe(bg_upsample, claheStrength, backgroundMask);
    apply_detail_enhance(bg_upsample, detailEnhanceStrength, backgroundMask);
    apply_sharpen(bg_upsample, sharpenStrength, backgroundMask);

    cv::imwrite(outputPath, bg_upsample);
#else
    ncnn::Mat gfpgan_result;
    gfpgan.process(img, gfpgan_result);

    cv::Mat restored_face;
    to_ocv(gfpgan_result, restored_face);

    // RESTORE_WHOLE_IMAGE=0 빌드(사용되지 않음)에서는 얼굴/배경 구분이
    // 없으므로, 이미지 전체를 "배경"으로 간주해 스크래치 제거를 적용합니다.
    cv::Mat fullMask = cv::Mat::ones(restored_face.size(), CV_8UC1) * 255;

    apply_white_balance(restored_face, whiteBalance, fullMask);
    apply_vignette_correction(restored_face, vignetteStrength, fullMask);
    apply_scratch_removal(restored_face, scratchStrength, fullMask);
    apply_denoise(restored_face, denoiseStrength, fullMask);
    apply_clahe(restored_face, claheStrength, fullMask);
    apply_detail_enhance(restored_face, detailEnhanceStrength, fullMask);
    apply_sharpen(restored_face, sharpenStrength, fullMask);

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
    enable_ansi_colors();  // Windows 콘솔에서 -h 출력 색상이 보이도록 VT100 처리를 켠다
    print_version();  // 매 실행마다 항상 찍음 - 구버전 exe를 새 버전으로 착각해서
                       // 헤매는 일을 막기 위함(스크린샷 한 장으로 바로 확인 가능)

    std::string imagepath;
    std::string outputpath;
    std::string modeldir = DEFAULT_MODEL_DIR;
    std::string realesrganModelDir = DEFAULT_REALESRGAN_MODEL_DIR;  // -rm
    std::string realcuganModelDir = DEFAULT_REALCUGAN_MODEL_DIR;    // -rcm
    std::string realesrganModelName = "realesrgan-x2plus";  // -rn, 이미지 종류에 맞게 선택
                                      // (명시적으로 안 주면 -s 값을 보고 아래에서 자동 결정됨)
                                      // realcugan-* 값을 주면 RealESRGAN이 아니라 RealCUGAN을 씀
    bool realesrganModelNameExplicit = false;  // 사용자가 -rn 을 직접 지정했는지 여부
    std::string format;
    int tilesize = 400;  // background upscale tile size, overridable via -t
    // 구형 GPU(fp16/int8 storage buffer 확장 미지원 또는 드라이버 버그)에서
    // 타일 결과가 깨질 때 fp32 경로로 강제 전환해 확인할 수 있는 진단용 옵션.
    // 기본값 false(기존 동작인 fp16/int8 최적화 사용) 유지.
    bool fp32Only = false;
    // RealCUGAN 전용 - 켜면 타일 경계의 SE(Squeeze-Excitation) 게이트를
    // 전체 이미지 기준으로 동기화해서 은은한 격자 얼룩을 없앰. 신경망을
    // 사실상 2번 통과해야 해서 처리 시간이 대략 2배로 늘어나므로 기본은 끔.
    // RealESRGAN 모델을 쓸 때는 이 값 자체가 무시됨(RealESRGAN엔 SE 레이어가
    // 없어서 애초에 해당 없음).
    int syncGap = 0;
    int denoiseStrength = 0;  // -dn, 0-100, default off
    int sharpenStrength = 0;  // -sp, 0-100, default off
    int claheStrength = 0;  // -cl, 0-100, default off (CLAHE 자동 대비 향상)
    int scale = 2;  // -s, 최종 출력 배율(원본 대비): 1/2/3/4, 기본값 2
    int maxFaces = 5;  // -mf, 이미지 한 장당 처리할 최대 얼굴 후보 수 (기본 5)
    int whiteBalance = 0;  // -wb, 0=off(기본)/1=on, 자동 화이트밸런스(Gray-World)
    int detailEnhanceStrength = 0;  // -de, 0-100, default off, edge-preserving 디테일 향상
    int vignetteStrength = 0;  // -vg, 0-100, default off, 비네팅(가장자리 어두워짐) 보정
    int scratchStrength = 0;  // -sr, 0-100, default off, 스크래치/먼지 제거(배경 전용, 인페인팅)
    int faceRestore = 1;  // -fr, 0=끔/1=켬(기본), 얼굴 검출 + GFPGAN 얼굴 복원 여부
    int aiDenoiseLevel = 0;  // -nn, 0=off(기본)/1/2/3, waifu2x cunet AI 노이즈 제거
    std::string waifu2xModelDir = DEFAULT_WAIFU2X_MODEL_DIR;  // -wm
    int cpuThreads = -1;  // -threads, OpenCV 후처리(CPU)용 스레드 수 제한, 기본 -1(무제한)
    int delayMs = 0;  // -delay-ms, 폴더 일괄 처리 시 이미지 한 장마다 대기(ms), 기본 0

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
        } else if (arg == "-rm" && i + 1 < argc) {
            realesrganModelDir = argv[++i];
        } else if (arg == "-rcm" && i + 1 < argc) {
            realcuganModelDir = argv[++i];
        } else if (arg == "-rn" && i + 1 < argc) {
            realesrganModelName = argv[++i];
            realesrganModelNameExplicit = true;
        } else if (arg == "-f" && i + 1 < argc) {
            format = argv[++i];
        } else if (arg == "-t" && i + 1 < argc) {
            tilesize = std::atoi(argv[++i]);
        } else if (arg == "-fp32" && i + 1 < argc) {
            fp32Only = std::atoi(argv[++i]) != 0;
        } else if (arg == "-syncgap" && i + 1 < argc) {
            syncGap = std::atoi(argv[++i]);
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
        } else if (arg == "-vg" && i + 1 < argc) {
            vignetteStrength = std::atoi(argv[++i]);
        } else if (arg == "-sr" && i + 1 < argc) {
            scratchStrength = std::atoi(argv[++i]);
        } else if (arg == "-fr" && i + 1 < argc) {
            faceRestore = std::atoi(argv[++i]);
        } else if (arg == "-nn" && i + 1 < argc) {
            aiDenoiseLevel = std::atoi(argv[++i]);
        } else if (arg == "-wm" && i + 1 < argc) {
            waifu2xModelDir = argv[++i];
        } else if (arg == "-threads" && i + 1 < argc) {
            cpuThreads = std::atoi(argv[++i]);
        } else if (arg == "-delay-ms" && i + 1 < argc) {
            delayMs = std::atoi(argv[++i]);
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

    if (tilesize < 0) {
        fprintf(stderr, "Error: -t tile-size must be 0 (no tiling) or a positive integer (got '%d')\n\n", tilesize);
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

    if (scale < 1 || scale > 4) {
        fprintf(stderr, "Error: -s scale must be 1, 2, 3 or 4 (default = 2) (got '%d')\n\n", scale);
        print_usage(argv[0]);
        return -1;
    }

    // -rn 을 직접 지정하지 않았다면, -s 값에 맞춰 기본 모델을 자동으로 고릅니다.
    // x2plus(네이티브 2배)로 -s 3/4를 만들면 네이티브 배율 이상으로 다시 확대해야
    // 해서 x4plus(네이티브 4배)를 그대로 쓰는 것보다 디테일이 아쉬울 수 있고,
    // 반대로 -s 1/2에 x4plus를 쓰면 어차피 다시 축소할 4배 연산을 굳이 다 돌리는
    // 셈이라 GPU가 낭비됩니다. 그래서 요청한 -s 값과 가장 가까운 네이티브 배율의
    // 모델을 기본으로 선택합니다.
    if (!realesrganModelNameExplicit) {
        realesrganModelName = (scale <= 2) ? "realesrgan-x2plus" : "realesrgan-x4plus";
    }

    // realcugan-*는 별도 클래스(RealCUGAN)로 처리되므로, 이름 검증도
    // RealESRGAN 계열과 RealCUGAN 계열로 나눠서 합니다.
    const bool useRealCUGAN = realesrganModelName.rfind("realcugan-", 0) == 0;

    if (!useRealCUGAN
        && realesrganModelName != "realesrgan-x4plus"
        && realesrganModelName != "realesrgan-x2plus"
        && realesrganModelName != "realesr-animevideov3") {
        fprintf(stderr, "Error: -rn model-name must be one of realesrgan-x4plus, "
                         "realesrgan-x2plus, realesr-animevideov3, realcugan-pro-2x, "
                         "realcugan-pro-2x-conservative, realcugan-pro-3x, "
                         "realcugan-pro-3x-conservative, realcugan-se-4x, "
                         "realcugan-se-4x-conservative (got '%s')\n\n",
                realesrganModelName.c_str());
        print_usage(argv[0]);
        return -1;
    }

    if (useRealCUGAN
        && realesrganModelName != "realcugan-pro-2x"
        && realesrganModelName != "realcugan-pro-2x-conservative"
        && realesrganModelName != "realcugan-pro-3x"
        && realesrganModelName != "realcugan-pro-3x-conservative"
        && realesrganModelName != "realcugan-se-4x"
        && realesrganModelName != "realcugan-se-4x-conservative") {
        fprintf(stderr, "Error: -rn realcugan model-name must be one of realcugan-pro-2x, "
                         "realcugan-pro-2x-conservative, realcugan-pro-3x, "
                         "realcugan-pro-3x-conservative, realcugan-se-4x, "
                         "realcugan-se-4x-conservative (got '%s')\n\n",
                realesrganModelName.c_str());
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

    if (vignetteStrength < 0 || vignetteStrength > 100) {
        fprintf(stderr, "Error: -vg vignette-correct must be between 0 and 100 (got '%d')\n\n", vignetteStrength);
        print_usage(argv[0]);
        return -1;
    }

    if (scratchStrength < 0 || scratchStrength > 100) {
        fprintf(stderr, "Error: -sr scratch-removal must be between 0 and 100 (got '%d')\n\n", scratchStrength);
        print_usage(argv[0]);
        return -1;
    }

    if (faceRestore != 0 && faceRestore != 1) {
        fprintf(stderr, "Error: -fr face-restore must be 0 (off) or 1 (on, default) (got '%d')\n\n", faceRestore);
        print_usage(argv[0]);
        return -1;
    }

    if (aiDenoiseLevel < 0 || aiDenoiseLevel > 3) {
        fprintf(stderr, "Error: -nn ai-denoise-level must be 0 (off, default), 1, 2, or 3 (got '%d')\n\n", aiDenoiseLevel);
        print_usage(argv[0]);
        return -1;
    }

    if (delayMs < 0) {
        fprintf(stderr, "Error: -delay-ms must be 0 or a positive integer (got '%d')\n\n", delayMs);
        print_usage(argv[0]);
        return -1;
    }

    // -threads: OpenCV 후처리(CPU) 전용 스레드 수 제한. GPU(Vulkan) 연산에는 영향 없음.
    // -1(기본)은 OpenCV가 시스템 기본값(보통 코어 수만큼)을 쓰도록 되돌리는 의미이며,
    // 0은 OpenCV의 스레딩 최적화 자체를 끄는 의미(사실상 단일 스레드)입니다.
    cv::setNumThreads(cpuThreads);

    if (!fs::exists(imagepath)) {
        fprintf(stderr, "Error: input path '%s' does not exist\n", imagepath.c_str());
        return -1;
    }

    // Strip a single trailing slash/backslash so path concatenation below is clean.
    if (!modeldir.empty() && (modeldir.back() == '/' || modeldir.back() == '\\')) {
        modeldir.pop_back();
    }
    if (!realesrganModelDir.empty() && (realesrganModelDir.back() == '/' || realesrganModelDir.back() == '\\')) {
        realesrganModelDir.pop_back();
    }
    if (!realcuganModelDir.empty() && (realcuganModelDir.back() == '/' || realcuganModelDir.back() == '\\')) {
        realcuganModelDir.pop_back();
    }

    GFPGAN gfpgan;
    gfpgan.load(modeldir + "/encoder.param", modeldir + "/encoder.bin", modeldir + "/style.bin");

#if RESTORE_WHOLE_IMAGE
    Face face_detector;
    face_detector.load(modeldir + "/yolov5-blazeface.param", modeldir + "/yolov5-blazeface.bin");

    RealESRGAN real_esrgan(0, false, fp32Only);
    RealCUGAN real_cugan(0, fp32Only);
    BackgroundUpscaler bg_upscaler;

    // 공식 realesrgan-ncnn-vulkan 배포본의 파일명을 변형 없이 그대로 사용합니다.
    // realesrgan-x4plus: 네트워크 구조 자체가 4배 고정이라, 파일명에 배율이
    //   붙지 않습니다. 최종 원하는 배율(-s)이 4가 아니면 4배로 처리한 뒤
    //   나중에 리사이즈합니다 (restore_one_image 참고).
    // realesrgan-x2plus: 위와 같은 계열(RRDB)이지만 네트워크 구조 자체가 2배
    //   고정입니다. x4plus 대비 GPU 연산량이 대략 절반이라, 저사양 GPU에서
    //   화질과 속도의 절충안으로 씁니다. -s가 2가 아니면 2배로 처리한 뒤
    //   나중에 리사이즈합니다(x4plus와 동일한 흐름, 기준 배율만 다름).
    // realesr-animevideov3: 배율별로 별도 모델(-x2/-x3/-x4)이 나뉘어 있어서,
    //   원하는 최종 배율(-s)에 맞는 파일을 바로 불러오면 되고, 후처리
    //   리사이즈가 필요 없습니다.
    // realcugan-*: 애니메이션/일러스트 전용(realesrgan-x4plus-anime를 대체).
    //   -rn 값 -> 실제 모델 파일명 매핑은 realcugan-models 폴더에 실제로
    //   들어있는 파일 이름(pro/se 폴더에서 골라온 5개) 기준입니다.
    //   noise 레벨 모델(denoise3x)은 애초에 채택하지 않았으므로 여기 없음
    //   (-AiDenoise로 이미 노이즈 제거를 마쳤다고 가정).
    if (useRealCUGAN) {
        std::string realcuganFile;
        int cuganScale;
        if (realesrganModelName == "realcugan-pro-2x") {
            realcuganFile = "up2x-no-denoise";
            cuganScale = 2;
        } else if (realesrganModelName == "realcugan-pro-2x-conservative") {
            realcuganFile = "up2x-conservative";
            cuganScale = 2;
        } else if (realesrganModelName == "realcugan-pro-3x") {
            realcuganFile = "up3x-no-denoise";
            cuganScale = 3;
        } else if (realesrganModelName == "realcugan-pro-3x-conservative") {
            realcuganFile = "up3x-conservative";
            cuganScale = 3;
        } else if (realesrganModelName == "realcugan-se-4x") {
            realcuganFile = "up4x-no-denoise";
            cuganScale = 4;
        } else { // realcugan-se-4x-conservative (up4x는 pro에 없어서 se에서만 가져옴)
            realcuganFile = "up4x-conservative";
            cuganScale = 4;
        }

        std::string realcuganParam = realcuganModelDir + "/" + realcuganFile + ".param";
        std::string realcuganModel = realcuganModelDir + "/" + realcuganFile + ".bin";

        if (real_cugan.load(realcuganParam, realcuganModel) < 0) {
            fprintf(stderr, "Error: failed to load RealCUGAN model '%s' from '%s'\n",
                    realesrganModelName.c_str(), realcuganModelDir.c_str());
            return -1;
        }

        bg_upscaler.set_backend(&real_cugan);
        bg_upscaler.set_scale(cuganScale);
        bg_upscaler.set_tile_size(tilesize);
        // RealCUGAN은 valid-conv(패딩 없음) 구조라, 타일 주변에 얼마나 여유
        // 컨텍스트(prepadding)를 줘야 하는지가 RealESRGAN(prepadding=10 고정)과
        // 전혀 다릅니다. 배율마다 정확히 이 값이어야 네트워크 출력 크기가
        // 딱 맞아떨어집니다 (공식 realcugan-ncnn-vulkan main.cpp 기준값 -
        // models-se/pro/nose 공통). 이 값이 안 맞으면 출력 타일 크기가
        // 어긋나서 이미지가 잘게 쪼개져 잘못 이어붙는 증상이 납니다.
        if (cuganScale == 2) real_cugan.tile_pad = 18;
        else if (cuganScale == 3) real_cugan.tile_pad = 14;
        else real_cugan.tile_pad = 19; // cuganScale == 4
        real_cugan.syncgap = syncGap;
    } else {
        std::string realesrganParam, realesrganModel;
        int esrganScale;
        if (realesrganModelName == "realesr-animevideov3") {
            // animevideov3는 2/3/4배 모델만 배포되고 1배(원본 크기) 모델은 없습니다.
            // -s 1을 고른 경우, 가장 작은 x2 모델을 불러온 뒤 restore_one_image의
            // 리사이즈 단계(scale != bg_upscaler.scale)에서 1배로 축소되도록 합니다.
            int modelScale = (scale == 1) ? 2 : scale;
            std::string suffix = "-x" + std::to_string(modelScale);
            realesrganParam = realesrganModelDir + "/" + realesrganModelName + suffix + ".param";
            realesrganModel = realesrganModelDir + "/" + realesrganModelName + suffix + ".bin";
            esrganScale = modelScale;
        } else {
            realesrganParam = realesrganModelDir + "/" + realesrganModelName + ".param";
            realesrganModel = realesrganModelDir + "/" + realesrganModelName + ".bin";
            // x4plus는 네트워크 구조상 4배 고정, x2plus는 2배 고정.
            // (x2plus는 입력 단에서 pixel-unshuffle로 공간 해상도를 먼저 절반으로
            // 줄이고 그만큼 채널을 늘린 뒤 RRDB를 거치는 구조라, x4plus와 파라미터
            // 수/레이어 수는 비슷해도 최종 네이티브 배율은 2배입니다.)
            esrganScale = (realesrganModelName == "realesrgan-x2plus") ? 2 : 4;
        }

        if (real_esrgan.load(realesrganParam, realesrganModel) < 0) {
            fprintf(stderr, "Error: failed to load RealESRGAN model '%s' from '%s'\n",
                    realesrganModelName.c_str(), realesrganModelDir.c_str());
            return -1;
        }

        bg_upscaler.set_backend(&real_esrgan);
        bg_upscaler.set_scale(esrganScale);
        bg_upscaler.set_tile_size(tilesize);
    }
#endif

    // -nn(AI 노이즈 제거): 0(기본, 꺼짐)이면 Waifu2xDenoise를 아예 만들지
    // 않고 nullptr로 둬서, GPU 로드/VRAM 낭비 없이 완전히 건너뜁니다.
    // RESTORE_WHOLE_IMAGE 매크로와 무관하게(두 빌드 경로 모두) 동작합니다.
    Waifu2xDenoise *waifu2x_denoise = nullptr;
    if (aiDenoiseLevel > 0) {
        waifu2x_denoise = new Waifu2xDenoise(0, fp32Only);
        // 공식 waifu2x-ncnn-vulkan models-cunet 배포본과 동일한 파일명
        // (noise1_model.param/.bin ~ noise3_model.param/.bin, scale 접미사
        // 없는 버전 - 배율은 안 바꾸고 노이즈만 제거).
        std::string noiseParam = waifu2xModelDir + "/noise" + std::to_string(aiDenoiseLevel) + "_model.param";
        std::string noiseModel = waifu2xModelDir + "/noise" + std::to_string(aiDenoiseLevel) + "_model.bin";
        if (waifu2x_denoise->load(noiseParam, noiseModel) < 0) {
            fprintf(stderr, "Error: failed to load waifu2x -nn model from '%s' (-wm to change the folder)\n\n", waifu2xModelDir.c_str());
            return -1;
        }
        // RealESRGAN과 같은 -t 값을 공유(간단하게 시작; 문제가 생기면 나중에
        // -nn 전용 타일 크기 옵션으로 분리 가능).
        waifu2x_denoise->tile_size = tilesize;
    }

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
        int attempted = 0;  // 확장자가 맞아서 실제로 시도된 파일 수 (실패 포함) - "파일이
                            // 아예 없음"과 "찾았지만 전부 실패함"을 구분하기 위함
        for (const auto &entry : fs::directory_iterator(imagepath)) {
            if (!is_image_file(entry.path())) continue;
            attempted++;

            std::string outExt = !format.empty() ? to_lower(format) : "png";
            std::string outName = entry.path().stem().string() + "." + outExt;
            std::string outPath = (fs::path(outDir) / outName).string();

            fprintf(stderr, "Processing %s -> %s\n", entry.path().string().c_str(), outPath.c_str());
#if RESTORE_WHOLE_IMAGE
            if (restore_one_image(gfpgan, face_detector, bg_upscaler, waifu2x_denoise, entry.path().string(), outPath,
                                   denoiseStrength, sharpenStrength, claheStrength, scale, maxFaces, whiteBalance, detailEnhanceStrength, vignetteStrength, scratchStrength, faceRestore)) {
#else
            if (restore_one_image(gfpgan, waifu2x_denoise, entry.path().string(), outPath,
                                   denoiseStrength, sharpenStrength, claheStrength, scale, maxFaces, whiteBalance, detailEnhanceStrength, vignetteStrength, scratchStrength, faceRestore)) {
#endif
                processed++;
            }

            // -delay-ms: 이미지 한 장이 끝날 때마다(성공/실패 무관) 대기.
            // GPU가 쉬지 않고 연속 풀가동되는 것을 완화하기 위한 옵션.
            if (delayMs > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
            }
        }

        if (attempted == 0) {
            fprintf(stderr, "No jpg/png/webp images found in '%s'\n", imagepath.c_str());
            return -1;
        }
        if (processed == 0) {
            fprintf(stderr, "Found %d jpg/png/webp image(s) in '%s' but every one failed to process - see the errors above.\n",
                    attempted, imagepath.c_str());
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
        if (!restore_one_image(gfpgan, face_detector, bg_upscaler, waifu2x_denoise, imagepath, outPath,
                                denoiseStrength, sharpenStrength, claheStrength, scale, maxFaces, whiteBalance, detailEnhanceStrength, vignetteStrength, scratchStrength, faceRestore)) {
#else
        if (!restore_one_image(gfpgan, waifu2x_denoise, imagepath, outPath,
                                denoiseStrength, sharpenStrength, claheStrength, scale, maxFaces, whiteBalance, detailEnhanceStrength, vignetteStrength, scratchStrength, faceRestore)) {
#endif
            return -1;
        }
        fprintf(stderr, "Saved %s\n", outPath.c_str());
    }

    return 0;
}
