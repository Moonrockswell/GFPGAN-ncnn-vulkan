#include <cstdio>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>
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
        "Usage: %s -i infile -o outfile [options]...\n"
        "  -h                   show this help\n"
        "  -i input-path        input image path (jpg/png/webp) or directory\n"
        "  -o output-path       output image path (jpg/png/webp) or directory\n"
        "  -m model-path        folder path to the pre-trained models (default=%s)\n"
        "  -f output format       output image format (jpg/png/webp, default=ext/png)\n"
        "  -t tile-size           background upscale tile size, must be > 0 (default = 400)\n"
        "                          smaller values reduce GPU memory/load per step,\n"
        "                          useful to avoid GPU timeouts (Vulkan device lost) on older GPUs\n"
        "*Unmodifiable Options*\n"
        " -s scale               upscale ratio (default=2)\n"
        " -n model name     GFPGANCleanv1-NoCE-C2 supports only one type of model\n"
        "\n"
        "If -o is omitted:\n"
        "  - single file input : saved next to the input as <name>-output.<ext>\n"
        "  - folder input       : saved into a new '<foldername>-output' subfolder\n"
        "                          inside the input folder, original filenames kept\n",
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
// Extension follows -f when given, otherwise the input file's own extension,
// falling back to png if that extension isn't a supported image format.
static std::string default_single_output(const std::string &inputPath, const std::string &format) {
    fs::path p(inputPath);
    std::string stem = p.stem().string();

    std::string ext;
    if (!format.empty()) {
        ext = to_lower(format);
    } else {
        std::string inExt = p.extension().string();
        if (!inExt.empty() && inExt[0] == '.') inExt = inExt.substr(1);
        inExt = to_lower(inExt);
        ext = is_supported_format(inExt) ? inExt : "png";
    }

    fs::path outPath = p.parent_path() / (stem + "-output." + ext);
    return outPath.string();
}

// Default output folder for a directory input when -o is not given:
// "<foldername>-output" created as a subfolder inside the input folder,
// e.g. ./image -> ./image/image-output
static std::string default_output_folder(const std::string &inputDir) {
    fs::path p(inputDir);
    std::string dirname = p.filename().string();
    if (dirname.empty()) {
        // handles a trailing slash, e.g. "./image/"
        dirname = p.parent_path().filename().string();
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
    int w_edge = int(std::sqrt(total_face_area) / 20);
    int erosion_radius = w_edge * 2;
    cv::Mat inv_mask_center;

    kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(erosion_radius, erosion_radius));
    cv::erode(inv_mask_erosion, inv_mask_center, kernel);

    int blur_size = w_edge * 2;
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

// Runs the full restoration pipeline on a single image and writes the result.
// Models are already loaded, so this can be called repeatedly for batch processing.
static bool restore_one_image(GFPGAN &gfpgan,
#if RESTORE_WHOLE_IMAGE
                               Face &face_detector, RealESRGAN &real_esrgan,
#endif
                               const std::string &inputPath, const std::string &outputPath) {
    cv::Mat img = cv::imread(inputPath, 1);
    if (img.empty()) {
        fprintf(stderr, "cv::imread %s failed\n", inputPath.c_str());
        return false;
    }

#if RESTORE_WHOLE_IMAGE
    cv::Mat bg_upsample;
    real_esrgan.tile_process(img, bg_upsample);

    std::vector<cv::Mat> trans_img;
    std::vector<cv::Mat> trans_matrix_inv;
    std::vector<Object> objects;
    face_detector.detect(img, objects);
    face_detector.align_warp_face(img, objects, trans_matrix_inv, trans_img);

    for (size_t i = 0; i < objects.size(); i++) {
        ncnn::Mat gfpgan_result;
        gfpgan.process(trans_img[i], gfpgan_result);

        cv::Mat restored_face;
        to_ocv(gfpgan_result, restored_face);

        paste_faces_to_input_image(restored_face, trans_matrix_inv[i], bg_upsample);
    }
    cv::imwrite(outputPath, bg_upsample);
#else
    ncnn::Mat gfpgan_result;
    gfpgan.process(img, gfpgan_result);

    cv::Mat restored_face;
    to_ocv(gfpgan_result, restored_face);
    cv::imwrite(outputPath, restored_face);
#endif

    return true;
}

int main(int argc, char **argv) {
    std::string imagepath;
    std::string outputpath;
    std::string modeldir = DEFAULT_MODEL_DIR;
    std::string format;
    int tilesize = 400;  // background upscale tile size, overridable via -t

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

            std::string outName = format.empty()
                                   ? entry.path().filename().string()
                                   : (entry.path().stem().string() + "." + to_lower(format));
            std::string outPath = (fs::path(outDir) / outName).string();

            fprintf(stderr, "Processing %s -> %s\n", entry.path().string().c_str(), outPath.c_str());
#if RESTORE_WHOLE_IMAGE
            if (restore_one_image(gfpgan, face_detector, real_esrgan, entry.path().string(), outPath)) {
#else
            if (restore_one_image(gfpgan, entry.path().string(), outPath)) {
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
        if (!restore_one_image(gfpgan, face_detector, real_esrgan, imagepath, outPath)) {
#else
        if (!restore_one_image(gfpgan, imagepath, outPath)) {
#endif
            return -1;
        }
        fprintf(stderr, "Saved %s\n", outPath.c_str());
    }

    return 0;
}
