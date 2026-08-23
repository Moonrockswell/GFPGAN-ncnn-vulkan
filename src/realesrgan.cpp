#include "realesrgan.h"

RealESRGAN::RealESRGAN() {
    net.opt.use_vulkan_compute = true;
    net.opt.num_threads = 4;
    // realesrgan-x4plus 모델은 네트워크 구조 자체가 4배 업스케일로 고정되어
    // 있습니다(2배/3배로 학습된 모델이 아님). 최종 출력 배율을 2/3배로
    // 낮추고 싶을 때는, 이 4배 네이티브 결과를 만든 뒤 호출부에서 원하는
    // 배율로 다시 축소(resize)하는 방식으로 처리합니다.
    scale = 4;
    tile_size = 400;
    tile_pad = 10;
}

RealESRGAN::~RealESRGAN() {
    net.clear();
}

int RealESRGAN::load(const std::string &param_path, const std::string &model_path) {
    int ret = net.load_param(param_path.c_str());
    if (ret < 0) {
        fprintf(stderr, "open param file %s failed\n", param_path.c_str());
        return -1;
    }
    ret = net.load_model(model_path.c_str());
    if (ret < 0) {
        fprintf(stderr, "open bin file %s failed\n", model_path.c_str());
        return -1;
    }

    return 0;
}

cv::Mat RealESRGAN::to_ocv(const cv::Mat &source, const ncnn::Mat &result) {
    cv::Mat cv_result_32F = cv::Mat::zeros(cv::Size(result.w, result.h), CV_32FC3);
    for (int i = 0; i < result.h; i++) {
        for (int j = 0; j < result.w; j++) {
            cv_result_32F.at<cv::Vec3f>(i, j)[2] = result.channel(0)[i * result.w + j];
            cv_result_32F.at<cv::Vec3f>(i, j)[1] = result.channel(1)[i * result.w + j];
            cv_result_32F.at<cv::Vec3f>(i, j)[0] = result.channel(2)[i * result.w + j];
        }
    }

    cv::Mat cv_result_8U;
    cv_result_32F.convertTo(cv_result_8U, CV_8UC3, 255.0, 0);

    return cv_result_8U;
}

int RealESRGAN::preprocess(const cv::Mat &img, cv::Mat &pad_img, int &img_pad_h, int &img_pad_w) {
    if (img.cols % 2 != 0) {
        img_pad_w = (2 - img.cols % 2);
    }
    if (img.rows % 2 != 0) {
        img_pad_h = (2 - img.rows % 2);
    }
    cv::copyMakeBorder(img, pad_img, 0, img_pad_h, 0, img_pad_w, cv::BORDER_CONSTANT, cv::Scalar(0));

    return 0;
}

int RealESRGAN::inference(const cv::Mat &in, ncnn::Mat &out, int w, int h) {
    ncnn::Mat ncnn_in = ncnn::Mat::from_pixels(in.data, ncnn::Mat::PIXEL_BGR2RGB, w, h);

    ncnn_in.substract_mean_normalize(0, norm_vals);
    ncnn::Extractor ex = net.create_extractor();
    ex.input("input", ncnn_in);
    ex.extract("output", out);

    return 0;
}

int RealESRGAN::tile_process(const cv::Mat &inimage, cv::Mat &outimage) {
    cv::Mat pad_inimage;
    int img_pad_w = 0, img_pad_h = 0;
    preprocess(inimage, pad_inimage, img_pad_w, img_pad_h);

    int tiles_x = std::ceil((float) inimage.cols / tile_size);
    int tiles_y = std::ceil((float) inimage.rows / tile_size);

    // 이전에는 여기 리터럴 2가 하드코딩되어 있어서, scale 멤버 변수를
    // 2가 아닌 값(예: realesrgan-x4plus의 4배)으로 바꾸면 출력 캔버스
    // 크기와 아래 타일 배치 좌표 계산(* scale)이 서로 어긋나는 잠재
    // 버그가 있었습니다. scale로 통일해서 실제 모델 배율과 항상 맞도록
    // 고쳤습니다.
    cv::Mat out = cv::Mat(cv::Size(pad_inimage.cols * scale, pad_inimage.rows * scale), CV_8UC3);
    for (int i = 0; i < tiles_y; i++) {
        for (int j = 0; j < tiles_x; j++) {
            int ofs_x = j * tile_size;
            int ofs_y = i * tile_size;

            int input_start_x = ofs_x;
            int input_end_x = std::min(ofs_x + tile_size, pad_inimage.cols);
            int input_start_y = ofs_y;
            int input_end_y = std::min(ofs_y + tile_size, pad_inimage.rows);

            int input_start_x_pad = std::max(input_start_x - tile_pad, 0);
            int input_end_x_pad = std::min(input_end_x + tile_pad, pad_inimage.cols);
            int input_start_y_pad = std::max(input_start_y - tile_pad, 0);
            int input_end_y_pad = std::min(input_end_y + tile_pad, pad_inimage.rows);

            int input_tile_width = input_end_x - input_start_x;
            int input_tile_height = input_end_y - input_start_y;

            cv::Mat input_tile = pad_inimage(
                    cv::Rect(input_start_x_pad, input_start_y_pad, input_end_x_pad - input_start_x_pad,
                             input_end_y_pad - input_start_y_pad)).clone();
            //infer
            ncnn::Mat ncnn_out;
            inference(input_tile, ncnn_out, input_end_x_pad - input_start_x_pad, input_end_y_pad - input_start_y_pad);
            //to mat
            cv::Mat out_tile = to_ocv(input_tile, ncnn_out);

            int output_start_x = input_start_x * scale;
            int output_end_x = input_end_x * scale;
            int output_start_y = input_start_y * scale;
            int output_end_y = input_end_y * scale;

            int output_start_x_tile = (input_start_x - input_start_x_pad) * scale;
            int output_end_x_tile = output_start_x_tile + input_tile_width * scale;
            int output_start_y_tile = (input_start_y - input_start_y_pad) * scale;
            int output_end_y_tile = output_start_y_tile + input_tile_height * scale;
            cv::Rect tile_roi = cv::Rect(output_start_x_tile, output_start_y_tile,
                                         output_end_x_tile - output_start_x_tile,
                                         output_end_y_tile - output_start_y_tile);
            cv::Rect out_roi = cv::Rect(output_start_x, output_start_y,
                                        output_end_x - output_start_x, output_end_y - output_start_y);
            out_tile(tile_roi).copyTo(out(out_roi));
        }
    }

    out.copyTo(outimage);
    return 0;

}
