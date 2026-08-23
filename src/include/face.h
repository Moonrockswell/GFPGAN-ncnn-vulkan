#ifndef FACE_H
#define FACE_H
#include <cfloat>
#include <cstdio>
#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "net.h"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float score;
    std::vector<cv::Point2f> pts;
    cv::Mat trans_inv;

};

class Face
{
public:
    Face();
    ~Face();
    int load(const std::string& param_path, const std::string& model_path);
    int detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold = 0.7f, float nms_threshold = 0.3f);
    // bg_upscale: 얼굴을 붙여넣을 배경 이미지가 원본 대비 몇 배로
    // 업스케일되어 있는지. RealESRGAN의 네이티브 배율(realesrgan-x4plus
    // 기준 4)과 항상 일치시켜야 얼굴이 배경 위 정확한 위치/크기로
    // 합성됩니다. 기본값 2는 이전 동작과의 호환용입니다.
    int align_warp_face(cv::Mat& img, const std::vector<Object>& objects, std::vector<cv::Mat>& trans_matrix_inv, std::vector<cv::Mat>& trans_img, int bg_upscale = 2);
    void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects);
private:
    ncnn::Net net;

};

#endif // FACE_H
