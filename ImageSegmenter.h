#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct DigitSegment {
    cv::Mat image; // 28x28 黑底白字，保持宽高比并按灰度重心居中。
    int x_start = 0, x_end = 0;
    cv::Rect bounds;
    int line = 0;
    bool touching = false;
};
class ImageSegmenter {
    cv::Mat original;
public:
    ImageSegmenter() = default;
    std::vector<DigitSegment> segment(const std::string& imagePath);
    std::vector<DigitSegment> segment(const cv::Mat& image);
    cv::Mat drawBoundingBoxes(const std::vector<DigitSegment>& segments);
};
