#pragma once
#include "Network.h"
#include "ImageSegmenter.h"
struct Recognition {
    std::string text;
    std::vector<int> digits; // 修改：-1 表示拒识，text 中对应 ?，保留位序。
    std::vector<double> scores;
    cv::Mat preview;
    int uncertain = 0;
    bool touching = false;
};
Recognition recognize(Network& net, const cv::Mat& image);
int runDesktop();
