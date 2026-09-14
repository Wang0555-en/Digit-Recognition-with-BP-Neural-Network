#include "ImageSegmenter.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>

std::vector<DigitSegment> ImageSegmenter::segment(const std::string& path) {
    // 通过文件流解码，支持 Windows 中文和带空格的 UTF-8 路径。
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open image");
    std::vector<uchar> bytes((std::istreambuf_iterator<char>(file)), {});
    return segment(cv::imdecode(bytes, cv::IMREAD_UNCHANGED));
}

std::vector<DigitSegment> ImageSegmenter::segment(const cv::Mat& source) {
    if (source.empty()) throw std::runtime_error("Invalid image");
    if (source.total() > 24000000) throw std::runtime_error("Image too large (24 megapixels maximum)");
    cv::Mat gray;
    if (source.channels() == 4) {
        // 透明 PNG 合成到白底，避免透明背景被当作笔画。
        cv::Mat bgr(source.size(), CV_8UC3);
        for (int y = 0; y < source.rows; ++y) for (int x = 0; x < source.cols; ++x) {
            auto p = source.at<cv::Vec4b>(y,x);
            for (int c = 0; c < 3; ++c) bgr.at<cv::Vec3b>(y,x)[c] = uchar((p[c]*p[3]+255*(255-p[3]))/255);
        }
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    } else if (source.channels() == 3) cv::cvtColor(source, gray, cv::COLOR_BGR2GRAY);
    else gray = source.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8U, gray.depth() == CV_16U ? 1.0/256 : 1);
    original = gray.clone();
    cv::Mat binary;
    double low, high; cv::minMaxLoc(gray, &low, &high);
    if (high - low < 12) return {};
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    // 边缘背景决定反色；适用于白纸黑字和黑板白字。
    int white = cv::countNonZero(binary.row(0)) + cv::countNonZero(binary.row(binary.rows-1))
              + cv::countNonZero(binary.col(0)) + cv::countNonZero(binary.col(binary.cols-1));
    bool lightBackground = white > binary.rows + binary.cols;
    if (lightBackground) { cv::bitwise_not(binary,binary); cv::bitwise_not(gray,gray); }
    cv::Mat labels, stats, centers;
    int n = cv::connectedComponentsWithStats(binary, labels, stats, centers);
    int largest = 0;
    for (int i=1;i<n;++i) largest = std::max(largest,stats.at<int>(i,cv::CC_STAT_AREA));
    std::vector<cv::Rect> components;
    cv::Mat cleaned = cv::Mat::zeros(binary.size(),CV_8U);
    for (int i=1;i<n;++i) {
        if (stats.at<int>(i,cv::CC_STAT_AREA) < std::max(2,largest/200)) continue;
        cv::Rect r(stats.at<int>(i,0),stats.at<int>(i,1),stats.at<int>(i,2),stats.at<int>(i,3));
        cleaned.setTo(255,labels==i); components.push_back(r);
    }
    // 产品输入是一行长数字串。整行做垂直投影，可以把上下分离的笔画
    // 仍归入同一位数字；结果用字符串保存，因此会保留前导零。
    if (components.empty()) return {};
    cv::Rect content = components.front();
    for (size_t i=1;i<components.size();++i) content |= components[i];
    std::vector<cv::Rect> lines{content};
    std::vector<DigitSegment> result;
    int row=0;
    for(auto line:lines) {
        int begin=-1;
        for(int x=line.x;x<=line.x+line.width;++x) {
            bool ink=x<line.x+line.width && cv::countNonZero(cleaned(cv::Rect(x,line.y,1,line.height)))>0;
            if(ink && begin<0) begin=x;
            if(!ink && begin>=0) {
                cv::Rect stripe(begin,line.y,x-begin,line.height);
                std::vector<cv::Point> points; cv::findNonZero(cleaned(stripe),points);
                auto tight=cv::boundingRect(points); tight.x+=begin;tight.y+=line.y;
                cv::Mat crop=gray(tight).clone();
                // 原代码按整图 Otsu 掩膜清零灰度边缘：同一数字单独输入与拼成
                // 长串时阈值不同，笔画被裁掉的程度不同。二值图只用于定位，
                // 网络输入保留框内原灰度，避免损伤抗锯齿笔画和单字/长串不一致。
                // 原代码将暗灰笔画直接除以255，输入幅度远低于 MNIST 的白色笔画。
                // 修改：按每位前景峰值归一化，保持灰度层次，减少亮度导致的误判。
                double peak=0;cv::minMaxLoc(crop,nullptr,&peak);
                if(peak>0) crop.convertTo(crop,CV_8U,255.0/peak);
                double scale=20.0/std::max(crop.rows,crop.cols);
                cv::Mat small;cv::resize(crop,small,cv::Size(std::max(1,int(std::round(crop.cols*scale))),std::max(1,int(std::round(crop.rows*scale)))),0,0,cv::INTER_AREA);
                cv::Mat tile=cv::Mat::zeros(28,28,CV_8U);
                small.copyTo(tile(cv::Rect((28-small.cols)/2,(28-small.rows)/2,small.cols,small.rows)));
                auto m=cv::moments(tile);cv::Mat centered;
                cv::Mat shift=(cv::Mat_<double>(2,3)<<1,0,13.5-m.m10/m.m00,0,1,13.5-m.m01/m.m00);
                cv::warpAffine(tile,centered,shift,tile.size(),cv::INTER_LINEAR);
                DigitSegment digit;digit.image=centered;digit.bounds=tight;digit.x_start=tight.x;digit.x_end=tight.x+tight.width-1;digit.line=row;
                digit.touching=tight.width>tight.height*1.25;
                result.push_back(digit);begin=-1;
            }
        }
        ++row;
    }
    if(result.size()>256) throw std::runtime_error("Too many digits (256 maximum)");
    return result;
}
cv::Mat ImageSegmenter::drawBoundingBoxes(const std::vector<DigitSegment>& segments) {
    cv::Mat color;cv::cvtColor(original,color,cv::COLOR_GRAY2BGR);
    for(size_t i=0;i<segments.size();++i) {
        cv::rectangle(color,segments[i].bounds,cv::Scalar(190,110,30),1);
        cv::putText(color,std::to_string(i+1),cv::Point(segments[i].bounds.x,std::max(12,segments[i].bounds.y-3)),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(190,110,30),1);
    }
    return color;
}
