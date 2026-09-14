#ifndef DATASET_WRITER_H
#define DATASET_WRITER_H

#include <fstream>
#include <string>
#include "matrix.h"

// 只负责保存数据，不参与神经网络的计算。
class DatasetWriter {
private:
    std::ofstream samples;
    std::ofstream history;

public:
    bool open(const std::string& sample_file, const std::string& history_file);
    bool writeSample(const Matrix& input, const Matrix& target);
    bool writeEpoch(int epoch, double learning_rate, double loss, double accuracy);
};

#endif
