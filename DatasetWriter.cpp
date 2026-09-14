#include "DatasetWriter.h"
#include <cmath>
#include <iomanip>

bool DatasetWriter::open(const std::string& sample_file, const std::string& history_file) {
    // 每次训练重新生成，避免把不同训练任务混在一起。
    samples.open(sample_file);
    history.open(history_file);
    if (!samples.is_open() || !history.is_open()) return false;

    history << "epoch,learning_rate,loss,test_accuracy" << std::endl;
    history << std::setprecision(17);
    return samples.good() && history.good();
}

bool DatasetWriter::writeSample(const Matrix& input, const Matrix& target) {
    if (input.getRows() != 1 || input.getCols() != 784 ||
        target.getRows() != 1 || target.getCols() != 10) return false;

    // target 中只有真实数字的位置是 1，例如数字 3 对应第 3 列。
    int label = -1;
    for (int i = 0; i < 10; ++i) {
        double value = target.getData(0, i);
        if (value == 1.0 && label == -1) label = i;
        else if (value != 0.0) return false;
    }
    if (label == -1) return false;

    // 先检查整条样本，避免无效数据写到一半才报错。
    for (int i = 0; i < 784; ++i) {
        double pixel = input.getData(0, i);
        if (!std::isfinite(pixel) || pixel < 0.0 || pixel > 1.0) return false;
    }

    samples << label;
    for (int i = 0; i < 784; ++i) {
        // 读取时除以 255，这里乘回去并四舍五入，还原整数灰度值。
        int pixel = (int)std::lround(input.getData(0, i) * 255.0);
        samples << ',' << pixel;
    }
    // endl 同时换行和刷新缓冲区，让样本随训练及时写入文件。
    samples << std::endl;
    return samples.good();
}

bool DatasetWriter::writeEpoch(int epoch, double learning_rate, double loss, double accuracy) {
    history << epoch << ',' << learning_rate << ',' << loss << ',' << accuracy << std::endl;
    return history.good();
}
