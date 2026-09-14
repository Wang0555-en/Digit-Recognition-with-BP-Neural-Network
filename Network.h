#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "Layer.h"

// 网络类：组合多个 Layer，实现完整的 BP 训练
class Network {
private:
    std::vector<Layer*> layers;   // 存储所有层（包括隐藏层和输出层）

public:
    Network(const std::vector<int>& layer_sizes);
    ~Network();

    Matrix forward(const Matrix& input);

    void backwardAndUpdate(const Matrix& target, double lr);

    void save(const std::string& filename) const;  // 保存到文件
    bool load(const std::string& filename);        // 失败时不修改已有参数
};

#endif
