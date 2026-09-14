#include "Network.h"
#include <cmath>
#include <iostream>
#include <fstream>  
#include <iostream>

// 辅助：Sigmoid 导数（因为已经存在 Sigmoid::backward，但需要对象，我们用静态函数）
static double sigmoid_deriv(double y) { return y * (1.0 - y); }

// ---------- 构造函数 ----------
Network::Network(const std::vector<int>& layer_sizes) {
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        int in = layer_sizes[i];
        int out = layer_sizes[i + 1];
        Activation* act = new Sigmoid();
        Layer* layer = new Layer(in, out, act);
        layers.push_back(layer);
    }
}

// ---------- 析构函数（释放所有层和激活函数） ----------
Network::~Network() {
    for (auto* layer : layers) {
        delete layer;  // 这里需要 Layer 的析构释放 activation，我们稍后修改 Layer 析构
    }
}

// ---------- 前向传播 ----------
Matrix Network::forward(const Matrix& input) {
    Matrix current = input;
    for (auto* layer : layers) {
        current = layer->forward(current);
    }
    return current;
}

// ---------- 反向传播 + 权重更新（单样本） ----------
void Network::backwardAndUpdate(const Matrix& target, double lr) {
    int num_layers = static_cast<int>(layers.size());

    // 1. 计算所有层的 delta（敏感度），从后往前
    std::vector<Matrix> deltas(num_layers);  // deltas[i] 对应 layers[i]

    // 最后一层（输出层）
    Layer* last_layer = layers[num_layers - 1];
    Matrix output = last_layer->getOutputCache();   // (1 × out)
    Matrix error = target-output;                 // (1 × out)
    int out_size = output.getCols();
    Matrix delta_out(1, out_size);
    for (int j = 0; j < out_size; ++j) {
        double out_val = output.getData(0, j);
        double err = error.getData(0, j);
        double deriv = sigmoid_deriv(out_val);
        delta_out.setData(0, j, err * deriv);
    }
    deltas[num_layers - 1] = delta_out;

    // 隐藏层（从后往前，除了输出层）
    for (int i = num_layers - 2; i >= 0; --i) {
        Layer* curr_layer = layers[i];
        Layer* next_layer = layers[i + 1];
        Matrix next_delta = deltas[i + 1];          // (1 × next_out)
        Matrix next_weights = next_layer->getWeights(); // (out × next_out)
        Matrix weights_T = next_weights.transpose(); // (next_out × out)

        // 未激活的 delta = next_delta * weights_T  (1 × out)
        Matrix delta_unscaled = next_delta * weights_T;

        // 乘上当前层的激活函数导数
        Matrix curr_out = curr_layer->getOutputCache(); // (1 × out)
        int curr_out_size = curr_out.getCols();
        Matrix delta_curr(1, curr_out_size);
        for (int j = 0; j < curr_out_size; ++j) {
            double out_val = curr_out.getData(0, j);
            double deriv = sigmoid_deriv(out_val);
            double unscaled = delta_unscaled.getData(0, j);
            delta_curr.setData(0, j, unscaled * deriv);
        }
        deltas[i] = delta_curr;
    }

    // 2. 利用 deltas 计算每个层的权重梯度并更新
    for (int i = 0; i < num_layers; ++i) {
        Layer* layer = layers[i];
        Matrix input_cache = layer->getInputCache();   // (1 × in)
        Matrix delta = deltas[i];                      // (1 × out)

        // 权重梯度 = input_cache^T * delta  (in × out)
        Matrix input_T = input_cache.transpose();      // (in × 1)
        Matrix delta_w = input_T * delta;              // (in × out)

        // 偏置梯度 = delta (1 × out)
        Matrix delta_b = delta;  // 直接拷贝

        // 调用 Layer 的更新函数
        layer->updateWeights(delta_w, delta_b, lr);
    }
}

void Network::save(const std::string& filename) const {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "保存失败：无法创建文件 " << filename << std::endl;
        return;
    }

    // 1. 先写入层数（便于加载时校验）
    int num_layers = static_cast<int>(layers.size());
    ofs.write((char*)&num_layers, sizeof(num_layers));

    // 2. 逐层写入权重和偏置
    for (auto* layer : layers) {
        Matrix w = layer->getWeights();
        Matrix b = layer->getBias();

        // 写入权重矩阵：先写行列数，再写所有数据
        int rows = w.getRows();
        int cols = w.getCols();
        ofs.write((char*)&rows, sizeof(rows));
        ofs.write((char*)&cols, sizeof(cols));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double val = w.getData(i, j);
                ofs.write((char*)&val, sizeof(val));
            }
        }

        // 写入偏置矩阵：先写行列数，再写所有数据
        int b_rows = b.getRows();
        int b_cols = b.getCols();
        ofs.write((char*)&b_rows, sizeof(b_rows));
        ofs.write((char*)&b_cols, sizeof(b_cols));
        for (int i = 0; i < b_rows; ++i) {
            for (int j = 0; j < b_cols; ++j) {
                double val = b.getData(i, j);
                ofs.write((char*)&val, sizeof(val));
            }
        }
    }
    ofs.close();
    std::cout << "模型已保存至: " << filename << std::endl;
}
bool Network::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    // 先验证完整模型，再更新参数，避免损坏文件导致随机预测或部分加载。
    auto readInt = [&](int& value) { return bool(file.read(reinterpret_cast<char*>(&value), sizeof(value))); };
    int count = 0;
    if (!readInt(count) || count != static_cast<int>(layers.size())) return false;
    std::vector<Matrix> weights, biases;
    for (auto* layer : layers) {
        for (int kind = 0; kind < 2; ++kind) {
            Matrix expected = kind == 0 ? layer->getWeights() : layer->getBias();
            int rows = 0, cols = 0;
            if (!readInt(rows) || !readInt(cols) || rows != expected.getRows() || cols != expected.getCols()) return false;
            Matrix value(rows, cols);
            for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) {
                double v = 0;
                if (!file.read(reinterpret_cast<char*>(&v), sizeof(v)) || !std::isfinite(v)) return false;
                value.setData(r, c, v);
            }
            (kind == 0 ? weights : biases).push_back(value);
        }
    }
    if (file.peek() != std::char_traits<char>::eof()) return false;
    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i]->setWeights(weights[i]);
        layers[i]->setBias(biases[i]);
    }
    return true;
}
