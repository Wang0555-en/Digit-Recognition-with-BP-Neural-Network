#include "Layer.h"
#include <cmath>   
#include <cstdlib>   
#include <ctime>    

double Sigmoid::forward(double x) { return 1.0 / (1.0 + exp(-x)); }

double Sigmoid::backward(double y) { return y * (1.0 - y); }

Layer::Layer(int in, int out, Activation* act): input_size(in), output_size(out), activation(act) 
{
    weights = Matrix(in, out);
    static bool seed_set = false;
    if (!seed_set)
    {
        srand((unsigned)time(0));
        seed_set = true;
    }
    for (int i = 0; i < in; i++) 
    {
        for (int j = 0; j < out; j++) 
        {
            double val = (rand() % 100) / 100.0 - 0.5;
            weights.setData(i, j, val);
        }
    }

    bias = Matrix(1, out, 0.0);

}

Layer::~Layer() { delete activation; }

Matrix Layer::forward(const Matrix& input_data) 
{
    input_cache = input_data; 
    // ----- 第2步：计算加权和  Z = input_data * weights -----
    // input_data 尺寸: (batch_size × input_size)
    // weights 尺寸: (input_size × output_size)
    // 结果 Z 尺寸: (batch_size × output_size)
    Matrix weighted_sum = input_data * weights; 

    // ----- 第3步：加上偏置 bias (1 × output_size) -----
    // 问题：A同学的 operator+ 要求两个矩阵尺寸完全一样。
    // 而 weighted_sum 是 (batch × out)，bias 是 (1 × out)，尺寸不同！
    // 解决办法：把 bias 逐行加到 weighted_sum 的每一行上去（这叫"广播"）
    int batch_size = weighted_sum.getRows();
    int out_cols = weighted_sum.getCols();

    for (int i = 0; i < batch_size; ++i) {          // 遍历每一行
        for (int j = 0; j < out_cols; ++j) {        // 遍历每一列
            // 从 bias 取出第0行第j列的值，加到当前元素上
            double bias_val = bias.getData(0, j);
            double cur_val = weighted_sum.getData(i, j);
            weighted_sum.setData(i, j, cur_val + bias_val);
        }
    }

    // ----- 第4步：对加权和的每个元素应用激活函数 -----
    // 因为 activation->forward 只处理单个 double，所以要遍历矩阵每个元素
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < out_cols; ++j) {
            double val = weighted_sum.getData(i, j);
            double activated = activation->forward(val); // 多态调用 Sigmoid
            weighted_sum.setData(i, j, activated);
        }
    }

    // ----- 第5步：缓存输出（反向传播要用）并返回 -----
    output_cache = weighted_sum;
    return output_cache;
}

// ============================================================
// 权重更新函数（给C同学调用）
// ============================================================
void Layer::updateWeights(const Matrix& delta_w, const Matrix& delta_b, double learning_rate) {
    // delta_w 尺寸必须与 weights 一致 (input_size × output_size)
    // delta_b 尺寸必须与 bias 一致 (1 × output_size)
    // 更新公式：新权重 = 旧权重 - 学习率 × 梯度  （注意是减号）
    // 因为我们用的是 error = target - pred，推导出的梯度已经带方向，所以这里用 +=
    // 但为了通用性，我们用标准的 新 = 旧 + lr * 梯度 (梯度已包含负号)
    // 为了安全，直接用 += 配合我们之前推导的 d_w (它已经是正负调整量)

    for (int i = 0; i < weights.getRows(); ++i) {
        for (int j = 0; j < weights.getCols(); ++j) {
            double old_w = weights.getData(i, j);
            double grad = delta_w.getData(i, j);
            weights.setData(i, j, old_w + learning_rate * grad);
        }
    }

    for (int j = 0; j < bias.getCols(); ++j) {
        double old_b = bias.getData(0, j);
        double grad = delta_b.getData(0, j);
        bias.setData(0, j, old_b + learning_rate * grad);
    }
}