#ifndef LAYER_H
#define LAYER_H
#include "matrix.h" 

class Activation
{
public:
    virtual double forward(double x) = 0;
    virtual double backward(double y) = 0;

    virtual ~Activation() {}
};

class Sigmoid : public Activation {
public:
    double forward(double x) override;
    double backward(double y) override;
};

class Layer {
private:
    int input_size;  
    int output_size;  

    Matrix weights;  
    Matrix bias;     

    Matrix input_cache;  
    Matrix output_cache; 

    Activation* activation; 

public:
    Layer(int in, int out, Activation* act);
    ~Layer();

    Matrix forward(const Matrix& input_data);

    Matrix getInputCache() const { return input_cache; }
    Matrix getOutputCache() const { return output_cache; }
    Matrix getWeights() const { return weights; }
    Matrix getBias() const { return bias; }

    void setWeights(const Matrix& w) { weights = w; }
    void setBias(const Matrix& b) { bias = b; }

    // 更新权重（传梯度和学习率）
    void updateWeights(const Matrix& delta_w, const Matrix& delta_b, double learning_rate);
};

#endif