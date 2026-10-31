#pragma once

#include <cstddef>
#include <cassert>
#include "module.hpp"
#include "ops.hpp"

struct ReLu : Module{
    int width, max_batch;
    Tensor x{};
    float* y = nullptr; 

    ReLu(int width, int max_batch) {
        this->width = width;
        this->max_batch = max_batch;
        this->y = ops_alloc((size_t)width*max_batch);
    }

    ReLu(const ReLu&) = delete;

    ReLu& operator=(const ReLu&) = delete;

    ~ReLu() override{
        ops_free(this->y);
    }

    Tensor forward(Tensor input) override{
        assert(input.cols == width && input.rows <= max_batch);
        this->x = input;
        ops_relu(this->y, this->x.data, (size_t)this->x.rows * this->x.cols);
        return Tensor{this->y, input.rows, input.cols};
    }

    Tensor backward(Tensor grad) override{
        ops_relu_backward(grad.data, this->x.data, (size_t)grad.rows * grad.cols);
        return Tensor{grad.data, grad.rows, grad.cols};
    }

    std::vector<size_t> param_sizes() override{ return {};}
    void bind(float* p, float* g) override{}
};