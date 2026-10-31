#pragma once

#include <vector>
#include <random>

struct Tensor{ 
    float* data;
    int rows, cols;
};

struct Module{
    virtual void init(float* host, std::mt19937& rng) {}
    virtual ~Module() = default;

    virtual Tensor forward(Tensor x) = 0;
    virtual Tensor backward(Tensor grad) =0;
    virtual std::vector<size_t> param_sizes(){ return {};}
    virtual void bind(float* p, float* g) {}
};


