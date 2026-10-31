#pragma once

#include "module.hpp"
#include "ops.hpp"
#include <cmath>
#include <cassert>
#include <random>

struct Linear : Module{
    int in, out, max_batch;
    float *W = nullptr, *b = nullptr, *dW = nullptr, *db = nullptr;
    Tensor x{};
    bool need_dx; //first layer set to false
    float *y = nullptr, *dx = nullptr;

    Linear(int in, int out, int max_batch, bool need_dx = true){
        this->in = in;
        this->out = out;
        this->max_batch = max_batch;
        this->y = ops_alloc(out * max_batch);
        this->need_dx = need_dx;
        if(this->need_dx) this->dx = ops_alloc(in * max_batch);
    }

    ~Linear(){
        ops_free(this->y);
        if(this->need_dx) ops_free(this->dx);
    }

    Linear(const Linear&) = delete;

    Linear& operator=(const Linear&) = delete;

    std::vector<size_t> param_sizes() override{
        return std::vector<size_t> {(size_t)this->in * (size_t)this->out, (size_t)this->out};
    }

    void bind(float* p, float* g) override{
        W = p;
        b = p + (size_t)in*out;

        dW = g;
        db = g + (size_t)in*out;
    }

    void init(float* host, std::mt19937& rng) override{
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f/this->in));
        //wegihts
        for(size_t i = 0; i < (size_t)this->in * this->out; i++){
            host[i] = dist(rng);
        }
        //bias
        for(int j = 0; j < this->out; j++){
            host[(size_t)this->in * this->out + j] = 0.0f;
        }
    }

    Tensor forward(Tensor input) override{
        this->x = input;
        //x*w
        ops_gemm(false, false, x.rows, this->out, this->in, this->x.data, this->W, this->y);
        ops_add_bias(this->y, this->b, this->x.rows, this->out);
        return {this->y, this->x.rows, this->out};
    }

    Tensor backward(Tensor g) override{
        // dW = x^T x g, dim : in * out
        // db = col_sum of g
        ops_gemm(true, false, this->in, this->out, g.rows, this->x.data, g.data, this->dW);
        ops_col_sum(this->db, g.data, g.rows, out);
        if(!need_dx) return {nullptr, g.rows, in};
        ops_gemm(false, true, g.rows, this->in, this->out, g.data, this->W, this->dx);
        return {dx, g.rows, in};
        
    }

};