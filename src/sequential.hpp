#pragma once

#include "module.hpp"
#include "ops.hpp"
#include <memory>

struct Sequential : Module{
    std::vector<std::unique_ptr<Module>> layers;
    float* params = nullptr;
    float* grads = nullptr;
    size_t n_params = 0;
    std::vector<size_t> offsets; // start of every W and b, plus n_params

    ~Sequential() override{
        ops_free(this->params);
        ops_free(this->grads);
    }

    void add(Module* m){
        this->layers.emplace_back(m);
    }

    void allocate(unsigned seed){
        for(auto& l : this->layers){
            for(size_t s : l->param_sizes()){
                this->offsets.push_back(this->n_params);
                this->n_params += s;
            }
        }
        this->offsets.push_back(this->n_params);
        this->params = ops_alloc(this->n_params);
        this->grads = ops_alloc(this->n_params);

        std::vector<float> host(this->n_params);
        std::mt19937 rng(seed);

        size_t off = 0;
        for(auto& l : layers){
            l->bind(this->params + off, this->grads + off);
            l->init(host.data() + off, rng);
            for(size_t s: l->param_sizes()) off += s;
        }
        ops_upload(params, host.data(), n_params); //to gpu or copy to cpu
    
    }

    Tensor forward(Tensor input) override{
        for(auto& l : this->layers) 
            input = l->forward(input);
        return input;
    }

    Tensor backward(Tensor g) override{
        for(int i = (int)layers.size() - 1; i>=0; i--)
            g = layers[i]->backward(g);
        return g;
    }
};