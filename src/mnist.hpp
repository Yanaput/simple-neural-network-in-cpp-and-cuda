#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include<iostream>

struct Mnist{
    int n;
    //images n * 784, labels n * 10, label_ids n
    std::vector<float> images, labels;
    std::vector<uint8_t> label_ids;
};

Mnist load_mnist(const std::string& img_path, const std::string& label_path);
