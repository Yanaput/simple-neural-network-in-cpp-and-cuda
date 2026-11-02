#include "mnist.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>

// Like assert, but never compiled out (NDEBUG) and prints a message.
static void check(bool cond, const std::string& msg){
    if(!cond){
        std::cerr << "MNIST loader error: " << msg << std::endl;
        std::exit(1);
    }
}

static uint32_t read_be_u32(std::ifstream& f){
    //read and move cursor 4 bytes each
    unsigned char buffer[4];
    f.read(reinterpret_cast<char*>(buffer), 4);
    check(f.gcount() == 4, "unexpected end of file while reading header");
    return (uint32_t(buffer[0]) << 24) | (uint32_t(buffer[1]) << 16)
         | (uint32_t(buffer[2]) << 8)  |  uint32_t(buffer[3]);
}

Mnist load_mnist(const std::string& img_path, const std::string& label_path){
    std::ifstream img(img_path, std::ios::binary);
    std::ifstream label(label_path, std::ios::binary);
    check(bool(img),   "cannot open " + img_path);
    check(bool(label), "cannot open " + label_path);

    uint32_t magic_img   = read_be_u32(img);
    uint32_t magic_label = read_be_u32(label);
    check(magic_img == 2051,   img_path + ": bad magic " + std::to_string(magic_img) + " (expected 2051)");
    check(magic_label == 2049, label_path + ": bad magic " + std::to_string(magic_label) + " (expected 2049)");

    int count_img   = read_be_u32(img);
    int count_label = read_be_u32(label);
    int rows = read_be_u32(img);
    int cols = read_be_u32(img);

    check(count_img == count_label, "image count " + std::to_string(count_img) +
                                    " != label count " + std::to_string(count_label));
    check(rows * cols == 784, "expected 28x28 images, got " +
                              std::to_string(rows) + "x" + std::to_string(cols));

    const int n = count_img;
    const size_t px = size_t(n) * 784;

    Mnist m;
    m.n = n;

    // raw bytes: 784 per image, 1 per label
    std::vector<uint8_t> raw_imgs(px);
    m.label_ids.resize(n);

    img.read(reinterpret_cast<char*>(raw_imgs.data()), px);
    check(size_t(img.gcount()) == px, img_path + ": file truncated (pixel data)");
    label.read(reinterpret_cast<char*>(m.label_ids.data()), n);
    check(label.gcount() == n, label_path + ": file truncated (label data)");

    // pixels -> [0, 1]
    m.images.resize(px);
    for(size_t i = 0; i < px; i++){
        m.images[i] = raw_imgs[i] / 255.0f;
    }

    // labels -> one-hot, n x 10
    m.labels.assign(size_t(n) * 10, 0.0f);
    for(int i = 0; i < n; i++){
        check(m.label_ids[i] < 10, "label out of range at index " + std::to_string(i));
        m.labels[size_t(i) * 10 + m.label_ids[i]] = 1.0f;
    }

    return m;
}
