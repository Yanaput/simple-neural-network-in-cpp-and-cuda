#include "ops.hpp"
#include "linear.hpp"
#include "relu.hpp"
#include "sequential.hpp"
#include "mnist.hpp"
#include "sync.hpp"
#include <random>
#include <algorithm>

int main(int argc, char* argv[]){
    int rank, P, local_rank;
    sync
    std::string mode = argc > 1 ? argv[1] : "None";
    std::string data_dir = argc > 2 ? argv[2] : "data";
    std::string tr_img = data_dir + "/train-images-idx3-ubyte";
    std::string tr_lbl = data_dir + "/train-labels-idx1-ubyte";
    std::string te_img = data_dir + "/t10k-images-idx3-ubyte";
    std::string te_lbl = data_dir + "/t10k-labels-idx1-ubyte";


    Sequential model;
    int B = 64;
    int seed = 42;
    int epoch = 10;

    model.add(new Linear(784, 512, B, false));
    model.add(new ReLu(512, B));
    model.add(new Linear(512, 256, B));
    model.add(new ReLu(256, B));
    model.add(new Linear(256, 10, B));
    model.allocate(seed);

    float* dout = ops_alloc((size_t)B * 10);
    
    std::mt19937 rng(seed);
    Mnist train = load_mnist(tr_img, tr_lbl);
    Mnist test = load_mnist(te_img, te_lbl);

    std::vector<int> perm(train.n);
    int* d_perm = ops_alloc_int(train.n)
    for(int step = 0; step < epoch; step++){
        perm =
    }
}