#include "ops.hpp"
#include "linear.hpp"
#include "relu.hpp"
#include "sequential.hpp"
#include "mnist.hpp"
#include "sync.hpp"
#include <random>
#include <algorithm>
#include <string>
#include <numeric>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>

struct EpochLog {
    int epoch;
    float loss;
    double time;
    double acc;
};

int main(int argc, char* argv[]){
    int rank, P, local_rank;
    sync_init(&argc, &argv, &rank, &P, &local_rank);
    ops_init(local_rank);

    // defaults, overridden by --flag value pairs
    std::string data_dir = "data";        // --data
    int B_global = 128;                   // --batch      global batch, all ranks together
    int epochs = 10;                      // --epochs
    float lr = 0.001f;                    // --lr
    unsigned seed = 42;                   // --seed       same on every rank
    Sync mode = Sync::None;               // --sync       none | host | perlayer | cudaaware | local
    int sync_every = 8;                   // --sync-every only used by Sync::Local

    // parse after sync_init: MPI_Init may strip its own arguments from argv
    for(int a = 1; a < argc; ++a){
        std::string flag = argv[a];
        if(a + 1 >= argc){
            if(rank == 0) std::fprintf(stderr, "error: %s needs a value\n", flag.c_str());
            sync_finalize(); return 1;
        }
        const char* v = argv[++a];
        if (flag == "--data") data_dir = v;
        else if(flag == "--batch") B_global = std::atoi(v);
        else if(flag == "--epochs") epochs = std::atoi(v);
        else if(flag == "--lr") lr = (float)std::atof(v);
        else if(flag == "--seed") seed = (unsigned)std::strtoul(v, nullptr, 10);
        else if(flag == "--sync") mode = parse_sync(v);
        else if(flag == "--sync-every") sync_every = std::atoi(v);
        else{
            if(rank == 0) std::fprintf(stderr, "error: unknown flag %s\n"
                "flags: --data --batch --epochs --lr --seed --sync --sync-every\n", flag.c_str());
            sync_finalize(); return 1;
        }
    }


    if(B_global <= 0 || B_global % P != 0){
        if(rank == 0) std::fprintf(stderr, "error: --batch %d must be positive and divisible by P=%d\n", B_global, P);
        sync_finalize(); return 1;
    }
    if(rank == 0)
        std::printf("data %s  batch %d (P=%d, %d per rank)  epochs %d  lr %g  seed %u  sync_every %d\n",
                    data_dir.c_str(), B_global, P, B_global / P, epochs, lr, seed, sync_every);

    std::string tr_img = data_dir + "/train-images-idx3-ubyte";
    std::string tr_lbl = data_dir + "/train-labels-idx1-ubyte";
    std::string te_img = data_dir + "/t10k-images-idx3-ubyte";
    std::string te_lbl = data_dir + "/t10k-labels-idx1-ubyte";

    Sequential model;
    const int   Bl = B_global / P;

    model.add(new Linear(784, 512, Bl, false));
    model.add(new ReLu(512, Bl));
    model.add(new Linear(512, 256, Bl));
    model.add(new ReLu(256, Bl));
    model.add(new Linear(256, 10, Bl));
    model.allocate(seed);

    float* dout = ops_alloc((size_t)Bl * 10);
    
    std::mt19937 rng(seed);
    Mnist train = load_mnist(tr_img, tr_lbl);
    Mnist test = load_mnist(te_img, te_lbl);

    float* train_x = ops_alloc((size_t)train.n * 784);
    float* train_y = ops_alloc((size_t)train.n * 10);
    ops_upload(train_x, train.images.data(), (size_t)train.n * 784);
    ops_upload(train_y, train.labels.data(), (size_t)train.n * 10);
    float* shuf_x = ops_alloc((size_t)train.n * 784);
    float* shuf_y = ops_alloc((size_t)train.n * 10);
    
    float* test_x = ops_alloc((size_t)test.n * 784);
    ops_upload(test_x, test.images.data(), (size_t)test.n * 784);


    std::vector<int> perm(train.n);
    int* d_perm = ops_alloc_int(train.n);
    const int steps_per_epoch = train.n / B_global;
    std::vector<float> out_host((size_t)Bl * 10);
    std::vector<EpochLog> log;


    for(int epoch = 0; epoch < epochs; epoch++){
        auto start_epoch = std::chrono::steady_clock::now();;
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        ops_upload_int(d_perm, perm.data(), train.n);
        ops_gather_rows(shuf_x, train_x, d_perm, train.n, 784);
        ops_gather_rows(shuf_y, train_y, d_perm, train.n, 10);
        
        float loss_sum = 0.0f;
        for(int i = 0; i < steps_per_epoch; ++i){
            size_t row = (size_t)i * B_global + (size_t)rank * Bl;
            Tensor x = {shuf_x + row * 784, Bl, 784};
            const float* y = shuf_y + row * 10;
            Tensor out = model.forward(x);
            float loss = ops_softmax_grad(dout, out.data, y, out.rows, 10);
            loss_sum += loss;

            model.backward({dout, out.rows, 10});
            sync_gradients(mode, model.grads, model.n_params, model.offsets);
            ops_sgd(model.params, model.grads, model.n_params, lr);
            int step = epoch * steps_per_epoch + i;
            
            if(mode == Sync::Local && (step+1) % sync_every == 0) 
                average_weights(model.params, model.n_params);
        

        }
        auto end_epoch = std::chrono::steady_clock::now();;
        std::chrono::duration<double> elapsed_time = end_epoch - start_epoch;
        double epoch_time = elapsed_time.count();


        int correct = 0;
        for (int s = 0; s < test.n; s += Bl) {
            int rows = std::min(Bl, test.n - s);
            Tensor o = model.forward({test_x + (size_t)s * 784, rows, 784});
            ops_download(out_host.data(), o.data, (size_t)rows * 10);
            for (int i = 0; i < rows; ++i) {
                const float* r = &out_host[(size_t)i * 10];
                int pred = (int)(std::max_element(r, r + 10) - r);   // argmax of softmax
                correct += (pred == test.label_ids[s + i]);
            }
        }
        std::string buffer;
        if (rank == 0){
            std::printf("epoch %d  loss %.4f  test acc %.4f\n",
                    epoch, loss_sum / steps_per_epoch, (double)correct / test.n);
            log.push_back(EpochLog{epoch, loss_sum / steps_per_epoch, epoch_time, (double)correct/test.n});
        }
    }

    if (rank == 0){
        std::ofstream outfile ("log.csv");
        outfile << "epoch, loss, time, acc\n";
        for(int i = 0; i < log.size(); i++){
            outfile <<std ::fixed << std::setprecision(4) << log[i].epoch << "," <<  log[i].loss << "," << log[i].time << "," << log[i].acc << "\n";
        }
        outfile.close();
    }


    ops_free(dout);
    ops_free(train_x);
    ops_free(train_y);
    ops_free(shuf_x);
    ops_free(shuf_y);
    ops_free(test_x);
    ops_free_int(d_perm);
    sync_finalize();
    return 0;
}