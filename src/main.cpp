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
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <vector>


struct EpochLog {
    int epoch;
    float loss;
    double acc;
    double time;      // training wall time for the epoch (excludes test eval); mean over ranks
    double compute;   // time - copy - mpi, mean over ranks
    double copy;      // host<->device copies: perm upload + host-staged sync, mean over ranks
    double mpi;       // MPI_Allreduce time including wait for slower ranks, mean over ranks
    double time_max;  // training wall time of the slowest rank
};

using Clock = std::chrono::steady_clock;
static double seconds_since(Clock::time_point t0){
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

void write_log(std::string log_path, std::vector<EpochLog> log, 
    int P, int B_global, std::string model_name, std::string sync_name, int steps_per_epoch){
    std::ofstream outfile(log_path);
    outfile << "epoch,P,batch,model,sync,loss,acc,time,compute,copy,mpi,step_ms,time_max\n";
    for(size_t i = 0; i < log.size(); i++){
        const EpochLog& e = log[i];
        outfile << std::fixed
                << e.epoch << "," << P << "," << B_global << "," << model_name << "," << sync_name << ","
                << std::setprecision(4) << e.loss << "," << e.acc << ","
                << std::setprecision(6) << e.time << "," << e.compute << ","
                << e.copy << "," << e.mpi << ","
                << e.time / steps_per_epoch * 1000.0 << "," << e.time_max << "\n";
    }
}

// Hidden-layer widths for --model; empty if the name is unknown.
// Sized so multiply-adds per step are about 1x, 2x, 4x, and 8x of the base model.
static std::vector<int> get_hidden(const std::string& name){
    if(name == "1x") return {512, 256};
    if(name == "2x") return {848, 424};
    if(name == "4x") return {1344, 672};
    if(name == "8x") return {2064, 1032};
    return {};
}

int main(int argc, char* argv[]){
    int rank, P, local_rank;
    sync_init(&argc, &argv, &rank, &P, &local_rank);
    ops_init(local_rank);

    // defaults, overridden by --flag value pairs
    std::string data_dir = "data";        // --data
    int B_global = 128;                   // --batch      global batch, all ranks together
    std::string model_name = "1x";        // --model      1x | 2x | 4x | 8x
    int epochs = 10;                      // --epochs
    float lr = 0.05f;                    // --lr
    unsigned seed = 42;                   // --seed       same on every rank
    Sync mode = Sync::None;               // --sync       none | host | perlayer | cudaaware | local
    int sync_every = 8;                   // --sync-every only used by Sync::Local
    std::string sync_name = "none";       // --sync value as typed, for the CSV
    std::string log_dir = "logs";         //log_dir
    std::string log_path;                 // --log        per-epoch timing CSV (rank 0 writes); default built below
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
        else if(flag == "--model") model_name = v;
        else if(flag == "--epochs") epochs = std::atoi(v);
        else if(flag == "--lr") lr = (float)std::atof(v);
        else if(flag == "--seed") seed = (unsigned)std::strtoul(v, nullptr, 10);
        else if(flag == "--sync") { mode = parse_sync(v); sync_name = v; }
        else if(flag == "--sync-every") sync_every = std::atoi(v);
        else if(flag == "--log_dir") log_dir = v;
        else if(flag == "--log") log_path = v;
        else{
            if(rank == 0) std::fprintf(stderr, "error: unknown flag %s\n"
                "flags: --data --batch --model --epochs --lr --seed --sync --sync-every --log_dir --log\n", flag.c_str());
            sync_finalize(); return 1;
        }
    }


    const std::vector<int> hidden = get_hidden(model_name);
    if(hidden.empty()){
        if(rank == 0) std::fprintf(stderr, "error: unknown --model '%s' (valid: 1x, 2x, 4x, 8x)\n", model_name.c_str());
        sync_finalize(); return 1;
    }

    if(B_global <= 0 || B_global % P != 0){
        if(rank == 0) std::fprintf(stderr, "error: --batch %d must be positive and divisible by P=%d\n", B_global, P);
        sync_finalize(); return 1;
    }
    if(rank == 0)
        std::printf("data %s  model %s  batch %d (P=%d, %d per rank)  epochs %d  lr %g  seed %u  sync_every %d\n",
                    data_dir.c_str(), model_name.c_str(), B_global, P, B_global / P, epochs, lr, seed, sync_every);

    
    std::string tr_img = data_dir + "/emnist-digits-train-images-idx3-ubyte";
    std::string tr_lbl = data_dir + "/emnist-digits-train-labels-idx1-ubyte";
    std::string te_img = data_dir + "/emnist-digits-test-images-idx3-ubyte";
    std::string te_lbl = data_dir + "/emnist-digits-test-labels-idx1-ubyte";

    Sequential model;
    const int Bl = B_global / P;

    int in = 784;
    for(size_t l = 0; l < hidden.size(); l++){
        model.add(new Linear(in, hidden[l], Bl, l > 0));   // the first layer needs no dx
        model.add(new ReLu(hidden[l], Bl));
        in = hidden[l];
    }
    model.add(new Linear(in, 10, Bl));
    model.allocate(seed);

    if(log_path.empty())
        log_path = log_dir + "/log_P" + std::to_string(P) + "_B" + std::to_string(B_global) +
                   "_" + model_name + "_params" + std::to_string(model.n_params) +
                   "_" + sync_name + ".csv";

    // create the folder the CSV actually goes into (--log may point outside log_dir)
    std::filesystem::path log_parent = std::filesystem::path(log_path).parent_path();
    if(rank == 0 && !log_parent.empty())
        std::filesystem::create_directories(log_parent);

    float* d_dout = ops_alloc((size_t)Bl * 10);
    
    std::mt19937 rng(seed);
    Mnist train = load_mnist(tr_img, tr_lbl);
    Mnist test = load_mnist(te_img, te_lbl);

    float* d_train_x = ops_alloc((size_t)train.n * 784);
    float* d_train_y = ops_alloc((size_t)train.n * 10);
    ops_upload(d_train_x, train.images.data(), (size_t)train.n * 784);
    ops_upload(d_train_y, train.labels.data(), (size_t)train.n * 10);
    
    float* d_shuf_x = ops_alloc((size_t)Bl * 784);
    float* d_shuf_y = ops_alloc((size_t)Bl * 10);
    
    float* d_test_x = ops_alloc((size_t)test.n * 784);
    ops_upload(d_test_x, test.images.data(), (size_t)test.n * 784);
    


    std::vector<int> perm(train.n);
    int* d_perm = ops_alloc_int(train.n);
    const int steps_per_epoch = train.n / B_global;
    std::vector<float> out_host((size_t)Bl * 10);
    std::vector<EpochLog> log;


    for(int epoch = 0; epoch < epochs; epoch++){
        ops_sync();                 // nothing from the previous epoch leaks into this one
        sync_times_take();          // reset copy/MPI accumulators
        auto start_epoch = Clock::now();
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        auto start_perm = Clock::now();
        ops_upload_int(d_perm, perm.data(), train.n);
        ops_sync();
        double perm_copy = seconds_since(start_perm);
        
        float loss_sum = 0.0f;
        for(int i = 0; i < steps_per_epoch; ++i){
            size_t row = (size_t)i * B_global + (size_t)rank * Bl;
            // this rank's Bl indices start at d_perm + row; gather just those rows
            ops_gather_rows(d_shuf_x, d_train_x, d_perm + row, Bl, 784);
            ops_gather_rows(d_shuf_y, d_train_y, d_perm + row, Bl, 10);
            Tensor x = {d_shuf_x, Bl, 784};
            const float* d_y = d_shuf_y;
            Tensor out = model.forward(x);
            float loss = ops_softmax_grad(d_dout, out.data, d_y, out.rows, 10);
            loss_sum += loss;

            model.backward({d_dout, out.rows, 10});
            sync_gradients(mode, model.grads, model.n_params, model.offsets);
            ops_sgd(model.params, model.grads, model.n_params, lr);
            int step = epoch * steps_per_epoch + i;
            
            if(mode == Sync::Local && (step+1) % sync_every == 0) 
                average_weights(model.params, model.n_params);
        

        }

        if (mode == Sync::Local) average_weights(model.params, model.n_params);

        ops_sync();                 // kernels are async: wait before reading the clock
        double epoch_time = seconds_since(start_epoch);
        SyncTimes st = sync_times_take();
        double t_copy = perm_copy + st.copy;
        double t_mpi = st.mpi;
        double t_compute = epoch_time - t_copy - t_mpi;
        // every rank must call these (collectives)
        double t_max = sync_max(epoch_time);
        // means, not per-part maxes, so that compute + copy + mpi = time
        epoch_time = sync_mean(epoch_time);
        t_compute = sync_mean(t_compute);
        t_copy = sync_mean(t_copy);
        t_mpi = sync_mean(t_mpi);

        float loss = sync_sum(loss_sum) / (P * steps_per_epoch);   // mean over all ranks and steps
        
        int correct = 0;
        for (int s = 0; s < test.n; s += Bl) {
            int rows = std::min(Bl, test.n - s);
            Tensor o = model.forward({d_test_x + (size_t)s * 784, rows, 784});
            ops_download(out_host.data(), o.data, (size_t)rows * 10);
            for (int i = 0; i < rows; ++i) {
                const float* r = &out_host[(size_t)i * 10];
                int pred = (int)(std::max_element(r, r + 10) - r);   // argmax of softmax
                correct += (pred == test.label_ids[s + i]);
            }
        }
        if (rank == 0){
            std::printf("epoch %d  loss %.4f  test acc %.4f  time %.3fs (compute %.3f  copy %.3f  mpi %.3f)\n",
                    epoch, loss, (double)correct / test.n,
                    epoch_time, t_compute, t_copy, t_mpi);
            log.push_back(EpochLog{epoch, loss, (double)correct / test.n,
                                   epoch_time, t_compute, t_copy, t_mpi, t_max});
        }
    }

    if (rank == 0){
        write_log(log_path, log, P, B_global, model_name, sync_name, steps_per_epoch);
    }

    // Weight checksum, printed by EVERY rank: all ranks must end with identical weights
    // (validation 3). sum|w| as well, since a plain sum can hide differences that cancel.
    {
        std::vector<float> w(model.n_params);
        ops_download(w.data(), model.params, model.n_params);
        double sum = 0.0, abs_sum = 0.0;
        for(float v : w){ sum += v; abs_sum += std::fabs(v); }
        std::printf("rank %d checksum sum %.10e  sum|w| %.10e\n", rank, sum, abs_sum);
        std::fflush(stdout);
    }


    ops_free(d_dout);
    ops_free(d_train_x);
    ops_free(d_train_y);
    ops_free(d_shuf_x);
    ops_free(d_shuf_y);
    ops_free(d_test_x);
    ops_free_int(d_perm);
    sync_finalize();
    return 0;
}