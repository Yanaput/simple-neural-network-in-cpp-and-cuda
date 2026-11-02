#ifdef USE_MPI
#include <mpi.h>
#endif

#include "ops.hpp"
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include "sync.hpp"
#include <cstddef>

static int g_P = 1;

void sync_init(int* argc, char** argv[], int* rank, int* P, int* local_rank){
#ifdef USE_MPI
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, P);
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, *rank, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, local_rank);
    MPI_Comm_free(&node);
#else
    *rank =0;
    *P = 1;
    *local_rank = 0;
#endif
    g_P = *P;
}

void sync_finalize(){
#ifdef USE_MPI
    MPI_Finalize();
#endif
}

Sync parse_sync(const char* s){
    if(!std::strcmp(s, "none")) return Sync::None;
    if(!std::strcmp(s, "host")) return Sync::Host;
    if(!std::strcmp(s, "perlayer")) return Sync::PerLayer;
    if(!std::strcmp(s, "cudaaware")) return Sync::CudaAware;
    if(!std::strcmp(s, "local")) return Sync::Local;

    std::fprintf(stderr, "error: unknown --sync mode '%s' (valid: none, host, perlayer, cudaaware, local)\n", s);
    std::exit(1);
}

void sync_gradients(Sync mode, float* grads, size_t n, const std::vector<size_t>& offsets){
    if(mode == Sync::None || mode== Sync::Local) return;
#ifdef USE_MPI
    static std::vector<float> host;
    host.resize(n);

    switch (mode){
        case Sync::Host: //copy to host, sum across ranks, then copy back
            ops_download(host.data(), grads, n);
            MPI_Allreduce(MPI_IN_PLACE, host.data(), n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            ops_upload(grads, host.data(), n);
            break;

        case Sync::PerLayer: // Allreduce per W or b
            ops_download(host.data(), grads, n);
            for(size_t i = 0; i + 1 < offsets.size(); ++i){
                MPI_Allreduce(MPI_IN_PLACE, host.data() + offsets[i], offsets[i + 1] - offsets[i]
                , MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            }
            ops_upload(grads, host.data(), n);
            break;
        
        case Sync::CudaAware: // MPI reads and writes GPU memory directly
            MPI_Allreduce(MPI_IN_PLACE, grads, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        default:
            break;
        }
#else
    std::fprintf(stderr, "error: this mode needs an MPI build)\n");
    std::exit(1);
#endif
}

void average_weights(float* params, size_t n){
    // Local SGD weight averaging: implemented later behind #ifdef USE_MPI
#ifdef USE_MPI
    static std::vector<float> host;
    host.resize(n);

    ops_download(host.data(), params, n);
    MPI_Allreduce(MPI_IN_PLACE, host.data(), n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    ops_scale(host.data(), n, 1.0f/g_P);
#else
    return;
#endif
    std::fprintf(stderr, "error: average_weights not available in this build (compiled without USE_MPI)\n");
    std::exit(1);
}

float  sync_sum(float x){
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, &x, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        return x;
#else
    return x;
#endif
}

double sync_max(double t){
    #ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, &t, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        return t;
#else
    return t;
#endif
}