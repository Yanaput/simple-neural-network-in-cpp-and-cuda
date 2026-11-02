
/*
sync.cpp

For Phase 1 it's just the Sync enum, plus sync_gradients returning right away for None. 
It's worth adding a declaration header now (sync.hpp, as in the plan's file layout).

The one design point: train_cpu has to build on Windows, where you have no MPI. 
So decide now how MPI stays out of the CPU build. 
The usual options are a #ifdef USE_MPI guard or a separate compile flag. 
That way adding Host in Phase 3 doesn't break train_cpu.
*/


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

void sync_init(int* argc, char* argv[], int* rank, int* P, int* local_rank){
    MPI_Init(argc, argv);
}

void sync_finalize(){

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

    // Host, PerLayer, CudaAware: implemented later behind #ifdef USE_MPI
    std::fprintf(stderr, "error: sync mode not available in this build (compiled without USE_MPI)\n");
    std::exit(1);
}

void average_weights(float* params, size_t n){
    // Local SGD weight averaging: implemented later behind #ifdef USE_MPI
    std::fprintf(stderr, "error: average_weights not available in this build (compiled without USE_MPI)\n");
    std::exit(1);
}