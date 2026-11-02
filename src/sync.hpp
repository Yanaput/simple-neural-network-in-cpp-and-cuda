#pragma once

#include <vector>
#include <cstddef>

enum class Sync { None, Host, PerLayer, CudaAware, Local };

Sync  parse_sync(const char* s);

void  sync_init(int* argc, char** argv[],
                int* rank,
                int* P,
                int* local_rank);                //   position on this node -> which GPU (ops_init)

void  sync_finalize();

// offsets = start of every W and b array in the flat buffer, plus n at the end
void  sync_gradients(Sync mode, float* grads, size_t n, const std::vector<size_t>& offsets);

void  average_weights(float* params, size_t n);  // Local SGD: params <- mean over ranks

float  sync_sum(float x);
double sync_max(double t);   // slowest rank's time, for the timing CSV
