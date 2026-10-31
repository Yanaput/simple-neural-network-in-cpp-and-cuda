#pragma once
#include <cstddef>
// Declared once here. ops_cpu.cpp implements them with plain loops;
// ops_cuda.cu implements them with cuBLAS and CUDA kernels.
// In train_gpu every float* below points to GPU memory.
float* ops_alloc(size_t n);
void   ops_free(float* p);
void   ops_upload(float* dst, const float* host_src, size_t n);
void   ops_download(float* host_dst, const float* src, size_t n);
void   ops_gemm(bool transA, bool transB, int M, int N, int K,
                const float* A, const float* B, float* C);   // row-major C = op(A) * op(B)
void   ops_add_bias(float* Z, const float* b, int rows, int cols);
void   ops_relu(float* A, const float* Z, size_t n);
void   ops_relu_backward(float* dZ, const float* Z, size_t n);
float  ops_softmax_grad(float* dZ, const float* Z, const float* Y, int rows, int cols); // returns loss
void   ops_col_sum(float* db, const float* dZ, int rows, int cols);
void   ops_scale(float* x, size_t n, float s);
void   ops_sgd(float* w, const float* g, size_t n, float lr);
