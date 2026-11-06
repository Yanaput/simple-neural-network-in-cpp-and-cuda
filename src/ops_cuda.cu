#include "ops.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <vector>

static int threads_per_block = 256;
// Override with -DTILE=<n>, up to 32 (the block limit is 1024 threads).
#ifndef TILE
#define TILE 16
#endif

#define CUDA_CHECK(call) do{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}while(0)


int get_block_num(int n, int threads){
    return (n + threads-1)/threads;
}

__global__ void cuda_relu(float* A, const float* Z, size_t n){
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(t_idx >= n) return;
    A[t_idx] = std::fmaxf(0.f, Z[t_idx]);
}

__global__ void cuda_relu_backward(float* dZ, const float* Z, size_t n){
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(t_idx >= n) return;
    dZ[t_idx] = Z[t_idx] > 0.f ? dZ[t_idx] : 0.f;
}

__global__ void cuda_scale(float* x, size_t n, float s){
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(t_idx >= n) return;
    x[t_idx] = x[t_idx] * s;
}

__global__ void cuda_sgd(float* w, const float* g, size_t n, float lr){
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(t_idx >= n) return;
    w[t_idx] = w[t_idx] - g[t_idx]*lr;
}

__global__ void cuda_add_bias(float* Z, const float* b, int rows, int cols){
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(t_idx >= rows*cols) return;
    int j = t_idx % cols; //col of i
    Z[t_idx] += b[j];
}

__global__ void cuda_gather_rows(float* dst, const float* src, const int* idx, int nrows, int cols){
    int r = blockIdx.x;
    int s = idx[r];
    for(int c = threadIdx.x; c < cols; c+=blockDim.x){
        dst[r*cols+c] = src[s*cols+c];
    } 
}

__global__ void cuda_col_sum(float* db, const float* dZ, int rows, int cols){
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(t_idx >= cols) return;
    float sum = 0.f;
    for(int i = 0; i < rows; i++){
        sum += dZ[i*cols+t_idx];
    }
    db[t_idx] = sum;
}

__global__ void cuda_softmax_grad(float* dZ, const float* Z, const float* Y, int rows, int cols, float* row_loss){
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(t_idx >= rows) return;
    const float* z = Z + t_idx*cols;
    const float* y = Y + t_idx*cols;
    float* dz = dZ + t_idx*cols;
    float max = z[0];
    for(int i = 0; i < cols; i++){
        if(z[i] > max) max = z[i];
    }

    float s = 0.f;
    float z_c = 0.f;
    for(int i = 0; i < cols; i++){
        float e = expf(z[i]-max);
        dz[i] = e;
        s += e;
        if(y[i] == 1.0f) z_c = z[i];
    }

    row_loss[t_idx] = max+logf(s) - z_c;
    
    for(int i = 0; i < cols; i++){
        dz[i] = (dz[i]/s - y[i]) / rows;
    }
}

__global__ void cuda_matmul(bool transA, bool transB, int M, int N, int K,
                const float* A, const float* B, float* C){
    __shared__ float s_A[TILE][TILE], s_B[TILE][TILE];
    int tx = threadIdx.x; // col
    int ty = threadIdx.y; // row
    int i = blockIdx.y * blockDim.y + ty; 
    int j = blockIdx.x * blockDim.x + tx;
    float sum = 0.f;

    for(int phase = 0; phase < (K+TILE-1)/TILE; phase++){
        int k_a = phase*TILE + tx;
        int k_b = phase*TILE + ty;
        if(i >= M || k_a >= K) s_A[ty][tx] = 0;
        else s_A[ty][tx] = transA ? A[k_a*M + i] : A[i*K + k_a];
        if(j >= N || k_b >= K) s_B[ty][tx] = 0;
        else s_B[ty][tx] = transB ? B[j*K + k_b] : B[k_b*N + j];
        __syncthreads();
        
        for(int k = 0; k < TILE; k++)
           sum += s_A[ty][k] * s_B[k][tx];
        __syncthreads();
    }

    if (i < M && j < N) C[i*N + j] = sum;
}

void ops_init(int local_rank){
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    CUDA_CHECK(cudaSetDevice(local_rank%device_count));
}

void ops_sync(){
    CUDA_CHECK(cudaDeviceSynchronize());
}

float* ops_alloc(size_t n){
    float* d_p; 
    CUDA_CHECK(cudaMalloc(&d_p, n* sizeof(float)));
    CUDA_CHECK(cudaMemset(d_p, 0, n*sizeof(float)));
    return d_p;
}

void ops_free(float* d_p){
    CUDA_CHECK(cudaFree(d_p));
}

void ops_upload(float* dst, const float* host_src, size_t n){
    CUDA_CHECK(cudaMemcpy(dst, host_src, n*sizeof(float), cudaMemcpyHostToDevice));
}

void ops_download(float* host_dst, const float* src, size_t n){
    CUDA_CHECK(cudaMemcpy(host_dst, src, n*sizeof(float), cudaMemcpyDeviceToHost));
}

void ops_matmul(bool transA, bool transB, int M, int N, int K,
                const float* A, const float* B, float* C){
    // row-major C = op(A) * op(B)
    dim3 block(TILE, TILE);
    dim3 grid(get_block_num(N, TILE), get_block_num(M, TILE));
    cuda_matmul<<<grid, block>>>(transA, transB, M, N, K , A, B, C);
    CUDA_CHECK(cudaGetLastError()); 
}

void ops_add_bias(float* Z, const float* b, int rows, int cols){
    cuda_add_bias<<<get_block_num(rows*cols, threads_per_block), threads_per_block>>>(Z, b, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void ops_relu(float* A, const float* Z, size_t n){
    cuda_relu<<<get_block_num(n, threads_per_block), threads_per_block>>>(A, Z, n);
    CUDA_CHECK(cudaGetLastError());
}

void ops_relu_backward(float* dZ, const float* Z, size_t n){
    cuda_relu_backward<<<get_block_num(n, threads_per_block), threads_per_block>>>(dZ, Z, n);
    CUDA_CHECK(cudaGetLastError());
}

float ops_softmax_grad(float* dZ, const float* Z, const float* Y, int rows, int cols){ // returns loss
    static float* d_row_loss = nullptr;
    static int cap = 0;
    std::vector<float> row_loss;

    if(rows > cap){
        if(d_row_loss) ops_free(d_row_loss);
        d_row_loss = ops_alloc(rows);
        cap = rows;
    }
    if((int)row_loss.size() < rows) row_loss.resize(rows);
    
    cuda_softmax_grad<<<get_block_num(rows, threads_per_block), threads_per_block>>>(dZ, Z, Y, rows, cols, d_row_loss);
    CUDA_CHECK(cudaGetLastError());

    ops_download(row_loss.data(), d_row_loss, rows);
    double total_loss = 0.f;
    for(int i = 0; i < rows; i++)
        total_loss += row_loss[i];
    
    return (float)(total_loss/rows);
} 

void ops_col_sum(float* db, const float* dZ, int rows, int cols){
    cuda_col_sum<<<get_block_num(cols, threads_per_block), threads_per_block>>>(db, dZ, rows, cols);
    CUDA_CHECK(cudaGetLastError());

}

void ops_scale(float* x, size_t n, float s){
    cuda_scale<<<get_block_num(n, threads_per_block), threads_per_block>>>(x, n, s);
    CUDA_CHECK(cudaGetLastError());
}

void ops_sgd(float* w, const float* g, size_t n, float lr){
    cuda_sgd<<<get_block_num(n, threads_per_block), threads_per_block>>>(w, g, n, lr);
    CUDA_CHECK(cudaGetLastError());
}

void ops_gather_rows(float* dst, const float* src, const int* idx, int nrows, int cols){
    cuda_gather_rows<<<nrows, threads_per_block>>>(dst, src, idx, nrows, cols);
    CUDA_CHECK(cudaGetLastError());
}

int* ops_alloc_int(size_t n){
    int* d_p; 
    CUDA_CHECK(cudaMalloc(&d_p, n* sizeof(int)));
    CUDA_CHECK(cudaMemset(d_p, 0, n*sizeof(int)));
    return d_p;
}

void ops_free_int(int *d_p){
    CUDA_CHECK(cudaFree(d_p));
}

void ops_upload_int(int* dst, const int* host_src, size_t n){
    CUDA_CHECK(cudaMemcpy(dst, host_src, n*sizeof(int), cudaMemcpyHostToDevice));
}