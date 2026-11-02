#include "ops.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>

float* ops_alloc(size_t n){
    return new float[n]();
}

void ops_free(float* p){
    delete[] p;
}

void ops_upload(float* dst, const float* host_src, size_t n){
    std::memcpy(dst, host_src, sizeof(float) * n);
}

void ops_download(float* host_dst, const float* src, size_t n){
    std::memcpy(host_dst, src, sizeof(float) * n);
}

void ops_gemm(bool transA, bool transB, int M, int N, int K, const float* A, const float* B, float* C){
    if(transB){
        for (int i = 0; i < M; i++){
            for (int j = 0; j < N; j++){
            float sum = 0.0f;
                for (int k = 0; k < K; k++){
                    int a_idx = transA ? k * M + i : i * K + k;
                    int b_idx = j * K + k;
                    sum += A[a_idx] * B[b_idx];
                }
            C[i * N + j] = sum;
            }
        }
    }
    else{
        // i-k-j order: row i of C += A(i,k) * row k of B, so the inner
        // loop walks rows of B and C at stride 1 (vectorizable at -O3).
        for (int i = 0; i < M; i++){
            float* c_row = C + (size_t)i * N;
            for (int j = 0; j < N; j++) c_row[j] = 0.0f;
            for (int k = 0; k < K; k++){
                float a = transA ? A[(size_t)k * M + i] : A[(size_t)i * K + k];
                const float* b_row = B + (size_t)k * N;
                for (int j = 0; j < N; j++){
                    c_row[j] += a * b_row[j];
                }
            }
        }
    }
}

void ops_add_bias(float* Z, const float* b, int rows, int cols){
    for(int row = 0; row < rows; row++){
        for(int col = 0; col < cols; col++){
            Z[cols * row + col] += b[col];
        }
    }
}

void ops_relu(float* A, const float* Z, size_t n){
    for(size_t i = 0; i < n; i++){
        A[i] = std::max(0.0f, Z[i]); 
    }
}

void ops_relu_backward(float* dZ, const float* Z, size_t n){
    for(size_t i = 0; i < n; i++){
        dZ[i] = Z[i] > 0.0f ? dZ[i] : 0.0f;
    }
}


// Fused softmax + cross-entropy. Z: rows x cols logits, Y: rows x cols one-hot labels.
// Per row i, with c = true class and m_i = max_j Z[i,j] (for numerical stability):
//   P[i,j]  = exp(Z[i,j] - m_i) / sum_k exp(Z[i,k] - m_i)
//   L_i     = -log P[i,c] = m_i + log(sum_k exp(Z[i,k] - m_i)) - Z[i,c]
//   dZ[i,j] = (P[i,j] - Y[i,j]) / rows
// Returns the mean loss (1/rows) * sum_i L_i. Dividing by the local batch size means
// the MPI gradient sync must average (not just sum) across ranks.
float ops_softmax_grad(float* dZ, const float* Z, const float* Y, int rows, int cols){
    double total = 0; // sum of per-row losses
    for(int i = 0; i < rows; i++){
        const float* z = Z + i*cols; // logits of sample i
        const float* y = Y + i*cols; // label of sample i
        float* dz = dZ + i*cols; // gradient of row
        float max = z[0];
        double s = 0.f;   // sum_k exp(z[k] - max)
        float z_c = 0.f;   // logit of the true class, z[c]
        // row max m_i
        for(int j = 0; j < cols; j++){
            if( z[j] > max){
                max = z[j];
            } 
        }
        // shifted exponentials (stored in dz as scratch), their sum s, and z[c]
        for(int j = 0; j < cols; j++){
            float e =  std::exp(z[j]-max);
            dz[j] = e;
            s += e;
            if(y[j] == 1.0f) z_c = z[j];
        }
        // row loss L_i = m_i + log(s) - z[c]
        total += max + std::log(s) - z_c;

        // turn the stored exponentials into the gradient (P - Y) / rows
        for(int j = 0; j < cols; j++){
            dz[j] = (dz[j]/s -y[j])/rows;
        }
    }
    return (float)(total/rows);
}

void ops_col_sum(float* db, const float* dZ, int rows, int cols){
    for(int j = 0; j < cols; j++){
        db[j] = 0;
    }

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            db[j] += dZ[i * cols + j];
        }
    }
}

void ops_scale(float* x, size_t n, float s){
    for(size_t i = 0; i < n; i++){
        x[i] = x[i]*s;
    }
}

void ops_sgd(float* w, const float* g, size_t n, float lr){
    for(size_t i = 0; i < n; i++){
        w[i] = w[i] - lr*g[i];
    }
}


// for idx
int* ops_alloc_int(size_t n){
    return new int[n]();
}

void ops_free_int(int* p){
    delete[] p;
}

void upload_int(int* dst, const int* host_src, size_t n){
    std::memcpy(dst, host_src, sizeof(float) * n);
}