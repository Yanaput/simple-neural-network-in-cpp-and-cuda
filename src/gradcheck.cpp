/*
Gradient check (CPU only, float)

Compares the analytic gradients from backward() with central finite differences
  numeric[i] = (L(p_i + eps) - L(p_i - eps)) / (2 eps)
on a tiny 4 -> 3 -> ReLU -> 2 model with batch 3.

Build and run:
  g++ -O3 -march=native .\src\gradcheck.cpp .\src\ops_cpu.cpp -o .\tests\gradcheck.exe
  .\tests\gradcheck.exe        exit code 0 = pass

Reading the output:
- eps sweep: with correct code the worst error falls, bottoms out around eps 1e-2..1e-3,
  then rises again (float round-off). A bug stays large at every eps.
- ratio analytic/numeric: ~3 (= B) or ~1/3 means mean-vs-sum mismatch, ~-1 a sign error.
- One isolated failure whose hidden unit has |Z1| near eps is a ReLU kink, not a bug:
  rerun with a different seed (first argument).
*/

#include "sequential.hpp"
#include "relu.hpp"
#include "linear.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

// Loss at the current parameters: forward + loss only, no backward.
// Takes the model by reference: Sequential holds unique_ptrs and can't be copied.
float loss_at_current_params(Sequential& model, float* x, float* y, float* g){
    Tensor logits = model.forward(Tensor{x, 3, 4});
    return ops_softmax_grad(g, logits.data, y, 3, 2);
}

const char* block_name(int k){
    static const char* names[] = {"W1", "b1", "W2", "b2"};
    return k < 4 ? names[k] : "?";
}

int main(int argc, char* argv[]){
    int SEED = argc > 1 ? std::atoi(argv[1]) : 42;
    const float ATOL = 1e-4f, RTOL = 1e-2f;
    const float EPS_REPORT = 1e-2f;               // eps used for the detailed table and pass/fail
    const float EPS_SWEEP[] = {1e-1f, 1e-2f, 1e-3f, 1e-4f};

    Sequential model;
    model.add(new Linear(4, 3, 3, false));
    model.add(new ReLu(3, 3));
    model.add(new Linear(3, 2, 3));

    model.allocate(SEED);

    std::mt19937 gen(SEED + 1);                   // separate stream from the weight init
    std::uniform_real_distribution<float> distrib_x(-1.0, 1.0);

    std::vector<float> x_h ((size_t)3*4), y_h ((size_t)3*2);

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 4; j++){
            x_h[i*4 + j] = distrib_x(gen);
        }
    }

    y_h = {
        0, 1,
        1, 0,
        1, 0
    };

    float* x = ops_alloc(x_h.size());
    float* y = ops_alloc(y_h.size());
    float* g = ops_alloc((size_t)3*2);
    ops_upload(x, x_h.data(), x_h.size());
    ops_upload(y, y_h.data(), y_h.size());

    // Steps 3-4: analytic gradients
    float loss = loss_at_current_params(model, x, y, g);
    model.backward(Tensor{g, 3, 2});

    std::vector<float> grad_h (model.n_params);
    ops_download(grad_h.data(), model.grads, model.n_params);

    // Smallest |Z1| (hidden pre-activation), to judge ReLU-kink risk.
    // Linear1's output buffer still holds Z1 from the forward above.
    Linear* l1 = static_cast<Linear*>(model.layers[0].get());
    float min_abs_z1 = INFINITY;
    for(int i = 0; i < 3*3; i++) min_abs_z1 = std::min(min_abs_z1, std::fabs(l1->y[i]));

    printf("seed %d   n_params %zu   loss %.6f   min|Z1| %.3e\n\n",
           SEED, model.n_params, loss, min_abs_z1);

    // Step 6: central differences. On the CPU build model.params is host memory.
    auto numeric_grad = [&](size_t i, float eps){
        float orig = model.params[i];
        model.params[i] = orig + eps;
        float lp = loss_at_current_params(model, x, y, g);
        model.params[i] = orig - eps;
        float lm = loss_at_current_params(model, x, y, g);
        model.params[i] = orig;
        return (lp - lm) / (2.0f * eps);
    };

    // Which W/b block index i belongs to (offsets has one entry per W and b, plus n_params).
    auto block_of = [&](size_t i){
        int k = 0;
        while(i >= model.offsets[k + 1]) k++;
        return k;
    };

    // Detailed table at EPS_REPORT
    printf("eps = %.0e\n", EPS_REPORT);
    printf("%-4s %4s %13s %13s %11s %11s %9s\n",
           "blk", "idx", "analytic", "numeric", "abs err", "rel err", "ratio");

    int n_fail = 0;
    int n_blocks = (int)model.offsets.size() - 1;
    std::vector<float> worst_rel(n_blocks, 0.0f);

    for(size_t i = 0; i < model.n_params; i++){
        float a = grad_h[i];
        float n = numeric_grad(i, EPS_REPORT);
        float abs_err = std::fabs(a - n);
        float scale = std::max(std::fabs(a), std::fabs(n));
        float rel_err = abs_err / std::max(scale, 1e-8f);
        bool ok = abs_err <= ATOL + RTOL * scale;
        int k = block_of(i);

        worst_rel[k] = std::max(worst_rel[k], rel_err);
        if(!ok) n_fail++;

        printf("%-4s %4zu %13.6e %13.6e %11.3e %11.3e %9.4f %s\n",
               block_name(k), i - model.offsets[k], a, n, abs_err, rel_err,
               n != 0.0f ? a / n : 0.0f, ok ? "" : "  <-- FAIL");
    }

    printf("\nworst rel err per block:");
    for(int k = 0; k < n_blocks; k++) printf("  %s %.3e", block_name(k), worst_rel[k]);
    printf("\n\n");

    // eps sweep, worst abs and rel error over all params
    printf("%-8s %11s %11s\n", "eps", "worst abs", "worst rel");
    for(float eps : EPS_SWEEP){
        float worst_abs = 0.0f, worst_r = 0.0f;
        for(size_t i = 0; i < model.n_params; i++){
            float a = grad_h[i];
            float n = numeric_grad(i, eps);
            float abs_err = std::fabs(a - n);
            float scale = std::max(std::fabs(a), std::fabs(n));
            worst_abs = std::max(worst_abs, abs_err);
            worst_r = std::max(worst_r, abs_err / std::max(scale, 1e-8f));
        }
        printf("%-8.0e %11.3e %11.3e\n", eps, worst_abs, worst_r);
    }

    printf("\n%s: %d / %zu params failed (atol %.0e, rtol %.0e, eps %.0e)\n",
           n_fail == 0 ? "PASS" : "FAIL", n_fail, model.n_params, ATOL, RTOL, EPS_REPORT);

    ops_free(x);
    ops_free(y);
    ops_free(g);
    return n_fail == 0 ? 0 : 1;
}
