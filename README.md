# MLP on EMNIST-digits with MPI + CUDA

A fully connected MLP (784 → h₁ → h₂ → 10, ReLU, softmax + cross-entropy, mini-batch SGD) trained on EMNIST-digits. The same source builds four programs:

| Binary | Sources | Runs on |
|---|---|---|
| `train_cpu` | `main.cpp ops_cpu.cpp sync.cpp mnist.cpp` | 1 CPU core |
| `train_cpu_mpi` | the same, plus `-DUSE_MPI` | one MPI rank per CPU core |
| `train_gpu` | `main.cpp ops_cuda.cu sync.cpp mnist.cpp` | 1 GPU |
| `train_gpu_mpi` | the same, plus `-DUSE_MPI` | one MPI rank per GPU |

Only two things change between the four builds, and each lives in one place:

- **Where the math runs:** `ops_cpu.cpp` (plain loops) or `ops_cuda.cu` (CUDA kernels). Both implement the functions declared in `ops.hpp`. In the GPU builds, every `float*` passed to an `ops_*` function is device memory.
- **Whether MPI is used:** `sync.cpp`. Without `-DUSE_MPI` every function in it becomes a single-process no-op.

The layer classes and the training loop never call CUDA or MPI directly, only `ops_*` and `sync_*`.

## Running on Vanda

First, on the login node (compute nodes have no internet and no `git`), clone the repo and download the data:

```bash
git clone https://github.com/Yanaput/simple-neural-network-in-cpp-and-cuda.git
cd simple-neural-network-in-cpp-and-cuda
bash scripts/get_emnist.sh        # downloads EMNIST-digits into data/
```

The commands below assume the repo is in your home folder. Every option of the training programs is listed under [Flags](#maincpp-training-program).

The GPU nodes have 2 × NVIDIA A40 (`sm_86`), and a job can use at most 2 GPUs. The CPU nodes have 2 × 36-core Xeon Platinum 8452Y.

Each command below starts an interactive job run the rest inside it. Larger model may require a longer walltime.

### CPU: serial and MPI (Intel compiler and Intel MPI)

```bash
# Serial: one core
qsub -I -l select=1:ncpus=1:mpiprocs=1:ompthreads=1:mem=16gb -l walltime=02:00:00

# MPI on one node: e.g. 32 ranks
qsub -I -l select=1:ncpus=32:mpiprocs=32:ompthreads=1:mem=64gb -l walltime=02:00:00

# MPI across nodes: e.g. 256 ranks = 8 nodes x 32
qsub -I -l select=8:ncpus=32:mpiprocs=32:ompthreads=1:mem=64gb -l place=scatter -l walltime=02:00:00
```

Inside the job:

```bash
cd ~/simple-neural-network-in-cpp-and-cuda
source /app1/ebenv
module purge
module load impi/2021.13.0-intel-compilers-2024.2.0
mkdir -p executable

# -fp-model=precise: icpx uses fast floating point by default; this keeps the results identical to g++/nvcc
icpx    -O3 -march=native -std=c++17 -fp-model=precise           src/main.cpp src/ops_cpu.cpp src/sync.cpp src/mnist.cpp -o executable/train_cpu
mpiicpx -O3 -march=native -std=c++17 -fp-model=precise -DUSE_MPI src/main.cpp src/ops_cpu.cpp src/sync.cpp src/mnist.cpp -o executable/train_cpu_mpi

# Quick check that the build works (one epoch, about 30 s)
executable/train_cpu --batch 1024 --model 1x --epochs 1 --log results_local/cpu_check.csv

# Serial
executable/train_cpu --batch 1024 --model 1x --epochs 10 --log results_local/cpu_1x.csv

# MPI. On multi-node jobs, start the remote ranks through PBS rather than ssh
# (ssh between nodes is refused on some CN nodes).
export I_MPI_HYDRA_BOOTSTRAP=rsh I_MPI_HYDRA_BOOTSTRAP_EXEC=/opt/pbs/bin/pbs_tmrsh
mpirun -np 32 executable/train_cpu_mpi --batch 1024 --model 1x --sync host --epochs 10 --log results_local/cpu_mpi_np32.csv
```

Each `train_cpu_mpi` rank needs about 1.8 GB of memory, because every rank loads the whole dataset. Intel MPI reads the PBS node list itself, so no `--hostfile` is needed.

### GPU: single GPU and MPI + CUDA (NVHPC with HPC-X)

```bash
# 1 or 2 GPUs on one node
qsub -I -l select=1:ncpus=2:mpiprocs=2:ompthreads=1:ngpus=2:mem=120gb -l walltime=01:00:00
```

Inside the job:

```bash
cd ~/simple-neural-network-in-cpp-and-cuda
module purge
module load nvhpc-hpcx-cuda12/26.1
mkdir -p executable build

# Single GPU
nvcc -O3 -std=c++17 -arch=sm_86 src/main.cpp src/ops_cuda.cu src/sync.cpp src/mnist.cpp -o executable/train_gpu

# MPI + CUDA: compile the kernels with nvcc, link with mpicxx against the CUDA runtime that matches nvcc
nvcc -O3 -std=c++17 -arch=sm_86 -c src/ops_cuda.cu -o build/ops_cuda.o
CUDA_VER=$(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
CUDA_LIB=$(dirname "$(readlink -f "$(which nvcc)")")/../../cuda/$CUDA_VER/targets/x86_64-linux/lib
mpicxx -O3 -std=c++17 -DUSE_MPI src/main.cpp src/sync.cpp src/mnist.cpp build/ops_cuda.o \
    -L"$CUDA_LIB" -lcudart -Wl,-rpath,"$CUDA_LIB" -o executable/train_gpu_mpi

# Quick check that the build works (one epoch, a few seconds)
executable/train_gpu --batch 1024 --model 1x --epochs 1 --log results_local/gpu_check.csv

# Single GPU
executable/train_gpu --batch 1024 --model 1x --epochs 10 --log results_local/gpu_1x.csv

# 2 GPUs, one rank each. --bind-to none avoids an hwloc_set_cpubind failure under HPC-X.
mpirun -np 2 --bind-to none --map-by ppr:2:node executable/train_gpu_mpi \
    --batch 1024 --model 1x --sync cudaaware --epochs 10 --log results_local/gpu_mpi_P2.csv
```

Add `-DTILE=8` or `-DTILE=32` to both `nvcc` lines to change the matmul tile size. `--sync cudaaware` needs a CUDA-aware MPI, which HPC-X is; `--sync host` and `--sync local` work with any MPI.

### Gradient check

```bash
g++ -O3 -march=native src/gradcheck.cpp src/ops_cpu.cpp -o executable/gradcheck
executable/gradcheck 42     # exit code 0 = pass
```

### Checking a run

- Every rank prints `rank r checksum sum … sum|w| …` at the end. With `--sync host` or `cudaaware`, all ranks must print the same two numbers.
- With the same `--batch` and `--seed`, a `host` or `cudaaware` run on P ranks should give the same loss column as a run on 1 rank, up to rounding.


## Source files (`src/`)

```
main.cpp ──► sequential.hpp ──► linear.hpp ─┐
   │                        └─► relu.hpp ───┼─► module.hpp
   │                                        └─► ops.hpp ◄── ops_cpu.cpp | ops_cuda.cu
   ├──► sync.hpp ◄── sync.cpp (MPI)
   └──► mnist.hpp ◄── mnist.cpp
gradcheck.cpp: separate test program (layers + ops_cpu.cpp only)
```

Conventions used throughout:

- Matrices are flat **row-major** `float` arrays: element (i, j) of an R×C matrix is `A[i*C + j]`.
- Weights are stored **in × out**, so the forward pass is `Y = X·W` with X of shape batch × in.
- All parameters live in **one contiguous buffer** and all gradients in a second buffer with the same layout, so a single `MPI_Allreduce` covers every gradient.

### `ops.hpp`: the operations interface

Declares every numeric operation once. `ops_cpu.cpp` and `ops_cuda.cu` define the same names and signatures, and the linker decides which one is used.

| Function | What it does |
|---|---|
| `ops_init(local_rank)` | GPU: selects device `local_rank % device_count`. CPU: no-op. |
| `ops_sync()` | GPU: `cudaDeviceSynchronize()`, called before every timer read because kernels are asynchronous. CPU: no-op. |
| `ops_alloc(n)` / `ops_free(p)` | Allocates `n` zeroed floats (`new[]` or `cudaMalloc`) and frees them. |
| `ops_upload(dst, host_src, n)` | Copies `n` floats from host memory into a buffer from `ops_alloc` (a `memcpy` on CPU, host → device on GPU). |
| `ops_download(host_dst, src, n)` | The reverse copy, into host memory. |
| `ops_matmul(transA, transB, M, N, K, A, B, C)` | Row-major `C = op(A)·op(B)`, where `op` optionally transposes. M, N, K are the sizes *after* `op`: op(A) is M×K and op(B) is K×N. Overwrites C (does not accumulate). |
| `ops_add_bias(Z, b, rows, cols)` | `Z[i][j] += b[j]` for every row. |
| `ops_relu(A, Z, n)` | `A = max(0, Z)` element-wise. |
| `ops_relu_backward(dZ, Z, n)` | Zeroes `dZ[i]` wherever `Z[i] ≤ 0`, in place. |
| `ops_softmax_grad(dZ, Z, Y, rows, cols)` | Fused softmax and cross-entropy. Writes `dZ = (softmax(Z) − Y) / rows` and returns the mean loss. Subtracts each row's max before `exp` for numerical stability. |
| `ops_col_sum(db, dZ, rows, cols)` | Column sums of `dZ` (the bias gradient). |
| `ops_scale(x, n, s)` | `x *= s` (used for the 1/P average after an all-reduce). |
| `ops_sgd(w, g, n, lr)` | SGD update `w -= lr · g`. |
| `ops_gather_rows(dst, src, idx, nrows, cols)` | `dst` row r ← `src` row `idx[r]`. Builds a shuffled mini-batch without copying the dataset. |
| `ops_alloc_int`, `ops_free_int`, `ops_upload_int` | The same as above for `int` buffers (the shuffled index array). |

### `ops_cpu.cpp`: CPU implementation

Plain C++ loops for every function in `ops.hpp`. Memory functions are `new[]`/`delete[]`/`memcpy`, and `ops_init`/`ops_sync` do nothing.

`ops_matmul` uses a loop order that keeps the innermost loop on contiguous memory:

- **Without `transB`** (forward `X·W` and `dW = Xᵀ·G`): i-k-j order. The inner loop does `C[i,:] += A[i,k] · B[k,:]`, reading a row of B and updating a row of C at stride 1, which `-O3` vectorizes.
- **With `transB`** (`dX = G·Wᵀ`): column j of Bᵀ is row j of B, so the i-j-k order is already stride 1, and the inner loop is a dot product of two rows. It accumulates into 8 independent partial sums so the compiler can vectorize it without `-ffast-math`.

`ops_softmax_grad` keeps the per-row exponentials in `dZ` as scratch space, then turns them into the gradient in a second pass.

### `ops_cuda.cu`: GPU implementation

The same functions with hand-written CUDA kernels (cuBLAS is not used). Every call is checked with `CUDA_CHECK`, which prints the error and exits.

| Kernel | Mapping |
|---|---|
| `cuda_relu`, `cuda_relu_backward`, `cuda_add_bias`, `cuda_scale`, `cuda_sgd` | One thread per element, 256 threads per block. |
| `cuda_matmul` | Shared-memory tiled. Each block computes a `TILE × TILE` tile of C with one thread per element. It loops over K in phases, loading one tile of op(A) and one of op(B) into shared memory per phase, padding out-of-range entries with 0. Transposes are handled in the load indexing, so no transposed copy is made. `TILE` is 16 by default; override it with `-DTILE=8` or `-DTILE=32` (32 is the maximum, since a block holds at most 1024 threads). |
| `cuda_gather_rows` | One block per batch row; the block's threads stride across the columns, so reads and writes are contiguous. |
| `cuda_col_sum` | One thread per column, looping over the rows. |
| `cuda_softmax_grad` | One thread per row (only 10 classes). Writes `dZ` and a per-row loss into a device buffer, which `ops_softmax_grad` downloads and sums on the host. The buffer is allocated once and kept in a `static`. |

Helpers: `get_block_num(n, threads)` returns the number of blocks needed to cover `n` elements.

### `module.hpp`: base types

| Type | Purpose |
|---|---|
| `Tensor` | A non-owning view `{data, rows, cols}` of a row-major matrix. |
| `Module` | Abstract layer. `forward(x)` and `backward(grad)` return the output Tensor and the gradient with respect to the input. `param_sizes()` lists the sizes of the layer's parameter blocks, `bind(p, g)` points the layer at its slice of the shared parameter and gradient buffers, and `init(host, rng)` writes initial values into a host-side copy. |

### `linear.hpp`: `Linear` layer

An affine layer `Y = X·W + b` with `W` of shape in × out.

- **Constructor** `Linear(in, out, max_batch, need_dx = true)` allocates the output buffer `y` and, if `need_dx`, the input-gradient buffer `dx`, both sized for `max_batch` rows. The first layer passes `need_dx = false`, since nothing needs the gradient with respect to the images. The copy constructor is deleted, because the layer owns these buffers.
- `param_sizes()` returns `{in·out, out}` (W, then b).
- `bind(p, g)` sets `W`, `b`, `dW` and `db` to offsets into the shared buffers. The layer never owns its parameters.
- `init(host, rng)` uses He initialization, W ~ N(0, 2/in), with b = 0.
- `forward(x)` saves `x` for backward, then `ops_matmul(false, false)` followed by `ops_add_bias`.
- `backward(g)` computes `dW = Xᵀ·G` with `ops_matmul(true, false)`, `db = col_sum(G)`, and, if needed, `dX = G·Wᵀ` with `ops_matmul(false, true)`.

### `relu.hpp`: `ReLu` layer

The activation layer, with no parameters. The constructor allocates the output buffer for `max_batch` rows.

- `forward(x)` saves the input pointer and computes `y = max(0, x)` with `ops_relu`.
- `backward(g)` masks the incoming gradient in place with `ops_relu_backward`, using the saved input, and returns it.

### `sequential.hpp`: `Sequential` container

Like `nn.Sequential`: holds the layers in order and owns the shared buffers.

- `add(m)` appends a layer and takes ownership of it (`unique_ptr`).
- `allocate(seed)` adds up every layer's `param_sizes()`, records the start of each W and b in `offsets` (with `n_params` appended at the end), allocates one `params` and one `grads` buffer, then binds each layer to its slice. It initializes the weights on the host with an `mt19937(seed)` and uploads them once. Every rank uses the same seed, so all ranks start with identical weights.
- `forward(x)` runs the layers first to last; `backward(g)` runs them last to first.
- The destructor frees `params` and `grads`.

### `sync.hpp` / `sync.cpp`: MPI and gradient synchronization

All MPI calls in the program are in this file, inside `#ifdef USE_MPI`. Without that flag, each function does the single-process equivalent: rank 0 of 1, and reductions return their input.

| Function | What it does |
|---|---|
| `enum class Sync { None, Host, CudaAware, Local }` | The gradient synchronization modes. |
| `parse_sync(s)` | Turns the `--sync` text (`none`, `host`, `cudaaware`, `local`) into a `Sync`; exits on anything else. |
| `sync_init(argc, argv, rank, P, local_rank)` | `MPI_Init`, then the rank and the number of ranks P. `local_rank` is the rank's position on its own node, found with `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)`, and `ops_init` uses it to pick a GPU. |
| `sync_finalize()` | `MPI_Finalize`. |
| `sync_gradients(mode, grads, n)` | Called between `backward` and `ops_sgd`. **Host:** download the gradients, `MPI_Allreduce` (sum) on host memory, upload, then scale by 1/P. **CudaAware:** `MPI_Allreduce` directly on the device buffer (needs a CUDA-aware MPI), then scale by 1/P. **None** and **Local:** return immediately. It calls `ops_sync()` first, so unfinished backward kernels aren't counted as communication time. |
| `average_weights(params, n)` | Local SGD: replaces each rank's weights with the mean over all ranks (download, all-reduce, upload, scale by 1/P). |
| `sync_sum(x)`, `sync_max(t)`, `sync_mean(t)` | Scalar all-reduces: the loss is summed over ranks, and epoch times are reported as the mean (so compute + copy + MPI adds up) and the max (the slowest rank). |
| `sync_times_take()` | Returns the copy and MPI seconds accumulated by `sync_gradients` and `average_weights` since the last call, as a `SyncTimes {copy, mpi}`, and resets them. |

### `mnist.hpp` / `mnist.cpp`: dataset loader

| Item | What it does |
|---|---|
| `struct Mnist` | `n` samples; `images` (n × 784 floats scaled to [0, 1]), `labels` (n × 10 one-hot) and `label_ids` (n class indices). |
| `load_mnist(img_path, label_path)` | Reads a pair of IDX files (the MNIST/EMNIST format). It checks the magic numbers (2051 and 2049), that the image and label counts match, that images are 28×28, that the files aren't truncated, and that labels are below 10; any failure prints a message and exits. |
| `read_be_u32(f)` (static) | Reads one big-endian 32-bit header field. |
| `check(cond, msg)` (static) | Like `assert`, but never compiled out. |

### `main.cpp`: training program

1. `sync_init` and `ops_init`, then parses the `--flag value` pairs (below). Checks that `--batch` is divisible by P.
2. Builds `Linear(784, h₁) → ReLu → Linear(h₁, h₂) → ReLu → Linear(h₂, 10)` with widths from `get_hidden(--model)`, then calls `allocate(seed)`.
3. Loads the EMNIST train and test sets from `--data`, uploads them once, and frees the host copies. The data stays on the device for the whole run.
4. Each epoch:
   - Shuffles the index array with the shared seed (so all ranks agree on the order) and uploads it.
   - For each step, rank r gathers its slice of B/P rows of the global batch with `ops_gather_rows`, then runs forward → `ops_softmax_grad` → backward → `sync_gradients` → `ops_sgd`. With `--sync local`, it also calls `average_weights` every `--sync-every` steps and at the end of the epoch.
   - Times the epoch and splits it into copy, MPI and compute (the remainder), each averaged over ranks, plus the slowest rank's total.
   - Evaluates test accuracy (argmax of the logits).
5. Rank 0 writes the CSV. Then **every** rank prints a weight checksum (`sum` and `sum|w|`). The checksums must be identical across ranks.

Helpers: `get_hidden(name)` maps `1x`/`2x`/`4x`/`8x` to the hidden widths, `write_log(...)` writes the CSV, and `seconds_since(t0)` reads the steady clock.

| Model | Hidden widths | Parameters |
|---|---|---|
| `1x` | 512 / 256 | 535,818 |
| `2x` | 848 / 424 | 1,029,906 |
| `4x` | 1344 / 672 | 1,965,610 |
| `8x` | 2064 / 1032 | 3,761,650 |

**Flags** (each takes a value):

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `data` | Folder holding the four `emnist-digits-*-ubyte` files. |
| `--batch` | `128` | **Global** batch size. Must be divisible by the number of ranks; each rank processes `batch / P` rows. |
| `--model` | `1x` | `1x`, `2x`, `4x` or `8x`. |
| `--epochs` | `10` | Number of epochs. |
| `--lr` | `0.05` | Learning rate. |
| `--seed` | `42` | Seed for the weight initialization and the shuffle. Keep it the same on every rank. |
| `--sync` | `none` | `none`, `host`, `cudaaware` or `local`. Use `none` for the non-MPI builds. |
| `--sync-every` | `8` | Steps between weight averages (only for `--sync local`). |
| `--log_dir` | `logs` | Folder for the default CSV name. |
| `--log` | `<log_dir>/log_P{P}_B{B}_{model}_params{n}_{sync}.csv` | Full CSV path. Its folder is created if needed. |

**CSV columns:** `epoch, P, batch, model, sync, loss, acc, time, compute, copy, mpi, step_ms, time_max`. `time` and its parts are seconds per epoch, averaged over ranks; `step_ms` is `time` divided by the steps per epoch; `time_max` is the slowest rank's epoch time.

### `gradcheck.cpp`: finite-difference gradient check

A separate program (CPU only). It builds a tiny `4 → 3 → ReLU → 2` network with batch 3 and compares every analytic gradient from `backward()` with a central finite difference, `(L(w+ε) − L(w−ε)) / 2ε`.

- `loss_at_current_params(model, x, y, g)` runs forward and the loss only.
- `block_name(k)` labels parameter block k (W1, b1, W2, b2).
- `main` prints a table at ε = 10⁻², with the analytic/numeric ratio (about B or 1/B suggests a mean-vs-sum mistake, about −1 a sign error), then the worst error for ε from 10⁻¹ to 10⁻⁴. It exits with 0 if every parameter is within `atol 1e-4 + rtol 1e-2`. The first argument sets the seed. A single failure on a hidden unit whose pre-activation is close to 0 is a ReLU kink, not a bug.