# GoTorch — Build Instructions

GoTorch contains its own C++ backend (`aten/`, `c10/`, `torch/csrc/`).
**No separate libtorch download needed.**

## Architecture

```
GoTorch/
├── aten/              ✅ C++ tensor ops (built-in)
├── c10/               ✅ C++ core lib (built-in)
├── torch/csrc/        ✅ C++ API (built-in)
│
├── csrc/go_binding/   ← C shim: Go ↔ C++ bridge
│   ├── torch_api.h
│   └── torch_api.cpp
│
└── torch/             ← Go packages (Python replaced)
    ├── tensor.go
    ├── nn/modules/
    ├── optim/
    └── ...
```

## Step 1: Build the C++ backend

```bash
git clone https://github.com/Sarkar-AGI/GoTorch
cd GoTorch

mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_PYTHON=OFF \
    -DBUILD_TEST=OFF

cmake --build . --target torch torch_cpu c10 -j$(nproc)
cd ..
```

## Step 2: Build the C shim

```bash
g++ -std=c++17 -c csrc/go_binding/torch_api.cpp \
    -I./torch/csrc/api/include \
    -I./build/include \
    -Ibuild/_deps/fmt-src/include \
    -o csrc/go_binding/torch_api.o

ar rcs csrc/go_binding/libtorch_go.a csrc/go_binding/torch_api.o
```

## Step 3: Set CGo flags

```bash
export GOTORCH=$(pwd)

export CGO_CFLAGS="-I${GOTORCH}/torch/csrc/api/include \
    -I${GOTORCH}/build/include \
    -I${GOTORCH}/csrc/go_binding"

export CGO_LDFLAGS="-L${GOTORCH}/build/lib \
    -L${GOTORCH}/csrc/go_binding \
    -Wl,-rpath,${GOTORCH}/build/lib \
    -ltorch -ltorch_cpu -lc10 \
    -ltorch_go"
```

## Step 4: Build & run

```bash
go build ./...
go run examples/train_mlp.go
```

## GPU (CUDA) support

```bash
# CUDA build — requires CUDA toolkit installed
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=ON \
    -DBUILD_PYTHON=OFF \
    -DBUILD_TEST=OFF

cmake --build . -j$(nproc)
```

## Go API → Internal C++ mapping

| GoTorch Go | GoTorch C++ (built-in) |
|---|---|
| `torch.Randn(N, D)` | `torch::randn({N, D})` in `aten/` |
| `modules.NewLinear(in, out, true)` | `torch::nn::Linear` in `torch/csrc/` |
| `optim.NewAdam(...)` | `torch::optim::Adam` in `torch/csrc/` |
| `cuda.IsAvailable()` | `torch::cuda::is_available()` in `c10/` |
