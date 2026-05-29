# GoTorch Release Notes

## v1.0.0

### Summary
Initial release of GoTorch — a Go deep learning framework backed by the
libtorch C++ tensor library.

### What's included

**Go layer (new)**
- `torch/` — Tensor type with full arithmetic, shape, autograd ops
- `torch/nn/modules/` — Linear, Conv2d, ReLU, GELU, SiLU, ELU, Sigmoid,
  Tanh, Dropout, BatchNorm1d/2d, LayerNorm, Embedding, Sequential, ModuleList
- `torch/nn/functional/` — Stateless functional ops
- `torch/optim/` — SGD, Adam, AdamW, RMSprop + StepLR, CosineAnnealingLR,
  ReduceLROnPlateau
- `torch/nn/modules/loss.go` — MSE, CrossEntropy, BCE, BCEWithLogits, NLL,
  L1, Huber losses
- `torch/utils/data/` — Dataset, TensorDataset, DataLoader, samplers
- `torch/autograd/` — WithNoGrad context, grad control
- `torch/cuda/` — Device management, availability checks

**C shim (new)**
- `csrc/go_binding/torch_api.h` — C interface
- `csrc/go_binding/torch_api.cpp` — libtorch C++ implementation

**C++ backend (unchanged)**
- All C++ kernel code in `aten/`, `c10/`, `torch/csrc/` is the upstream
  libtorch library, used unmodified under its BSD license.

### Removed
- All Python source files (`torch/**/*.py`) — replaced by Go equivalents

### Breaking changes from the original Python interface
Everything. GoTorch is a Go API. See README.md for the full translation table.
