# GoTorch — Agent Guidelines

## Project identity

**GoTorch** = libtorch C++ backend + Go interface (Python layer removed).

- Language: **Go** (not Python)
- Build system: `go build` + CGo + libtorch
- Test: `go test ./...`

## What changed from the original Python interface

| Component | Status |
|---|---|
| `aten/` C++ tensor ops | ✅ Unchanged |
| `c10/` C++ core | ✅ Unchanged |
| `torch/csrc/` C++ API | ✅ Unchanged |
| `csrc/go_binding/` | 🆕 New — C shim |
| `torch/**/*.py` | 🗑️ Deleted |
| `torch/**/*.go` | 🆕 New — Go replacements |

## When modifying code

- **Never** write Python code (`.py` files are deleted)
- **Never** modify `aten/`, `c10/`, or `torch/csrc/` (C++ core)
- **Always** add to `csrc/go_binding/torch_api.h` before calling from Go
- New layers go in `torch/nn/modules/`
- New ops go in `torch/tensor.go`

## Build commands

```bash
go build ./...           # build all Go packages
go test ./...            # run all tests
go vet ./...             # lint
gofmt -w ./torch/        # format
```

## Go import paths

```go
"github.com/Sarkar-AGI/GoTorch/torch"              // tensors
"github.com/Sarkar-AGI/GoTorch/torch/nn/modules"   // layers
"github.com/Sarkar-AGI/GoTorch/torch/nn/functional" // F.*
"github.com/Sarkar-AGI/GoTorch/torch/optim"         // optimizers
"github.com/Sarkar-AGI/GoTorch/torch/autograd"      // grad control
"github.com/Sarkar-AGI/GoTorch/torch/cuda"          // CUDA
"github.com/Sarkar-AGI/GoTorch/torch/utils/data"    // DataLoader
```
