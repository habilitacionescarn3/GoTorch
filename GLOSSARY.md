# GoTorch Glossary

Terms used throughout GoTorch and their original Python equivalents.

## Core types

| GoTorch (Go) | Original Python API | Description |
|---|---|---|
| `*torch.Tensor` | `torch.Tensor` | Multi-dimensional array |
| `torch.RawTensor` (`C.Tensor`) | — | Opaque C++ tensor handle used across CGo boundary |
| `modules.Module` (interface) | `nn.Module` | Base interface for all layers |
| `*modules.Sequential` | `nn.Sequential` | Ordered container of layers |
| `optim.Optimizer` (interface) | `optim.Optimizer` | Base optimizer interface |
| `data.Dataset` (interface) | `Dataset` | Base dataset interface |
| `*data.DataLoader` | `DataLoader` | Mini-batch iterator |

## Packages

| Go package | Python equivalent |
|---|---|
| `github.com/Sarkar-AGI/GoTorch/torch` | `import torch` |
| `torch/nn/modules` | `import torch.nn as nn` |
| `torch/nn/functional` | `import torch.nn.functional as F` |
| `torch/optim` | `import torch.optim as optim` |
| `torch/autograd` | `import torch.autograd` |
| `torch/cuda` | `import torch.cuda` |
| `torch/utils/data` | `from torch.utils.data import ...` |

## Patterns

| Concept | Go | Python |
|---|---|---|
| No-grad context | `autograd.WithNoGrad(func() { ... })` | `with torch.no_grad():` |
| Backward pass | `torch.BackwardRaw(loss)` | `loss.backward()` |
| Zero gradients | `optimizer.ZeroGrad()` | `optimizer.zero_grad()` |
| Optimizer step | `optimizer.Step()` | `optimizer.step()` |
| Training mode | `model.Train()` | `model.train()` |
| Eval mode | `model.Eval()` | `model.eval()` |
| Get parameters | `model.Parameters()` | `model.parameters()` |
| Tensor item | `t.Item()` | `t.item()` |
| Tensor shape | `t.Shape()` | `t.shape` |
| Requires grad | `t.SetRequiresGrad(true)` | `t.requires_grad_(True)` |
| Detach | `t.Detach()` | `t.detach()` |
| Move to GPU | `t.To(torch.CUDA)` | `t.to('cuda')` |

## Build terms

| Term | Description |
|---|---|
| **CGo** | Go's mechanism for calling C code from Go |
| **C shim** | `csrc/go_binding/torch_api.{h,cpp}` — bridges Go ↔ C++ |
| **libtorch** | GoTorch built-in C++ backend (`aten/`, `c10/`, `torch/csrc/`) |
| **ATen** | C++ tensor library (`aten/`) |
| **c10** | C++ core utility library (`c10/`) |
| **LIBTORCH** | Environment variable pointing to libtorch install |
| `CGO_CFLAGS` | C compiler flags for CGo (libtorch include paths) |
| `CGO_LDFLAGS` | Linker flags for CGo (libtorch lib paths) |
