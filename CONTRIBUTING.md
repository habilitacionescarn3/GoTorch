# Contributing to GoTorch

GoTorch is a Go deep learning framework backed by the libtorch C++ library.
We welcome contributions to the Go layer, C shim, tests, and documentation.

## What can I contribute?

- **Go bindings** — new ops in `torch/tensor.go`, new layers in `torch/nn/modules/`
- **C shim** — new functions in `csrc/go_binding/torch_api.h` + `torch_api.cpp`
- **Tests** — Go test files (`*_test.go`)
- **Examples** — training scripts in `examples/`
- **Documentation** — `.md` files

> The C++ backend (`aten/`, `c10/`, `torch/csrc/`) is the upstream libtorch
> library and is not modified in GoTorch. Contributions to that layer belong
> to the upstream open-source project.

## Development setup

```bash
# 1. Install Go 1.21+
go version




# 3. Set CGo flags
export GOTORCH=$(pwd)
export CGO_CFLAGS="-I${GOTORCH}/torch/csrc/api/include \
    
    -I${GOTORCH}/csrc/go_binding"
export CGO_LDFLAGS="-L${GOTORCH}/build/lib \
    -Wl,-rpath,${GOTORCH}/build/lib \
    -ltorch -ltorch_cpu -lc10"

# 4. Build and test
go build ./...
go test ./...
```

## Adding a new layer

**Step 1** — Add C signature to `csrc/go_binding/torch_api.h`:
```c
Module gotorch_nn_prelu_new(int64_t num_parameters, double init);
Tensor gotorch_nn_prelu_forward(Module m, Tensor input);
```

**Step 2** — Implement in `csrc/go_binding/torch_api.cpp`:
```cpp
Module gotorch_nn_prelu_new(int64_t num_parameters, double init) {
    auto opts = torch::nn::PReLUOptions()
        .num_parameters(num_parameters).init(init);
    return reinterpret_cast<Module>(new torch::nn::PReLU(opts));
}
Tensor gotorch_nn_prelu_forward(Module m, Tensor input) {
    auto* mod = reinterpret_cast<torch::nn::PReLU*>(m);
    return newT((*mod)->forward(*T(input)));
}
```

**Step 3** — Add Go wrapper in `torch/nn/modules/activation.go`:
```go
type PReLU struct {
    module   C.Module
    training bool
}

func NewPReLU(numParameters int, init float64) *PReLU {
    m := C.gotorch_nn_prelu_new(C.int64_t(numParameters), C.double(init))
    return &PReLU{module: m, training: true}
}

func (p *PReLU) Forward(x C.Tensor) C.Tensor {
    return C.gotorch_nn_prelu_forward(p.module, x)
}
func (p *PReLU) Parameters() []C.Tensor { return nil }
func (p *PReLU) ZeroGrad()               {}
func (p *PReLU) Train()                  { p.training = true }
func (p *PReLU) Eval()                   { p.training = false }
func (p *PReLU) Name() string            { return "PReLU()" }
```

**Step 4** — Write a test in `torch/nn/modules/activation_test.go`:
```go
func TestPReLU(t *testing.T) {
    layer := NewPReLU(1, 0.25)
    x := torch.RandnRaw(4, 8)
    out := layer.Forward(x)
    shape := torch.ShapeRaw(out)
    if shape[0] != 4 || shape[1] != 8 {
        t.Errorf("unexpected shape %v", shape)
    }
}
```

## Adding a new tensor op

**Step 1** — `csrc/go_binding/torch_api.h`:
```c
Tensor gotorch_cumsum(Tensor t, int64_t dim);
```

**Step 2** — `csrc/go_binding/torch_api.cpp`:
```cpp
Tensor gotorch_cumsum(Tensor t, int64_t dim) {
    return newT(T(t)->cumsum(dim));
}
```

**Step 3** — `torch/tensor.go`:
```go
// Cumsum returns the cumulative sum along dim.
func (t *Tensor) Cumsum(dim int) *Tensor {
    return newTensor(C.gotorch_cumsum(t.ptr, C.int64_t(dim)))
}
```

## Code style

- Go: `gofmt` — run `gofmt -w ./torch/` before committing
- C++: C++17, match existing style in `csrc/go_binding/`
- Every Go function comment must name what it does (not "Mirrors ...")
- Keep CGo boundary thin — no logic in `torch_api.cpp`, only dispatch to libtorch

## Commit messages

```
feat(nn): add PReLU activation layer
fix(tensor): fix Shape() for 0-dim tensors
docs: update BUILD.md with macOS instructions
test(optim): add Adam convergence test
```

## Pull Requests

1. Fork: `https://github.com/Sarkar-AGI/GoTorch`
2. Branch: `git checkout -b feat/add-prelu`
3. Test: `go test ./...` must pass
4. Lint: `go vet ./...` must pass
5. Open PR with description of what was added

## License

By contributing to GoTorch, you agree your contributions will be released
under the BSD 3-Clause License.

GoTorch is an independent project and is not affiliated with Meta Platforms,
Inc., The Linux Foundation, or the upstream open-source project this C++
backend originates from.
