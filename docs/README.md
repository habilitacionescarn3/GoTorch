# GoTorch Documentation

API documentation is generated from Go source files using `go doc`.

```bash
# View package docs
go doc github.com/Sarkar-AGI/GoTorch/torch
go doc github.com/Sarkar-AGI/GoTorch/torch/nn/modules
go doc github.com/Sarkar-AGI/GoTorch/torch/optim

# Serve docs locally
godoc -http=:6060
# Then visit: http://localhost:6060/pkg/github.com/Sarkar-AGI/GoTorch/
```

## Guides

- [BUILD.md](../BUILD.md) — Build instructions
- [README.md](../README.md) — Quick start and API reference
- [GLOSSARY.md](../GLOSSARY.md) — Term definitions
- [CONTRIBUTING.md](../CONTRIBUTING.md) — How to contribute

## C++ API docs

The underlying libtorch C++ API documentation is at:
https://pytorch.org/cppdocs/
