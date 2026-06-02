// Package functional — All functional operations. Mirrors: torch.nn.functional
// This file re-exports common ops and documents the package.
package functional

// Package functional provides stateless neural network operations.
// All functions in this package are pure functions with no trainable parameters.
// They mirror torch.nn.functional (F) in Python PyTorch.
//
// Usage:
//   import F "github.com/Sarkar-AGI/GoTorch/torch/nn/functional"
//
//   out := F.ReLU(x)
//   out := F.Conv2d(x, weight, bias, 1, 1, 1, 1)
//   loss := F.CrossEntropy(logits, targets, F.ReduceMean)
