// Package autograd — Automatic differentiation. Mirrors: torch.autograd
package autograd

import "github.com/Sarkar-AGI/GoTorch/torch"

// Backward computes gradients for a tensor. Mirrors: tensor.backward()
func Backward(t *torch.Tensor) { t.Backward() }

// BackwardWithGrad computes gradients with external grad.
func BackwardWithGrad(t, grad *torch.Tensor) { t.BackwardWithGrad(grad) }

// Grad returns gradient of a tensor. Mirrors: tensor.grad
func Grad(t *torch.Tensor) *torch.Tensor { return t.Grad() }

// ZeroGrad zeroes gradient of a tensor.
func ZeroGrad(t *torch.Tensor) { t.ZeroGrad() }

// ZeroAllGrads zeroes gradients of all tensors.
func ZeroAllGrads(params []*torch.Tensor) {
	for _, p := range params { p.ZeroGrad() }
}
