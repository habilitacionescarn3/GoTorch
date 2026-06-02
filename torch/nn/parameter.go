// Package nn — Parameter type. Mirrors: torch.nn.Parameter
package nn

import "github.com/Sarkar-AGI/GoTorch/torch"

// Parameter is a tensor that is a module parameter.
// Mirrors: nn.Parameter(data, requires_grad=True)
type Parameter struct {
	Tensor       *torch.Tensor
	RequiresGrad bool
	Name         string
}

// NewParameter creates a Parameter.
func NewParameter(data *torch.Tensor, requiresGrad bool) *Parameter {
	if data != nil { data.SetRequiresGrad(requiresGrad) }
	return &Parameter{Tensor: data, RequiresGrad: requiresGrad}
}

// Data returns the underlying tensor.
func (p *Parameter) Data() *torch.Tensor { return p.Tensor }

// Grad returns the gradient.
func (p *Parameter) Grad() *torch.Tensor {
	if p.Tensor == nil { return nil }
	return p.Tensor.Grad()
}

// ZeroGrad zeroes the gradient.
func (p *Parameter) ZeroGrad() {
	if p.Tensor != nil { p.Tensor.ZeroGrad() }
}
