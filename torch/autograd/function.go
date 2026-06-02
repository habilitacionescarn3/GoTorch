// Package autograd — Custom autograd functions. Mirrors: torch.autograd.Function
package autograd

import "github.com/Sarkar-AGI/GoTorch/torch"

// Function is the interface for custom autograd functions.
// Mirrors: torch.autograd.Function
type Function interface {
	Forward(inputs ...*torch.Tensor) *torch.Tensor
	Backward(gradOutput *torch.Tensor) []*torch.Tensor
}

// Context stores information for backward pass.
// Mirrors: ctx in autograd Function
type Context struct {
	SavedTensors []*torch.Tensor
	NeedsInput   []bool
}

// SaveForBackward saves tensors for backward pass.
func (c *Context) SaveForBackward(tensors ...*torch.Tensor) {
	c.SavedTensors = append(c.SavedTensors, tensors...)
}

// GetSavedTensors retrieves saved tensors.
func (c *Context) GetSavedTensors() []*torch.Tensor { return c.SavedTensors }
