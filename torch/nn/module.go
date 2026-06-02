// Package nn — Module interface. Mirrors: torch.nn.Module
package nn

import "github.com/Sarkar-AGI/GoTorch/torch"

// Module is the base interface for all neural network layers.
type Module interface {
	Forward(x *torch.Tensor) *torch.Tensor
	Parameters() []*torch.Tensor
	ZeroGrad()
	Train()
	Eval()
	Name() string
}

// BaseModule provides default no-op implementations.
type BaseModule struct{ training bool }

func (b *BaseModule) Parameters() []*torch.Tensor { return nil }
func (b *BaseModule) ZeroGrad()                   {}
func (b *BaseModule) Train()                      { b.training = true }
func (b *BaseModule) Eval()                       { b.training = false }
func (b *BaseModule) IsTraining() bool            { return b.training }
func (b *BaseModule) Name() string                { return "Module" }
