// Package modules — Container modules. Mirrors: torch.nn.modules.container
package modules

import (
	"fmt"
	"github.com/Sarkar-AGI/GoTorch/torch"
)

// Sequential chains modules in order. Mirrors: nn.Sequential
type Sequential struct {
	layers   []interface{ Forward(*torch.Tensor) *torch.Tensor }
	training bool
}
func NewSequential(layers ...interface{ Forward(*torch.Tensor) *torch.Tensor }) *Sequential {
	return &Sequential{layers: layers, training: true}
}
func (s *Sequential) Forward(x *torch.Tensor) *torch.Tensor {
	for _, l := range s.layers { x = l.Forward(x) }
	return x
}
func (s *Sequential) Train()        { s.training = true }
func (s *Sequential) Eval()         { s.training = false }
func (s *Sequential) Len() int      { return len(s.layers) }
func (s *Sequential) Name() string  { return fmt.Sprintf("Sequential(%d layers)", len(s.layers)) }
