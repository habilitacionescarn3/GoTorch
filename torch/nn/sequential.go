// Package nn — Sequential container. Mirrors: torch.nn.Sequential
package nn

import (
	"fmt"
	"github.com/Sarkar-AGI/GoTorch/torch"
)

// Sequential chains modules in order. Mirrors: nn.Sequential(*layers)
type Sequential struct {
	layers   []Module
	training bool
}

// NewSequential creates a Sequential. Mirrors: nn.Sequential(layer1, layer2, ...)
func NewSequential(layers ...Module) *Sequential {
	return &Sequential{layers: layers, training: true}
}

// Add appends a module to the end.
func (s *Sequential) Add(m Module) { s.layers = append(s.layers, m) }

// Get returns the module at index i.
func (s *Sequential) Get(i int) Module { return s.layers[i] }

// Len returns the number of modules.
func (s *Sequential) Len() int { return len(s.layers) }

// Forward passes input through all layers in order.
func (s *Sequential) Forward(x *torch.Tensor) *torch.Tensor {
	for _, l := range s.layers { x = l.Forward(x) }
	return x
}

func (s *Sequential) Parameters() []*torch.Tensor {
	var p []*torch.Tensor
	for _, l := range s.layers { p = append(p, l.Parameters()...) }
	return p
}

func (s *Sequential) ZeroGrad() { for _, l := range s.layers { l.ZeroGrad() } }
func (s *Sequential) Train()    { s.training = true; for _, l := range s.layers { l.Train() } }
func (s *Sequential) Eval()     { s.training = false; for _, l := range s.layers { l.Eval() } }
func (s *Sequential) Name() string { return fmt.Sprintf("Sequential(%d layers)", len(s.layers)) }
