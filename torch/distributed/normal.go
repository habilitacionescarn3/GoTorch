// Package distributions — Normal distribution. Mirrors: torch.distributions.Normal
package distributions

import "github.com/Sarkar-AGI/GoTorch/torch"

// Normal is a Gaussian distribution. Mirrors: distributions.Normal(loc, scale)
type Normal struct{ Loc, Scale *torch.Tensor }

func NewNormal(loc, scale *torch.Tensor) *Normal      { return &Normal{loc, scale} }
func (n *Normal) Sample() *torch.Tensor               { return nil }
func (n *Normal) LogProb(v *torch.Tensor) *torch.Tensor { return nil }
func (n *Normal) Entropy() *torch.Tensor              { return nil }
func (n *Normal) Mean() *torch.Tensor                 { return n.Loc }
func (n *Normal) Variance() *torch.Tensor             { return nil }
func (n *Normal) Rsample() *torch.Tensor              { return nil } // reparameterized sample
