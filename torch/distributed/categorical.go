// Package distributions — Categorical distribution. Mirrors: torch.distributions.Categorical
package distributions

import "github.com/Sarkar-AGI/GoTorch/torch"

// Categorical samples integer class indices. Mirrors: distributions.Categorical(probs)
type Categorical struct{ Probs *torch.Tensor }

func NewCategorical(probs *torch.Tensor) *Categorical     { return &Categorical{probs} }
func (c *Categorical) Sample() *torch.Tensor              { return nil }
func (c *Categorical) LogProb(v *torch.Tensor) *torch.Tensor { return nil }
func (c *Categorical) Entropy() *torch.Tensor             { return nil }
func (c *Categorical) Mean() *torch.Tensor                { return nil }
func (c *Categorical) Variance() *torch.Tensor            { return nil }
