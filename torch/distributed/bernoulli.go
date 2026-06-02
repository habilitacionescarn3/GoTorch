// Package distributions — Bernoulli distribution. Mirrors: torch.distributions.Bernoulli
package distributions

import "github.com/Sarkar-AGI/GoTorch/torch"

// Bernoulli samples binary values. Mirrors: distributions.Bernoulli(probs)
type Bernoulli struct{ Probs *torch.Tensor }

func NewBernoulli(probs *torch.Tensor) *Bernoulli       { return &Bernoulli{probs} }
func (b *Bernoulli) Sample() *torch.Tensor              { return nil }
func (b *Bernoulli) LogProb(v *torch.Tensor) *torch.Tensor { return nil }
func (b *Bernoulli) Entropy() *torch.Tensor             { return nil }
func (b *Bernoulli) Mean() *torch.Tensor                { return b.Probs }
func (b *Bernoulli) Variance() *torch.Tensor            { return nil }
