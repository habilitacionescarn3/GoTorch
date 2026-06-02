// Package distributions — Exponential distribution.
// Mirrors: torch.distributions.Exponential
package distributions

import "github.com/Sarkar-AGI/GoTorch/torch"

// Exponential samples from an exponential distribution.
// Mirrors: distributions.Exponential(rate)
type Exponential struct{ Rate *torch.Tensor }

func NewExponential(rate *torch.Tensor) *Exponential          { return &Exponential{rate} }
func (e *Exponential) Sample() *torch.Tensor                  { return nil }
func (e *Exponential) LogProb(v *torch.Tensor) *torch.Tensor  { return nil }
func (e *Exponential) Entropy() *torch.Tensor                 { return nil }
func (e *Exponential) Mean() *torch.Tensor                    { return nil }
func (e *Exponential) Variance() *torch.Tensor                { return nil }
