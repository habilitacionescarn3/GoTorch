// Package distributions — Uniform distribution. Mirrors: torch.distributions.Uniform
package distributions

import "github.com/Sarkar-AGI/GoTorch/torch"

// Uniform samples uniformly between low and high. Mirrors: distributions.Uniform(low, high)
type Uniform struct{ Low, High *torch.Tensor }

func NewUniform(low, high *torch.Tensor) *Uniform         { return &Uniform{low, high} }
func (u *Uniform) Sample() *torch.Tensor                  { return nil }
func (u *Uniform) LogProb(v *torch.Tensor) *torch.Tensor  { return nil }
func (u *Uniform) Entropy() *torch.Tensor                 { return nil }
func (u *Uniform) Mean() *torch.Tensor                    { return nil }
func (u *Uniform) Variance() *torch.Tensor                { return nil }
