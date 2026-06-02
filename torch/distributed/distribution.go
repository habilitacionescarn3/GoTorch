// Package distributions — Probability distributions. Mirrors: torch.distributions
package distributions

import "github.com/Sarkar-AGI/GoTorch/torch"

// Distribution is the base interface for all probability distributions.
// Mirrors: torch.distributions.Distribution
type Distribution interface {
	Sample() *torch.Tensor
	LogProb(value *torch.Tensor) *torch.Tensor
	Entropy() *torch.Tensor
	Mean() *torch.Tensor
	Variance() *torch.Tensor
}
