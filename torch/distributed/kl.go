// Package distributions — KL divergence. Mirrors: torch.distributions.kl
package distributions

import "github.com/Sarkar-AGI/GoTorch/torch"

// KLDivergence computes KL(p || q). Mirrors: torch.distributions.kl_divergence
func KLDivergence(p, q Distribution) *torch.Tensor { return nil }

// RegisterKL registers a KL divergence implementation for two distribution types.
func RegisterKL(pType, qType string, fn func(p, q Distribution) *torch.Tensor) {}
