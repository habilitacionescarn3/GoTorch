// Package distributions — Multivariate distributions.
// Mirrors: torch.distributions.multivariate_normal
package distributions

import "github.com/Sarkar-AGI/GoTorch/torch"

// MultivariateNormal is a multivariate Gaussian distribution.
// Mirrors: distributions.MultivariateNormal(loc, covariance_matrix)
type MultivariateNormal struct {
	Loc               *torch.Tensor
	CovarianceMatrix  *torch.Tensor
}

func NewMultivariateNormal(loc, cov *torch.Tensor) *MultivariateNormal {
	return &MultivariateNormal{loc, cov}
}
func (m *MultivariateNormal) Sample() *torch.Tensor               { return nil }
func (m *MultivariateNormal) LogProb(v *torch.Tensor) *torch.Tensor { return nil }
func (m *MultivariateNormal) Entropy() *torch.Tensor              { return nil }
func (m *MultivariateNormal) Mean() *torch.Tensor                 { return m.Loc }
func (m *MultivariateNormal) Variance() *torch.Tensor             { return nil }
func (m *MultivariateNormal) Rsample() *torch.Tensor              { return nil }
