// Package modules — Distance layers. Mirrors: torch.nn.modules.distance
package modules

import "github.com/Sarkar-AGI/GoTorch/torch"

type CosineSimilarity struct{ Dim int; Eps float64 }
func NewCosineSimilarity(dim int, eps float64) *CosineSimilarity { return &CosineSimilarity{dim, eps} }
func (c *CosineSimilarity) Forward(x *torch.Tensor) *torch.Tensor     { return nil }
func (c *CosineSimilarity) Forward2(x1, x2 *torch.Tensor) *torch.Tensor { return nil }
func (c *CosineSimilarity) Parameters() []*torch.Tensor               { return nil }
func (c *CosineSimilarity) ZeroGrad()                                  {}
func (c *CosineSimilarity) Train()                                     {}
func (c *CosineSimilarity) Eval()                                      {}
func (c *CosineSimilarity) Name() string                               { return "CosineSimilarity" }

type PairwiseDistance struct{ P, Eps float64; Keepdim bool }
func NewPairwiseDistance(p, eps float64) *PairwiseDistance { return &PairwiseDistance{p, eps, false} }
func (pd *PairwiseDistance) Forward(x *torch.Tensor) *torch.Tensor      { return nil }
func (pd *PairwiseDistance) Forward2(x1, x2 *torch.Tensor) *torch.Tensor { return nil }
func (pd *PairwiseDistance) Parameters() []*torch.Tensor                 { return nil }
func (pd *PairwiseDistance) ZeroGrad()                                   {}
func (pd *PairwiseDistance) Train()                                      {}
func (pd *PairwiseDistance) Eval()                                       {}
func (pd *PairwiseDistance) Name() string                                { return "PairwiseDistance" }
