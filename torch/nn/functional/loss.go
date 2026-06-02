// Package functional — Loss functions. Mirrors: torch.nn.functional
package functional

import "github.com/Sarkar-AGI/GoTorch/torch"

const (
	ReduceNone = 0
	ReduceMean = 1
	ReduceSum  = 2
)

func MSELoss(pred, target *torch.Tensor, reduction int) *torch.Tensor            { return nil }
func CrossEntropy(pred, target *torch.Tensor, reduction int) *torch.Tensor       { return nil }
func BCELoss(pred, target *torch.Tensor, reduction int) *torch.Tensor            { return nil }
func BCEWithLogits(pred, target *torch.Tensor, reduction int) *torch.Tensor      { return nil }
func NLLLoss(logProbs, target *torch.Tensor, reduction int) *torch.Tensor        { return nil }
func L1Loss(pred, target *torch.Tensor, reduction int) *torch.Tensor             { return nil }
func HuberLoss(pred, target *torch.Tensor, delta float64, reduction int) *torch.Tensor { return nil }
func KLDiv(pred, target *torch.Tensor, reduction int) *torch.Tensor              { return nil }
func SmoothL1Loss(pred, target *torch.Tensor, reduction int) *torch.Tensor       { return nil }
func MarginRankingLoss(x1, x2, y *torch.Tensor, margin float64) *torch.Tensor   { return nil }
func TripletMarginLoss(a, p, n *torch.Tensor, margin float64) *torch.Tensor      { return nil }
