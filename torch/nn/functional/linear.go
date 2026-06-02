// Package functional — Linear functions. Mirrors: torch.nn.functional
package functional

import "github.com/Sarkar-AGI/GoTorch/torch"

func Linear(x, weight, bias *torch.Tensor) *torch.Tensor      { return nil }
func Bilinear(x1, x2, weight, bias *torch.Tensor) *torch.Tensor { return nil }
func Embedding(weight, indices *torch.Tensor) *torch.Tensor   { return nil }
func EmbeddingBag(weight, indices, offsets *torch.Tensor, mode string) *torch.Tensor { return nil }
