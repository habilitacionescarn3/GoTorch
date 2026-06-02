// Package modules — Embedding layers. Mirrors: torch.nn.modules.sparse
package modules

import (
	"fmt"
	"github.com/Sarkar-AGI/GoTorch/torch"
)

type Embedding struct {
	NumEmbeddings, EmbeddingDim, PaddingIdx int
	Sparse   bool
	training bool
}
func NewEmbedding(num, dim int) *Embedding { return &Embedding{NumEmbeddings: num, EmbeddingDim: dim, PaddingIdx: -1, training: true} }
func NewEmbeddingFull(num, dim, padIdx int, sparse bool) *Embedding {
	return &Embedding{num, dim, padIdx, sparse, true}
}
func (e *Embedding) Forward(x *torch.Tensor) *torch.Tensor { return nil }
func (e *Embedding) Parameters() []*torch.Tensor           { return nil }
func (e *Embedding) ZeroGrad()                             {}
func (e *Embedding) Train()                                { e.training = true }
func (e *Embedding) Eval()                                 { e.training = false }
func (e *Embedding) Name() string                          { return fmt.Sprintf("Embedding(%d, %d)", e.NumEmbeddings, e.EmbeddingDim) }

type EmbeddingBag struct {
	NumEmbeddings, EmbeddingDim int
	Mode     string
	training bool
}
func NewEmbeddingBag(num, dim int, mode string) *EmbeddingBag { return &EmbeddingBag{num, dim, mode, true} }
func (e *EmbeddingBag) Forward(x *torch.Tensor) *torch.Tensor         { return nil }
func (e *EmbeddingBag) ForwardBag(input, offsets *torch.Tensor) *torch.Tensor { return nil }
func (e *EmbeddingBag) Parameters() []*torch.Tensor                    { return nil }
func (e *EmbeddingBag) ZeroGrad()                                      {}
func (e *EmbeddingBag) Train()                                         { e.training = true }
func (e *EmbeddingBag) Eval()                                          { e.training = false }
func (e *EmbeddingBag) Name() string                                   { return fmt.Sprintf("EmbeddingBag(%d, %d, mode=%s)", e.NumEmbeddings, e.EmbeddingDim, e.Mode) }
