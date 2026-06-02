// Package modules — Dropout layers. Mirrors: torch.nn.modules.dropout
package modules

import (
	"fmt"
	"github.com/Sarkar-AGI/GoTorch/torch"
)

type Dropout struct{ P float64; training bool }
func NewDropout(p float64) *Dropout { return &Dropout{p, true} }
func (d *Dropout) Forward(x *torch.Tensor) *torch.Tensor { return nil }
func (d *Dropout) Parameters() []*torch.Tensor           { return nil }
func (d *Dropout) ZeroGrad()                             {}
func (d *Dropout) Train()                                { d.training = true }
func (d *Dropout) Eval()                                 { d.training = false }
func (d *Dropout) Name() string                          { return fmt.Sprintf("Dropout(p=%.2f)", d.P) }

type Dropout2d struct{ P float64; training bool }
func NewDropout2d(p float64) *Dropout2d { return &Dropout2d{p, true} }
func (d *Dropout2d) Forward(x *torch.Tensor) *torch.Tensor { return nil }
func (d *Dropout2d) Parameters() []*torch.Tensor           { return nil }
func (d *Dropout2d) ZeroGrad()                             {}
func (d *Dropout2d) Train()                                { d.training = true }
func (d *Dropout2d) Eval()                                 { d.training = false }
func (d *Dropout2d) Name() string                          { return fmt.Sprintf("Dropout2d(p=%.2f)", d.P) }

type Dropout3d struct{ P float64; training bool }
func NewDropout3d(p float64) *Dropout3d { return &Dropout3d{p, true} }
func (d *Dropout3d) Forward(x *torch.Tensor) *torch.Tensor { return nil }
func (d *Dropout3d) Parameters() []*torch.Tensor           { return nil }
func (d *Dropout3d) ZeroGrad()                             {}
func (d *Dropout3d) Train()                                { d.training = true }
func (d *Dropout3d) Eval()                                 { d.training = false }
func (d *Dropout3d) Name() string                          { return fmt.Sprintf("Dropout3d(p=%.2f)", d.P) }

type AlphaDropout struct{ P float64; training bool }
func NewAlphaDropout(p float64) *AlphaDropout { return &AlphaDropout{p, true} }
func (d *AlphaDropout) Forward(x *torch.Tensor) *torch.Tensor { return nil }
func (d *AlphaDropout) Parameters() []*torch.Tensor           { return nil }
func (d *AlphaDropout) ZeroGrad()                             {}
func (d *AlphaDropout) Train()                                { d.training = true }
func (d *AlphaDropout) Eval()                                 { d.training = false }
func (d *AlphaDropout) Name() string                          { return fmt.Sprintf("AlphaDropout(p=%.2f)", d.P) }
