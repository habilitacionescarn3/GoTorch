// Package modules — Upsampling layers. Mirrors: torch.nn.modules.upsampling
package modules

import (
	"fmt"
	"github.com/Sarkar-AGI/GoTorch/torch"
)

type Upsample struct {
	ScaleFactor float64
	Mode        string
	OutH, OutW  int
}
func NewUpsample(scaleFactor float64, mode string) *Upsample { return &Upsample{ScaleFactor: scaleFactor, Mode: mode} }
func NewUpsampleSize(outH, outW int, mode string) *Upsample  { return &Upsample{OutH: outH, OutW: outW, Mode: mode} }
func (u *Upsample) Forward(x *torch.Tensor) *torch.Tensor    { return nil }
func (u *Upsample) Parameters() []*torch.Tensor              { return nil }
func (u *Upsample) ZeroGrad()                                {}
func (u *Upsample) Train()                                   {}
func (u *Upsample) Eval()                                    {}
func (u *Upsample) Name() string                             { return fmt.Sprintf("Upsample(mode=%s)", u.Mode) }
