// Package modules — Loss functions. Mirrors: torch.nn.modules.loss
package modules

import "github.com/Sarkar-AGI/GoTorch/torch"

type MSELoss struct{ Reduction int }
func NewMSELoss(reduction int) *MSELoss { return &MSELoss{reduction} }
func (m *MSELoss) Forward(pred, target *torch.Tensor) *torch.Tensor { return nil }
func (m *MSELoss) ForwardSingle(x *torch.Tensor) *torch.Tensor      { return nil }
func (m *MSELoss) Parameters() []*torch.Tensor                      { return nil }
func (m *MSELoss) ZeroGrad()                                        {}
func (m *MSELoss) Train()                                           {}
func (m *MSELoss) Eval()                                            {}
func (m *MSELoss) Name() string                                     { return "MSELoss" }

type CrossEntropyLoss struct{ Reduction int }
func NewCrossEntropyLoss(reduction int) *CrossEntropyLoss { return &CrossEntropyLoss{reduction} }
func (c *CrossEntropyLoss) Forward(pred, target *torch.Tensor) *torch.Tensor { return nil }
func (c *CrossEntropyLoss) ForwardSingle(x *torch.Tensor) *torch.Tensor      { return nil }
func (c *CrossEntropyLoss) Parameters() []*torch.Tensor                      { return nil }
func (c *CrossEntropyLoss) ZeroGrad()                                        {}
func (c *CrossEntropyLoss) Train()                                           {}
func (c *CrossEntropyLoss) Eval()                                            {}
func (c *CrossEntropyLoss) Name() string                                     { return "CrossEntropyLoss" }

type BCELoss struct{ Reduction int }
func NewBCELoss(reduction int) *BCELoss { return &BCELoss{reduction} }
func (b *BCELoss) Forward(pred, target *torch.Tensor) *torch.Tensor { return nil }
func (b *BCELoss) ForwardSingle(x *torch.Tensor) *torch.Tensor      { return nil }
func (b *BCELoss) Parameters() []*torch.Tensor                      { return nil }
func (b *BCELoss) ZeroGrad()                                        {}
func (b *BCELoss) Train()                                           {}
func (b *BCELoss) Eval()                                            {}
func (b *BCELoss) Name() string                                     { return "BCELoss" }

type BCEWithLogitsLoss struct{ Reduction int }
func NewBCEWithLogitsLoss(reduction int) *BCEWithLogitsLoss { return &BCEWithLogitsLoss{reduction} }
func (b *BCEWithLogitsLoss) Forward(pred, target *torch.Tensor) *torch.Tensor { return nil }
func (b *BCEWithLogitsLoss) ForwardSingle(x *torch.Tensor) *torch.Tensor      { return nil }
func (b *BCEWithLogitsLoss) Parameters() []*torch.Tensor                      { return nil }
func (b *BCEWithLogitsLoss) ZeroGrad()                                        {}
func (b *BCEWithLogitsLoss) Train()                                           {}
func (b *BCEWithLogitsLoss) Eval()                                            {}
func (b *BCEWithLogitsLoss) Name() string                                     { return "BCEWithLogitsLoss" }

type NLLLoss struct{ Reduction int }
func NewNLLLoss(reduction int) *NLLLoss { return &NLLLoss{reduction} }
func (n *NLLLoss) Forward(pred, target *torch.Tensor) *torch.Tensor { return nil }
func (n *NLLLoss) ForwardSingle(x *torch.Tensor) *torch.Tensor      { return nil }
func (n *NLLLoss) Parameters() []*torch.Tensor                      { return nil }
func (n *NLLLoss) ZeroGrad()                                        {}
func (n *NLLLoss) Train()                                           {}
func (n *NLLLoss) Eval()                                            {}
func (n *NLLLoss) Name() string                                     { return "NLLLoss" }

type L1Loss struct{ Reduction int }
func NewL1Loss(reduction int) *L1Loss { return &L1Loss{reduction} }
func (l *L1Loss) Forward(pred, target *torch.Tensor) *torch.Tensor { return nil }
func (l *L1Loss) ForwardSingle(x *torch.Tensor) *torch.Tensor      { return nil }
func (l *L1Loss) Parameters() []*torch.Tensor                      { return nil }
func (l *L1Loss) ZeroGrad()                                        {}
func (l *L1Loss) Train()                                           {}
func (l *L1Loss) Eval()                                            {}
func (l *L1Loss) Name() string                                     { return "L1Loss" }

type HuberLoss struct{ Delta float64; Reduction int }
func NewHuberLoss(delta float64, reduction int) *HuberLoss { return &HuberLoss{delta, reduction} }
func (h *HuberLoss) Forward(pred, target *torch.Tensor) *torch.Tensor { return nil }
func (h *HuberLoss) ForwardSingle(x *torch.Tensor) *torch.Tensor      { return nil }
func (h *HuberLoss) Parameters() []*torch.Tensor                      { return nil }
func (h *HuberLoss) ZeroGrad()                                        {}
func (h *HuberLoss) Train()                                           {}
func (h *HuberLoss) Eval()                                            {}
func (h *HuberLoss) Name() string                                     { return "HuberLoss" }

type KLDivLoss struct{ Reduction int }
func NewKLDivLoss(reduction int) *KLDivLoss { return &KLDivLoss{reduction} }
func (k *KLDivLoss) Forward(pred, target *torch.Tensor) *torch.Tensor { return nil }
func (k *KLDivLoss) ForwardSingle(x *torch.Tensor) *torch.Tensor      { return nil }
func (k *KLDivLoss) Parameters() []*torch.Tensor                      { return nil }
func (k *KLDivLoss) ZeroGrad()                                        {}
func (k *KLDivLoss) Train()                                           {}
func (k *KLDivLoss) Eval()                                            {}
func (k *KLDivLoss) Name() string                                     { return "KLDivLoss" }
