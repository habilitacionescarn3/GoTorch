// Package nn — ModuleList container. Mirrors: torch.nn.ModuleList
package nn

import "github.com/Sarkar-AGI/GoTorch/torch"

// ModuleList holds an ordered list of modules. Mirrors: nn.ModuleList
type ModuleList struct{ modules []Module }

func NewModuleList(modules ...Module) *ModuleList { return &ModuleList{modules: modules} }
func (ml *ModuleList) Get(i int) Module           { return ml.modules[i] }
func (ml *ModuleList) Append(m Module)            { ml.modules = append(ml.modules, m) }
func (ml *ModuleList) Len() int                   { return len(ml.modules) }
func (ml *ModuleList) Forward(x *torch.Tensor) *torch.Tensor { return x }
func (ml *ModuleList) Parameters() []*torch.Tensor {
	var p []*torch.Tensor
	for _, m := range ml.modules { p = append(p, m.Parameters()...) }
	return p
}
func (ml *ModuleList) ZeroGrad() { for _, m := range ml.modules { m.ZeroGrad() } }
func (ml *ModuleList) Train()    { for _, m := range ml.modules { m.Train() } }
func (ml *ModuleList) Eval()     { for _, m := range ml.modules { m.Eval() } }
func (ml *ModuleList) Name() string { return "ModuleList" }
