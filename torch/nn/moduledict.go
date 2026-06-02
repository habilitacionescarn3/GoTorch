// Package nn — ModuleDict container. Mirrors: torch.nn.ModuleDict
package nn

import "github.com/Sarkar-AGI/GoTorch/torch"

// ModuleDict holds a dictionary of named modules. Mirrors: nn.ModuleDict
type ModuleDict struct{ modules map[string]Module }

func NewModuleDict() *ModuleDict { return &ModuleDict{modules: make(map[string]Module)} }
func (md *ModuleDict) Set(name string, m Module) { md.modules[name] = m }
func (md *ModuleDict) Get(name string) Module    { return md.modules[name] }
func (md *ModuleDict) Keys() []string {
	keys := make([]string, 0, len(md.modules))
	for k := range md.modules { keys = append(keys, k) }
	return keys
}
func (md *ModuleDict) Forward(x *torch.Tensor) *torch.Tensor { return x }
func (md *ModuleDict) Parameters() []*torch.Tensor {
	var p []*torch.Tensor
	for _, m := range md.modules { p = append(p, m.Parameters()...) }
	return p
}
func (md *ModuleDict) ZeroGrad() { for _, m := range md.modules { m.ZeroGrad() } }
func (md *ModuleDict) Train()    { for _, m := range md.modules { m.Train() } }
func (md *ModuleDict) Eval()     { for _, m := range md.modules { m.Eval() } }
func (md *ModuleDict) Name() string { return "ModuleDict" }
