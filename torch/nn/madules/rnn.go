// Package modules — Recurrent layers. Mirrors: torch.nn.modules.rnn
package modules

import "github.com/Sarkar-AGI/GoTorch/torch"

type LSTMOutput struct{ Output, Hn, Cn *torch.Tensor }
type GRUOutput  struct{ Output, Hn *torch.Tensor }

type LSTM struct {
	InputSize, HiddenSize, NumLayers int
	Bias, BatchFirst, Bidirectional  bool
	Dropout  float64
	training bool
}
func NewLSTM(inputSize, hiddenSize, numLayers int, bias, batchFirst bool, dropout float64, bidirectional bool) *LSTM {
	return &LSTM{inputSize, hiddenSize, numLayers, bias, batchFirst, bidirectional, dropout, true}
}
func (l *LSTM) ForwardLSTM(x, h0, c0 *torch.Tensor) LSTMOutput { return LSTMOutput{} }
func (l *LSTM) Forward(x *torch.Tensor) *torch.Tensor           { return l.ForwardLSTM(x, nil, nil).Output }
func (l *LSTM) Parameters() []*torch.Tensor                     { return nil }
func (l *LSTM) ZeroGrad()                                       {}
func (l *LSTM) Train()                                          { l.training = true }
func (l *LSTM) Eval()                                           { l.training = false }
func (l *LSTM) Name() string                                    { return "LSTM" }

type GRU struct {
	InputSize, HiddenSize, NumLayers int
	Bias, BatchFirst, Bidirectional  bool
	Dropout  float64
	training bool
}
func NewGRU(inputSize, hiddenSize, numLayers int, bias, batchFirst bool, dropout float64, bidirectional bool) *GRU {
	return &GRU{inputSize, hiddenSize, numLayers, bias, batchFirst, bidirectional, dropout, true}
}
func (g *GRU) ForwardGRU(x, h0 *torch.Tensor) GRUOutput { return GRUOutput{} }
func (g *GRU) Forward(x *torch.Tensor) *torch.Tensor     { return g.ForwardGRU(x, nil).Output }
func (g *GRU) Parameters() []*torch.Tensor               { return nil }
func (g *GRU) ZeroGrad()                                 {}
func (g *GRU) Train()                                    { g.training = true }
func (g *GRU) Eval()                                     { g.training = false }
func (g *GRU) Name() string                              { return "GRU" }

type RNN struct {
	InputSize, HiddenSize, NumLayers int
	NonLinearity string
	training     bool
}
func NewRNN(inputSize, hiddenSize, numLayers int, nonlinearity string) *RNN {
	return &RNN{inputSize, hiddenSize, numLayers, nonlinearity, true}
}
func (r *RNN) Forward(x *torch.Tensor) *torch.Tensor { return nil }
func (r *RNN) Parameters() []*torch.Tensor           { return nil }
func (r *RNN) ZeroGrad()                             {}
func (r *RNN) Train()                                { r.training = true }
func (r *RNN) Eval()                                 { r.training = false }
func (r *RNN) Name() string                          { return "RNN" }
