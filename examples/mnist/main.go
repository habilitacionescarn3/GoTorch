// examples/mnist/main.go — MNIST classifier using the single-import GoTorch API.
//
// Python equivalent:
//   import torch
//   import torch.nn as nn
//   import torch.optim as optim
//
// Go equivalent:
//   import gt "github.com/Sarkar-AGI/GoTorch"
//
// Build & run:
//   go run ./examples/mnist/

package main

import (
	"fmt"
	gt "github.com/Sarkar-AGI/GoTorch"
)

// ─── MLP model ────────────────────────────────────────────────────────────────

// MLP mirrors:
//   nn.Sequential(
//     nn.Linear(784, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
//     nn.Linear(256, 128), nn.ReLU(),
//     nn.Linear(128, 10),
//   )
type MLP struct{ net *gt.Sequential }

func NewMLP() *MLP {
	return &MLP{net: gt.NewSequential(
		gt.NewLinear(784, 256, true),
		gt.NewBatchNorm1d(256),
		gt.NewReLUModule(),
		gt.NewDropout(0.3),
		gt.NewLinear(256, 128, true),
		gt.NewReLUModule(),
		gt.NewLinear(128, 10, true),
	)}
}

func (m *MLP) Forward(x *gt.Tensor) *gt.Tensor { return m.net.Forward(x) }
func (m *MLP) Parameters() []*gt.Tensor        { return m.net.Parameters() }
func (m *MLP) Train()                          { m.net.Train() }
func (m *MLP) Eval()                           { m.net.Eval() }

// ─── CNN model ────────────────────────────────────────────────────────────────

// CNN mirrors:
//   features    = nn.Sequential(Conv2d, BN2d, ReLU, MaxPool2d, Conv2d, ReLU, AdaptiveAvgPool2d)
//   classifier  = nn.Sequential(Flatten, Linear, ReLU, Linear)
type CNN struct {
	features   *gt.Sequential
	classifier *gt.Sequential
}

func NewCNN() *CNN {
	return &CNN{
		features: gt.NewSequential(
			gt.NewConv2d(1, 32, 3, 1, 1, true),
			gt.NewBatchNorm2d(32),
			gt.NewReLUModule(),
			gt.NewMaxPool2d(2, 2, 0, 1),
			gt.NewConv2d(32, 64, 3, 1, 1, true),
			gt.NewReLUModule(),
			gt.NewAdaptiveAvgPool2d(4, 4),
		),
		classifier: gt.NewSequential(
			gt.NewFlatten(1, -1),
			gt.NewLinear(64*4*4, 128, true),
			gt.NewReLUModule(),
			gt.NewLinear(128, 10, true),
		),
	}
}

func (c *CNN) Forward(x *gt.Tensor) *gt.Tensor {
	return c.classifier.Forward(c.features.Forward(x))
}
func (c *CNN) Parameters() []*gt.Tensor {
	return append(c.features.Parameters(), c.classifier.Parameters()...)
}
func (c *CNN) Train() { c.features.Train(); c.classifier.Train() }
func (c *CNN) Eval()  { c.features.Eval(); c.classifier.Eval() }

// ─── TensorDataset ────────────────────────────────────────────────────────────

// TensorDataset wraps two tensors. Mirrors: torch.utils.data.TensorDataset
type TensorDataset struct {
	inputs  *gt.Tensor
	targets *gt.Tensor
	n       int
}

func (td *TensorDataset) Len() int { return td.n }
func (td *TensorDataset) GetItem(i int) (*gt.Tensor, *gt.Tensor) {
	x := td.inputs.Slice(0, int64(i), int64(i+1), 1).SqueezeDim(0)
	y := td.targets.Slice(0, int64(i), int64(i+1), 1).SqueezeDim(0)
	return x, y
}

// ─── Training loop ────────────────────────────────────────────────────────────

func train(
	forward func(*gt.Tensor) *gt.Tensor,
	params func() []*gt.Tensor,
	opt gt.Optimizer,
	loader *gt.DataLoader,
	epochs int,
) {
	sched := gt.NewStepLR(opt, 5, 0.5)
	for epoch := 1; epoch <= epochs; epoch++ {
		var totalLoss float64
		batches := 0
		loader.Reset()
		for loader.HasNext() {
			batch := loader.Next()
			opt.ZeroGrad()
			logits := forward(batch.Inputs)
			loss := gt.CrossEntropyLoss(logits, batch.Targets, gt.ReduceMean)
			loss.Backward()
			opt.Step()
			totalLoss += loss.Item()
			batches++
		}
		sched.Step()
		if batches > 0 {
			fmt.Printf("  epoch %2d/%d  loss=%.4f  lr=%.6f\n",
				epoch, epochs, totalLoss/float64(batches), opt.GetLR())
		}
	}
}

// ─── Main ─────────────────────────────────────────────────────────────────────

func main() {
	fmt.Println(gt.Version())
	fmt.Printf("CUDA: %v  (GPUs: %d)\n\n", gt.CUDAIsAvailable(), gt.CUDADeviceCount())

	// ── Synthetic data (replace with real MNIST loader) ──────────────────────
	// Python: x = torch.randn(512, 784); y = torch.zeros(512, dtype=torch.long)
	N := 512
	xTrain := gt.Randn(N, 784)
	yTrain := gt.Zeros(N).Cast(gt.Int64)
	ds := &TensorDataset{inputs: xTrain, targets: yTrain, n: N}
	loader := gt.NewDataLoader(ds, 64, true)

	// ── MLP ──────────────────────────────────────────────────────────────────
	fmt.Println("=== MLP ===")
	mlp := NewMLP()
	mlp.Train()
	// Python: opt = optim.Adam(mlp.parameters(), lr=1e-3)
	optMLP := gt.NewAdam(mlp.Parameters(), gt.AdamOptions{LR: 1e-3})
	train(mlp.Forward, mlp.Parameters, optMLP, loader, 8)

	// Inference
	mlp.Eval()
	gt.WithNoGrad(func() {
		// Python: with torch.no_grad(): preds = mlp(x_test).argmax(dim=1)
		xTest := gt.Randn(16, 784)
		preds := mlp.Forward(xTest).Argmax(1, false)
		fmt.Printf("MLP predictions shape: %v\n\n", preds.Shape())
	})

	// ── CNN ──────────────────────────────────────────────────────────────────
	fmt.Println("=== CNN ===")
	cnn := NewCNN()
	cnn.Train()
	// Python: opt = optim.AdamW(cnn.parameters(), lr=5e-4, weight_decay=1e-2)
	optCNN := gt.NewAdamW(cnn.Parameters(), gt.AdamOptions{LR: 5e-4, WeightDecay: 1e-2})

	// Single forward/backward to demonstrate CNN
	// Python: x = torch.randn(8, 1, 28, 28)
	imgs := gt.Randn(8, 1, 28, 28)
	logits := cnn.Forward(imgs)
	targets := gt.Zeros(8).Cast(gt.Int64)
	loss := gt.CrossEntropyLoss(logits, targets, gt.ReduceMean)
	optCNN.ZeroGrad()
	loss.Backward()
	optCNN.Step()
	fmt.Printf("CNN loss=%.4f  output shape=%v\n\n", loss.Item(), logits.Shape())

	// ── LSTM text classifier ──────────────────────────────────────────────────
	fmt.Println("=== LSTM ===")
	// Python: emb  = nn.Embedding(10000, 64)
	//         lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.1)
	//         fc   = nn.Linear(128, 2)
	emb  := gt.NewEmbedding(10000, 64)
	lstm := gt.NewLSTM(64, 128, 2, true, true, 0.1, false)
	fc   := gt.NewLinear(128, 2, true)
	lstm.Train()

	// Python: tokens = torch.randint(0, 10000, (4, 20))
	tokens := gt.Zeros(4, 20).Cast(gt.Int64)
	embedded := emb.Forward(tokens)              // (4, 20, 64)
	lstmOut := lstm.ForwardLSTM(embedded, nil, nil)
	// Take last hidden state: h_n[-1]  → (4, 128)
	lastHidden := lstmOut.Hn.Slice(0, 0, 1, 1).SqueezeDim(0)
	lstmLogits := fc.Forward(lastHidden)
	fmt.Printf("LSTM logits shape: %v\n\n", lstmLogits.Shape())

	// ── Transformer encoder ───────────────────────────────────────────────────
	fmt.Println("=== Transformer Encoder ===")
	// Python: enc_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, dropout=0.1)
	//         encoder   = nn.TransformerEncoder(enc_layer, num_layers=4)
	encLayer := gt.NewTransformerEncoderLayer(128, 8, 512, 0.1)
	encoder  := gt.NewTransformerEncoder(encLayer, 4)
	encoder.Train()

	// Python: src = torch.randn(16, 4, 128)  # (seq_len, batch, d_model)
	src     := gt.Randn(16, 4, 128)
	encoded := encoder.Forward(src)
	fmt.Printf("Transformer output shape: %v\n\n", encoded.Shape())

	// ── ReduceLROnPlateau example ─────────────────────────────────────────────
	fmt.Println("=== ReduceLROnPlateau ===")
	optRL := gt.NewAdam(mlp.Parameters(), gt.AdamOptions{LR: 1e-2})
	sched := gt.NewReduceLROnPlateau(optRL, "min", 0.5, 2, 1e-5)
	valLosses := []float64{1.0, 0.95, 0.94, 0.94, 0.94, 0.93}
	for i, vl := range valLosses {
		sched.StepWithMetric(vl)
		fmt.Printf("  step %d  val_loss=%.3f  lr=%.6f\n", i+1, vl, optRL.GetLR())
	}

	fmt.Println("\nAll examples completed successfully.")
}
