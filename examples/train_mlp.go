// examples/train_mlp.go — full training loop using GoTorch.
// This is the Go equivalent of a typical GoTorch training script.
//
// ══════════════════════════════════════════════════════════
// Python (original)          →   Go (GoTorch)
// ══════════════════════════════════════════════════════════
// import torch              →   import "github.com/Sarkar-AGI/GoTorch/torch"
// import torch.nn as nn     →   import ".../torch/nn/modules"
// import torch.optim        →   import ".../torch/optim"
// nn.Sequential(...)        →   modules.NewSequential(...)
// nn.Linear(in, out)        →   modules.NewLinear(in, out, true)
// nn.ReLU()                 →   modules.NewReLU()
// nn.Dropout(0.2)           →   modules.NewDropout(0.2)
// optimizer.zero_grad()     →   optimizer.ZeroGrad()
// loss.backward()           →   torch.Backward(loss)
// optimizer.step()          →   optimizer.Step()
// model.train()             →   model.Train()
// model.eval()              →   model.Eval()
// with torch.no_grad():     →   autograd.WithNoGrad(func() { ... })
// ══════════════════════════════════════════════════════════
//
// Build (requires libtorch installed):
//   export LIBTORCH=/path/to/libtorch
//   export CGO_CFLAGS="-I${LIBTORCH}/include -I${LIBTORCH}/include/torch/csrc/api/include"
//   export CGO_LDFLAGS="-L${LIBTORCH}/lib -Wl,-rpath,${LIBTORCH}/lib -ltorch -ltorch_cpu -lc10"
//   go run examples/train_mlp.go

package main

import (
	"fmt"

	"github.com/Sarkar-AGI/GoTorch/torch"
	"github.com/Sarkar-AGI/GoTorch/torch/autograd"
	"github.com/Sarkar-AGI/GoTorch/torch/cuda"
	"github.com/Sarkar-AGI/GoTorch/torch/nn/modules"
	"github.com/Sarkar-AGI/GoTorch/torch/optim"
)

// MLP is a simple multi-layer perceptron. The equivalent in the original Python interface:
//
//	class MLP(nn.Module):
//	    def __init__(self):
//	        super().__init__()
//	        self.net = nn.Sequential(
//	            nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
//	            nn.Linear(256, 128), nn.ReLU(),
//	            nn.Linear(128, 10)
//	        )
//	    def forward(self, x):
//	        return self.net(x)
type MLP struct {
	net *modules.Sequential
}

func NewMLP(inputSize, hiddenSize, numClasses int) *MLP {
	return &MLP{
		net: modules.NewSequential(
			modules.NewLinear(inputSize, hiddenSize, true),
			modules.NewReLU(),
			modules.NewDropout(0.2),
			modules.NewLinear(hiddenSize, hiddenSize/2, true),
			modules.NewReLU(),
			modules.NewLinear(hiddenSize/2, numClasses, true),
		),
	}
}

func (m *MLP) Forward(x torch.RawTensor) torch.RawTensor { return m.net.Forward(x) }
func (m *MLP) Parameters() []torch.RawTensor             { return m.net.Parameters() }
func (m *MLP) Train()                                    { m.net.Train() }
func (m *MLP) Eval()                                     { m.net.Eval() }

func main() {
	fmt.Println("╔══════════════════════════════════════╗")
	fmt.Println("║        GoTorch Training Demo          ║")
	fmt.Println("║  C++ libtorch backend, Go interface   ║")
	fmt.Println("╚══════════════════════════════════════╝")
	fmt.Println()

	// Device selection — mirrors:
	//   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device := torch.CPU
	if cuda.IsAvailable() {
		device = torch.CUDA
		fmt.Printf("CUDA available — using GPU (device count: %d)\n", cuda.DeviceCount())
	} else {
		fmt.Println("CUDA not available — using CPU")
	}
	_ = device

	// Synthetic dataset — mirrors:
	//   X_train = torch.randn(1000, 784)
	//   y_train = torch.randint(0, 10, (1000,))
	fmt.Println("\n[1] Generating synthetic data...")
	XTrain := torch.Randn(200, 64) // (N, features)
	_ = XTrain
	fmt.Println("    X_train shape: [200, 64]")
	fmt.Println("    y_train shape: [200]  (class indices 0-3)")

	// Model — mirrors: model = MLP(784, 256, 10).to(device)
	fmt.Println("\n[2] Building model...")
	model := NewMLP(64, 32, 4)
	fmt.Println("   ", model.net.Name())

	// Optimizer — mirrors: optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	fmt.Println("\n[3] Setting up optimizer...")
	optimizer := optim.NewAdam(model.Parameters(), optim.AdamOptions{LR: 1e-3})
	fmt.Println("   ", optimizer.String())

	// Loss — mirrors: criterion = nn.CrossEntropyLoss()
	criterion := modules.NewCrossEntropyLoss(modules.ReduceMean)
	fmt.Println("   ", criterion.Name())

	// LR Scheduler — mirrors: scheduler = CosineAnnealingLR(optimizer, T_max=50)
	// scheduler := optim.NewCosineAnnealingLR(optimizer, 50, 0)

	// ─── Training loop ───────────────────────────────────────────────────────
	// Mirrors:
	//   for epoch in range(num_epochs):
	//       model.train()
	//       optimizer.zero_grad()
	//       output = model(X)
	//       loss = criterion(output, y)
	//       loss.backward()
	//       optimizer.step()
	fmt.Println("\n[4] Training...")
	numEpochs := 50
	batchSize := 32
	nSamples := 200
	nFeatures := 64
	nClasses := 4

	for epoch := 1; epoch <= numEpochs; epoch++ {
		model.Train()
		var epochLoss float64
		numBatches := 0

		for start := 0; start < nSamples; start += batchSize {
			end := start + batchSize
			if end > nSamples {
				end = nSamples
			}
			bs := end - start

			// Synthetic batch — in real code, use DataLoader
			X := torch.Randn(bs, nFeatures)
			y := torch.Zeros(bs) // dummy targets (Long tensor needed for cross_entropy)

			// Forward pass
			output := model.Forward(X.RawPtr())
			loss := criterion.Forward(output, y.RawPtr())

			// Backward + update — mirrors:
			//   optimizer.zero_grad(); loss.backward(); optimizer.step()
			optimizer.ZeroGrad()
			torch.BackwardRaw(loss)
			optimizer.Step()

			epochLoss += torch.ItemRaw(loss)
			numBatches++
		}

		if epoch%10 == 0 {
			avgLoss := epochLoss / float64(numBatches)
			fmt.Printf("    Epoch [%2d/%d]  Loss: %.4f\n", epoch, numEpochs, avgLoss)
		}
	}

	// ─── Evaluation — mirrors: ────────────────────────────────────────────────
	//   model.eval()
	//   with torch.no_grad():
	//       output = model(X_val)
	fmt.Println("\n[5] Evaluation (torch.no_grad)...")
	model.Eval()
	autograd.WithNoGrad(func() {
		XVal := torch.Randn(20, nFeatures)
		output := model.Forward(XVal.RawPtr())
		fmt.Printf("    Val output shape: [20, %d]\n", nClasses)
		_ = output
	})

	fmt.Println("\n✓ Training complete!")
	fmt.Println()
	fmt.Println("── API Translation Reference ─────────────────────────────────")
	fmt.Println("  Python                              Go")
	fmt.Println("  torch.randn(N, D)                → torch.Randn(N, D)")
	fmt.Println("  torch.zeros(N)                   → torch.Zeros(N)")
	fmt.Println("  nn.Linear(in, out)               → modules.NewLinear(in, out, true)")
	fmt.Println("  nn.ReLU()                        → modules.NewReLU()")
	fmt.Println("  nn.Dropout(p)                    → modules.NewDropout(p)")
	fmt.Println("  nn.Sequential(...)               → modules.NewSequential(...)")
	fmt.Println("  nn.CrossEntropyLoss()            → modules.NewCrossEntropyLoss(ReduceMean)")
	fmt.Println("  torch.optim.Adam(params, lr=...) → optim.NewAdam(params, AdamOptions{LR:...})")
	fmt.Println("  optimizer.zero_grad()            → optimizer.ZeroGrad()")
	fmt.Println("  loss.backward()                  → torch.Backward(loss)")
	fmt.Println("  optimizer.step()                 → optimizer.Step()")
	fmt.Println("  model.train() / model.eval()     → model.Train() / model.Eval()")
	fmt.Println("  with torch.no_grad(): ...        → autograd.WithNoGrad(func() { ... })")
	fmt.Println("  torch.cuda.is_available()        → cuda.IsAvailable()")
	fmt.Println("  torch.save(model, path)          → torch.SaveModule(model, path)")
	fmt.Println("─────────────────────────────────────────────────────────────")
}
