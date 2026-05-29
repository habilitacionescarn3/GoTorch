// examples/image_classification/main.go
// ResNet-style block using Conv2d + BatchNorm2d + skip connection.
//
// Python equivalent:
//   class ResBlock(nn.Module):
//     def __init__(self, channels):
//       self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
//       self.bn1   = nn.BatchNorm2d(channels)
//       self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
//       self.bn2   = nn.BatchNorm2d(channels)
//     def forward(self, x):
//       residual = x
//       x = F.relu(self.bn1(self.conv1(x)))
//       x = self.bn2(self.conv2(x))
//       return F.relu(x + residual)   # skip connection

package main

import (
	"fmt"
	gt "github.com/Sarkar-AGI/GoTorch"
)

// ResBlock is one residual block. Mirrors the Python class above.
type ResBlock struct {
	conv1 *gt.Conv2d
	bn1   *gt.BatchNorm2d
	conv2 *gt.Conv2d
	bn2   *gt.BatchNorm2d
}

func NewResBlock(channels int) *ResBlock {
	return &ResBlock{
		conv1: gt.NewConv2d(channels, channels, 3, 1, 1, false),
		bn1:   gt.NewBatchNorm2d(channels),
		conv2: gt.NewConv2d(channels, channels, 3, 1, 1, false),
		bn2:   gt.NewBatchNorm2d(channels),
	}
}

func (r *ResBlock) Forward(x *gt.Tensor) *gt.Tensor {
	residual := x
	// Python: x = F.relu(self.bn1(self.conv1(x)))
	x = gt.ReLU(r.bn1.Forward(r.conv1.Forward(x)))
	// Python: x = self.bn2(self.conv2(x))
	x = r.bn2.Forward(r.conv2.Forward(x))
	// Python: return F.relu(x + residual)
	return gt.ReLU(gt.Add(x, residual))
}

func (r *ResBlock) Parameters() []*gt.Tensor {
	var p []*gt.Tensor
	p = append(p, r.conv1.Parameters()...)
	p = append(p, r.bn1.Parameters()...)
	p = append(p, r.conv2.Parameters()...)
	p = append(p, r.bn2.Parameters()...)
	return p
}

func (r *ResBlock) Train() {
	r.conv1.Train(); r.bn1.Train()
	r.conv2.Train(); r.bn2.Train()
}

func (r *ResBlock) Eval() {
	r.conv1.Eval(); r.bn1.Eval()
	r.conv2.Eval(); r.bn2.Eval()
}

// MiniResNet stacks stem + residual blocks + classifier.
// Python:
//   stem      = nn.Sequential(Conv2d(3,64,7,2,3), BN2d(64), ReLU, MaxPool2d(3,2,1))
//   layer1    = nn.Sequential(ResBlock(64), ResBlock(64))
//   avgpool   = nn.AdaptiveAvgPool2d((1,1))
//   fc        = nn.Linear(64, num_classes)
type MiniResNet struct {
	stem   *gt.Sequential
	layer1 []*ResBlock
	avgpool *gt.AdaptiveAvgPool2d
	fc     *gt.Linear
}

func NewMiniResNet(numClasses int) *MiniResNet {
	return &MiniResNet{
		stem: gt.NewSequential(
			gt.NewConv2d(3, 64, 7, 2, 3, false),
			gt.NewBatchNorm2d(64),
			gt.NewReLUModule(),
			gt.NewMaxPool2d(3, 2, 1, 1),
		),
		layer1:  []*ResBlock{NewResBlock(64), NewResBlock(64)},
		avgpool: gt.NewAdaptiveAvgPool2d(1, 1),
		fc:      gt.NewLinear(64, numClasses, true),
	}
}

func (m *MiniResNet) Forward(x *gt.Tensor) *gt.Tensor {
	x = m.stem.Forward(x)
	for _, blk := range m.layer1 {
		x = blk.Forward(x)
	}
	x = m.avgpool.Forward(x)
	x = x.Flatten(1, -1) // (batch, 64)
	return m.fc.Forward(x)
}

func (m *MiniResNet) Parameters() []*gt.Tensor {
	var p []*gt.Tensor
	p = append(p, m.stem.Parameters()...)
	for _, blk := range m.layer1 {
		p = append(p, blk.Parameters()...)
	}
	p = append(p, m.fc.Parameters()...)
	return p
}

func (m *MiniResNet) Train() {
	m.stem.Train()
	for _, blk := range m.layer1 { blk.Train() }
	m.fc.Train()
}

func (m *MiniResNet) Eval() {
	m.stem.Eval()
	for _, blk := range m.layer1 { blk.Eval() }
	m.fc.Eval()
}

func main() {
	fmt.Println("=== Image Classification (MiniResNet) ===")

	model := NewMiniResNet(10)
	model.Train()

	// Python: optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
	opt := gt.NewSGD(model.Parameters(), gt.SGDOptions{
		LR:          0.01,
		Momentum:    0.9,
		WeightDecay: 1e-4,
	})
	// Python: scheduler = CosineAnnealingLR(optimizer, T_max=20)
	sched := gt.NewCosineAnnealingLR(opt, 20, 1e-4)

	// Synthetic ImageNet-like batch: (batch=8, channels=3, H=224, W=224)
	// Python: x = torch.randn(8, 3, 224, 224)
	x := gt.Randn(8, 3, 224, 224)
	y := gt.Zeros(8).Cast(gt.Int64)

	for step := 1; step <= 5; step++ {
		opt.ZeroGrad()
		logits := model.Forward(x)
		loss := gt.CrossEntropyLoss(logits, y, gt.ReduceMean)
		loss.Backward()
		opt.Step()
		sched.Step()
		fmt.Printf("  step %d  loss=%.4f  lr=%.6f\n", step, loss.Item(), opt.GetLR())
	}

	// Inference
	model.Eval()
	gt.WithNoGrad(func() {
		xTest := gt.Randn(4, 3, 224, 224)
		logits := model.Forward(xTest)
		preds := logits.Argmax(1, false)
		fmt.Printf("Predictions: %v\n", preds.Shape())
	})

	fmt.Println("Done.")
}
