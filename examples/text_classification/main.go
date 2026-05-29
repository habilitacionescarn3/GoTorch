// examples/text_classification/main.go
// Sentiment classifier: Embedding → BiLSTM → Linear
//
// Python equivalent:
//   class SentimentModel(nn.Module):
//     def __init__(self, vocab_size, emb_dim, hidden, num_classes):
//       self.emb  = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
//       self.lstm = nn.LSTM(emb_dim, hidden, num_layers=2, batch_first=True,
//                           bidirectional=True, dropout=0.2)
//       self.fc   = nn.Linear(hidden * 2, num_classes)
//       self.drop = nn.Dropout(0.3)
//
//     def forward(self, x):
//       e = self.drop(self.emb(x))               # (B, T, E)
//       _, (h, _) = self.lstm(e)                 # h: (4, B, H)
//       h = torch.cat([h[-2], h[-1]], dim=1)      # (B, 2H) — last bidirectional state
//       return self.fc(self.drop(h))

package main

import (
	"fmt"
	gt "github.com/Sarkar-AGI/GoTorch"
)

const (
	vocabSize  = 20000
	embDim     = 128
	hiddenSize = 256
	numClasses = 2
	seqLen     = 50
	batchSize  = 32
)

type SentimentModel struct {
	emb  *gt.Embedding
	lstm *gt.LSTM
	fc   *gt.Linear
	drop *gt.Dropout
}

func NewSentimentModel() *SentimentModel {
	return &SentimentModel{
		// Python: nn.Embedding(vocabSize, embDim, padding_idx=0)
		emb:  gt.NewEmbeddingFull(vocabSize, embDim, 0, false),
		// Python: nn.LSTM(embDim, hiddenSize, num_layers=2, batch_first=True,
		//                 bidirectional=True, dropout=0.2)
		lstm: gt.NewLSTM(embDim, hiddenSize, 2, true, true, 0.2, true),
		// Python: nn.Linear(hiddenSize*2, numClasses)
		fc:   gt.NewLinear(hiddenSize*2, numClasses, true),
		// Python: nn.Dropout(0.3)
		drop: gt.NewDropout(0.3),
	}
}

func (m *SentimentModel) Forward(tokens *gt.Tensor) *gt.Tensor {
	// embedded: (batch, seq_len, emb_dim)
	embedded := m.drop.Forward(m.emb.Forward(tokens))

	// lstm output: h_n shape = (num_layers*2, batch, hidden)
	out := m.lstm.ForwardLSTM(embedded, nil, nil)

	// Concat last two layers of h_n (bidirectional last state)
	// h_n[-2]: forward last,  h_n[-1]: backward last
	n := int64(out.Hn.Shape()[0])
	hFwd := out.Hn.Slice(0, n-2, n-1, 1).SqueezeDim(0) // (batch, hidden)
	hBwd := out.Hn.Slice(0, n-1, n, 1).SqueezeDim(0)   // (batch, hidden)
	h := gt.Cat([]*gt.Tensor{hFwd, hBwd}, 1)             // (batch, hidden*2)

	return m.fc.Forward(m.drop.Forward(h))
}

func (m *SentimentModel) Parameters() []*gt.Tensor {
	var p []*gt.Tensor
	p = append(p, m.emb.Parameters()...)
	p = append(p, m.lstm.Parameters()...)
	p = append(p, m.fc.Parameters()...)
	return p
}

func (m *SentimentModel) Train() {
	m.emb.Train()
	m.lstm.Train()
	m.fc.Train()
	m.drop.Train()
}

func (m *SentimentModel) Eval() {
	m.emb.Eval()
	m.lstm.Eval()
	m.fc.Eval()
	m.drop.Eval()
}

func main() {
	fmt.Println("=== Text Classification (BiLSTM) ===")

	model := NewSentimentModel()
	model.Train()

	// Python: optimizer = optim.Adam(model.parameters(), lr=2e-4)
	opt := gt.NewAdam(model.Parameters(), gt.AdamOptions{LR: 2e-4})
	// Python: scheduler = CosineAnnealingLR(optimizer, T_max=10)
	sched := gt.NewCosineAnnealingLR(opt, 10, 1e-6)

	// Synthetic data: token ids (integers 0–vocabSize-1)
	// Python: tokens  = torch.randint(0, vocabSize, (batchSize, seqLen))
	//         targets = torch.randint(0, 2, (batchSize,))
	tokens  := gt.Zeros(batchSize, seqLen).Cast(gt.Int64)
	targets := gt.Zeros(batchSize).Cast(gt.Int64)

	for epoch := 1; epoch <= 5; epoch++ {
		opt.ZeroGrad()

		logits := model.Forward(tokens)
		loss := gt.CrossEntropyLoss(logits, targets, gt.ReduceMean)
		loss.Backward()
		opt.Step()
		sched.Step()

		fmt.Printf("  epoch %d  loss=%.4f  lr=%.7f\n", epoch, loss.Item(), opt.GetLR())
	}

	// Inference
	model.Eval()
	gt.WithNoGrad(func() {
		testTokens := gt.Zeros(4, seqLen).Cast(gt.Int64)
		logits := model.Forward(testTokens)
		preds := logits.Argmax(1, false)
		fmt.Printf("Predictions shape: %v\n", preds.Shape())
	})
}
