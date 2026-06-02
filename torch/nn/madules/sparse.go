// Package modules — Sparse layers (alias for embedding). Mirrors: torch.nn.modules.sparse
package modules

// Sparse module types are defined in embedding.go
// This file documents the sparse module package.

// SparseMode defines embedding bag reduction modes.
type SparseMode int

const (
	SparseModeSum  SparseMode = 0
	SparseModeMean SparseMode = 1
	SparseModeMax  SparseMode = 2
)

func (s SparseMode) String() string {
	switch s {
	case SparseModeSum:  return "sum"
	case SparseModeMean: return "mean"
	case SparseModeMax:  return "max"
	default:             return "mean"
	}
}
