// Package torch — Data type constants. Mirrors: torch.dtype
package torch

// DType represents a tensor data type.
type DType int

const (
	Float32  DType = 6
	Float64  DType = 7
	Int32    DType = 3
	Int64    DType = 4
	Bool     DType = 11
	Float16  DType = 5
	BFloat16 DType = 15
	Int8     DType = 1
	UInt8    DType = 0
)

func (d DType) String() string {
	switch d {
	case Float32:  return "torch.float32"
	case Float64:  return "torch.float64"
	case Int32:    return "torch.int32"
	case Int64:    return "torch.int64"
	case Bool:     return "torch.bool"
	case Float16:  return "torch.float16"
	case BFloat16: return "torch.bfloat16"
	default:       return "torch.unknown"
	}
}
func (d DType) IsFloating() bool { return d == Float32 || d == Float64 || d == Float16 || d == BFloat16 }
func (d DType) IsInteger() bool  { return d == Int32 || d == Int64 || d == Int8 || d == UInt8 }
func (d DType) ItemSize() int {
	switch d {
	case Float32, Int32: return 4
	case Float64, Int64: return 8
	default:             return 1
	}
}
