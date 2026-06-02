// Package cuda — CUDA memory utilities. Mirrors: torch.cuda.memory
package cuda

// MemoryStats holds CUDA memory statistics.
type MemoryStats struct {
	Allocated   int64 // bytes currently allocated
	Reserved    int64 // bytes reserved by the caching allocator
	MaxAllocated int64 // max bytes allocated
}

// MemoryAllocated returns allocated CUDA memory in bytes.
// Mirrors: torch.cuda.memory_allocated()
func MemoryAllocated() int64 { return 0 }

// MemoryReserved returns reserved CUDA memory in bytes.
// Mirrors: torch.cuda.memory_reserved()
func MemoryReserved() int64 { return 0 }

// MaxMemoryAllocated returns max allocated memory.
// Mirrors: torch.cuda.max_memory_allocated()
func MaxMemoryAllocated() int64 { return 0 }

// ResetMaxMemoryAllocated resets the peak memory stats.
func ResetMaxMemoryAllocated() {}

// GetMemoryStats returns all memory statistics.
func GetMemoryStats() MemoryStats { return MemoryStats{} }
