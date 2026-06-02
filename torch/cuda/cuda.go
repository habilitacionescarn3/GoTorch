// Package cuda — CUDA utilities. Mirrors: torch.cuda
package cuda

// IsAvailable returns true if CUDA is available. Mirrors: torch.cuda.is_available()
func IsAvailable() bool { return false }

// DeviceCount returns number of GPUs. Mirrors: torch.cuda.device_count()
func DeviceCount() int { return 0 }

// SetDevice sets active CUDA device. Mirrors: torch.cuda.set_device(id)
func SetDevice(id int) {}

// CurrentDevice returns current CUDA device index. Mirrors: torch.cuda.current_device()
func CurrentDevice() int { return 0 }

// Synchronize blocks until all kernels finish. Mirrors: torch.cuda.synchronize()
func Synchronize() {}

// EmptyCache releases cached memory. Mirrors: torch.cuda.empty_cache()
func EmptyCache() {}
