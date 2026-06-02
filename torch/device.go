// Package torch — Device type. Mirrors: torch.device
package torch

import "fmt"

type Device struct{ Type DeviceType; Index int }
type DeviceType int

const (
	CPU  DeviceType = 0
	CUDA DeviceType = 1
	MPS  DeviceType = 2
)

func CPUDevice() Device           { return Device{CPU, -1} }
func CUDADevice(i int) Device     { return Device{CUDA, i} }
func MPSDevice() Device           { return Device{MPS, -1} }
func (d Device) IsCPU() bool      { return d.Type == CPU }
func (d Device) IsCUDA() bool     { return d.Type == CUDA }
func (d Device) IsMPS() bool      { return d.Type == MPS }
func (d Device) String() string {
	switch d.Type {
	case CUDA:
		if d.Index >= 0 { return fmt.Sprintf("cuda:%d", d.Index) }
		return "cuda"
	case MPS: return "mps"
	default:  return "cpu"
	}
}
