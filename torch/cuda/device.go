// Package cuda — CUDA device context. Mirrors: torch.cuda.device
package cuda

// DeviceContext manages the active CUDA device.
// Mirrors: with torch.cuda.device(device_id):
type DeviceContext struct{ deviceID int }

// NewDeviceContext creates a device context.
func NewDeviceContext(deviceID int) *DeviceContext { return &DeviceContext{deviceID} }

// Enter sets the device as active.
func (d *DeviceContext) Enter() { SetDevice(d.deviceID) }

// Exit restores the previous device.
func (d *DeviceContext) Exit() {}

// WithDevice runs fn on the specified CUDA device.
func WithDevice(deviceID int, fn func()) {
	prev := CurrentDevice()
	SetDevice(deviceID)
	defer SetDevice(prev)
	fn()
}
