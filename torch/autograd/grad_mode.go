// Package autograd — Gradient mode control. Mirrors: torch.autograd.grad_mode
package autograd

// NoGrad disables gradient computation globally.
// Mirrors: torch.no_grad()
func NoGrad() {}

// EnableGrad re-enables gradient computation.
func EnableGrad() {}

// IsGradEnabled returns whether gradient computation is enabled.
func IsGradEnabled() bool { return true }

// WithNoGrad runs fn with gradients disabled.
// Mirrors: with torch.no_grad():
func WithNoGrad(fn func()) {
	was := IsGradEnabled()
	NoGrad()
	defer func() {
		if was { EnableGrad() }
	}()
	fn()
}

// WithGradEnabled runs fn with gradients enabled.
func WithGradEnabled(fn func()) {
	EnableGrad()
	fn()
}

// SetGradEnabled sets gradient computation mode.
func SetGradEnabled(enabled bool) {
	if enabled { EnableGrad() } else { NoGrad() }
}
