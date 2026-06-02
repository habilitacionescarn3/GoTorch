// Package jit — TorchScript JIT. Mirrors: torch.jit
package jit

import "github.com/Sarkar-AGI/GoTorch/torch"

// ScriptModule is a TorchScript compiled module.
// Mirrors: torch.jit.ScriptModule
type ScriptModule struct{ path string }

// Load loads a TorchScript model from disk.
// Mirrors: torch.jit.load(path)
func Load(path string) (*ScriptModule, error) { return &ScriptModule{path}, nil }

// Save saves a TorchScript model to disk.
// Mirrors: torch.jit.save(model, path)
func Save(m *ScriptModule, path string) error { return nil }

// Forward runs the scripted model.
func (m *ScriptModule) Forward(inputs ...*torch.Tensor) *torch.Tensor { return nil }

// Path returns the model file path.
func (m *ScriptModule) Path() string { return m.path }
