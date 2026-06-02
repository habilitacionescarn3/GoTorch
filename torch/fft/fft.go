// Package fft — Fast Fourier Transform. Mirrors: torch.fft
package fft

import "github.com/Sarkar-AGI/GoTorch/torch"

// FFT computes 1-D discrete Fourier transform. Mirrors: torch.fft.fft
func FFT(t *torch.Tensor, n int, dim int) *torch.Tensor { return nil }

// IFFT computes inverse 1-D DFT. Mirrors: torch.fft.ifft
func IFFT(t *torch.Tensor, n int, dim int) *torch.Tensor { return nil }

// FFT2 computes 2-D DFT. Mirrors: torch.fft.fft2
func FFT2(t *torch.Tensor) *torch.Tensor { return nil }

// IFFT2 computes inverse 2-D DFT. Mirrors: torch.fft.ifft2
func IFFT2(t *torch.Tensor) *torch.Tensor { return nil }

// RFFT computes 1-D DFT of real input. Mirrors: torch.fft.rfft
func RFFT(t *torch.Tensor, n int) *torch.Tensor { return nil }

// IRFFT computes inverse RFFT. Mirrors: torch.fft.irfft
func IRFFT(t *torch.Tensor, n int) *torch.Tensor { return nil }
