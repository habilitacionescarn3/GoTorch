# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest  | ✅ Yes    |

## Reporting a Vulnerability

To report a security vulnerability in GoTorch (Go layer, CGo bridge):

- Open a GitHub issue at https://github.com/Sarkar-AGI/GoTorch/issues
- Use the `security` label
- Or email the maintainer directly via GitHub

## Scope

| Component | Maintainer |
|---|---|
| Go layer (`torch/**/*.go`) | GoTorch maintainers |
| C shim (`csrc/go_binding/`) | GoTorch maintainers |
| C++ backend (`aten/`, `c10/`, `torch/csrc/`) | Upstream open-source project |

## C++ backend

GoTorch contains its own C++ backend. Security issues in the C++ layer
should be reported via GitHub issues at https://github.com/Sarkar-AGI/GoTorch
