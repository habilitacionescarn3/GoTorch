# _dynamo — Note

The `_dynamo` directory contains the original PyTorch TorchDynamo C++ source.
Python files have been removed. This directory's C++ components are used by
libtorch internally and are not directly exposed in the GoTorch Go API.

GoTorch does not provide a Go wrapper for TorchDynamo at this time.
