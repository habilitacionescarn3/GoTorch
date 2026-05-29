module github.com/Sarkar-AGI/GoTorch

go 1.21

// GoTorch uses CGo to call its built-in C++ backend (aten/, c10/, torch/csrc/).
// No separate library download needed — everything is in this repository.
//
// Build steps:
//   1. cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=OFF -DBUILD_TEST=OFF
//   2. cmake --build . --target torch torch_cpu c10 -j$(nproc)
//   3. export GOTORCH=$(pwd)
//   4. export CGO_CFLAGS="-I${GOTORCH}/torch/csrc/api/include -I${GOTORCH}/csrc/go_binding"
//   5. export CGO_LDFLAGS="-L${GOTORCH}/build/lib -Wl,-rpath,${GOTORCH}/build/lib -ltorch -ltorch_cpu -lc10"
//   6. go build ./...
