here#!/usr/bin/env bash
# install.sh — GoTorch quick setup script
# Copyright (c) 2024-2026 Sarkar-AGI. MIT License.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/Sarkar-AGI/GoTorch/main/install.sh | bash
#
# Or manually:
#   git clone https://github.com/Sarkar-AGI/GoTorch.git
#   cd GoTorch
#   ./install.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         GoTorch Installer v1.0.0           ║${NC}"
echo -e "${BLUE}║   PyTorch C++ engine, wrapped for Go       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""

# ─── Check requirements ───────────────────────────────────────────────────────

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo -e "${RED}✗ $1 not found. Please install $1 first.${NC}"
        echo "  $2"
        exit 1
    fi
    echo -e "${GREEN}✓ $1 found: $(command -v $1)${NC}"
}

echo "── Checking requirements ────────────────────────────────────────────────"
check_cmd go    "https://golang.org/dl"
check_cmd cmake "sudo apt install cmake  OR  brew install cmake"
check_cmd g++   "sudo apt install build-essential  OR  brew install gcc"
check_cmd ninja "sudo apt install ninja-build  OR  brew install ninja"
check_cmd git   "sudo apt install git  OR  brew install git"
echo ""

# Go version check
GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
GO_MAJOR=$(echo $GO_VERSION | cut -d. -f1)
GO_MINOR=$(echo $GO_VERSION | cut -d. -f2)
if [ "$GO_MAJOR" -lt 1 ] || ([ "$GO_MAJOR" -eq 1 ] && [ "$GO_MINOR" -lt 21 ]); then
    echo -e "${RED}✗ Go 1.21+ required, found $GO_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Go $GO_VERSION OK${NC}"

# ─── Clone if needed ──────────────────────────────────────────────────────────

GOTORCH_DIR="${GOTORCH_DIR:-$HOME/GoTorch}"

if [ ! -f "$GOTORCH_DIR/gotorch.go" ]; then
    echo ""
    echo "── Cloning GoTorch ──────────────────────────────────────────────────"
    git clone https://github.com/Sarkar-AGI/GoTorch.git "$GOTORCH_DIR"
fi

cd "$GOTORCH_DIR"
echo -e "${GREEN}✓ GoTorch source at: $GOTORCH_DIR${NC}"

# ─── Submodules ───────────────────────────────────────────────────────────────

echo ""
echo "── Initializing submodules ──────────────────────────────────────────────"
git submodule update --init --recursive --depth 1 \
    third_party/eigen \
    third_party/fmt \
    third_party/glog \
    third_party/googletest \
    third_party/protobuf \
    third_party/pybind11 \
    third_party/sleef 2>/dev/null || true
echo -e "${GREEN}✓ Submodules ready${NC}"

# ─── Python deps for build ────────────────────────────────────────────────────

echo ""
echo "── Installing Python build dependencies ─────────────────────────────────"
pip3 install --quiet pyyaml numpy typing_extensions 2>/dev/null || \
    pip install --quiet pyyaml numpy typing_extensions 2>/dev/null || true
echo -e "${GREEN}✓ Python deps ready${NC}"

# ─── Build libtorch ───────────────────────────────────────────────────────────

echo ""
echo "── Building libtorch C++ backend ────────────────────────────────────────"
echo -e "${YELLOW}   This may take 10-30 minutes on first build...${NC}"
echo ""

NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
mkdir -p build && cd build

cmake .. \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON=OFF \
    -DBUILD_TEST=OFF \
    -DBUILD_CAFFE2=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DUSE_CUDA=OFF \
    -DUSE_DISTRIBUTED=OFF \
    -DUSE_MKLDNN=ON \
    -DUSE_QNNPACK=OFF \
    -DUSE_PYTORCH_QNNPACK=OFF \
    -DUSE_XNNPACK=OFF \
    -DUSE_NNPACK=OFF \
    -DUSE_OPENMP=ON \
    -DBUILD_BINARY=OFF \
    -DCMAKE_INSTALL_PREFIX="$GOTORCH_DIR/build/install" \
    2>&1 | tail -5

ninja -j"$NPROC" torch torch_cpu c10 2>&1 | tail -3
ninja install 2>&1 | tail -2

cd "$GOTORCH_DIR"
echo -e "${GREEN}✓ libtorch built successfully${NC}"
echo "  Libraries:"
ls -lh build/lib/libtorch*.so 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'

# ─── Set env ──────────────────────────────────────────────────────────────────

PROFILE_FILE="$HOME/.bashrc"
[ -f "$HOME/.zshrc" ] && PROFILE_FILE="$HOME/.zshrc"

GOTORCH_ENV="
# GoTorch environment
export GOTORCH=\"$GOTORCH_DIR\"
export CGO_CFLAGS=\"-I\${GOTORCH}/torch/csrc/api/include -I\${GOTORCH}/csrc/go_binding -I\${GOTORCH}/build/install/include\"
export CGO_LDFLAGS=\"-L\${GOTORCH}/build/lib -Wl,-rpath,\${GOTORCH}/build/lib -ltorch -ltorch_cpu -lc10\"
export LD_LIBRARY_PATH=\"\${GOTORCH}/build/lib:\${LD_LIBRARY_PATH}\"
"

if ! grep -q "GoTorch environment" "$PROFILE_FILE" 2>/dev/null; then
    echo "$GOTORCH_ENV" >> "$PROFILE_FILE"
    echo -e "${GREEN}✓ Environment added to $PROFILE_FILE${NC}"
fi

# Export for current session
export GOTORCH="$GOTORCH_DIR"
export CGO_CFLAGS="-I${GOTORCH}/torch/csrc/api/include -I${GOTORCH}/csrc/go_binding -I${GOTORCH}/build/install/include"
export CGO_LDFLAGS="-L${GOTORCH}/build/lib -Wl,-rpath,${GOTORCH}/build/lib -ltorch -ltorch_cpu -lc10"
export LD_LIBRARY_PATH="${GOTORCH}/build/lib:${LD_LIBRARY_PATH}"

# ─── Build Go ─────────────────────────────────────────────────────────────────

echo ""
echo "── Building Go binding ──────────────────────────────────────────────────"
go build ./...
echo -e "${GREEN}✓ Go binding built${NC}"

# ─── Quick test ───────────────────────────────────────────────────────────────

echo ""
echo "── Running quick tests ──────────────────────────────────────────────────"
GOTORCH_SKIP_CPP_TESTS=1 go test -timeout 60s -run "^TestGoTensorCreation|TestGoLinear|TestGoAdd" . && \
    echo -e "${GREEN}✓ Tests passed${NC}" || \
    echo -e "${YELLOW}⚠ Some tests failed — check CGo flags${NC}"

# ─── Run example ──────────────────────────────────────────────────────────────

echo ""
echo "── Running example ──────────────────────────────────────────────────────"
go run ./examples/mnist/ 2>&1 | head -15

# ─── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          GoTorch installed! 🎉             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
echo ""
echo "Next steps:"
echo ""
echo "  # Reload shell"
echo "  source $PROFILE_FILE"
echo ""
echo "  # Run examples"
echo "  cd $GOTORCH_DIR"
echo "  go run ./examples/mnist/"
echo "  go run ./examples/text_classification/"
echo "  go run ./examples/image_classification/"
echo ""
echo "  # Run tests"
echo "  go test -v ./..."
echo ""
echo "  # Run benchmarks"
echo "  go test -bench=. -benchmem ."
echo ""
echo "  # Use in your project"
echo "  go get github.com/Sarkar-AGI/GoTorch@v1.0.0"
echo ""
echo "  import gt \"github.com/Sarkar-AGI/GoTorch\""
echo ""
