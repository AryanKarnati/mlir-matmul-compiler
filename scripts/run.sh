#!/usr/bin/env bash
#
# Lower src/matmul.mlir from the linalg dialect down to the llvm dialect,
# then JIT-execute it. Prints the resulting matrix.
#
# Usage:  ./scripts/run.sh [path/to/file.mlir]   (defaults to src/matmul.mlir)

set -euo pipefail

# --- locate mlir-opt and derive the build's lib directory from it ---
MLIR_OPT="$(command -v mlir-opt || true)"
if [[ -z "$MLIR_OPT" ]]; then
  echo "error: mlir-opt not on PATH. Add <your-llvm-build>/bin to PATH first." >&2
  exit 1
fi
BIN_DIR="$(cd "$(dirname "$MLIR_OPT")" && pwd)"
LIB_DIR="$(cd "$BIN_DIR/../lib" && pwd)"

# --- the runner was renamed mlir-cpu-runner -> mlir-runner in early 2025 ---
RUNNER="$(command -v mlir-runner || command -v mlir-cpu-runner || true)"
if [[ -z "$RUNNER" ]]; then
  echo "error: neither mlir-runner nor mlir-cpu-runner found on PATH." >&2
  exit 1
fi

# --- the runtime lib that defines printMemrefF32 (.so on Linux/WSL) ---
RUNNER_UTILS="$(ls "$LIB_DIR"/libmlir_runner_utils.* 2>/dev/null | head -n1 || true)"
if [[ -z "$RUNNER_UTILS" ]]; then
  echo "error: libmlir_runner_utils not found in $LIB_DIR" >&2
  echo "       build it from your llvm build dir: ninja mlir_runner_utils mlir_c_runner_utils" >&2
  exit 1
fi

SRC="${1:-src/matmul.mlir}"

echo ">> opt:    $MLIR_OPT"
echo ">> runner: $RUNNER"
echo ">> utils:  $RUNNER_UTILS"
echo ">> source: $SRC"
echo

# The lowering pipeline: linalg -> loops -> ... -> llvm dialect, then JIT.
"$MLIR_OPT" "$SRC" \
  -convert-linalg-to-loops \
  -linalg-fuse-elementwise-ops \
  -affine-loop-tile \
  -convert-scf-to-cf \
  -expand-strided-metadata \
  -lower-affine \
  -convert-arith-to-llvm \
  -finalize-memref-to-llvm \
  -convert-func-to-llvm \
  -convert-cf-to-llvm \
  -reconcile-unrealized-casts \
| "$RUNNER" -O3 -e main -entry-point-result=void \
    -shared-libs="$RUNNER_UTILS"