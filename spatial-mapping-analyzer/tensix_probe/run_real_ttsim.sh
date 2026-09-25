#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="$REPO_ROOT/spatial-mapping-analyzer"
: "${TT_METAL_HOME:=/tt-metal}"
TTSIM_LIBRARY="${TTSIM_LIBRARY:-$REPO_ROOT/.ci/ttsim/libttsim.so}"
BUILD_DIR="$ROOT/build/tensix_probe"
RESULTS_DIR="$ROOT/results/ci-tensix"

if [[ ! -f "$TTSIM_LIBRARY" ]]; then
    echo "missing TT-Sim library: $TTSIM_LIBRARY" >&2
    exit 1
fi
if [[ ! -d "$TT_METAL_HOME" ]]; then
    echo "missing TT_METAL_HOME: $TT_METAL_HOME" >&2
    exit 1
fi

CMAKE_EXTRA_ARGS=()
if [[ -n "${TT_METAL_SOURCE_DIR:-}" ]]; then
    CMAKE_EXTRA_ARGS+=("-DTT_METAL_SOURCE_DIR=$TT_METAL_SOURCE_DIR")
    CMAKE_EXTRA_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$TT_METAL_SOURCE_DIR/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake")
    CONFIG="source-tree"
else
    if [[ -n "${TT_METALIUM_CONFIG_DIR:-}" ]]; then
        CONFIG="$(find "$TT_METALIUM_CONFIG_DIR" -maxdepth 1 -type f \( -iname 'tt-metalium-config.cmake' -o -iname 'TT-MetaliumConfig.cmake' \) -print -quit)"
    else
        CONFIG="$(find "$TT_METAL_HOME" /usr /opt -type f \( -iname 'tt-metalium-config.cmake' -o -iname 'TT-MetaliumConfig.cmake' \) -print -quit 2>/dev/null || true)"
    fi
    if [[ -z "$CONFIG" ]]; then
        echo "TT-Metalium CMake package not found" >&2
        exit 1
    fi
    CMAKE_EXTRA_ARGS+=("-DTT-Metalium_DIR=$(dirname "$CONFIG")")
fi

echo "TT_METAL_HOME=$TT_METAL_HOME"
echo "TT-Metalium config=$CONFIG"
echo "TT-Sim library=$TTSIM_LIBRARY"

/usr/bin/python3 -m pip install -r "$ROOT/requirements.txt"

rm -rf "$BUILD_DIR" "$RESULTS_DIR"
cmake -S "$ROOT/tensix_probe" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    "${CMAKE_EXTRA_ARGS[@]}"
cmake --build "$BUILD_DIR" -j2

PROBE="$BUILD_DIR/spatial_tensix_probe"
test -x "$PROBE"

cd "$ROOT"
/usr/bin/python3 run_analyzer.py \
    --backend tensix-probe \
    --arch examples/wormhole_tensix_probe.yaml \
    --program examples/bf16_tile_add.yaml \
    --search --candidate-limit 4 --iterations 4 \
    --tt-metal-home "$TT_METAL_HOME" \
    --tensix-probe-binary "$PROBE" \
    --tt-sim-library "$TTSIM_LIBRARY" \
    --tt-sim-timeout 180 \
    --output "$RESULTS_DIR"

/usr/bin/python3 - <<'PY'
import json
from pathlib import Path

root = Path("results/ci-tensix")
summary = json.loads((root / "summary.json").read_text())
assert summary["status"] == "ok", summary
assert len(summary["successful_trial_ids"]) == 4, summary
assert summary["best_trial_id"] is None, summary

cores = []
for trial_id in summary["successful_trial_ids"]:
    report = json.loads((root / trial_id / "report.json").read_text())
    assert report["status"] == "ok", report
    assert report["correctness"] == "passed", report
    assert report["objective"] is None, report
    cores.append(tuple(report["extensions"]["physical_core"]))

assert len(set(cores)) >= 2, cores
print("Verified real Tensix placement cores:", cores)
PY
