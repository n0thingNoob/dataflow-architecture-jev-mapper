# Tensix Placement Probe

This optional probe validates one narrow path:

```text
Mapping IR logical placement
        ↓
logical core -> CoreCoord
        ↓
TT-Metal Program
        ↓
TT_METAL_SIMULATOR
        ↓
official TT-Sim
        ↓
Tensix UNPACK / MATH / PACK
```

It is intentionally **not** a performance backend. It runs one 32x32 BF16 tile
`add` and checks correctness on the requested worker core. No latency/cycle
objective is produced.

## Prerequisites

Use a built TT-Metal checkout that provides the `TT-Metalium` CMake package:

```bash
export TT_METAL_HOME=/path/to/tt-metal
```

Build the probe:

```bash
cd spatial-mapping-analyzer
cmake -S tensix_probe -B build/tensix_probe \
  -DCMAKE_PREFIX_PATH="$TT_METAL_HOME/build"
cmake --build build/tensix_probe -j
```

If your TT-Metalium package is installed elsewhere, point `CMAKE_PREFIX_PATH`
at the directory containing `tt-metalium-config.cmake`.

## Run through the analyzer

Build the pinned Wormhole TT-Sim library first, then:

```bash
python run_analyzer.py \
  --backend tensix-probe \
  --arch examples/wormhole_tensix_probe.yaml \
  --program examples/bf16_tile_add.yaml \
  --search --candidate-limit 4 --iterations 4 \
  --tt-metal-home "$TT_METAL_HOME" \
  --tensix-probe-binary build/tensix_probe/spatial_tensix_probe
```

The backend creates a per-trial simulator directory containing the pinned
`libttsim.so` and Wormhole SoC descriptor, sets slow dispatch, invokes the
probe, and verifies that the core reported by the probe matches the Mapping IR.

Current scope is deliberately small:
- Wormhole only
- one BF16 add op
- one tile
- one explicitly placed core
- correctness only
- no performance metric
