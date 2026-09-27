# Tensix Placement Probe

This optional probe validates one narrow path:

```text
Mapping IR logical placement
        ↓
Mapping logical core ID -> TT-Metal logical worker CoreCoord
        ↓
worker_core_from_logical_core -> observed worker CoreCoord
        ↓
TT-Metal Program
        ↓
TT_METAL_SIMULATOR
        ↓
official TT-Sim
        ↓
Tensix UNPACK / MATH / PACK
```

TT-Sim remains correctness-only. The two-core chain can additionally run on a
real Wormhole device and read `DEVICE KERNEL DURATION [ns]` from TT-Metal's
device profiler. Only that device-profiler value is emitted as a measured
ranking objective; host wall-clock and simulator timing are never used as labels.

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


## Real Wormhole measured objective

Build the probes against a Tracy-enabled TT-Metal build, then run the two-core
chain on a physical Wormhole device:

```bash
python run_analyzer.py \
  --backend tensix-chain \
  --tensix-runtime device \
  --arch examples/wormhole_tensix_probe.yaml \
  --program examples/bf16_two_add_chain.yaml \
  --search --candidate-limit 4 --iterations 4 \
  --tt-metal-home "$TT_METAL_HOME" \
  --tt-metal-revision "$(git -C "$TT_METAL_HOME" rev-parse HEAD)" \
  --tensix-chain-binary build/tensix_probe/spatial_tensix_chain_probe
```

Device mode enables `TT_METAL_DEVICE_PROFILER=1`,
`TT_METAL_PROFILER_MID_RUN_DUMP=1`, and
`TT_METAL_PROFILER_CPP_POST_PROCESS=1`. Each successful trial must return a
positive `device_kernel_duration_ns` tagged with
`measurement_source=tt_metal_device_profiler`; otherwise the backend rejects
the result instead of manufacturing a timing label. Device mode also requires
an explicit TT-Metal revision and records that revision, the probe binary SHA256,
and profiler configuration in `measurement_context`.

The TT-Sim CI path deliberately stays on `--tensix-runtime ttsim` semantics
and therefore must keep `objective: null`. It cross-checks requested TT-Metal
logical cores against worker coordinates returned by TT-Metal, but it is not a
physical-device performance validation.
