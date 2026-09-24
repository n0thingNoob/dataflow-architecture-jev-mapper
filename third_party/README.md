# Third-party dependencies

## Tenstorrent simulator

`ttsim/` is a real Git submodule of <https://github.com/tenstorrent/ttsim>, pinned
to commit `40bb1a2ad6a755279c4628ddc65e30b10721fdef` (upstream commit message:
"Public source drop for v1.10.9 release"). The parent repository's gitlink pins
the revision; it does not automatically follow upstream `main`.

From the parent repository root:

```bash
git submodule update --init --recursive
git submodule status third_party/ttsim
```

After switching branches or pulling a dependency update, run the update command
again to check out the recorded revision. To intentionally update the dependency,
check out a reviewed upstream commit inside `third_party/ttsim`, then commit its
gitlink change in the parent repository and update the revision documented here.

### Optional source build

The upstream build requires Python 3.8+ and g++ with C++20 support. From the
parent repository root:

```bash
cd third_party/ttsim
python make.py src/_out/release_wh/libttsim.so
```

This builds only the single-chip Wormhole library used by the dummy runner.
The upstream full `:build` target also builds Blackhole and other variants.
Build products stay inside the submodule and
are covered by upstream ignore rules. Follow the pinned upstream README for
platform requirements and details. CI builds the pinned Wormhole library and
executes the real dummy workload; the simulator source is not modified.

### Relationship to the analyzer

- `third_party/ttsim/` contains the upstream simulator source and build system.
- `spatial-mapping-analyzer/backends/tt_sim.py` is our adapter for Mapping IR,
  execution configuration, process invocation and report conversion.

`--backend tt-sim` now generates a fixed RV32I program and dummy input files,
loads the real Wormhole library in a subprocess, and verifies BRISC integer
addition on physical core (1,1). It uses the documented PCI/BAR C ABI directly,
so this smoke test needs no TT-Metal installation, SoC descriptor or RISC-V
cross compiler. See the analyzer README for commands and the exact scope.

Mapping-to-TT-Metal lowering, fusion and full tensor-program execution are still
pending. The dummy result verifies simulator transport and execution, while
mapping performance scores remain explicitly synthetic.

Upstream licensing and notices are preserved inside the submodule; no upstream
source is copied into the analyzer package.
