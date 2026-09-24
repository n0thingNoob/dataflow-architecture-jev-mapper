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
./make.py :build
```

The upstream build produces `src/_out/release_wh/libttsim.so` and
`src/_out/release_bh/libttsim.so`. Build products stay inside the submodule and
are covered by upstream ignore rules. Follow the pinned upstream README for
platform requirements and details; this PR verifies submodule checkout, not
simulator compilation or execution.

### Relationship to the analyzer

- `third_party/ttsim/` contains the upstream simulator source and build system.
- `spatial-mapping-analyzer/backends/tt_sim.py` is our adapter for Mapping IR,
  execution configuration, process invocation and report conversion.

Adding the source does not implement the adapter. A built TT-Metal runner,
compatible SoC descriptor, deterministic mapping lowering and report extraction
are still needed. `--backend tt-sim` therefore continues to return `unsupported`;
the runnable E2E uses the explicitly labeled mock backend.

Upstream licensing and notices are preserved inside the submodule; no upstream
source is copied into the analyzer package.
