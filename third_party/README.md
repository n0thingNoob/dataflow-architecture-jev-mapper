# TT-Sim dependency

`ttsim/` is the official [Tenstorrent simulator](https://github.com/tenstorrent/ttsim)
as a Git submodule, pinned to `40bb1a2ad6a755279c4628ddc65e30b10721fdef`.
The parent gitlink controls the version; upstream `main` is not followed automatically.

From the repository root (Python 3.8+ and g++ with C++20 support):

```bash
git submodule update --init --recursive
cd third_party/ttsim
python make.py src/_out/release_wh/libttsim.so
```

This builds the single-chip Wormhole library used by the
[real dummy E2E](../spatial-mapping-analyzer/README.md#real-tt-sim-dummy-e2e).
The adapter uses the official PCI/BAR ABI directly; full TT-Metal program lowering
is separate future work. Mock runs do not require this build.

To upgrade, check out a reviewed commit inside the submodule, commit its gitlink
change in the parent, update the revision above and run the real E2E. CI builds
and executes the pinned source without modifications. Upstream licensing, notices
and build-output ignore rules remain in the submodule.
