# Dataflow Architecture JEV Mapper

Standalone spatial mapping experiments. Start with the runnable
[spatial-mapping-analyzer](spatial-mapping-analyzer/README.md).

The official [Tenstorrent simulator](https://github.com/tenstorrent/ttsim) is
included as a pinned Git submodule at `third_party/ttsim`. After checking out
this branch, initialize dependencies from the repository root:

```bash
git submodule update --init --recursive
```

For a new clone, use `git clone --recurse-submodules`. See
[third-party setup](third_party/README.md) for the pinned revision and build commands.
The mock analyzer can run without initializing or building the simulator.

The TT-Sim backend also runs a real, generated BRISC dummy program through the
official simulator library. See the analyzer's
[TT-Sim dummy E2E instructions](spatial-mapping-analyzer/README.md#real-tt-sim-dummy-e2e).

For actual program execution with CPU reference checks, run the
[tensor GEMM–ReLU–GEMM and scalar diamond examples](spatial-mapping-analyzer/README.md#tensor-and-scalar-program-e2e).
These use int32 BRISC kernels; Tensix FPU/SFPU lowering remains future work.
