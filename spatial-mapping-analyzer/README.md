# spatial-mapping-analyzer

Standalone passthrough mapping loop with mock, TT-Sim dummy and real int32 DAG
backends. The Analyzer preserves the DAG and assigns one core per op. Search,
fusion and Tensix FPU/SFPU lowering are not implemented; ranking scores remain
synthetic. This project does not integrate with a compiler.

## Quick start

Python 3.10+; CI uses 3.12. From the repository root:

```bash
cd spatial-mapping-analyzer
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_analyzer.py --arch examples/wormhole.yaml \
  --program examples/matmul_relu_matmul.yaml --iterations 10 --backend mock
```

Each run creates a unique directory under `results/`. Use `--output PATH` for a
specific **new** directory; existing directories are refused to prevent stale
results. Mock executes no kernels and returns constant cost 1; ties select the
first trial. The example is a 64x64 BF16 Matmul → ReLU → Matmul chain.

## Real TT-Sim dummy E2E

Initialize and build the pinned official Wormhole simulator from the repository
root (g++ with C++20 support is required):

```bash
git submodule update --init --recursive
cd third_party/ttsim
python make.py src/_out/release_wh/libttsim.so
cd ../../spatial-mapping-analyzer
python run_analyzer.py --arch examples/wormhole.yaml \
  --program examples/matmul_relu_matmul.yaml --backend tt-sim --iterations 3
```

The backend generates `dummy.json`, a 32-byte RV32I `dummy.bin`, and
`dummy_data.bin`. An isolated subprocess loads the real library through its
PCI/BAR C ABI, starts BRISC on physical core (1,1), and reads back the computed
result. The example adds region count 3 to allocated-core count 3 and verifies
**6**, with completion flag 1. Before clocking, the output must still be pending.
No TT-Metal installation, SoC descriptor or cross compiler is needed for this dummy.

The **original program DAG and logical mapping are not executed**. Original-program
correctness remains `not_checked`; dummy correctness is recorded separately.
Hardware metrics stay null/unsupported, and selection still uses constant synthetic
cost 1. The observed seven API steps are a dummy diagnostic, not hardware latency.

Use `--tt-sim-library PATH` to override the submodule release library and
`--tt-sim-timeout SECONDS` to change the 30-second per-trial limit. The dummy
requires `extensions.target_family: wormhole`. Errors never fall back to mock.
For direct debugging, the child is `python -m backends.tt_sim_runner` from this
directory; each trial's `invocation.json` records its complete command.

## Contracts and module boundaries

For real tensor/scalar execution, see the program E2E commands below.

| Component | Source of truth / responsibility |
| --- | --- |
| Architecture, program | `specs/models.py`: resource budget, tensors, op ports and dependency edges |
| Mapping IR | `mapping_ir/models.py`: ordered regions, core counts and fusion declarations |
| Analyzer | `analyzer/base.py`: propose a mapping using architecture, program and trial history |
| Validator | `validator/checks.py`: semantic checks and structured errors |
| Backend, report | `backends/base.py`: execution protocol, objectives, metrics and report invariants |
| TT-Sim dummy | `tt_sim_dummy.py`: shared workload constants, file generation and validation |
| TT-Sim execution | `tt_sim.py`: subprocess/report handling; `tt_sim_runner.py`: device ABI |
| Program execution | `tt_sim_program.py`: CPU comparison/report; `program_workload.py`: capability checks, reference and lowering |
| Program kernels | `rv32_kernels.py`: bounded RV32IM loops; `tt_sim_program_runner.py`: ready waves and device execution |
| Orchestration | `pipeline.py`: proposal/validation boundary, execution boundary, persistence and ranking |

Backend filenames in this table are relative to `backends/`. Analyzer, Validator,
Mapping IR and Backend remain separate. To replace passthrough, implement the
existing Analyzer or Backend protocol and inject it into `pipeline.run(...)`.

The four JSON Schemas are generated from the runtime models, **not hand-maintained
or committed**. Export them when needed:

```bash
python export_schemas.py
# specs/schemas/{arch,program,mapping,report}.schema.json
```

Contracts use version `0.1`, reject unknown fields and provide a top-level
`extensions` namespace. Program scope is static 2D matmul, shape-preserving relu
and non-broadcasting add, with matching float32/BF16/int32 dtypes. Scalars use
`shape: []`. Edges must match producer/consumer tensor ports exactly.

Mapping uses disjoint reserved cores in topological order; total allocation must
fit the budget. The default `exclusive_cores_tensor_barrier` serializes regions.
`exclusive_cores_dependency_barrier` allows independent regions to execute together,
with full tensors available before consumers start. Fusion remains unsupported.
Architecture examples are logical budgets, not official SoC descriptors. General
physical placement and tile-streaming feasibility remain future work.

## Tensor and scalar program E2E

After building the same Wormhole library, from this directory:

```bash
python run_analyzer.py --arch examples/wormhole_brisc.yaml \
  --program examples/gemm_relu_gemm.yaml --backend tt-sim-program \
  --iterations 1 --seed 0
python run_analyzer.py --arch examples/wormhole_brisc.yaml \
  --program examples/scalar_diamond.yaml --inputs examples/scalar_diamond.inputs.json \
  --backend tt-sim-program --execution-policy exclusive_cores_dependency_barrier \
  --iterations 1
```

The tensor task is `Y = ReLU(A @ B) @ C` with 32x32 int32 matrices. Without an
input fixture, a fixed seed generates values in [-2, 2]; all actual inputs are
saved. The scalar diamond is `p=a+b; left=p+c; right=ReLU(p); y=left+right`.
The supplied inputs produce `p=-2, left=3, right=0, y=3`. Input JSON maps tensor
names to flat, row-major int32 arrays; a scalar has one value. Int32 arithmetic
wraps modulo 2^32 and ReLU compares the signed result.

Each region gets one BRISC on a distinct physical Tensix tile. Independent ops
are released before the same clock call; the host waits for the ready wave and
copies completed outputs into consumer SRAM. Scalar execution traces prove that
Left and Right overlap and Join starts after both complete. Keeping the default
tensor-barrier policy runs the same diamond serially for comparison.

The backend emits short RV32IM kernels, runs them on the **unmodified real TT-Sim**,
and compares every intermediate tensor and output with an independent Python
reference. Expected values never enter device memory or firmware. Additional
artifacts are `inputs.json`, `reference.json`, `correctness.json`, per-op binaries,
`program_execution.json`, and a runner result containing core assignments, waves
and start/completion observations. The usual feedback/history/best-mapping loop
also applies; repeated trials intentionally use identical inputs and mappings.

This backend is bounded to int32 scalars/vectors/matrices with dimensions <=32,
at most eight one-op/one-core regions, and a 16,388-byte local memory footprint.
It uses fixed physical columns 1,2,3,4,6,7,8,9 at row 1. Wormhole BRISC has no
floating-point ISA, so float/BF16 workloads are explicitly rejected by this path.
It verifies program correctness and simulated branch overlap using BRISC, while
accelerator GEMM/ReLU kernels and device-side NoC transfers remain unimplemented.
Completion is sampled every API step for scalar waves, every 32 steps for tensor
waves. These observations exclude host transfers and are **not hardware latency**;
performance metrics remain unsupported and ranking cost remains synthetic 1.

## Artifacts and failure behavior

Each trial saves input/mapping snapshots, `validation.json`, `report.json` and
`trial.json`. Errors have `code`, `path`, and `message`. TT-Sim also saves the
dummy files, `invocation.json`, stdout/stderr logs and the runner result when
available. Reports record file/library hashes and the scope of execution.

The run saves `history.jsonl`, `summary.json` and, when eligible, `best_mapping.yaml`.
Completed trials become feedback for the next proposal; the passthrough policy
deliberately ignores it. Invalid mappings never reach the backend. Plugin failures
are recorded and later trials continue. Input and filesystem errors stop the run.
No eligible result means no best mapping and exit 2; success exits 0.

## Maintenance and verification

```bash
python -m unittest discover -s tests -v
```

The tests cover mock/real CLI runs, feedback, input/graph validation, generated
schema export, ranking, provenance, simulator failure isolation, numerical
correctness, overflow, corrupted kernels and diamond overlap. Real integration
tests skip without a built library; `TT_SIM_TEST_LIBRARY` can select another build.
CI requires the pinned library, runs all tests and all E2Es, and uploads run
evidence plus the four schemas as the `tt-sim-dummy-e2e` artifact.

Keep schema changes in the models, dummy protocol changes in `tt_sim_dummy.py`,
and device register changes in the isolated runner. New mapping policies belong in
`analyzer/`. Tensix FPU/SFPU lowering, device-side transfers and trustworthy
performance labels are the next backend work; generated schemas, binaries,
results and logs should stay out of Git.
