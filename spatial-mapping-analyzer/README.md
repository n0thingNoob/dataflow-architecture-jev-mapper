# spatial-mapping-analyzer

A runnable **passthrough scaffold** for a standalone mapping analyzer. This change
establishes the contracts and end-to-end orchestration before implementing search,
fusion, full program lowering, or a learned model. Alongside the mock backend,
it can execute a generated dummy program on real TT-Sim. It does not integrate
with a compiler.

## Run the complete loop

Python 3.10+ is required; CI uses Python 3.12.

```bash
cd spatial-mapping-analyzer
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_analyzer.py \
  --arch examples/wormhole.yaml \
  --program examples/matmul_relu_matmul.yaml \
  --iterations 10 \
  --backend mock \
  --output results/demo
```

The output directory must be new. Omit `--output` to create a unique directory
under this project's `results/`. This prevents a failed run from reusing an older
`best_mapping.yaml`. Generated results are gitignored.

The example describes `Matmul0 -> ReLU1 -> Matmul2`, with explicit external inputs
`x`, `w0`, and `w1`, and output `y`; all tensors are 64x64 BF16. No tensor values are
executed in this scaffold. `wormhole.yaml` describes a logical eight-core budget
for the mock demo, **not an official SoC descriptor or calibrated timing model**.
Unknown architecture parameters remain null.

An actual ten-trial run of this scaffold prints each trial with the same score:

```text
Backend: mock; output: results/demo
trial_0000  ok  cost=1 arbitrary (synthetic)
trial_0001  ok  cost=1 arbitrary (synthetic)
trial_0002  ok  cost=1 arbitrary (synthetic)
trial_0003  ok  cost=1 arbitrary (synthetic)
trial_0004  ok  cost=1 arbitrary (synthetic)
trial_0005  ok  cost=1 arbitrary (synthetic)
trial_0006  ok  cost=1 arbitrary (synthetic)
trial_0007  ok  cost=1 arbitrary (synthetic)
trial_0008  ok  cost=1 arbitrary (synthetic)
trial_0009  ok  cost=1 arbitrary (synthetic)
Best: trial_0000
Passthrough only: identical candidates, constant synthetic score; hardware cycles unavailable.
```

**This validates the software loop, not mapping quality or hardware performance.**
The analyzer emits one topologically ordered region per op, one core per region,
and no fusion. It deliberately ignores feedback; identical candidates are retained
so every loop iteration is observable. The backend returns a constant synthetic
score of 1. Stable ties select the earliest trial; no speedup is claimed.

## Modules and extension boundaries

| Path | Responsibility |
| --- | --- |
| `specs/models.py` | Architecture and program contracts |
| `specs/schemas/` | Four generated JSON Schema definitions |
| `mapping_ir/models.py` | Regions, core allocations and fusion declarations |
| `analyzer/base.py` | `Analyzer.propose_mapping(architecture, program, feedback)` protocol |
| `analyzer/passthrough.py` | Identity mapping policy |
| `validator/checks.py` | Input and mapping semantic checks, structured errors |
| `backends/base.py` | `Backend.run(architecture, program, mapping, workdir)` and report contracts |
| `backends/mock.py` | Explicit synthetic result, no numerical computation |
| `backends/tt_sim.py` | Dummy file generation, isolated real simulator invocation and report conversion |
| `backends/tt_sim_dummy.py` | Fixed RV32I program and mapping-derived dummy input files |
| `backends/tt_sim_runner.py` | Official Wormhole PCI/BAR ABI, BRISC launch and output verification |
| `pipeline.py` | Proposal, validation, execution, feedback history and best-result selection |
| `run_analyzer.py` | CLI entry point |
| `examples/` | Minimal architecture and matmul/relu/matmul DAG |
| `tests/` | Mock CLI, real simulator integration and boundary/negative-path tests |
| `results/` | Generated run directories (created on demand) |

`pipeline.run(...)` accepts injected Analyzer and Backend implementations. On
iteration N, the analyzer receives a deep copy of all completed trial records
0..N-1, including validation errors and reports. Copies of the original program
and architecture prevent module-side mutations from changing stored semantics.
The pipeline reparses proposals and reports at module boundaries.

## Four file contracts

Runtime validation uses Pydantic models; YAML is just their input/output encoding.
The JSON Schema files are generated from these same models:

| File | Definition | Main fields |
| --- | --- | --- |
| `arch.yaml` | [`arch.schema.json`](specs/schemas/arch.schema.json) | grid, available cores, local memory, compute units, topology, multicast, optional timing parameters |
| `program.yaml` | [`program.schema.json`](specs/schemas/program.schema.json) | tensors with shapes/dtypes, program inputs/outputs, op IDs/types/ports, explicit dependency edges |
| `mapping.yaml` | [`mapping.schema.json`](specs/schemas/mapping.schema.json) | program ID, input hashes, execution policy, ordered regions, core counts, fusion declarations |
| `report.json` | [`report.schema.json`](specs/schemas/report.schema.json) | backend/version, mapping hash, status, correctness, objective/source/unit, optional metrics |

Version is `0.1`. Unknown fields are rejected; top-level `extensions` provides an
explicit extension namespace. Schema definitions cover structure; the runtime
validator additionally checks cross-field/graph constraints. Regenerate schemas
after editing the models with `python export_schemas.py`; tests detect drift.

Program scope: static 2D matmul, shape-preserving relu, and same-shape add without
broadcasting; matching float32 or BF16 dtypes. Each node has one output. Every
producer/consumer tensor use must have exactly one edge with its input port index.
Input shape validity does not establish TT kernel compatibility.

Mapping scope: `exclusive_cores_tensor_barrier` means region core sets are
disjoint and reserved, and successor regions wait for full-tensor dependencies.
Regions and their ops are topologically ordered. The sum of core allocations
must fit the budget. This is not a tile-streaming pipeline. There is no physical
placement, buffer allocation or memory-capacity proof yet. The passthrough policy
requires at least as many cores as ops; otherwise validation fails without a
hidden remapping fallback.

`fusions` declares a future `matmul_relu` extension, but the current validator
rejects every nonempty fusion list as `UNSUPPORTED_FUSION` because there is no
fusion lowering. Grouping and fusion remain separate concepts. Program ops are
never fused or rewritten in `program.yaml`.

Reports explicitly distinguish `synthetic`, `estimated`, and `measured` objectives.
Mock metrics (cycles, latency, utilization, stalls, communication, buffers) have
`value: null`, `status: unsupported`, and a reason. Correctness is `not_checked`.
No timer is presented as accelerator latency. The pipeline rejects report hashes
for another mapping and incompatible objective definitions within one run.

## Result artifacts and errors

Each `trial_0000/`, `trial_0001/`, ... contains:

- `arch.yaml`, `program.yaml`, and the proposed `mapping.yaml` (when emitted).
- `validation.json` with `valid` and errors containing `code`, `path`, `message`.
- `report.json`, including explicit `skipped`, `unsupported` or `error` outcomes.
- `trial.json` with the full snapshot and the IDs of feedback trials provided.

The run directory also contains `history.jsonl`, `summary.json`, and, if there is
an eligible result, `best_mapping.yaml`. Every completed trial is written before
the next proposal. In mock trials, `objective.source` is `synthetic` and
`measured_cost` is null. Do not use these records as measured performance labels.

Invalid proposals never reach the backend. Proposal/backend failures are recorded
and later iterations continue. Exit status is 0 when at least one result has a
comparable objective, and 2 for invalid input, output errors, unsupported-only or
all-failed runs. An unsuccessful run creates no best mapping. Early input errors
are machine-readable JSON on stderr. Durability is per completed trial, not
transactional crash recovery; resume is not implemented.

## Real TT-Sim dummy E2E

The official simulator source is pinned as a Git submodule at
[`../third_party/ttsim`](../third_party/ttsim). From the repository root, run
`git submodule update --init --recursive`; see
[dependency setup and build instructions](../third_party/README.md).
The submodule supplies upstream code; `backends/tt_sim.py` generates dummy input
files, invokes an isolated runner and converts its verified result into a report.

Build the pinned single-chip Wormhole library, then run from the analyzer directory:

```bash
# From the repository root; Linux with Python and g++ supporting C++20.
git submodule update --init --recursive
cd third_party/ttsim
python make.py src/_out/release_wh/libttsim.so
cd ../../spatial-mapping-analyzer
python -m pip install -r requirements.txt
python run_analyzer.py \
  --arch examples/wormhole.yaml \
  --program examples/matmul_relu_matmul.yaml \
  --backend tt-sim --iterations 3 --output results/tt-sim-demo
```

The default library path is `third_party/ttsim/src/_out/release_wh/libttsim.so`.
Override it with `--tt-sim-library /absolute/path/libttsim_wh.so` if needed. The
runner wall-time limit defaults to 30 seconds per trial (`--tt-sim-timeout`).
Missing/incompatible libraries, timeouts, simulator failures and bad outputs
produce errors with no ranking objective; there is no mock fallback. An
explicit `extensions.target_family: wormhole` is required by this dummy adapter.

The actual execution in each trial is:

1. Preserve the original program/mapping snapshots. Write `dummy.json`, a 32-byte
   `dummy.bin` containing eight RV32I instructions, and 16-byte `dummy_data.bin`.
2. Put region count and total allocated core count in the two input words. For
   the supplied mapping these are 3 and 3. The output word starts at `0xFFFFFFFF`
   and the completion flag at 0, so loading the files alone cannot pass the test.
3. Launch `backends/tt_sim_runner.py` in a child process. It loads the official
   `libttsim.so` via ctypes and verifies the Wormhole PCI device ID.
4. Use the documented PCI/BAR interface to load SRAM on physical Tensix core
   (1,1), then release BRISC reset. Clock the simulator until BRISC writes the
   result and completion flag, or a bounded step budget is exhausted.
5. Read back and check **6** with completion **1**, save `runner_result.json`, and
   return the report to the existing feedback/history/best-selection loop.

This runs the actual Wormhole chip simulator, not the upstream generic RV64
computer simulator. No TT-Metal installation, SoC descriptor, cross compiler or
modified upstream source is needed for this direct ABI smoke test. The register
and TLB definitions are tied to the pinned upstream revision. See the
[official library API](https://github.com/tenstorrent/ttsim/blob/40bb1a2ad6a755279c4628ddc65e30b10721fdef/docs/libttsim_api.md).

**Only the dummy is executed.** The matmul/relu/matmul DAG, fusion, logical core
allocation and spatial pipeline are not lowered onto hardware yet. Accordingly,
`report.correctness` stays `not_checked` for the original program, while
`report.extensions.dummy_correctness` is `passed` for the actual dummy. Report
extensions record `program_dag_executed: false` and `mapping_lowered: false`.

The report includes the library/firmware/data hashes, actual result, completion
flag and API step count. A local run completed in 7 API steps; this is a dummy
execution diagnostic, **not accelerator latency or mapped-DAG cycle count**.
Hardware performance fields remain null/unsupported. The ranking score remains
constant `passthrough_cost = 1`, source `synthetic`, so the first successful tied
mapping is saved as `best_mapping.yaml` without a performance-improvement claim.

Each launched trial additionally saves `invocation.json` (exact argv, directory,
timeout and library hash), `stdout.log`, `stderr.log`, and `runner_result.json`
when the child exits normally. A fatal simulator error may prevent that last
file from being written; the adapter still records failure and preserves logs.
See [the observed run](docs/tt_sim_dummy_run.md) for the validated output.

Full program execution will require a separate TT-Metal lowering/runner path or
another supported backend lowering, with tensor reference checks and trustworthy
performance reporting. Initializing/building TT-Sim remains optional for mock runs.

## Next implementation steps

1. Add a candidate policy for grouping, legal fusion, and core allocations; keep
   a generator/ranker boundary for cost models that score rather than generate.
2. Extend the working dummy adapter with deterministic TT-Metal lowering, backend capability checks and numerical
   reference validation. Preserve the original DAG and record the execution plan.
3. Establish a trustworthy performance objective. Simulator steps, simulator host
   runtime and hardware latency are distinct; validate timing/ranking fidelity
   before collecting performance training labels.
4. Add deduplication, seeds for randomized policies, pinned toolchain/backend
   revisions and broader program/architecture coverage for reproducible datasets.
5. Replace the policy with a learned/Jev-style adapter once its concrete API and
   input/output behavior are defined. No external model API is assumed here.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite covers real CLI subprocess runs, all expected artifacts, feedback
delivery, DAG preservation, schema drift, malformed mappings, core capacity,
dependency order, unsupported fusion, shape/edge consistency, backend failure
recovery, report provenance and stable best selection. Real simulator tests
add dummy result verification, changed inputs and isolation of an actual fatal
simulator error. They skip locally if no library is built; set
`TT_SIM_TEST_LIBRARY` to test a non-default library. Boundary tests always run.

GitHub Actions verifies checkout of the pinned submodule, builds the real
Wormhole library, runs all 21 tests (including the real integration tests), then
executes a ten-iteration mock demo and a three-iteration real dummy E2E. The
`tt-sim-dummy-e2e` artifact contains the generated files, logs and reports.
