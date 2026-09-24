# spatial-mapping-analyzer

A runnable **passthrough scaffold** for a standalone mapping analyzer. This change
establishes the contracts and end-to-end orchestration before implementing search,
fusion, kernel execution, or a learned model. It does not integrate with a compiler.

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
| `backends/tt_sim.py` | Explicitly unsupported TT-Sim adapter placeholder |
| `pipeline.py` | Proposal, validation, execution, feedback history and best-result selection |
| `run_analyzer.py` | CLI entry point |
| `examples/` | Minimal architecture and matmul/relu/matmul DAG |
| `tests/test_e2e.py` | CLI and boundary/negative-path tests |
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

## TT-Sim status

**This PR does not launch TT-Sim.** The command below records unsupported reports,
creates no best mapping, and exits 2; it never silently switches to mock:

```bash
python run_analyzer.py \
  --arch examples/wormhole.yaml \
  --program examples/matmul_relu_matmul.yaml \
  --backend tt-sim --iterations 1 --output results/tt-sim-check
```

The intended real adapter will lower validated mappings into deterministic
execution plans, instantiate parameterized TT-Metal kernels, and launch a runner
in an isolated subprocess with a timeout. Official ttsim execution uses
`TT_METAL_SIMULATOR=/path/to/libttsim_wh.so`, an accompanying `soc_descriptor.yaml`,
and typically `TT_METAL_SLOW_DISPATCH_MODE=1` with a built TT-Metal executable.
There is no official Mapping-YAML ingestion interface. See the
[official ttsim setup](https://github.com/tenstorrent/ttsim#running-with-tt-metalium)
and [library API](https://github.com/tenstorrent/ttsim/blob/main/docs/libttsim_api.md).
Those dependencies are intentionally not required to run this scaffold.

## Next implementation steps

1. Add a candidate policy for grouping, legal fusion, and core allocations; keep
   a generator/ranker boundary for cost models that score rather than generate.
2. Add deterministic TT-Metal lowering, backend capability checks and numerical
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
recovery, report provenance, stable best selection and unsupported TT-Sim.
GitHub Actions runs the suite and the ten-iteration demo.
