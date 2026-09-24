# spatial-mapping-analyzer

Standalone passthrough mapping loop with mock and real TT-Sim dummy backends.
The Analyzer preserves the DAG and assigns one core per op. Search, fusion and
full tensor-program lowering are not implemented; all ranking scores are explicitly
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

| Component | Source of truth / responsibility |
| --- | --- |
| Architecture, program | `specs/models.py`: resource budget, tensors, op ports and dependency edges |
| Mapping IR | `mapping_ir/models.py`: ordered regions, core counts and fusion declarations |
| Analyzer | `analyzer/base.py`: propose a mapping using architecture, program and trial history |
| Validator | `validator/checks.py`: semantic checks and structured errors |
| Backend, report | `backends/base.py`: execution protocol, objectives, metrics and report invariants |
| TT-Sim dummy | `tt_sim_dummy.py`: shared workload constants, file generation and validation |
| TT-Sim execution | `tt_sim.py`: subprocess/report handling; `tt_sim_runner.py`: device ABI |
| Orchestration | `pipeline.py`: proposal/validation boundary, execution boundary, persistence and ranking |

All paths in the two TT-Sim rows are relative to `backends/`. Analyzer, Validator,
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
and non-broadcasting add, with matching float32/BF16 dtypes. Edges must match
producer/consumer tensor ports exactly.

Mapping uses disjoint reserved cores and ordered full-tensor barriers; total
allocation must fit the budget. Fusion declarations are currently rejected.
The example architecture is a logical budget, not an official SoC descriptor.
Physical placement, buffer capacity and tile-streaming feasibility remain unchecked.

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

The 21 tests cover mock/real CLI runs, feedback, input/graph validation, generated
schema export, ranking, provenance and simulator failure isolation. Real integration
tests skip without a built library; `TT_SIM_TEST_LIBRARY` can select another build.
CI requires the pinned library, runs all tests and both E2Es, and uploads run
evidence plus the four schemas as the `tt-sim-dummy-e2e` artifact.

Keep schema changes in the models, dummy protocol changes in `tt_sim_dummy.py`,
and device register changes in the isolated runner. New mapping policies belong in
`analyzer/`. Full DAG lowering and trustworthy performance labels are the next
backend work; generated schemas, binaries, results and logs should stay out of Git.
