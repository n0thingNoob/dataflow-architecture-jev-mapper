# Execution and Measurement Identity Contract

This project treats mapping search, backend execution, measurement, and training
data as separate contracts. Do not collapse these identities into one field.

## Identity layers

1. **Mapping IR identity**
   - `fingerprint(mapping)`
   - Describes the syntactic Mapping IR.

2. **Requested execution signature**
   - Produced by `backend.candidate_execution_signature(...)`.
   - Used only to decide whether two Mapping IR candidates can produce distinct
     executions for that backend.
   - Must be deterministic and independent of runtime noise.

3. **Observed execution identity**
   - Returned by the backend after lowering/runtime mapping.
   - For TT-Metal this distinguishes:
     - Mapping logical core IDs,
     - TT-Metal logical worker coordinates,
     - observed worker/NoC coordinates.
   - Measured objectives are invalid without an observed execution identity.

4. **Measurement context**
   - Records how a measured objective was obtained.
   - Includes measurement version, analysis name, runtime implementation
     revision, probe binary hash, and profiler configuration.
   - Changing measurement semantics requires a new measurement version.
   - Changing backend execution semantics requires a new backend version.

## Dataset identities

`MeasuredMappingRecord` separates:

- `observation_id`: stable identity of one run/trial observation. It does not
  contain the measured value.
- `content_hash`: integrity hash of the full record. If the same observation ID
  appears with a different content hash, export must fail.
- `program_group_id`: groups all mappings/measurements of the same program.
- `execution_group_id`: groups repeated measurements of the same effective
  execution.

Repeated measurements are preserved as distinct observations, but they share
the same execution group.

## Cross-validation rules

Model evaluation must not use random row-level train/test splitting.

Minimum rule:

- split by `program_group_id`;
- never place observations from the same `execution_group_id` in different
  folds.

Recommended later evaluations include leave-one-DAG-family-out and
leave-one-measurement-context-out tests.

## Measurement validation

Before using physical-device labels for training:

- verify requested TT-Metal logical coordinates against backend output;
- verify observed worker coordinates are returned by TT-Metal, not inferred by
  Python;
- retain all raw repeated observations;
- randomize candidate execution order during repeated measurements to reduce
  warm-up, thermal, clock, and runtime-state confounding;
- aggregate repeated observations with robust statistics such as median and MAD,
  while keeping raw observations as the source of truth;
- cross-check TT-Metal profiler API extraction against
  `cpp_device_perf_report.csv` before accepting a device measurement.

The GitHub `tt-metal-ttsim` job validates real TT-Metal code against pinned
TT-Sim. It is not physical-device performance validation.
