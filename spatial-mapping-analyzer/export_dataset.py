"""Export measured successful trials as validated JSONL training records."""
import argparse
import hashlib
import json
from pathlib import Path

from dataset import (
    MeasuredMappingRecord,
    MeasuredObjective,
    RequestedExecutionIdentity,
    content_hash,
    make_execution_group_id,
    make_observation_id,
    make_program_group_id,
)
from mapping_ir import Mapping
from report import Report
from specs import Architecture, Program, fingerprint


def _legacy_run_id(history_path):
    digest = hashlib.sha256(history_path.read_bytes()).hexdigest()
    return f"legacy_{digest}"


def _record_from_trial(raw, fallback_run_id):
    report = Report.model_validate(raw["report"])
    objective = report.objective
    if (
        report.status != "ok"
        or report.correctness != "passed"
        or objective is None
        or objective.source != "measured"
    ):
        return None

    validation = raw.get("validation")
    if not isinstance(validation, dict) or validation.get("valid") is not True:
        raise ValueError("Measured trial must have successful mapping validation")

    trial_objective = raw.get("objective")
    if trial_objective != objective.model_dump():
        raise ValueError("Trial objective does not match backend report objective")

    architecture = Architecture.model_validate(raw["architecture"])
    program = Program.model_validate(raw["program"])
    mapping = Mapping.model_validate(raw["mapping"])
    mapping_hash = fingerprint(mapping)
    if report.mapping_hash != mapping_hash:
        raise ValueError("Measured trial report mapping hash does not match mapping")

    signature = raw.get("requested_execution_signature")
    if signature is None:
        raise ValueError(
            "Measured trial is missing requested_execution_signature"
        )
    if report.observed_execution is None or report.measurement_context is None:
        raise ValueError(
            "Measured trial is missing observed execution or measurement context"
        )

    requested_execution = RequestedExecutionIdentity(
        backend=report.backend,
        backend_version=report.backend_version,
        signature=signature,
    )
    measurement = MeasuredObjective.model_validate(objective.model_dump())
    run_id = raw.get("run_id") or fallback_run_id
    trial_id = raw["trial_id"]

    record_data = {
        "schema_version": "0.1",
        "extensions": {},
        "observation_id": make_observation_id(run_id, trial_id),
        "run_id": run_id,
        "trial_id": trial_id,
        "program_group_id": make_program_group_id(program),
        "execution_group_id": make_execution_group_id(
            program,
            architecture,
            requested_execution.model_dump(),
            report.observed_execution.model_dump(),
        ),
        "architecture": architecture.model_dump(),
        "program": program.model_dump(),
        "mapping": mapping.model_dump(),
        "mapping_hash": mapping_hash,
        "requested_execution": requested_execution.model_dump(),
        "observed_execution": report.observed_execution.model_dump(),
        "measurement_context": report.measurement_context.model_dump(),
        "measurement": measurement.model_dump(),
    }
    record_data["content_hash"] = "content_" + content_hash(record_data)
    return MeasuredMappingRecord.model_validate(record_data)


def load_run_records(run_dir):
    history_path = run_dir / "history.jsonl"
    if not history_path.is_file():
        raise ValueError(f"Run has no history.jsonl: {run_dir}")

    fallback_run_id = _legacy_run_id(history_path)
    records = []
    with history_path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                record = _record_from_trial(raw, fallback_run_id)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid trial at {history_path}:{line_number}: {exc}"
                ) from exc
            if record is not None:
                records.append(record)
    return records


def export_dataset(run_dirs, output):
    by_observation = {}
    for run_dir in run_dirs:
        for record in load_run_records(run_dir):
            existing = by_observation.get(record.observation_id)
            if (
                existing is not None
                and existing.content_hash != record.content_hash
            ):
                raise ValueError(
                    f"Conflicting dataset observation: {record.observation_id}"
                )
            by_observation[record.observation_id] = record

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for observation_id in sorted(by_observation):
            record = by_observation[observation_id]
            stream.write(
                json.dumps(record.model_dump(), allow_nan=False, sort_keys=True)
                + "\n"
            )
    return len(by_observation)


def main():
    parser = argparse.ArgumentParser(
        description="Export measured mapping trials as validated training JSONL"
    )
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        required=True,
        help="Analyzer run directory; repeat to combine runs",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    count = export_dataset(args.run, args.output)
    print(f"Exported {count} measured mapping records to {args.output}")


if __name__ == "__main__":
    main()
