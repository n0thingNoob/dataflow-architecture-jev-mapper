import json
from copy import deepcopy
from pathlib import Path

from pydantic import ValidationError

from backends.base import Report
from mapping_ir.models import Mapping
from specs.io import fingerprint, write_json, write_yaml
from validator.checks import error, schema_errors, validate_inputs, validate_mapping


def run(architecture, program, analyzer, backend, iterations: int, output: Path):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    input_errors = validate_inputs(architecture, program)
    if input_errors:
        raise ValueError(input_errors)
    # Never merge with old artifacts: a failed run must not inherit a stale best mapping.
    output.mkdir(parents=True, exist_ok=False)
    history = []
    best = None
    objective_contract = None
    for index in range(iterations):
        trial_id = f"trial_{index:04d}"
        directory = output / trial_id
        directory.mkdir()
        trial = {"trial_id": trial_id, "architecture": architecture.model_dump(),
                 "program": program.model_dump(), "mapping": None, "validation": None,
                 "report": None, "objective": None, "measured_cost": None,
                 "feedback_trial_ids": [t["trial_id"] for t in history]}
        write_yaml(directory / "arch.yaml", trial["architecture"])
        write_yaml(directory / "program.yaml", trial["program"])
        mapping = None
        try:
            proposal = analyzer.propose_mapping(
                architecture.model_copy(deep=True), program.model_copy(deep=True), deepcopy(history))
            raw = proposal.model_dump() if isinstance(proposal, Mapping) else proposal
            trial["mapping"] = raw
            write_yaml(directory / "mapping.yaml", raw)
            mapping = Mapping.model_validate(raw)
            errors = validate_mapping(architecture, program, mapping)
        except ValidationError as exc:
            errors = schema_errors(exc)
        except Exception as exc:
            errors = [error("ANALYZER_ERROR", "analyzer", str(exc))]
        trial["validation"] = {"valid": not errors, "errors": errors}
        write_json(directory / "validation.json", trial["validation"])
        if not errors:
            try:
                result = backend.run(architecture.model_copy(deep=True), program.model_copy(deep=True),
                                     mapping.model_copy(deep=True), directory)
                report = Report.model_validate(result.model_dump() if isinstance(result, Report) else result)
                if report.mapping_hash != fingerprint(mapping):
                    raise ValueError("Backend report refers to a different mapping")
                if report.backend != backend.name:
                    raise ValueError("Backend identity mismatch")
                if report.status == "ok" and report.objective and report.correctness != "failed":
                    objective = report.objective
                    contract = (report.backend, report.backend_version, objective.name, objective.unit, objective.source)
                    if objective_contract is not None and contract != objective_contract:
                        raise ValueError("Cannot compare different objective definitions in one run")
                    objective_contract = contract
                    trial["objective"] = objective.model_dump()
                    if objective.source == "measured":
                        trial["measured_cost"] = objective.value
                    if best is None or objective.value < best["objective"]["value"]:
                        best = trial
            except Exception as exc:
                report = Report(backend=backend.name, backend_version="unknown", status="error",
                                mapping_hash=fingerprint(mapping), message=str(exc))
        else:
            report = Report(backend=backend.name, backend_version="unknown", status="skipped",
                            mapping_hash=fingerprint(mapping) if mapping is not None else "unavailable",
                            message="Mapping validation failed; backend was not invoked")
        trial["report"] = report.model_dump()
        write_json(directory / "report.json", trial["report"])
        write_json(directory / "trial.json", trial)
        history.append(trial)
        # Each trial is durable before starting the next proposal.
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(trial, allow_nan=False) + "\n")
    summary = {"schema_version": "0.1", "backend": backend.name,
               "status": "ok" if best else "no_valid_result",
               "best_trial_id": best["trial_id"] if best else None,
               "best_objective": best["objective"] if best else None,
               "trials": [{"trial_id": t["trial_id"], "valid": t["validation"]["valid"],
                           "status": t["report"]["status"], "objective": t["objective"]} for t in history]}
    if best:
        write_yaml(output / "best_mapping.yaml", best["mapping"])
    write_json(output / "summary.json", summary)
    return summary
