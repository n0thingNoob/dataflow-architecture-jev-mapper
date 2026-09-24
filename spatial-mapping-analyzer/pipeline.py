"""The loop owns validation, trial persistence and ranking; plugins own decisions/execution."""
import json
from copy import deepcopy
from pathlib import Path

from pydantic import ValidationError

from analyzer.base import Analyzer
from backends.base import Backend, Report
from mapping_ir.models import Mapping
from specs.io import fingerprint, write_json, write_yaml
from specs.models import Architecture, Program
from validator.checks import InvalidInput, error, schema_errors, validate_inputs, validate_mapping


def _propose(architecture, program, analyzer, history):
    """Isolate plugin errors; keep structurally invalid JSON proposals for review."""
    raw = None
    try:
        proposal = analyzer.propose_mapping(
            architecture.model_copy(deep=True), program.model_copy(deep=True), deepcopy(history))
        raw = proposal.model_dump() if isinstance(proposal, Mapping) else proposal
        json.dumps(raw, allow_nan=False)  # Trial history must remain serializable.
        mapping = Mapping.model_validate(raw)
        return raw, mapping, validate_mapping(architecture, program, mapping)
    except ValidationError as exc:
        return raw, None, schema_errors(exc)
    except Exception as exc:
        return None, None, [error("ANALYZER_ERROR", "analyzer", str(exc))]


def _execute(architecture, program, mapping, backend, directory, errors, objective_key):
    report_fields = {"backend": backend.name, "backend_version": "unknown",
                     "mapping_hash": fingerprint(mapping) if mapping is not None else "unavailable"}
    if errors:
        return Report(**report_fields, status="skipped", message="Mapping validation failed; backend was not invoked")
    try:
        result = backend.run(architecture.model_copy(deep=True), program.model_copy(deep=True),
                             mapping.model_copy(deep=True), directory)
        report = Report.model_validate(result.model_dump() if isinstance(result, Report) else result)
        if report.mapping_hash != report_fields["mapping_hash"] or report.backend != backend.name:
            raise ValueError("Backend report identity does not match this trial")
        if report.objective and objective_key is not None and report.objective_key() != objective_key:
            raise ValueError("Cannot compare different objective definitions in one run")
        return report
    except Exception as exc:
        return Report(**report_fields, status="error", message=str(exc))


def run(architecture: Architecture, program: Program, analyzer: Analyzer, backend: Backend,
        iterations: int, output: Path) -> dict:
    errors = validate_inputs(architecture, program)
    if iterations < 1:
        errors.append(error("ITERATIONS", "iterations", "Must be positive"))
    if errors:
        raise InvalidInput(errors)
    output.mkdir(parents=True, exist_ok=False)  # Refuse stale results from an earlier run.
    history, best, objective_key = [], None, None
    for index in range(iterations):
        trial_id = f"trial_{index:04d}"
        directory = output / trial_id
        directory.mkdir()
        inputs = {"architecture": architecture.model_dump(), "program": program.model_dump()}
        write_yaml(directory / "arch.yaml", inputs["architecture"])
        write_yaml(directory / "program.yaml", inputs["program"])
        raw, mapping, errors = _propose(architecture, program, analyzer, history)
        if raw is not None:
            write_yaml(directory / "mapping.yaml", raw)
        validation = {"valid": not errors, "errors": errors}
        write_json(directory / "validation.json", validation)
        report = _execute(architecture, program, mapping, backend, directory, errors, objective_key)
        objective = report.objective
        trial = {"trial_id": trial_id, **inputs, "mapping": raw, "validation": validation,
                 "report": report.model_dump(), "objective": objective.model_dump() if objective else None,
                 "measured_cost": objective.value if objective and objective.source == "measured" else None,
                 "feedback_trial_ids": [t["trial_id"] for t in history]}
        if objective:
            objective_key = report.objective_key()
            if best is None or objective.value < best["objective"]["value"]:
                best = trial
        write_json(directory / "report.json", trial["report"])
        write_json(directory / "trial.json", trial)
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(trial, allow_nan=False) + "\n")
        history.append(trial)
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
