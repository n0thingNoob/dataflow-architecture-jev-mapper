"""The loop owns validation, trial persistence and ranking; plugins own decisions/execution."""
import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from backend_contract import json_safe_signature
from mapping_ir import Mapping
from report import Report
from specs import Architecture, Program, fingerprint, write_json, write_yaml
from validator import InvalidInput, error, schema_errors, validate_inputs, validate_mapping


def _propose(architecture, program, analyzer, history):
    """Isolate plugin errors; keep structurally invalid JSON proposals for review."""
    raw = None
    try:
        proposal = analyzer.propose_mapping(
            architecture.model_copy(deep=True), program.model_copy(deep=True), deepcopy(history))
        raw = proposal.model_dump() if isinstance(proposal, Mapping) else proposal
        json.dumps(raw, allow_nan=False)
        mapping = Mapping.model_validate(raw)
        return raw, mapping, validate_mapping(architecture, program, mapping)
    except ValidationError as exc:
        return raw, None, schema_errors(exc)
    except Exception as exc:
        return None, None, [error("ANALYZER_ERROR", "analyzer", str(exc))]


def _requested_execution_signature(
    architecture, program, mapping, backend, errors
):
    if errors or mapping is None:
        return None, errors
    signature = getattr(backend, "candidate_execution_signature", None)
    if signature is None:
        return None, errors
    try:
        value = signature(
            architecture.model_copy(deep=True),
            program.model_copy(deep=True),
            mapping.model_copy(deep=True),
        )
        return json_safe_signature(value), errors
    except Exception as exc:
        return None, [
            *errors,
            error(
                "EXECUTION_SIGNATURE_ERROR",
                "backend.candidate_execution_signature",
                str(exc),
            ),
        ]


def _execute(architecture, program, mapping, backend, directory, errors, objective_key):
    report_fields = {
        "backend": backend.name,
        "backend_version": getattr(backend, "backend_version", "unknown"),
        "mapping_hash": (
            fingerprint(mapping) if mapping is not None else "unavailable"
        ),
    }
    if errors:
        return Report(**report_fields, status="skipped", message="Mapping validation failed; backend was not invoked")
    try:
        result = backend.run(architecture.model_copy(deep=True), program.model_copy(deep=True),
                             mapping.model_copy(deep=True), directory)
        report = Report.model_validate(result.model_dump() if isinstance(result, Report) else result)
        if (
            report.mapping_hash != report_fields["mapping_hash"]
            or report.backend != backend.name
            or report.backend_version != report_fields["backend_version"]
        ):
            raise ValueError("Backend report identity does not match this trial")
        if report.objective and objective_key is not None and report.objective_key() != objective_key:
            raise ValueError("Cannot compare different objective definitions in one run")
        return report
    except Exception as exc:
        return Report(**report_fields, status="error", message=str(exc))


def run(
    architecture: Architecture,
    program: Program,
    analyzer,
    backend,
    iterations: int,
    output: Path,
    rank_objectives: bool = True,
) -> dict:
    errors = validate_inputs(architecture, program)
    if iterations < 1:
        errors.append(error("ITERATIONS", "iterations", "Must be positive"))
    if errors:
        raise InvalidInput(errors)

    output.mkdir(parents=True, exist_ok=False)
    run_id = f"run_{uuid4().hex}"
    history, best, objective_key = [], None, None
    successful_trials = []

    for index in range(iterations):
        trial_id = f"trial_{index:04d}"
        directory = output / trial_id
        directory.mkdir()
        inputs = {"architecture": architecture.model_dump(), "program": program.model_dump()}
        write_yaml(directory / "arch.yaml", inputs["architecture"])
        write_yaml(directory / "program.yaml", inputs["program"])

        raw, mapping, errors = _propose(
            architecture, program, analyzer, history
        )
        if raw is not None:
            write_yaml(directory / "mapping.yaml", raw)
        requested_execution_signature, errors = _requested_execution_signature(
            architecture, program, mapping, backend, errors
        )
        validation = {"valid": not errors, "errors": errors}
        write_json(directory / "validation.json", validation)

        report = _execute(
            architecture,
            program,
            mapping,
            backend,
            directory,
            errors,
            objective_key,
        )
        objective = report.objective
        trial = {
            "run_id": run_id,
            "trial_id": trial_id,
            **inputs,
            "mapping": raw,
            "requested_execution_signature": requested_execution_signature,
            "validation": validation,
            "report": report.model_dump(),
            "feedback_trial_ids": [t["trial_id"] for t in history],
        }

        if report.status == "ok":
            successful_trials.append(trial_id)
        if objective:
            objective_key = report.objective_key()
            if rank_objectives and (
                best is None
                or objective.value < best["report"]["objective"]["value"]
            ):
                best = trial

        write_json(directory / "report.json", trial["report"])
        write_json(directory / "trial.json", trial)
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(trial, allow_nan=False) + "\n")
        history.append(trial)

    summary = {
        "schema_version": "0.1",
        "run_id": run_id,
        "backend": backend.name,
        "status": "ok" if successful_trials else "no_successful_result",
        "successful_trial_ids": successful_trials,
        "best_trial_id": best["trial_id"] if best else None,
        "best_objective": best["report"]["objective"] if best else None,
        "trials": [
            {
                "trial_id": t["trial_id"],
                "valid": t["validation"]["valid"],
                "status": t["report"]["status"],
                "objective": t["report"]["objective"],
            }
            for t in history
        ],
    }
    if best:
        write_yaml(output / "best_mapping.yaml", best["mapping"])
    write_json(output / "summary.json", summary)
    return summary
