#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml
from pydantic import ValidationError

from analyzer.passthrough import PassthroughAnalyzer
from backends.mock import MockBackend
from backends.tt_sim import TTSimBackend
from pipeline import run
from specs.io import read_yaml
from specs.models import Architecture, Program
from validator.checks import error, schema_errors, validate_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Standalone passthrough mapping loop; mock by default")
    parser.add_argument("--arch", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--backend", choices=["mock", "tt-sim"], default="mock")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        architecture = Architecture.model_validate(read_yaml(args.arch))
        program = Program.model_validate(read_yaml(args.program))
        errors = validate_inputs(architecture, program)
        if args.iterations < 1:
            errors.append(error("ITERATIONS", "iterations", "Must be positive"))
        if errors:
            print(json.dumps({"status": "invalid_input", "errors": errors}), file=sys.stderr)
            return 2
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = args.output or Path(__file__).parent / "results" / f"{stamp}-{uuid4().hex[:8]}"
        backend = MockBackend() if args.backend == "mock" else TTSimBackend()
        summary = run(architecture, program, PassthroughAnalyzer(), backend, args.iterations, output)
    except ValidationError as exc:
        print(json.dumps({"status": "invalid_input", "errors": schema_errors(exc)}), file=sys.stderr)
        return 2
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "error", "errors": [error("RUN_ERROR", "run", str(exc))]}), file=sys.stderr)
        return 2
    print(f"Backend: {backend.name}; output: {output}")
    for trial in summary["trials"]:
        objective = trial["objective"]
        cost = f"{objective['value']:g} {objective['unit']} ({objective['source']})" if objective else "unavailable"
        print(f"{trial['trial_id']}  {trial['status']}  cost={cost}")
    print(f"Best: {summary['best_trial_id'] or 'none'}")
    if backend.name == "mock":
        print("Passthrough only: identical candidates, constant synthetic score; hardware cycles unavailable.")
    else:
        print("TT-Sim adapter is unsupported in this scaffold; no simulator was invoked.")
    return 0 if summary["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
