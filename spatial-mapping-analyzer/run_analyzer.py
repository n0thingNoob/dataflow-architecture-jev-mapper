"""CLI: load inputs, select the fixed mapping policy, run the loop."""
import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

import yaml
from pydantic import ValidationError

from analyzer import PassthroughAnalyzer
from pipeline import run
from specs import Architecture, Program, read_yaml
from tt_sim import ROOT, TTSimBackend
from validator import InvalidInput, error, schema_errors


def main():
    parser = argparse.ArgumentParser(description="Run a tensor/scalar DAG on real TT-Sim")
    parser.add_argument("--arch", type=Path, default=ROOT / "examples/wormhole.yaml")
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, help="JSON: input name -> flat int32 array; otherwise generate seeded inputs")
    parser.add_argument("--parallel", action="store_true", help="Start independent regions together")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tt-sim-library", type=Path)
    parser.add_argument("--tt-sim-timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, help="A new directory; existing results are never overwritten")
    args = parser.parse_args()
    try:
        architecture = Architecture.model_validate(read_yaml(args.arch))
        program = Program.model_validate(read_yaml(args.program))
        inputs = json.loads(args.inputs.read_text()) if args.inputs else None
        backend = TTSimBackend(args.tt_sim_library, args.tt_sim_timeout, inputs, args.seed)
        output = args.output or ROOT / "results" / f"run-{uuid4().hex[:12]}"
        summary = run(architecture, program, PassthroughAnalyzer(args.parallel), backend, args.iterations, output)
    except InvalidInput as exc:
        errors = exc.errors
    except ValidationError as exc:
        errors = schema_errors(exc)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        errors = [error("RUN_ERROR", "run", str(exc))]
    else:
        print(f"Results: {output}")
        for trial in summary["trials"]:
            print(f"{trial['trial_id']}  {trial['status']}")
        print(f"Best: {summary['best_trial_id'] or 'none'} (constant synthetic cost; no hardware performance claim)")
        return 0 if summary["status"] == "ok" else 2
    print(json.dumps({"status": "error", "errors": errors}), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
