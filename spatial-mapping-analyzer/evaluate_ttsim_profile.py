"""Evaluate HeuristicScorer against TT-Sim profiler estimates."""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from mapping_ir import Mapping
from scorer import HeuristicScorer
from specs import Architecture, Program, fingerprint, write_json

EXPECTED_OBJECTIVE = "ttsim_profile_kernel_duration"
TIE_REL_TOL = 1e-12
TIE_ABS_TOL = 1e-12


def load_profiled_trials(run_dir):
    history = run_dir / "history.jsonl"
    if not history.is_file():
        raise ValueError(f"Missing run history: {history}")

    trials = []
    with history.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                trial = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {history}:{line_number}"
                ) from exc
            report = trial.get("report") or {}
            objective = report.get("objective")
            if report.get("status") != "ok":
                raise ValueError(
                    f"Profile experiment contains non-ok trial at line {line_number}"
                )
            if (
                not isinstance(objective, dict)
                or objective.get("name") != EXPECTED_OBJECTIVE
                or objective.get("source") != "estimated"
                or objective.get("unit") != "ns"
            ):
                raise ValueError(
                    f"Trial at line {line_number} is not a TT-Sim profiler estimate"
                )
            value = objective.get("value")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(
                    f"Trial at line {line_number} has invalid profiler duration"
                )
            trials.append(trial)

    if not trials:
        raise ValueError("Profile experiment contains no trials")
    return trials


def aggregate_profiled_mappings(trials):
    groups = defaultdict(list)
    representatives = {}

    architecture_hash = None
    program_hash = None
    for trial in trials:
        architecture = Architecture.model_validate(trial["architecture"])
        program = Program.model_validate(trial["program"])
        mapping = Mapping.model_validate(trial["mapping"])
        current_arch_hash = fingerprint(architecture)
        current_program_hash = fingerprint(program)
        if architecture_hash is None:
            architecture_hash = current_arch_hash
            program_hash = current_program_hash
        elif (
            architecture_hash != current_arch_hash
            or program_hash != current_program_hash
        ):
            raise ValueError(
                "TT-Sim profile experiment mixes programs or architectures"
            )

        mapping_hash = fingerprint(mapping)
        report_mapping_hash = trial["report"].get("mapping_hash")
        if report_mapping_hash != mapping_hash:
            raise ValueError(
                "TT-Sim profile trial mapping hash does not match report"
            )
        groups[mapping_hash].append(
            float(trial["report"]["objective"]["value"])
        )
        representatives.setdefault(
            mapping_hash,
            (architecture, program, mapping, trial),
        )

    rows = []
    for mapping_hash in sorted(groups):
        architecture, program, mapping, trial = representatives[mapping_hash]
        values = groups[mapping_hash]
        rows.append(
            {
                "mapping_hash": mapping_hash,
                "architecture": architecture,
                "program": program,
                "mapping": mapping,
                "requested_execution_signature": trial.get(
                    "requested_execution_signature"
                ),
                "repeat_count": len(values),
                "profile_values_ns": values,
                "median_profile_ns": float(statistics.median(values)),
                "mad_profile_ns": float(
                    statistics.median(
                        abs(value - statistics.median(values))
                        for value in values
                    )
                ),
            }
        )
    return rows


def _top_score_indices(scores):
    best = min(scores)
    return [
        index
        for index, score in enumerate(scores)
        if math.isclose(
            score,
            best,
            rel_tol=TIE_REL_TOL,
            abs_tol=TIE_ABS_TOL,
        )
    ]


def _relative_regret(value, oracle):
    return (value - oracle) / oracle


def evaluate_profile_run(run_dir, scorer=None):
    scorer = scorer or HeuristicScorer()
    rows = aggregate_profiled_mappings(load_profiled_trials(run_dir))
    architecture = rows[0]["architecture"]
    program = rows[0]["program"]
    mappings = [row["mapping"] for row in rows]
    scores = scorer.score_candidates(architecture, program, mappings)
    if len(scores) != len(rows):
        raise ValueError("Scorer must return one score per mapping")

    for row, score in zip(rows, scores):
        row["heuristic_score"] = float(score)
        row["heuristic_breakdown"] = scorer.score_breakdown(
            architecture,
            program,
            row["mapping"],
        )

    top_indices = _top_score_indices(scores)
    selected = min(
        top_indices,
        key=lambda index: rows[index]["mapping_hash"],
    )
    oracle = min(
        range(len(rows)),
        key=lambda index: (
            rows[index]["median_profile_ns"],
            rows[index]["mapping_hash"],
        ),
    )
    top_best = min(
        top_indices,
        key=lambda index: (
            rows[index]["median_profile_ns"],
            rows[index]["mapping_hash"],
        ),
    )
    top_worst = max(
        top_indices,
        key=lambda index: (
            rows[index]["median_profile_ns"],
            rows[index]["mapping_hash"],
        ),
    )

    oracle_ns = rows[oracle]["median_profile_ns"]
    selected_ns = rows[selected]["median_profile_ns"]
    top_best_ns = rows[top_best]["median_profile_ns"]
    top_worst_ns = rows[top_worst]["median_profile_ns"]

    score_groups = defaultdict(list)
    for row in rows:
        score_groups[row["heuristic_score"]].append(
            row["median_profile_ns"]
        )

    serializable_rows = []
    for row in rows:
        serializable_rows.append(
            {
                "mapping_hash": row["mapping_hash"],
                "placement": [
                    region.placement[0]
                    for region in row["mapping"].regions
                ],
                "requested_execution_signature": (
                    row["requested_execution_signature"]
                ),
                "repeat_count": row["repeat_count"],
                "profile_values_ns": row["profile_values_ns"],
                "median_profile_ns": row["median_profile_ns"],
                "mad_profile_ns": row["mad_profile_ns"],
                "heuristic_score": row["heuristic_score"],
                "heuristic_breakdown": row["heuristic_breakdown"],
            }
        )

    return {
        "schema_version": "0.1",
        "experiment": "heuristic-vs-ttsim-profiler",
        "source_kind": "simulator_estimate",
        "warning": (
            "TT-Sim profiler values are simulator diagnostics and are not "
            "physical Wormhole performance measurements."
        ),
        "scorer": scorer.name,
        "architecture_hash": fingerprint(architecture),
        "program_hash": fingerprint(program),
        "candidate_count": len(rows),
        "repeat_counts": sorted(
            {row["repeat_count"] for row in rows}
        ),
        "top_score": float(min(scores)),
        "top_score_tie_count": len(top_indices),
        "selected_mapping_hash": rows[selected]["mapping_hash"],
        "oracle_mapping_hash": rows[oracle]["mapping_hash"],
        "selected_profile_ns": selected_ns,
        "oracle_profile_ns": oracle_ns,
        "selected_relative_regret": _relative_regret(
            selected_ns, oracle_ns
        ),
        "top_tie_best_profile_ns": top_best_ns,
        "top_tie_worst_profile_ns": top_worst_ns,
        "top_tie_regret_min": _relative_regret(
            top_best_ns, oracle_ns
        ),
        "top_tie_regret_max": _relative_regret(
            top_worst_ns, oracle_ns
        ),
        "score_groups": [
            {
                "heuristic_score": score,
                "candidate_count": len(values),
                "median_profile_ns": float(statistics.median(values)),
                "min_profile_ns": float(min(values)),
                "max_profile_ns": float(max(values)),
            }
            for score, values in sorted(score_groups.items())
        ],
        "candidates": serializable_rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare HeuristicScorer with TT-Sim profiler estimates"
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = evaluate_profile_run(args.run)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
