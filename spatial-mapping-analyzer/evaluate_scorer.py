"""Evaluate the heuristic scorer on comparable measured mappings."""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from dataset import content_hash
from dataset_split import load_records
from scorer import HeuristicScorer
from specs import fingerprint

TIE_REL_TOL = 1e-12
TIE_ABS_TOL = 1e-12


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _measurement_key(record):
    return (
        record.measurement.name,
        record.measurement.unit,
        record.measurement.source,
        _canonical_json(
            record.measurement_context.model_dump(mode="json")
        ),
    )


def aggregate_executions(records):
    """Median repeated observations without crossing measurement contexts."""
    groups = defaultdict(list)
    representatives = {}
    mapping_hashes = defaultdict(set)
    for record in records:
        key = (record.execution_group_id, _measurement_key(record))
        groups[key].append(record.measurement.value)
        mapping_hashes[key].add(record.mapping_hash)
        representatives.setdefault(key, record)

    ambiguous = [
        key for key, hashes in mapping_hashes.items()
        if len(hashes) != 1
    ]
    if ambiguous:
        raise ValueError(
            "Execution group contains multiple Mapping IR identities"
        )

    aggregated = [
        (representatives[key], statistics.median(values))
        for key, values in groups.items()
    ]
    return sorted(
        aggregated,
        key=lambda item: (
            item[0].program_group_id,
            fingerprint(item[0].architecture),
            item[0].execution_group_id,
            _measurement_key(item[0]),
        ),
    )


def _comparison_key(record):
    return (
        record.program_group_id,
        fingerprint(record.architecture),
        *_measurement_key(record),
    )


def _relative_regret(latency, oracle_latency):
    return (latency - oracle_latency) / oracle_latency


def _top_score_indices(scores):
    if not scores:
        raise ValueError("Cannot evaluate an empty candidate set")
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


def evaluate(records, scorer, split_manifest=None, split=None):
    records = list(records)
    if split is not None and split_manifest is None:
        raise ValueError("A named split requires a split manifest")

    allowed = None
    if split_manifest is not None:
        manifest_groups = split_manifest.get("groups")
        if not isinstance(manifest_groups, dict):
            raise ValueError("Split manifest is missing groups")
        dataset_groups = {record.program_group_id for record in records}
        missing_groups = dataset_groups - set(manifest_groups)
        if missing_groups:
            raise ValueError(
                "Split manifest does not cover every dataset program group"
            )
        allowed = {
            group_id
            for group_id, assigned in manifest_groups.items()
            if split is None or assigned == split
        }

    by_case = defaultdict(list)
    for record, measured in aggregate_executions(records):
        if allowed is not None and record.program_group_id not in allowed:
            continue
        by_case[_comparison_key(record)].append((record, measured))

    cases = []
    for comparison_key, unsorted_rows in sorted(by_case.items()):
        rows = sorted(
            unsorted_rows,
            key=lambda row: (
                row[0].execution_group_id,
                row[0].mapping_hash,
            ),
        )
        architecture = rows[0][0].architecture
        program = rows[0][0].program
        candidates = [record.mapping for record, _ in rows]
        measured = [value for _, value in rows]
        scores = scorer.score_candidates(architecture, program, candidates)
        if len(scores) != len(rows):
            raise ValueError("Scorer must return one score per candidate")

        top_indices = _top_score_indices(scores)
        selected = min(
            top_indices,
            key=lambda index: (
                rows[index][0].execution_group_id,
                rows[index][0].mapping_hash,
            ),
        )
        oracle = min(
            range(len(rows)),
            key=lambda index: (
                measured[index],
                rows[index][0].execution_group_id,
            ),
        )
        top_best = min(
            top_indices,
            key=lambda index: (
                measured[index],
                rows[index][0].execution_group_id,
            ),
        )
        top_worst = max(
            top_indices,
            key=lambda index: (
                measured[index],
                rows[index][0].execution_group_id,
            ),
        )

        oracle_latency = measured[oracle]
        selected_latency = measured[selected]
        top_best_latency = measured[top_best]
        top_worst_latency = measured[top_worst]
        evaluation_group_id = "evaluation_" + content_hash(
            {
                "program_group_id": comparison_key[0],
                "architecture_hash": comparison_key[1],
                "measurement": comparison_key[2:5],
                "measurement_context": comparison_key[5],
            }
        )
        cases.append(
            {
                "evaluation_group_id": evaluation_group_id,
                "program_group_id": rows[0][0].program_group_id,
                "architecture_hash": fingerprint(architecture),
                "candidate_count": len(rows),
                "top_score": float(min(scores)),
                "top_score_tie_count": len(top_indices),
                "selected_execution_group_id": (
                    rows[selected][0].execution_group_id
                ),
                "oracle_execution_group_id": rows[oracle][0].execution_group_id,
                "selected_latency": selected_latency,
                "oracle_latency": oracle_latency,
                "selected_relative_regret": _relative_regret(
                    selected_latency, oracle_latency
                ),
                "top_tie_best_latency": top_best_latency,
                "top_tie_worst_latency": top_worst_latency,
                "top_tie_regret_min": _relative_regret(
                    top_best_latency, oracle_latency
                ),
                "top_tie_regret_max": _relative_regret(
                    top_worst_latency, oracle_latency
                ),
            }
        )

    selected_regrets = [
        case["selected_relative_regret"] for case in cases
    ]
    tie_min_regrets = [case["top_tie_regret_min"] for case in cases]
    tie_max_regrets = [case["top_tie_regret_max"] for case in cases]
    return {
        "schema_version": "0.1",
        "scorer": getattr(scorer, "name", type(scorer).__name__),
        "split": split,
        "evaluation_group_count": len(cases),
        "mean_selected_relative_regret": (
            statistics.fmean(selected_regrets)
            if selected_regrets
            else None
        ),
        "median_selected_relative_regret": (
            statistics.median(selected_regrets)
            if selected_regrets
            else None
        ),
        "mean_top_tie_regret_min": (
            statistics.fmean(tie_min_regrets)
            if tie_min_regrets
            else None
        ),
        "mean_top_tie_regret_max": (
            statistics.fmean(tie_max_regrets)
            if tie_max_regrets
            else None
        ),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a heuristic scorer against measured device latency"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.split and args.split_manifest is None:
        parser.error("--split requires --split-manifest")
    manifest = (
        json.loads(args.split_manifest.read_text())
        if args.split_manifest
        else None
    )
    result = evaluate(
        load_records(args.dataset),
        HeuristicScorer(),
        split_manifest=manifest,
        split=args.split,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
