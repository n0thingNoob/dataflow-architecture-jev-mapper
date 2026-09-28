"""Deterministic leakage-safe train/validation/test splits for measured datasets."""
import argparse
import hashlib
import json
from pathlib import Path

from dataset import MeasuredMappingRecord

SPLITS = ("train", "validation", "test")


def load_records(path):
    records = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(MeasuredMappingRecord.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid dataset record at {path}:{line_number}: {exc}"
                ) from exc
    return records


def _group_sort_key(program_group_id, seed):
    return hashlib.sha256(
        f"{seed}:{program_group_id}".encode()
    ).digest()


def _balanced_split_counts(group_count):
    if group_count < 0:
        raise ValueError("Group count cannot be negative")
    if group_count == 0:
        return 0, 0, 0
    if group_count == 1:
        return 1, 0, 0
    if group_count == 2:
        return 1, 0, 1

    train = min(group_count - 2, max(1, int(group_count * 0.70 + 0.5)))
    remaining = group_count - train
    validation = min(
        remaining - 1,
        max(1, int(group_count * 0.15 + 0.5)),
    )
    test = remaining - validation
    return train, validation, test


def build_split_manifest(records, seed="jev-split-v2"):
    records = list(records)
    program_groups = {
        record.program_group_id
        for record in records
    }
    ordered_groups = sorted(
        program_groups,
        key=lambda group_id: (_group_sort_key(group_id, seed), group_id),
    )
    train_count, validation_count, test_count = _balanced_split_counts(
        len(ordered_groups)
    )

    assignment = {}
    boundaries = (
        ("train", train_count),
        ("validation", validation_count),
        ("test", test_count),
    )
    offset = 0
    for split, count in boundaries:
        for group_id in ordered_groups[offset:offset + count]:
            assignment[group_id] = split
        offset += count

    execution_to_split = {}
    for record in records:
        split = assignment[record.program_group_id]
        previous = execution_to_split.setdefault(
            record.execution_group_id, split
        )
        if previous != split:
            raise ValueError("Execution group would leak across splits")

    return {
        "schema_version": "0.1",
        "strategy": "balanced-program-group-hash-v2",
        "seed": seed,
        "groups": dict(sorted(assignment.items())),
        "counts": {
            split: sum(value == split for value in assignment.values())
            for split in SPLITS
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create deterministic leakage-safe dataset splits"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", default="jev-split-v2")
    args = parser.parse_args()

    manifest = build_split_manifest(load_records(args.dataset), args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
