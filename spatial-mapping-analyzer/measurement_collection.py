"""Randomized repeated measurement collection over effective backend candidates."""
import random
from collections import defaultdict
from statistics import median

from candidate_generator import generate_candidates
from dataset import content_hash
from export_dataset import export_dataset, load_run_records
from pipeline import run
from specs import fingerprint, write_json


class ScheduledAnalyzer:
    """Replay a fixed mapping schedule through the normal pipeline."""

    def __init__(self, mappings):
        if not mappings:
            raise ValueError("Measurement schedule cannot be empty")
        self.mappings = list(mappings)

    def propose_mapping(self, architecture, program, feedback=None):
        index = len(feedback or [])
        if index >= len(self.mappings):
            raise ValueError("Measurement schedule exhausted")
        return self.mappings[index]


def randomized_schedule(candidates, repeats, seed):
    if repeats < 2:
        raise ValueError("Measurement collection requires at least two repeats")
    schedule = [mapping for mapping in candidates for _ in range(repeats)]
    random.Random(seed).shuffle(schedule)
    return schedule


def aggregate_records(records):
    groups = defaultdict(list)
    for record in records:
        groups[record.execution_group_id].append(record)

    aggregates = []
    for execution_group_id in sorted(groups):
        group = groups[execution_group_id]
        definitions = {
            (
                item.measurement.name,
                item.measurement.unit,
                item.measurement.source,
            )
            for item in group
        }
        contexts = {
            content_hash(item.measurement_context.model_dump())
            for item in group
        }
        if len(definitions) != 1 or len(contexts) != 1:
            raise ValueError(
                f"Inconsistent measurements in execution group {execution_group_id}"
            )

        values = [item.measurement.value for item in group]
        center = float(median(values))
        mad = float(median(abs(value - center) for value in values))
        name, unit, source = next(iter(definitions))
        aggregates.append(
            {
                "execution_group_id": execution_group_id,
                "program_group_id": group[0].program_group_id,
                "observation_ids": sorted(
                    item.observation_id for item in group
                ),
                "mapping_hashes": sorted(
                    {item.mapping_hash for item in group}
                ),
                "count": len(values),
                "measurement": {
                    "name": name,
                    "unit": unit,
                    "source": source,
                },
                "measurement_context_hash": (
                    "context_" + next(iter(contexts))
                ),
                "median": center,
                "mad": mad,
                "min": float(min(values)),
                "max": float(max(values)),
            }
        )
    return aggregates


def collect_measurements(
    architecture,
    program,
    backend,
    candidate_limit,
    repeats,
    seed,
    output,
):
    candidates = generate_candidates(
        architecture,
        program,
        limit=candidate_limit,
        capabilities=backend.search_capabilities,
        candidate_execution_signature=backend.candidate_execution_signature,
    )
    if not candidates:
        raise ValueError("Backend produced no measurement candidates")

    schedule = randomized_schedule(candidates, repeats, seed)
    summary = run(
        architecture,
        program,
        ScheduledAnalyzer(schedule),
        backend,
        len(schedule),
        output,
        rank_objectives=False,
    )
    if len(summary["successful_trial_ids"]) != len(schedule):
        raise ValueError(
            "Measurement collection contains failed or skipped trials"
        )

    dataset_path = output / "measured_mappings.jsonl"
    exported = export_dataset([output], dataset_path)
    if exported != len(schedule):
        raise ValueError(
            "Not every successful trial produced a measured dataset record"
        )

    records = load_run_records(output)
    aggregates = aggregate_records(records)
    result = {
        "schema_version": "0.1",
        "run_id": summary["run_id"],
        "seed": seed,
        "repeats": repeats,
        "candidate_count": len(candidates),
        "trial_count": len(schedule),
        "schedule_mapping_hashes": [
            fingerprint(mapping) for mapping in schedule
        ],
        "dataset": dataset_path.name,
        "aggregates": aggregates,
    }
    write_json(output / "measurement_summary.json", result)
    return result
