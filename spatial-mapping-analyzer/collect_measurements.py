"""CLI for randomized repeated physical-device measurement collection."""
import argparse
from pathlib import Path
from uuid import uuid4

from measurement_collection import collect_measurements
from specs import Architecture, Program, read_yaml
from tt_metal_chain import TTMetalChainBackend
from tt_metal_probe import TTMetalProbeBackend
from tt_sim import ROOT


def main():
    parser = argparse.ArgumentParser(
        description="Collect repeated TT-Metal device measurements"
    )
    parser.add_argument(
        "--backend",
        choices=["single-add", "two-add-chain"],
        default="two-add-chain",
    )
    parser.add_argument(
        "--arch",
        type=Path,
        default=ROOT / "examples/wormhole_tensix_probe.yaml",
    )
    parser.add_argument("--program", type=Path)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--tt-metal-home", type=Path, required=True)
    parser.add_argument("--tt-metal-revision", required=True)
    parser.add_argument("--tensix-probe-binary", type=Path)
    parser.add_argument("--tensix-chain-binary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    architecture = Architecture.model_validate(read_yaml(args.arch))
    if args.backend == "single-add":
        if args.tensix_probe_binary is None:
            parser.error("single-add requires --tensix-probe-binary")
        program_path = args.program or ROOT / "examples/bf16_tile_add.yaml"
        candidate_limit = args.candidate_limit or architecture.available_cores
        backend = TTMetalProbeBackend(
            args.tt_metal_home,
            args.tensix_probe_binary,
            timeout_seconds=args.timeout,
            runtime="device",
            tt_metal_revision=args.tt_metal_revision,
        )
    else:
        if args.tensix_chain_binary is None:
            parser.error("two-add-chain requires --tensix-chain-binary")
        program_path = args.program or ROOT / "examples/bf16_two_add_chain.yaml"
        candidate_limit = args.candidate_limit or (
            architecture.available_cores * (architecture.available_cores - 1)
        )
        backend = TTMetalChainBackend(
            args.tt_metal_home,
            args.tensix_chain_binary,
            timeout_seconds=args.timeout,
            runtime="device",
            tt_metal_revision=args.tt_metal_revision,
        )

    program = Program.model_validate(read_yaml(program_path))
    output = args.output or ROOT / "results" / f"measure-{uuid4().hex[:12]}"
    result = collect_measurements(
        architecture,
        program,
        backend,
        candidate_limit,
        args.repeats,
        args.seed,
        output,
    )
    print(f"Results: {output}")
    print(
        f"Collected {result['trial_count']} observations across "
        f"{result['candidate_count']} effective candidates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
