"""Optional TT-Metal backend for validating Mapping IR -> Tensix core placement.

This backend is deliberately correctness-only. It requires an externally built
TT-Metal installation and the small spatial_tensix_probe executable.
"""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from report import Report, unsupported_metrics
from specs import fingerprint, write_json
from validator import validate_inputs, validate_mapping

ROOT = Path(__file__).resolve().parent
DEFAULT_TTSIM_LIBRARY = ROOT.parent / "third_party/ttsim/src/_out/release_wh/libttsim.so"
SOC_DESCRIPTOR_RELATIVE = Path("tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml")


def logical_core_to_coord(architecture, logical_core):
    if logical_core < 0 or logical_core >= architecture.available_cores:
        raise ValueError("Logical core is outside architecture capacity")
    x = logical_core % architecture.grid.cols
    y = logical_core // architecture.grid.cols
    if y >= architecture.grid.rows:
        raise ValueError("Logical core does not fit architecture grid")
    return x, y


def check_probe_supported(architecture, program, mapping):
    errors = validate_inputs(architecture, program) or validate_mapping(
        architecture, program, mapping
    )
    if errors:
        raise ValueError(str(errors))
    if architecture.extensions.get("target_family") != "wormhole":
        raise ValueError("Tensix probe currently supports Wormhole only")
    if len(program.ops) != 1 or program.ops[0].op != "add":
        raise ValueError("Tensix probe supports exactly one add op")
    if len(mapping.regions) != 1 or mapping.regions[0].ops != [program.ops[0].id]:
        raise ValueError("Tensix probe requires one region containing the add op")
    region = mapping.regions[0]
    if region.cores != 1 or region.placement is None or len(region.placement) != 1:
        raise ValueError("Tensix probe requires one explicitly placed core")

    op = program.ops[0]
    tensors = [program.tensors[name] for name in [*op.inputs, op.output]]
    if any(tensor.dtype != "bfloat16" or tensor.shape != [32, 32] for tensor in tensors):
        raise ValueError("Tensix probe supports one 32x32 BF16 tile per tensor")

    return logical_core_to_coord(architecture, region.placement[0])


class TTMetalProbeBackend:
    name = "tt-metal-tensix-probe"

    def __init__(
        self,
        tt_metal_home,
        probe_binary,
        library=None,
        timeout_seconds=120,
    ):
        self.tt_metal_home = Path(tt_metal_home).resolve()
        self.probe_binary = Path(probe_binary).resolve()
        self.library = Path(library).resolve() if library else DEFAULT_TTSIM_LIBRARY
        if timeout_seconds <= 0:
            raise ValueError("Probe timeout must be positive")
        self.timeout_seconds = timeout_seconds

    def _failure(self, mapping, code, message, status="error"):
        return Report(
            backend=self.name,
            backend_version="bf16-add-v1",
            status=status,
            mapping_hash=fingerprint(mapping),
            message=message,
            extensions={"error_code": code, "compute_path": "Tensix via TT-Metal"},
        )

    def _prepare_simulator_directory(self, workdir):
        descriptor = self.tt_metal_home / SOC_DESCRIPTOR_RELATIVE
        for path, label in [
            (self.tt_metal_home, "TT_METAL_HOME"),
            (self.probe_binary, "Tensix probe binary"),
            (self.library, "TT-Sim library"),
            (descriptor, "Wormhole SoC descriptor"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

        simulator_dir = workdir / "tt_metal_simulator"
        simulator_dir.mkdir()
        simulator_library = simulator_dir / "libttsim.so"
        shutil.copy2(self.library, simulator_library)
        shutil.copy2(descriptor, simulator_dir / "soc_descriptor.yaml")
        return simulator_library

    def run(self, architecture, program, mapping, workdir):
        workdir = workdir.resolve()
        try:
            core_x, core_y = check_probe_supported(
                architecture, program, mapping
            )
        except ValueError as exc:
            return self._failure(
                mapping, "UNSUPPORTED_PROGRAM", str(exc), status="unsupported"
            )

        try:
            simulator_library = self._prepare_simulator_directory(workdir)
        except (OSError, FileNotFoundError) as exc:
            return self._failure(mapping, "PROBE_ENVIRONMENT", str(exc))

        result_path = workdir / "tensix_probe_result.json"
        command = [
            str(self.probe_binary),
            "--core-x",
            str(core_x),
            "--core-y",
            str(core_y),
            "--result",
            str(result_path),
        ]
        env = os.environ.copy()
        env.update(
            {
                "TT_METAL_HOME": str(self.tt_metal_home),
                "TT_METAL_SIMULATOR": str(simulator_library),
                "TT_METAL_SLOW_DISPATCH_MODE": "1",
            }
        )
        write_json(
            workdir / "tensix_probe_invocation.json",
            {
                "argv": command,
                "cwd": str(self.tt_metal_home),
                "core": [core_x, core_y],
                "ttsim_sha256": hashlib.sha256(
                    simulator_library.read_bytes()
                ).hexdigest(),
            },
        )

        try:
            with (workdir / "tensix_probe_stdout.log").open("w") as stdout, (
                workdir / "tensix_probe_stderr.log"
            ).open("w") as stderr:
                process = subprocess.run(
                    command,
                    cwd=self.tt_metal_home,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            if process.returncode != 0:
                return self._failure(
                    mapping,
                    "PROBE_FAILED",
                    f"Tensix probe exited with {process.returncode}",
                )
            result = json.loads(result_path.read_text())
        except subprocess.TimeoutExpired:
            return self._failure(
                mapping,
                "PROBE_TIMEOUT",
                f"Tensix probe exceeded {self.timeout_seconds:g} seconds",
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return self._failure(mapping, "PROBE_RESULT_ERROR", str(exc))

        if (
            not isinstance(result, dict)
            or result.get("core") != [core_x, core_y]
            or result.get("elements") != 1024
            or type(result.get("passed")) is not bool
        ):
            return self._failure(
                mapping, "PROBE_RESULT_MISMATCH", "Probe result does not match requested mapping"
            )

        passed = result["passed"]
        return Report(
            backend=self.name,
            backend_version="bf16-add-v1",
            status="ok" if passed else "error",
            mapping_hash=fingerprint(mapping),
            correctness="passed" if passed else "failed",
            objective=None,
            metrics=unsupported_metrics(
                "Placement probe validates Tensix execution but does not expose a timing objective"
            ),
            message=(
                f"BF16 add executed on Tensix core ({core_x}, {core_y})"
                if passed
                else "Tensix BF16 add result mismatch"
            ),
            extensions={
                "compute_path": "Tensix UNPACK/MATH/PACK via TT-Metal",
                "transfers": "TT-Metal DRAM and circular buffers",
                "logical_core": mapping.regions[0].placement[0],
                "physical_core": [core_x, core_y],
                "probe_result": result,
            },
        )
