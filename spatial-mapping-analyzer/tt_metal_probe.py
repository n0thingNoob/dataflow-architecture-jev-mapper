"""Optional TT-Metal backend for validating Mapping IR -> Tensix placement."""
import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from backend_contract import BackendCapabilities, ObservedExecutionIdentity
from report import MeasurementContext, Metric, Objective, Report, unsupported_metrics
from specs import fingerprint, write_json
from validator import validate_inputs, validate_mapping

ROOT = Path(__file__).resolve().parent
DEFAULT_TTSIM_LIBRARY = ROOT.parent / "third_party/ttsim/src/_out/release_wh/libttsim.so"
SOC_DESCRIPTOR_RELATIVE = Path("tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml")


def profiler_csv_durations(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        field = "DEVICE KERNEL DURATION [ns]"
        if field not in (reader.fieldnames or []):
            raise ValueError(f"Profiler report is missing {field}")
        values = []
        for row in reader:
            raw = (row.get(field) or "").strip()
            if raw:
                values.append(int(raw))
    if not values:
        raise ValueError("Profiler report contains no device kernel duration")
    return values


def logical_core_id_to_tt_metal_logical_core(architecture, logical_core):
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
    if any(
        tensor.dtype != "bfloat16" or tensor.shape != [32, 32]
        for tensor in tensors
    ):
        raise ValueError("Tensix probe supports one 32x32 BF16 tile per tensor")

    return logical_core_id_to_tt_metal_logical_core(
        architecture, region.placement[0]
    )


class TTMetalProbeBackend:
    name = "tt-metal-tensix-probe"
    backend_version = "bf16-add-v3"
    measurement_version = "tt-metal-device-kernel-duration-v2"
    compute_path = "Tensix via TT-Metal"
    probe_label = "Tensix probe"
    search_capabilities = BackendCapabilities(
        topological_order=False,
        execution_policy=False,
        placement=True,
    )

    def candidate_execution_signature(self, architecture, program, mapping):
        core_x, core_y = check_probe_supported(architecture, program, mapping)
        return self.name, core_x, core_y

    def __init__(
        self,
        tt_metal_home,
        probe_binary,
        library=None,
        timeout_seconds=120,
        runtime="ttsim",
        tt_metal_revision=None,
    ):
        self.tt_metal_home = Path(tt_metal_home).resolve()
        self.probe_binary = Path(probe_binary).resolve()
        self.library = (
            Path(library).resolve() if library else DEFAULT_TTSIM_LIBRARY
        )
        if timeout_seconds <= 0:
            raise ValueError("Probe timeout must be positive")
        if runtime not in {"ttsim", "ttsim-profile", "device"}:
            raise ValueError(
                "Tensix runtime must be 'ttsim', 'ttsim-profile' or 'device'"
            )
        if runtime in {"ttsim-profile", "device"} and not tt_metal_revision:
            raise ValueError(
                "Profiled runtimes require an explicit TT-Metal revision"
            )
        self.timeout_seconds = timeout_seconds
        self.runtime = runtime
        self.tt_metal_revision = tt_metal_revision

    def _failure(self, mapping, code, message, status="error"):
        return Report(
            backend=self.name,
            backend_version=self.backend_version,
            status=status,
            mapping_hash=fingerprint(mapping),
            message=message,
            extensions={"error_code": code, "compute_path": self.compute_path},
        )

    def _prepare_simulator_directory(self, workdir):
        descriptor = self.tt_metal_home / SOC_DESCRIPTOR_RELATIVE
        for path, label in [
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

    def _prepare_runtime_environment(self, workdir):
        for path, label in [
            (self.tt_metal_home, "TT_METAL_HOME"),
            (self.probe_binary, "Tensix probe binary"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

        probe_sha256 = hashlib.sha256(self.probe_binary.read_bytes()).hexdigest()
        env = os.environ.copy()
        env.update(
            {
                "TT_METAL_HOME": str(self.tt_metal_home),
                "TT_METAL_FORCE_JIT_COMPILE": "1",
                "TT_METAL_DISABLE_SFPLOADMACRO": "1",
            }
        )

        simulator_library = None
        if self.runtime in {"ttsim", "ttsim-profile"}:
            simulator_library = self._prepare_simulator_directory(workdir)
            env.update(
                {
                    "TT_METAL_SIMULATOR": str(simulator_library),
                    "TT_METAL_SIMULATOR_HOME": str(simulator_library.parent),
                    "TT_METAL_SLOW_DISPATCH_MODE": "1",
                }
            )
        else:
            for name in (
                "TT_METAL_SIMULATOR",
                "TT_METAL_SIMULATOR_HOME",
                "TT_METAL_SLOW_DISPATCH_MODE",
            ):
                env.pop(name, None)

        if self.runtime == "ttsim":
            for name in (
                "SPATIAL_MEASURE_DEVICE",
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_PROFILER_MID_RUN_DUMP",
                "TT_METAL_PROFILER_CPP_POST_PROCESS",
                "TT_METAL_PROFILER_DIR",
            ):
                env.pop(name, None)
            provenance = {
                "runtime": "ttsim",
                "probe_binary_sha256": probe_sha256,
                "ttsim_sha256": hashlib.sha256(
                    simulator_library.read_bytes()
                ).hexdigest(),
            }
        else:
            profiler_dir = workdir / "tt_metal_profiler"
            profiler_dir.mkdir()
            profiler_config = {
                "TT_METAL_DEVICE_PROFILER": "1",
                "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
                "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
            }
            env.update(
                {
                    "SPATIAL_MEASURE_DEVICE": "1",
                    "TT_METAL_PROFILER_DIR": str(profiler_dir),
                    **profiler_config,
                }
            )
            provenance = {
                "runtime": self.runtime,
                "measurement_source": (
                    "ttsim_device_profiler"
                    if self.runtime == "ttsim-profile"
                    else "tt_metal_device_profiler"
                ),
                "tt_metal_revision": self.tt_metal_revision,
                "probe_binary_sha256": probe_sha256,
                "profiler_configuration": profiler_config,
                "profiler_report": str(
                    profiler_dir / ".logs" / "cpp_device_perf_report.csv"
                ),
            }
            if simulator_library is not None:
                provenance["ttsim_sha256"] = hashlib.sha256(
                    simulator_library.read_bytes()
                ).hexdigest()

        return env, provenance

    def _profile_result(self, mapping, result, provenance, artifacts=None):
        duration = result.get("device_kernel_duration_ns")
        if (
            result.get("measurement_source") != "tt_metal_device_profiler"
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or duration <= 0
        ):
            return None, None, self._failure(
                mapping,
                "PROBE_RESULT_MISMATCH",
                "Profiled result is missing a valid TT-Metal profiler duration",
            )

        profiler_report = Path(provenance["profiler_report"])
        try:
            csv_durations = profiler_csv_durations(profiler_report)
        except (OSError, ValueError) as exc:
            return None, None, self._failure(
                mapping, "PROFILER_CROSS_CHECK_FAILED", str(exc)
            )
        if set(csv_durations) != {int(duration)}:
            return None, None, self._failure(
                mapping,
                "PROFILER_CROSS_CHECK_FAILED",
                "Profiler API duration does not match cpp_device_perf_report.csv",
            )

        is_device = self.runtime == "device"
        objective = Objective(
            name=(
                "device_kernel_duration"
                if is_device
                else "ttsim_profile_kernel_duration"
            ),
            value=float(duration),
            unit="ns",
            source="measured" if is_device else "estimated",
        )
        measurement_context = None
        if is_device:
            measurement_context = MeasurementContext(
                measurement_version=self.measurement_version,
                runtime="device",
                source="tt_metal_device_profiler",
                analysis="DEVICE KERNEL DURATION [ns]",
                implementation_revision=provenance["tt_metal_revision"],
                executable_sha256=provenance["probe_binary_sha256"],
                artifacts=artifacts or {},
                configuration={
                    **provenance["profiler_configuration"],
                    "cross_check": "cpp_device_perf_report.csv",
                },
            )
        return objective, measurement_context, None

    def _run_probe(
        self,
        mapping,
        workdir,
        command,
        result_path,
        artifact_prefix,
        invocation,
    ):
        try:
            env, provenance = self._prepare_runtime_environment(workdir)
        except OSError as exc:
            return None, None, self._failure(
                mapping, "PROBE_ENVIRONMENT", str(exc)
            )

        write_json(
            workdir / f"{artifact_prefix}_invocation.json",
            {
                "argv": command,
                "cwd": str(self.tt_metal_home),
                **invocation,
                **provenance,
            },
        )

        try:
            with (workdir / f"{artifact_prefix}_stdout.log").open("w") as stdout, (
                workdir / f"{artifact_prefix}_stderr.log"
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
                return None, None, self._failure(
                    mapping,
                    "PROBE_FAILED",
                    f"{self.probe_label} exited with {process.returncode}",
                )
            return json.loads(result_path.read_text()), provenance, None
        except subprocess.TimeoutExpired:
            return None, None, self._failure(
                mapping,
                "PROBE_TIMEOUT",
                f"{self.probe_label} exceeded {self.timeout_seconds:g} seconds",
            )
        except (OSError, ValueError) as exc:
            return None, None, self._failure(
                mapping, "PROBE_RESULT_ERROR", str(exc)
            )

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

        tt_metal_logical_core = [core_x, core_y]
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
        result, provenance, failure = self._run_probe(
            mapping,
            workdir,
            command,
            result_path,
            "tensix_probe",
            {
                "mapping_logical_core_id": mapping.regions[0].placement[0],
                "tt_metal_logical_core": tt_metal_logical_core,
            },
        )
        if failure:
            return failure

        if (
            not isinstance(result, dict)
            or result.get("tt_metal_logical_core") != tt_metal_logical_core
            or not isinstance(result.get("worker_core"), list)
            or len(result["worker_core"]) != 2
            or result.get("elements") != 1024
            or type(result.get("passed")) is not bool
        ):
            return self._failure(
                mapping,
                "PROBE_RESULT_MISMATCH",
                "Probe result does not match requested mapping",
            )

        passed = result["passed"]
        observed_execution = ObservedExecutionIdentity(
            kind="tt-metal-worker-core-v1",
            value={
                "mapping_logical_core_id": mapping.regions[0].placement[0],
                "tt_metal_logical_core": tt_metal_logical_core,
                "worker_core": result["worker_core"],
            },
        )
        objective = None
        measurement_context = None
        metrics = unsupported_metrics("This runtime does not expose this metric")
        if self.runtime in {"ttsim-profile", "device"}:
            objective, measurement_context, failure = self._profile_result(
                mapping, result, provenance
            )
            if failure:
                return failure
            metrics["latency"] = Metric(
                value=objective.value,
                unit=objective.unit,
                status="available",
            )
        return Report(
            backend=self.name,
            backend_version=self.backend_version,
            status="ok" if passed else "error",
            mapping_hash=fingerprint(mapping),
            correctness="passed" if passed else "failed",
            objective=objective if passed else None,
            observed_execution=observed_execution,
            measurement_context=measurement_context if passed else None,
            metrics=metrics,
            message=(
                f"BF16 add executed on TT-Metal logical core {(core_x, core_y)}"
                if passed
                else "Tensix BF16 add result mismatch"
            ),
            extensions={
                "compute_path": "Tensix UNPACK/MATH/PACK via TT-Metal",
                "transfers": "TT-Metal DRAM and circular buffers",
                "runtime": self.runtime,
                "measurement_source": (
                    provenance.get("measurement_source", "unavailable")
                    if objective is not None
                    else "unavailable"
                ),
                "mapping_logical_core_id": mapping.regions[0].placement[0],
                "tt_metal_logical_core": tt_metal_logical_core,
                "worker_core": result["worker_core"],
                "runtime_provenance": provenance,
                "probe_result": result,
            },
        )
