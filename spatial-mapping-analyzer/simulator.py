"""Isolated process for the upstream Wormhole libttsim C ABI.

Register/TLB definitions are tied to the pinned upstream source. The runner is a
correctness harness: API clock steps are diagnostic only, not accelerator cycles.
"""
import argparse
import ctypes
import hashlib
import json
import struct
import sys
from pathlib import Path

from kernels import A, B, DONE, OUTPUT

RESET_REGISTER = 0xFFB121B0
ALL_RESET = 0x47800
BRISC_RESET = 0x800
TLB_CONFIG_OFFSET = 0x1FC00000
WINDOW_MASK = 0xFFFFF


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Wormhole:
    def __init__(self, library):
        self.core = (1, 1)
        self.lib = ctypes.CDLL(str(library))
        signatures = {
            "libttsim_init": ([], None),
            "libttsim_exit": ([], None),
            "libttsim_clock": ([ctypes.c_uint32], None),
            "libttsim_pci_config_rd32": ([ctypes.c_uint32, ctypes.c_uint32], ctypes.c_uint32),
            "libttsim_pci_mem_wr_bytes": ([ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint32], None),
            "libttsim_pci_mem_rd_bytes": ([ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint32], None),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result

    def initialize(self):
        self.lib.libttsim_init()
        self.device_id = self.lib.libttsim_pci_config_rd32(0, 0)
        if self.device_id != 0x401E1E52:
            raise ValueError(f"Expected a Wormhole library, found PCI ID {self.device_id:#x}")
        low = self.lib.libttsim_pci_config_rd32(0, 0x10)
        high = self.lib.libttsim_pci_config_rd32(0, 0x14)
        self.bar0 = (high << 32) | (low & ~0xF)

    def pci_write(self, address, data):
        buffer = ctypes.create_string_buffer(data)
        self.lib.libttsim_pci_mem_wr_bytes(address, buffer, len(data))

    def select_window(self, address):
        coordinate = self.core[0] | (self.core[1] << 6)
        config = (address >> 20) | (coordinate << 16)
        self.pci_write(self.bar0 + TLB_CONFIG_OFFSET, struct.pack("<Q", config))
        return self.bar0 + (address & WINDOW_MASK)

    def write(self, address, data):
        self.pci_write(self.select_window(address), data)

    def write32(self, address, value):
        self.write(address, struct.pack("<I", value))

    def read(self, address, size):
        physical = self.select_window(address)
        buffer = ctypes.create_string_buffer(size)
        self.lib.libttsim_pci_mem_rd_bytes(physical, buffer, size)
        return buffer.raw


def pack(values):
    return struct.pack(f"<{len(values)}i", *values)


def prepare(simulator, op, values, directory):
    simulator.core = tuple(op["core"])
    firmware = (directory / op["firmware"]).read_bytes()
    if digest(firmware) != op["firmware_sha256"]:
        raise ValueError("Kernel checksum mismatch")
    simulator.write32(RESET_REGISTER, ALL_RESET)
    writes = [(0, firmware), (DONE, bytes(4)), (OUTPUT, b"\xa5" * (op["elements"] * 4))]
    writes += [(address, pack(values[name])) for address, name in zip((A, B), op["inputs"])]
    for address, data in writes:
        simulator.write(address, data)
        if simulator.read(address, len(data)) != data:
            raise RuntimeError("Device load/readback failed before execution")


def _ready_operations(pending, values):
    return [op for op in pending if all(name in values for name in op["inputs"])]


def _launch(simulator, operations, values, directory, steps, trace_starts):
    for op in operations:
        prepare(simulator, op, values, directory)
    for op in operations:
        simulator.core = tuple(op["core"])
        simulator.write32(RESET_REGISTER, ALL_RESET & ~BRISC_RESET)
        if simulator.read(DONE, 4) != bytes(4):
            raise RuntimeError("Completion changed before simulator clocking")
        trace_starts[op["id"]] = steps


def execute(library, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    pending = list(manifest["operations"])
    values = dict(manifest["inputs"])
    tensors, trace, waves = {}, [], []
    active = []
    trace_starts = {}
    steps = 0
    parallel = manifest["execution_policy"] == "exclusive_cores_dependency_barrier"

    simulator = Wormhole(library)
    simulator.initialize()
    try:
        while pending or active:
            ready = _ready_operations(pending, values)
            if not parallel and active:
                ready = []
            elif not parallel:
                ready = ready[:1]

            if ready:
                _launch(simulator, ready, values, manifest_path.parent, steps, trace_starts)
                waves.append([op["id"] for op in ready])
                active.extend(ready)
                for op in ready:
                    pending.remove(op)

            if not active:
                raise ValueError("No ready operations; invalid DAG or missing inputs")

            quantum = 1 if all(op["elements"] == 1 for op in active) else 32
            count = min(quantum, manifest["max_clock_steps"] - steps)
            if count <= 0:
                raise RuntimeError("Program exhausted simulator step budget")
            simulator.lib.libttsim_clock(count)
            steps += count

            completed = []
            for op in active:
                simulator.core = tuple(op["core"])
                done = struct.unpack("<I", simulator.read(DONE, 4))[0]
                if done == 0:
                    continue
                if done != 1:
                    raise RuntimeError("Invalid completion flag")
                data = simulator.read(OUTPUT, op["elements"] * 4)
                values[op["output"]] = list(struct.unpack(f'<{op["elements"]}i', data))
                tensors[op["output"]] = values[op["output"]]
                simulator.write32(RESET_REGISTER, ALL_RESET)
                trace.append({
                    "op": op["id"],
                    "region": op["region"],
                    "core": op["core"],
                    "start_api_step": trace_starts[op["id"]],
                    "completion_observed_api_step": steps,
                    "poll_interval_steps": quantum,
                    "completion_flag": done,
                })
                completed.append(op)
            for op in completed:
                active.remove(op)

        return {
            "status": "ok",
            "mapping_hash": manifest["mapping_hash"],
            "manifest_sha256": digest(manifest_path.read_bytes()),
            "library_sha256": digest(library.read_bytes()),
            "tensors": tensors,
            "outputs": {name: values[name] for name in manifest["outputs"]},
            "trace": trace,
            "waves": waves,
            "api_steps": steps,
            "timing_scope": "Observed API steps; host transfers excluded; not hardware cycles",
        }
    finally:
        simulator.lib.libttsim_exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = execute(args.library.resolve(strict=True), args.manifest.resolve(strict=True))
        code = 0
    except Exception as exc:
        result = {"status": "error", "message": str(exc)}
        code = 1
    args.result.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
