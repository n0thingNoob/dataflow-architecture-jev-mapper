"""Isolated process for the upstream Wormhole libttsim C ABI.

Register/TLB definitions are tied to the pinned upstream source:
  src/libttsim.cpp: tlb_translate(), pci_mem_wr_cur()
  src/tile.cpp: RISCV_DEBUG_REGS_SOFT_RESET_0 handling
  data/wh/tile_regs.json: reset register addresses and reset mask

Use the supported PCI/BAR interface. The tile-relative ABI deliberately rejects
Wormhole accesses. libttsim may terminate the process on an error, so this module
must always run in a subprocess, never in the analyzer process.
"""
import argparse
import ctypes
import hashlib
import json
import struct
import sys
from pathlib import Path

from backends.tt_sim_dummy import CORE, DATA_ADDRESS, WORKLOAD, load_dummy

RESET_REGISTER = 0xFFB121B0
ALL_RESET = 0x47800
BRISC_RESET = 0x800
TLB_CONFIG_OFFSET = 0x1FC00000
WINDOW_MASK = 0xFFFFF  # Wormhole TLB 0 is a 1 MiB window.


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Wormhole:
    def __init__(self, library):
        self.core = (CORE["x"], CORE["y"])
        self.lib = ctypes.CDLL(str(library))
        signatures = {
            "libttsim_init": ([], None), "libttsim_exit": ([], None),
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
        # Unicast physical NoC coordinate, relaxed ordering, local page.
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


def execute(library, manifest_path):
    manifest, firmware, data = load_dummy(manifest_path)
    lhs, rhs, initial_output, _ = struct.unpack("<4I", data)
    simulator = Wormhole(library)
    simulator.initialize()
    try:
        simulator.write32(RESET_REGISTER, ALL_RESET)
        simulator.write(0, firmware)
        simulator.write(DATA_ADDRESS, data)
        if simulator.read(0, len(firmware)) != firmware or simulator.read(DATA_ADDRESS, len(data)) != data:
            raise RuntimeError("Simulator memory load/readback failed")
        simulator.write32(RESET_REGISTER, ALL_RESET & ~BRISC_RESET)
        # Prove that writes alone did not manufacture the result.
        before = struct.unpack("<4I", simulator.read(DATA_ADDRESS, 16))
        if before != (lhs, rhs, initial_output, 0):
            raise RuntimeError("Output changed before simulator clocking")
        completed = 0
        actual = initial_output
        steps = 0
        for steps in range(1, manifest["max_clock_steps"] + 1):
            simulator.lib.libttsim_clock(1)
            _, _, actual, completed = struct.unpack("<4I", simulator.read(DATA_ADDRESS, 16))
            if completed:
                break
        simulator.write32(RESET_REGISTER, ALL_RESET)
        expected = manifest["expected_result"]
        if completed != 1 or actual != expected:
            raise RuntimeError(f"Dummy execution failed: done={completed}, actual={actual}, expected={expected}, steps={steps}")
        return {
            "schema_version": "0.1", "status": "ok", "workload": WORKLOAD,
            "mapping_hash": manifest["mapping_hash"], "firmware_sha256": digest(firmware),
            "data_sha256": digest(data), "library_sha256": digest(library.read_bytes()),
            "library": str(library), "pci_device_id": f"0x{simulator.device_id:08x}",
            "core": manifest["core"], "inputs": {"lhs": lhs, "rhs": rhs},
            "expected_result": expected, "actual_result": actual, "completion_flag": completed,
            "dummy_correctness": "passed", "program_dag_executed": False,
            "clock_steps_to_completion": steps,
            "clock_steps_note": "API steps for this dummy program; not hardware latency or mapped-DAG cycles",
        }
    finally:
        simulator.lib.libttsim_exit()


def main(execute_fn=execute):
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = execute_fn(args.library.resolve(strict=True), args.manifest.resolve(strict=True))
        code = 0
    except Exception as exc:
        result = {"status": "error", "message": str(exc)}
        code = 1
    args.result.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
