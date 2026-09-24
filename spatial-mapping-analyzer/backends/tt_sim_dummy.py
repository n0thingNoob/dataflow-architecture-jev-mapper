"""Fixed RV32I workload used to verify the real Wormhole simulator transport.

This is deliberately not a lowering of matmul/relu/add from the program DAG.
The instructions read two uint32 inputs, add them, store the result and signal
completion. No cross compiler is needed for these eight fixed instructions.
"""
import hashlib
import json
import struct

from specs.io import fingerprint, write_json

WORKLOAD = "wormhole_brisc_add_u32_v1"
CORE = {"x": 1, "y": 1, "processor": "BRISC"}
DATA_ADDRESS = 0x1000
PENDING = 0xFFFFFFFF
INSTRUCTIONS = [
    0x000012B7,  # lui  t0, 0x1       ; input/output block at 0x1000
    0x0002A303,  # lw   t1, 0(t0)     ; lhs
    0x0042A383,  # lw   t2, 4(t0)     ; rhs
    0x00730E33,  # add  t3, t1, t2
    0x01C2A423,  # sw   t3, 8(t0)     ; result, initially PENDING
    0x00100E93,  # addi t4, zero, 1
    0x01D2A623,  # sw   t4, 12(t0)    ; completion flag, initially 0
    0x0000006F,  # jal  zero, 0       ; park until host asserts reset
]


def generate_dummy(architecture, program, mapping, directory):
    lhs = len(mapping.regions)
    rhs = sum(region.cores for region in mapping.regions)
    if max(lhs, rhs) > 0xFFFFFFFF:
        raise ValueError("Dummy inputs must fit uint32")
    firmware = struct.pack("<8I", *INSTRUCTIONS)
    data = struct.pack("<4I", lhs, rhs, PENDING, 0)
    (directory / "dummy.bin").write_bytes(firmware)
    (directory / "dummy_data.bin").write_bytes(data)
    manifest = {
        "schema_version": "0.1", "workload": WORKLOAD,
        "architecture_hash": fingerprint(architecture), "program_hash": fingerprint(program),
        "mapping_hash": fingerprint(mapping), "program_dag_executed": False,
        "core": CORE,
        "firmware": "dummy.bin", "firmware_sha256": hashlib.sha256(firmware).hexdigest(),
        "data": "dummy_data.bin", "data_sha256": hashlib.sha256(data).hexdigest(),
        "inputs": {"lhs": lhs, "rhs": rhs},
        "input_meaning": "region_count + total_allocated_cores (transport smoke test only)",
        "expected_result": (lhs + rhs) & 0xFFFFFFFF,
        "max_clock_steps": 256,
    }
    write_json(directory / "dummy.json", manifest)
    return manifest


def load_dummy(path):
    """Validate the same file contract emitted by generate_dummy before touching the device."""
    manifest = json.loads(path.read_text())
    if manifest["workload"] != WORKLOAD or manifest["core"] != CORE:
        raise ValueError("Unsupported dummy workload/core")
    if manifest["firmware"] != "dummy.bin" or manifest["data"] != "dummy_data.bin":
        raise ValueError("Dummy file names must match the fixed workload contract")
    firmware = (path.parent / "dummy.bin").read_bytes()
    data = (path.parent / "dummy_data.bin").read_bytes()
    for name, content, size in [("firmware", firmware, 32), ("data", data, 16)]:
        if len(content) != size or hashlib.sha256(content).hexdigest() != manifest[f"{name}_sha256"]:
            raise ValueError(f"Dummy {name} size/checksum mismatch")
    lhs, rhs, output, done = struct.unpack("<4I", data)
    if output != PENDING or done != 0:
        raise ValueError("Dummy output must start pending with completion cleared")
    if manifest["inputs"] != {"lhs": lhs, "rhs": rhs} or manifest["expected_result"] != ((lhs + rhs) & 0xFFFFFFFF):
        raise ValueError("Manifest inputs/result do not match dummy data")
    budget = manifest["max_clock_steps"]
    if type(budget) is not int or not 1 <= budget <= 10000:
        raise ValueError("Invalid simulator step budget")
    return manifest, firmware, data


def check_result(result, manifest, library_hash):
    """Bind the child result to this workload and verify its computed output."""
    expected = {key: manifest[key] for key in
                ("workload", "mapping_hash", "firmware_sha256", "data_sha256", "inputs", "expected_result")}
    expected.update(status="ok", library_sha256=library_hash, actual_result=manifest["expected_result"],
                    completion_flag=1, dummy_correctness="passed", program_dag_executed=False)
    if not isinstance(result, dict) or any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("Runner result failed identity or dummy-output checks")
