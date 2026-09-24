"""Execute lowered DAG operations on BRISC cores; host forwards completed tensors.

Ready operations start together under dependency barriers. Full tensor transfers
use PCI/BAR reads/writes; no NoC kernel or accelerator performance is modeled here.
"""
import json
import struct

from backends.rv32_kernels import A, B, OUTPUT, DONE
from backends.tt_sim_runner import ALL_RESET, BRISC_RESET, RESET_REGISTER, Wormhole, digest, main


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


def execute(library, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    pending = list(manifest["operations"])
    values = dict(manifest["inputs"])
    tensors, trace, waves, steps = {}, [], [], 0
    simulator = Wormhole(library)
    simulator.initialize()
    try:
        while pending:
            ready = [op for op in pending if all(name in values for name in op["inputs"])]
            if manifest["execution_policy"] == "exclusive_cores_tensor_barrier":
                ready = ready[:1]
            if not ready:
                raise ValueError("No ready operations; invalid DAG or missing inputs")
            for op in ready:
                prepare(simulator, op, values, manifest_path.parent)
            # No clock call occurs between these releases. Each core is initially pending.
            for op in ready:
                simulator.core = tuple(op["core"])
                simulator.write32(RESET_REGISTER, ALL_RESET & ~BRISC_RESET)
                if simulator.read(DONE, 4) != bytes(4):
                    raise RuntimeError("Completion changed before simulator clocking")
            start = steps
            waves.append([op["id"] for op in ready])
            active = list(ready)
            quantum = 1 if all(op["elements"] == 1 for op in ready) else 32
            while active:
                count = min(quantum, manifest["max_clock_steps"] - steps)
                if count <= 0:
                    raise RuntimeError("Program exhausted simulator step budget")
                simulator.lib.libttsim_clock(count)
                steps += count
                for op in list(active):
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
                    trace.append({"op": op["id"], "region": op["region"], "core": op["core"],
                                  "start_api_step": start, "completion_observed_api_step": steps,
                                  "poll_interval_steps": quantum, "completion_flag": done})
                    active.remove(op)
                    pending.remove(op)
        return {"status": "ok", "mapping_hash": manifest["mapping_hash"],
                "manifest_sha256": digest(manifest_path.read_bytes()),
                "library_sha256": digest(library.read_bytes()), "tensors": tensors,
                "outputs": {name: values[name] for name in manifest["outputs"]},
                "trace": trace, "waves": waves, "api_steps": steps,
                "timing_scope": "Observed API steps; host transfers excluded; not hardware cycles"}
    finally:
        simulator.lib.libttsim_exit()


if __name__ == "__main__":
    raise SystemExit(main(execute))
