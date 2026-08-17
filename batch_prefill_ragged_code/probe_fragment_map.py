#!/usr/bin/env python3
"""Build and run the Priority-1 C500 BF16 MMA fragment ownership probe."""
from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "ragged_prefill_stage_fg_fragment_map.cu"
LIB = HERE / "ragged_prefill_stage_fg_fragment_map.so"
OUT = HERE / "stage_fg_fragment_map.txt"


def main() -> int:
    cmd = [
        "mxcc", "-O3", "-std=c++17", "--offload-arch=xcore1000",
        "-I/opt/maca/tools/cu-bridge/include", "-shared", "-fPIC",
        str(SOURCE), "-o", str(LIB),
    ]
    subprocess.run(cmd, check=True, cwd=HERE.parent)
    library = ctypes.CDLL(str(LIB))
    probe = library.run_fragment_map_probe
    probe.argtypes = [ctypes.c_void_p, ctypes.c_int]
    probe.restype = None

    # 64 lanes x 4 BF16 slots form the complete B fragment.  A one-hot B
    # entry produces exactly the 16 C registers belonging to its N column.
    result = torch.empty((256, 64, 4), dtype=torch.float32, device="cuda")
    for idx in range(256):
        probe(ctypes.c_void_p(result[idx].data_ptr()), idx)
    torch.cuda.synchronize()
    cpu = result.cpu()

    groups: dict[tuple[tuple[int, int], ...], list[int]] = {}
    for idx in range(256):
        positions = tuple((int(i), int(j)) for i, j in torch.nonzero(cpu[idx] != 0, as_tuple=False))
        groups.setdefault(positions, []).append(idx)

    lines = [f"probe_groups={len(groups)}"]
    for n, (positions, slots) in enumerate(sorted(groups.items(), key=lambda item: item[1])):
        lines.append(f"group={n} slots={slots} outputs={list(positions)}")
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT)
    print("groups", len(groups), "group_sizes", sorted(len(x) for x in groups.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
