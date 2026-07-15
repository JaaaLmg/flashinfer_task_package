#!/usr/bin/env python3
"""Correctness and latency benchmark for the stage-A CUDA MACA baseline.

The shared library is loaded with ctypes so that the exact OJ C ABI is tested.
Reference outputs come from FlashInfer BatchPrefillWithRaggedKVCacheWrapper.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import flashinfer
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "code" / "ragged_prefill_baseline.cu"
LIBRARY = ROOT / "code" / "ragged_prefill_baseline.so"
DEFAULT_RESULT = ROOT / "code" / "stage_a_benchmark_results.csv"

H_Q = 32
H_KV = 4
D_QK = 128
D_VO = 128
RTOL = 1.6e-2
ATOL = 1.6e-2


@dataclass(frozen=True)
class Case:
    case_id: int
    name: str
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        return len(self.q_lens)

    @property
    def seq_len(self) -> int:
        return max(max(self.q_lens), max(self.kv_lens))

    @property
    def total_q(self) -> int:
        return sum(self.q_lens)

    @property
    def total_kv(self) -> int:
        return sum(self.kv_lens)


def cases() -> list[Case]:
    """Shapes from the current problem statement.

    The statement publishes totals/maxima but not every ragged indptr value.
    For ragged cases we construct deterministic positive lengths that exactly
    match the published batch size, totals and maximum lengths.
    """

    return [
        Case(1, "ragged_long", (987,) + (479,) * 11 + (478,) * 21,
             (987,) + (479,) * 11 + (478,) * 21),
        Case(2, "equal_b1_l1024", (1024,), (1024,)),
        Case(3, "equal_b1_l4096", (4096,), (4096,)),
        Case(4, "equal_b1_l16384", (16384,), (16384,)),
        Case(5, "equal_b4_l1024", (1024,) * 4, (1024,) * 4),
        Case(6, "equal_b4_l4096", (4096,) * 4, (4096,) * 4),
        Case(7, "equal_b16_l1024", (1024,) * 16, (1024,) * 16),
        Case(8, "equal_b16_l2048", (2048,) * 16, (2048,) * 16),
        Case(9, "q_lt_kv_uniform", (512,) * 4, (1024,) * 4),
        Case(10, "q_lt_kv_mixed", (640, 384, 320, 192), (1280, 1024, 768, 512)),
        Case(11, "q_lt_kv_two", (512, 512), (2048, 1024)),
        Case(12, "ragged_medium", (873,) + (438,) * 16 + (437,) * 10,
             (873,) + (438,) * 16 + (437,) * 10),
        Case(13, "ragged_short", (123,) + (61,) * 6 + (60,) * 8,
             (123,) + (61,) * 6 + (60,) * 8),
        Case(14, "single_token", (1,), (1,)),
        Case(15, "non_power_of_two", (65, 33), (65, 33)),
    ]


def make_indptr(lengths: Iterable[int], device: torch.device) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return torch.tensor(values, dtype=torch.int32, device=device)


def compile_library(force: bool = False) -> None:
    if LIBRARY.exists() and not force and LIBRARY.stat().st_mtime >= SOURCE.stat().st_mtime:
        return
    cmd = [
        "mxcc",
        "-O3",
        "-std=c++17",
        "--offload-arch=xcore1000",
        "-I/opt/maca/tools/cu-bridge/include",
        "-shared",
        "-fPIC",
        str(SOURCE),
        "-o",
        str(LIBRARY),
    ]
    print("[build]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def load_kernel() -> ctypes._CFuncPtr:
    library = ctypes.CDLL(str(LIBRARY))
    run = library.run_kernel
    run.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int64] * 7
    run.restype = None
    # Keep the CDLL alive for as long as the function object is alive.
    run._library = library  # type: ignore[attr-defined]
    return run


def launch(
    run: ctypes._CFuncPtr,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    case: Case,
) -> None:
    run(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(qo_indptr.data_ptr()),
        ctypes.c_void_p(kv_indptr.data_ptr()),
        case.batch_size,
        case.seq_len,
        H_Q,
        H_KV,
        D_QK,
        D_VO,
        1,
    )


def elapsed_ms(fn, repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def choose_repeats(first_ms: float, max_repeats: int) -> int:
    if first_ms <= 0:
        return 1
    # Target roughly 100 ms of measurements without repeating a slow stage-A
    # case many times.
    return max(1, min(max_repeats, math.ceil(100.0 / first_ms)))


def visible_pairs(case: Case) -> int:
    total = 0
    for q_len, kv_len in zip(case.q_lens, case.kv_lens):
        for q_pos in range(q_len):
            total += min(kv_len, max(0, kv_len - q_len + q_pos + 1))
    return total


def benchmark_case(
    run: ctypes._CFuncPtr,
    case: Case,
    workspace: torch.Tensor,
    seed: int,
    max_repeats: int,
) -> dict[str, object]:
    device = torch.device("cuda")
    torch.manual_seed(seed + case.case_id)
    qo_indptr = make_indptr(case.q_lens, device)
    kv_indptr = make_indptr(case.kv_lens, device)
    q = torch.randn((case.total_q, H_Q, D_QK), dtype=torch.bfloat16, device=device)
    k = torch.randn((case.total_kv, H_KV, D_QK), dtype=torch.bfloat16, device=device)
    v = torch.randn((case.total_kv, H_KV, D_VO), dtype=torch.bfloat16, device=device)
    output = torch.empty((case.total_q, H_Q, D_VO), dtype=torch.bfloat16, device=device)
    reference = torch.empty_like(output)

    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="auto"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        H_Q,
        H_KV,
        D_QK,
        D_VO,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    # Reference and one exact candidate run.  Synchronization also catches
    # asynchronous launch errors before correctness is inspected.
    wrapper.run(q, k, v, out=reference)
    launch(run, q, k, v, output, qo_indptr, kv_indptr, case)
    torch.cuda.synchronize()

    abs_error = (output.float() - reference.float()).abs()
    tolerance = ATOL + RTOL * reference.float().abs()
    finite = bool(torch.isfinite(abs_error).all().item())
    within = abs_error <= tolerance
    match_ratio = float(within.float().mean().item()) if finite else 0.0
    max_abs_error = float(abs_error.max().item()) if finite else float("inf")
    severe = abs_error > (8.0 * tolerance)
    severe_count = int(severe.sum().item()) if finite else output.numel()
    required_ratio = 1.0 if case.case_id in (14, 15) else 0.99
    passed = finite and match_ratio >= required_ratio and severe_count == 0

    first_candidate_ms = elapsed_ms(
        lambda: launch(run, q, k, v, output, qo_indptr, kv_indptr, case), 1
    )
    candidate_repeats = choose_repeats(first_candidate_ms, max_repeats)
    candidate_ms = (
        first_candidate_ms
        if candidate_repeats == 1
        else elapsed_ms(
            lambda: launch(run, q, k, v, output, qo_indptr, kv_indptr, case),
            candidate_repeats,
        )
    )

    # FlashInfer is fast enough to use more repetitions, but keep the same
    # adaptive policy to avoid distorting tiny cases with a fixed huge count.
    first_reference_ms = elapsed_ms(lambda: wrapper.run(q, k, v, out=reference), 1)
    reference_repeats = choose_repeats(first_reference_ms, max(20, max_repeats))
    reference_ms = (
        first_reference_ms
        if reference_repeats == 1
        else elapsed_ms(lambda: wrapper.run(q, k, v, out=reference), reference_repeats)
    )

    pairs = visible_pairs(case)
    flops = 2.0 * pairs * H_Q * (D_QK + D_VO)
    tflops = flops / (candidate_ms * 1.0e9) if candidate_ms > 0 else 0.0
    slowdown = candidate_ms / reference_ms if reference_ms > 0 else float("inf")

    return {
        "case_id": case.case_id,
        "name": case.name,
        "batch_size": case.batch_size,
        "seq_len": case.seq_len,
        "total_q": case.total_q,
        "total_kv": case.total_kv,
        "max_q": max(case.q_lens),
        "max_kv": max(case.kv_lens),
        "visible_pairs": pairs,
        "candidate_ms": candidate_ms,
        "flashinfer_ms": reference_ms,
        "slowdown_vs_flashinfer": slowdown,
        "effective_tflops": tflops,
        "candidate_repeats": candidate_repeats,
        "flashinfer_repeats": reference_repeats,
        "match_ratio": match_ratio,
        "max_abs_error": max_abs_error,
        "severe_error_count": severe_count,
        "passed": passed,
    }


def parse_case_filter(value: str) -> set[int] | None:
    if value.lower() == "all":
        return None
    return {int(item) for item in value.split(",") if item.strip()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="all", help="all or comma-separated case IDs")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--max-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--force-build", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/MACA GPU is required")

    selected_ids = parse_case_filter(args.cases)
    selected = [case for case in cases() if selected_ids is None or case.case_id in selected_ids]
    if not selected:
        raise ValueError("No cases selected")

    compile_library(force=args.force_build)
    run = load_kernel()
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    print(
        f"[env] device={torch.cuda.get_device_name(0)} torch={torch.__version__} "
        f"flashinfer={getattr(flashinfer, '__version__', 'unknown')}",
        flush=True,
    )
    rows: list[dict[str, object]] = []
    for case in selected:
        print(
            f"[case {case.case_id:02d}] {case.name}: B={case.batch_size}, "
            f"total_q={case.total_q}, total_kv={case.total_kv}, max={case.seq_len}",
            flush=True,
        )
        try:
            row = benchmark_case(run, case, workspace, args.seed, args.max_repeats)
        except Exception as exc:
            row = {
                "case_id": case.case_id,
                "name": case.name,
                "batch_size": case.batch_size,
                "seq_len": case.seq_len,
                "total_q": case.total_q,
                "total_kv": case.total_kv,
                "max_q": max(case.q_lens),
                "max_kv": max(case.kv_lens),
                "visible_pairs": visible_pairs(case),
                "candidate_ms": "",
                "flashinfer_ms": "",
                "slowdown_vs_flashinfer": "",
                "effective_tflops": "",
                "candidate_repeats": 0,
                "flashinfer_repeats": 0,
                "match_ratio": 0.0,
                "max_abs_error": "inf",
                "severe_error_count": -1,
                "passed": False,
                "error": repr(exc),
            }
            print(f"  ERROR: {exc!r}", file=sys.stderr, flush=True)
            # A device-side fault can poison the CUDA context; save what was
            # collected and stop instead of reporting misleading later rows.
            rows.append(row)
            break
        rows.append(row)
        print(
            f"  passed={row['passed']} match={row['match_ratio']:.6f} "
            f"max_err={row['max_abs_error']:.6g} candidate={row['candidate_ms']:.3f}ms "
            f"flashinfer={row['flashinfer_ms']:.3f}ms",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = args.output.with_suffix(".meta.txt")
    passed_count = sum(bool(row.get("passed")) for row in rows)
    metadata_path.write_text(
        "\n".join(
            [
                f"timestamp={datetime.now().isoformat()}",
                f"host={platform.node()}",
                f"python={sys.version.split()[0]}",
                f"torch={torch.__version__}",
                f"flashinfer={getattr(flashinfer, '__version__', 'unknown')}",
                f"device={torch.cuda.get_device_name(0)}",
                f"source={SOURCE}",
                f"library={LIBRARY}",
                f"results={args.output}",
                f"passed={passed_count}/{len(rows)}",
                "ragged_lengths=deterministic lengths matching published totals/maxima; exact hidden OJ indptr is unavailable",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[result] {args.output}", flush=True)
    print(f"[meta]   {metadata_path}", flush=True)
    return 0 if passed_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
