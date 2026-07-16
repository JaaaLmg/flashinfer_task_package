#!/usr/bin/env python3
"""Correctness and latency benchmark for problem 20002's exact C ABI."""

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

import flashinfer
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "batch_prefill_paged_code"
DEFAULT_SOURCE = CODE_DIR / "paged_prefill_baseline.cu"
DEFAULT_LIBRARY = CODE_DIR / "paged_prefill_baseline.so"
DEFAULT_RESULT = CODE_DIR / "stage_a_baseline_results.csv"

H_Q = 32
H_KV = 4
PAGE_SIZE = 16
RTOL = 1.6e-2
ATOL = 1.6e-2


@dataclass(frozen=True)
class Case:
    case_id: int
    name: str
    batch_size: int
    seq_len: int
    head_dim: int

    @property
    def num_pages(self) -> int:
        return self.batch_size * math.ceil(self.seq_len / PAGE_SIZE)


def cases() -> list[Case]:
    return [
        Case(1, "partial_page_b1_l257_d128", 1, 257, 128),
        Case(2, "b1_l1024_d128", 1, 1024, 128),
        Case(3, "b4_l1024_d128", 4, 1024, 128),
        Case(4, "b1_l4096_d128", 1, 4096, 128),
        Case(5, "b4_l4096_d128", 4, 4096, 128),
        Case(6, "b16_l1024_d128", 16, 1024, 128),
        Case(7, "b1_l8192_d128", 1, 8192, 128),
        Case(8, "b1_l1024_d256", 1, 1024, 256),
        Case(9, "b4_l1024_d256", 4, 1024, 256),
        Case(10, "b1_l4096_d256", 1, 4096, 256),
        Case(11, "b16_l1024_d256", 16, 1024, 256),
        Case(12, "b1_l16384_d128", 1, 16384, 128),
        Case(13, "b16_l4096_d128", 16, 4096, 128),
        Case(14, "b16_l8192_d128", 16, 8192, 128),
        Case(15, "b4_l16384_d128", 4, 16384, 128),
        Case(16, "b64_l1024_d128", 64, 1024, 128),
        Case(17, "b4_l4096_d256", 4, 4096, 256),
        Case(18, "b16_l4096_d256", 16, 4096, 256),
        Case(19, "b64_l1024_d256", 64, 1024, 256),
        Case(20, "b4_l8192_d256", 4, 8192, 256),
        Case(21, "b16_l16384_d128", 16, 16384, 128),
        Case(22, "b4_l8192_d128", 4, 8192, 128),
    ]


def compile_library(source: Path, library: Path, force: bool = False) -> None:
    if library.exists() and not force and library.stat().st_mtime >= source.stat().st_mtime:
        return
    cmd = [
        "mxcc", "-O3", "-std=c++17", "--offload-arch=xcore1000",
        "-I/opt/maca/tools/cu-bridge/include", "-shared", "-fPIC",
        str(source), "-o", str(library),
    ]
    print("[build]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def load_kernel(library_path: Path) -> ctypes._CFuncPtr:
    library = ctypes.CDLL(str(library_path))
    run = library.run_kernel
    run.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int64] * 7
    run.restype = None
    run._library = library  # type: ignore[attr-defined]
    return run


def launch(
    run: ctypes._CFuncPtr,
    q: torch.Tensor,
    kv_data: torch.Tensor,
    output: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    last_page_len: torch.Tensor,
    case: Case,
) -> None:
    run(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(kv_data.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(qo_indptr.data_ptr()),
        ctypes.c_void_p(kv_indptr.data_ptr()),
        ctypes.c_void_p(kv_indices.data_ptr()),
        ctypes.c_void_p(last_page_len.data_ptr()),
        case.batch_size,
        case.seq_len,
        H_Q,
        H_KV,
        case.head_dim,
        PAGE_SIZE,
        0,
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
    return max(1, min(max_repeats, math.ceil(200.0 / first_ms)))


def make_metadata(case: Case, device: torch.device, seed: int, identity_pages: bool = False):
    pages_per_request = math.ceil(case.seq_len / PAGE_SIZE)
    qo_indptr = torch.arange(
        0, (case.batch_size + 1) * case.seq_len, case.seq_len,
        dtype=torch.int32, device=device,
    )
    kv_indptr = torch.arange(
        0, (case.batch_size + 1) * pages_per_request, pages_per_request,
        dtype=torch.int32, device=device,
    )
    last_len = (case.seq_len - 1) % PAGE_SIZE + 1
    last_page_len = torch.full(
        (case.batch_size,), last_len, dtype=torch.int32, device=device,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1000 + case.case_id)
    extra_pages = 7
    num_physical_pages = case.num_pages + extra_pages
    if identity_pages:
        kv_indices = torch.arange(case.num_pages, dtype=torch.int32, device=device)
    else:
        permutation = torch.randperm(num_physical_pages, generator=generator, dtype=torch.int32)
        kv_indices = permutation[:case.num_pages].to(device)
    return qo_indptr, kv_indptr, kv_indices, last_page_len, num_physical_pages


def benchmark_case(
    run: ctypes._CFuncPtr,
    case: Case,
    workspace: torch.Tensor,
    seed: int,
    max_repeats: int,
    identity_pages: bool,
) -> dict[str, object]:
    device = torch.device("cuda")
    torch.manual_seed(seed + case.case_id)
    qo_indptr, kv_indptr, kv_indices, last_page_len, num_physical_pages = make_metadata(
        case, device, seed, identity_pages
    )
    q = torch.randn(
        (case.batch_size * case.seq_len, H_Q, case.head_dim),
        dtype=torch.bfloat16, device=device,
    )
    kv_data = torch.randn(
        (num_physical_pages, 2, PAGE_SIZE, H_KV, case.head_dim),
        dtype=torch.bfloat16, device=device,
    )
    output = torch.empty_like(q)
    reference = torch.empty_like(q)

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="auto"
    )
    wrapper.plan(
        qo_indptr, kv_indptr, kv_indices, last_page_len,
        H_Q, H_KV, case.head_dim, PAGE_SIZE,
        causal=False, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
    )
    wrapper.run(q, kv_data, out=reference)
    launch(run, q, kv_data, output, qo_indptr, kv_indptr, kv_indices, last_page_len, case)
    torch.cuda.synchronize()

    finite = True
    matched = 0
    severe_count = 0
    max_abs_error = 0.0
    output_flat = output.view(-1)
    reference_flat = reference.view(-1)
    check_chunk = 16 * 1024 * 1024
    for start in range(0, output.numel(), check_chunk):
        stop = min(start + check_chunk, output.numel())
        output_part = output_flat[start:stop].float()
        reference_part = reference_flat[start:stop].float()
        abs_error = (output_part - reference_part).abs()
        tolerance = ATOL + RTOL * reference_part.abs()
        part_finite = bool(torch.isfinite(abs_error).all().item())
        finite = finite and part_finite
        if not part_finite:
            break
        matched += int((abs_error <= tolerance).sum().item())
        severe_count += int((abs_error > 8.0 * tolerance).sum().item())
        max_abs_error = max(max_abs_error, float(abs_error.max().item()))
    match_ratio = matched / output.numel() if finite else 0.0
    if not finite:
        max_abs_error = float("inf")
        severe_count = output.numel()
    passed = finite and match_ratio >= 0.99 and severe_count == 0

    first_candidate_ms = elapsed_ms(
        lambda: launch(
            run, q, kv_data, output, qo_indptr, kv_indptr,
            kv_indices, last_page_len, case,
        ),
        1,
    )
    candidate_repeats = choose_repeats(first_candidate_ms, max_repeats)
    candidate_ms = first_candidate_ms if candidate_repeats == 1 else elapsed_ms(
        lambda: launch(
            run, q, kv_data, output, qo_indptr, kv_indptr,
            kv_indices, last_page_len, case,
        ),
        candidate_repeats,
    )
    first_reference_ms = elapsed_ms(lambda: wrapper.run(q, kv_data, out=reference), 1)
    reference_repeats = choose_repeats(first_reference_ms, max(20, max_repeats))
    reference_ms = first_reference_ms if reference_repeats == 1 else elapsed_ms(
        lambda: wrapper.run(q, kv_data, out=reference), reference_repeats
    )

    flops = 4.0 * case.batch_size * case.seq_len * case.seq_len * H_Q * case.head_dim
    tflops = flops / (candidate_ms * 1.0e9)
    return {
        "case_id": case.case_id,
        "name": case.name,
        "batch_size": case.batch_size,
        "seq_len": case.seq_len,
        "head_dim": case.head_dim,
        "logical_pages": case.num_pages,
        "candidate_ms": candidate_ms,
        "flashinfer_ms": reference_ms,
        "slowdown_vs_flashinfer": candidate_ms / reference_ms,
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
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--max-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--identity-pages", action="store_true",
                        help="Use the official benchmark's arange page table")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/MACA GPU is required")
    selected_ids = parse_case_filter(args.cases)
    selected = [case for case in cases() if selected_ids is None or case.case_id in selected_ids]
    if not selected:
        raise ValueError("No cases selected")

    source = args.source.resolve()
    library = args.library.resolve()
    compile_library(source, library, force=args.force_build)
    run = load_kernel(library)
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
            f"L={case.seq_len}, D={case.head_dim}", flush=True,
        )
        try:
            row = benchmark_case(
                run, case, workspace, args.seed, args.max_repeats, args.identity_pages
            )
        except Exception as exc:
            row = {
                "case_id": case.case_id, "name": case.name,
                "batch_size": case.batch_size, "seq_len": case.seq_len,
                "head_dim": case.head_dim, "passed": False, "error": repr(exc),
            }
            print(f"  ERROR: {exc!r}", file=sys.stderr, flush=True)
            rows.append(row)
            break
        rows.append(row)
        print(
            f"  passed={row['passed']} match={row['match_ratio']:.6f} "
            f"max_err={row['max_abs_error']:.6g} candidate={row['candidate_ms']:.3f}ms "
            f"flashinfer={row['flashinfer_ms']:.3f}ms", flush=True,
        )
        del row
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = args.output.with_suffix(".meta.txt")
    passed_count = sum(bool(row.get("passed")) for row in rows)
    metadata_path.write_text(
        "\n".join([
            f"timestamp={datetime.now().isoformat()}",
            f"host={platform.node()}",
            f"python={sys.version.split()[0]}",
            f"torch={torch.__version__}",
            f"flashinfer={getattr(flashinfer, '__version__', 'unknown')}",
            f"device={torch.cuda.get_device_name(0)}",
            f"source={source}", f"library={library}", f"results={args.output}",
            f"seed={args.seed}", f"passed={passed_count}/{len(rows)}",
            f"kv_indices={'identity' if args.identity_pages else 'random permutation'}; "
            "causal=0; page_size=16",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"[result] {args.output}", flush=True)
    print(f"[meta]   {metadata_path}", flush=True)
    return 0 if passed_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
