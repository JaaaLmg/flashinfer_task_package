#!/usr/bin/env python3
"""Correctness and latency benchmark for problem 20004's exact C ABI (paged decode)."""

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
CODE_DIR = ROOT / "paged_decode"
DEFAULT_SOURCE = CODE_DIR / "paged_decode_baseline.cu"
DEFAULT_LIBRARY = CODE_DIR / "paged_decode_baseline.so"
DEFAULT_RESULT = CODE_DIR / "stage_a_baseline_results.csv"

H_QO = 32
PAGE_SIZE = 16
RTOL = 1.6e-2
ATOL = 1.6e-2


def kv_heads_for(head_dim: int) -> int:
    """Matches benchmark/bench_batch_decode.py: 8 KV heads at d=64, otherwise 4."""
    return 8 if head_dim == 64 else 4


@dataclass(frozen=True)
class Case:
    case_id: int
    name: str
    batch_size: int
    seq_len_kv: int
    head_dim: int

    @property
    def num_kv_heads(self) -> int:
        return kv_heads_for(self.head_dim)

    @property
    def pages_per_request(self) -> int:
        return math.ceil(self.seq_len_kv / PAGE_SIZE)

    @property
    def num_pages(self) -> int:
        return self.batch_size * self.pages_per_request


def cases() -> list[Case]:
    return [
        Case(1, "b1_l1024_d128", 1, 1024, 128),
        Case(2, "b4_l1024_d128", 4, 1024, 128),
        Case(3, "b16_l1024_d128", 16, 1024, 128),
        Case(4, "b64_l1024_d128", 64, 1024, 128),
        Case(5, "b128_l1024_d128", 128, 1024, 128),
        Case(6, "b1_l4096_d128", 1, 4096, 128),
        Case(7, "b4_l4096_d128", 4, 4096, 128),
        Case(8, "b16_l4096_d128", 16, 4096, 128),
        Case(9, "b1_l16384_d128", 1, 16384, 128),
        Case(10, "b4_l16384_d128", 4, 16384, 128),
        Case(11, "b16_l16384_d128", 16, 16384, 128),
        Case(12, "b1_l512_d64", 1, 512, 64),
        Case(13, "b16_l1024_d64", 16, 1024, 64),
        Case(14, "b64_l4096_d64", 64, 4096, 64),
        Case(15, "b4_l16384_d64", 4, 16384, 64),
        Case(16, "b1_l1024_d256", 1, 1024, 256),
        Case(17, "b16_l1024_d256", 16, 1024, 256),
        Case(18, "b64_l1024_d256", 64, 1024, 256),
        Case(19, "b4_l4096_d256", 4, 4096, 256),
        Case(20, "b4_l16384_d256", 4, 16384, 256),
        Case(21, "odd_b3_l1023_d128", 3, 1023, 128),
        Case(22, "odd_b5_l257_d256", 5, 257, 256),
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
    run.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int64] * 6
    run.restype = None
    run._library = library  # type: ignore[attr-defined]
    return run


def launch(
    run: ctypes._CFuncPtr,
    q: torch.Tensor,
    kv_data: torch.Tensor,
    output: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    last_page_len: torch.Tensor,
    case: Case,
) -> None:
    run(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(kv_data.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(kv_indptr.data_ptr()),
        ctypes.c_void_p(kv_indices.data_ptr()),
        ctypes.c_void_p(last_page_len.data_ptr()),
        case.batch_size,
        case.seq_len_kv,
        H_QO,
        case.num_kv_heads,
        case.head_dim,
        PAGE_SIZE,
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


def make_metadata(case: Case, device: torch.device, seed: int, identity_pages: bool):
    """One decode query per request; kv_indptr/last_page_len describe the paged KV."""
    kv_indptr = torch.arange(
        0, (case.batch_size + 1) * case.pages_per_request, case.pages_per_request,
        dtype=torch.int32, device=device,
    )
    last_len = (case.seq_len_kv - 1) % PAGE_SIZE + 1
    last_page_len = torch.full(
        (case.batch_size,), last_len, dtype=torch.int32, device=device,
    )
    extra_pages = 7
    num_physical_pages = case.num_pages + extra_pages
    if identity_pages:
        kv_indices = torch.arange(case.num_pages, dtype=torch.int32, device=device)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 1000 + case.case_id)
        permutation = torch.randperm(num_physical_pages, generator=generator, dtype=torch.int32)
        kv_indices = permutation[:case.num_pages].to(device)
    return kv_indptr, kv_indices, last_page_len, num_physical_pages


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
    kv_indptr, kv_indices, last_page_len, num_physical_pages = make_metadata(
        case, device, seed, identity_pages
    )
    q = torch.randn(
        (case.batch_size, H_QO, case.head_dim),
        dtype=torch.bfloat16, device=device,
    )
    kv_data = torch.randn(
        (num_physical_pages, 2, PAGE_SIZE, case.num_kv_heads, case.head_dim),
        dtype=torch.bfloat16, device=device,
    )
    output = torch.empty_like(q)
    reference = torch.empty_like(q)

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", use_tensor_cores=True
    )
    wrapper.plan(
        kv_indptr, kv_indices, last_page_len,
        H_QO, case.num_kv_heads, case.head_dim, PAGE_SIZE,
        data_type=torch.bfloat16, q_data_type=torch.bfloat16,
    )
    wrapper.run(q, kv_data, out=reference)
    launch(run, q, kv_data, output, kv_indptr, kv_indices, last_page_len, case)
    torch.cuda.synchronize()

    output_flat = output.view(-1).float()
    reference_flat = reference.view(-1).float()
    abs_error = (output_flat - reference_flat).abs()
    tolerance = ATOL + RTOL * reference_flat.abs()
    finite = bool(torch.isfinite(abs_error).all().item())
    if finite:
        matched = int((abs_error <= tolerance).sum().item())
        severe_count = int((abs_error > 8.0 * tolerance).sum().item())
        max_abs_error = float(abs_error.max().item())
        match_ratio = matched / output.numel()
    else:
        severe_count = output.numel()
        max_abs_error = float("inf")
        match_ratio = 0.0
    passed = finite and match_ratio >= 0.99 and severe_count == 0

    first_candidate_ms = elapsed_ms(
        lambda: launch(run, q, kv_data, output,
                       kv_indptr, kv_indices, last_page_len, case),
        1,
    )
    candidate_repeats = choose_repeats(first_candidate_ms, max_repeats)
    candidate_ms = first_candidate_ms if candidate_repeats == 1 else elapsed_ms(
        lambda: launch(run, q, kv_data, output,
                       kv_indptr, kv_indices, last_page_len, case),
        candidate_repeats,
    )
    first_reference_ms = elapsed_ms(lambda: wrapper.run(q, kv_data, out=reference), 1)
    reference_repeats = choose_repeats(first_reference_ms, max(20, max_repeats))
    reference_ms = first_reference_ms if reference_repeats == 1 else elapsed_ms(
        lambda: wrapper.run(q, kv_data, out=reference), reference_repeats
    )

    # QK and PV each contribute one multiply and one add per (head, kv token, dim).
    flops = 4.0 * case.batch_size * case.seq_len_kv * H_QO * case.head_dim
    # KV cache dominates the traffic: K and V rows of every visible token, read once.
    kv_bytes = (2.0 * 2.0 * case.batch_size * case.seq_len_kv
                * case.num_kv_heads * case.head_dim)
    return {
        "case_id": case.case_id,
        "name": case.name,
        "batch_size": case.batch_size,
        "seq_len_kv": case.seq_len_kv,
        "head_dim": case.head_dim,
        "num_kv_heads": case.num_kv_heads,
        "logical_pages": case.num_pages,
        "candidate_ms": candidate_ms,
        "flashinfer_ms": reference_ms,
        "slowdown_vs_flashinfer": candidate_ms / reference_ms,
        "effective_tflops": flops / (candidate_ms * 1.0e9),
        "kv_bandwidth_GB_s": kv_bytes / (candidate_ms * 1.0e6),
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
            f"L={case.seq_len_kv}, D={case.head_dim}, "
            f"H_kv={case.num_kv_heads}", flush=True,
        )
        try:
            row = benchmark_case(
                run, case, workspace, args.seed, args.max_repeats, args.identity_pages
            )
        except Exception as exc:
            rows.append({
                "case_id": case.case_id, "name": case.name,
                "batch_size": case.batch_size, "seq_len_kv": case.seq_len_kv,
                "head_dim": case.head_dim, "passed": False, "error": repr(exc),
            })
            print(f"  ERROR: {exc!r}", file=sys.stderr, flush=True)
            break
        rows.append(row)
        print(
            f"  passed={row['passed']} match={row['match_ratio']:.6f} "
            f"max_err={row['max_abs_error']:.6g} candidate={row['candidate_ms']:.3f}ms "
            f"flashinfer={row['flashinfer_ms']:.3f}ms "
            f"bw={row['kv_bandwidth_GB_s']:.1f}GB/s", flush=True,
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
            f"page_size={PAGE_SIZE}; num_qo_heads={H_QO}; layout=NHD",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"[result] {args.output}", flush=True)
    print(f"[meta]   {metadata_path}", flush=True)
    return 0 if passed_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
