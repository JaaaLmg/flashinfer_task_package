#!/usr/bin/env python3
"""Correctness and latency benchmark for problem 20003's exact C ABI (MLA paged attention)."""

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
CODE_DIR = ROOT / "mla_paged_attention_code"
DEFAULT_SOURCE = CODE_DIR / "mla_paged_optimized.cu"
# Keep generated artifacts out of the cleaned source directory by default.
DEFAULT_LIBRARY = Path("/tmp/mla_paged_optimized_no_cheat.so")
DEFAULT_RESULT = Path("/tmp/mla_paged_results.csv")

HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 1
RTOL = 1.6e-2
ATOL = 1.6e-2


@dataclass(frozen=True)
class Case:
    case_id: int
    name: str
    batch_size: int
    seq_len: int
    num_heads: int


def cases() -> list[Case]:
    return [
        Case(1, "b1_l1024_h64", 1, 1024, 64),
        Case(2, "b4_l1024_h64", 4, 1024, 64),
        Case(3, "b16_l1024_h64", 16, 1024, 64),
        Case(4, "b64_l1024_h64", 64, 1024, 64),
        Case(5, "b1_l4096_h64", 1, 4096, 64),
        Case(6, "b4_l4096_h64", 4, 4096, 64),
        Case(7, "b16_l4096_h64", 16, 4096, 64),
        Case(8, "b1_l8192_h64", 1, 8192, 64),
        Case(9, "b4_l8192_h64", 4, 8192, 64),
        Case(10, "b1_l16384_h64", 1, 16384, 64),
        Case(11, "b4_l16384_h64", 4, 16384, 64),
        Case(12, "b16_l16384_h64", 16, 16384, 64),
        Case(13, "b1_l1024_h128", 1, 1024, 128),
        Case(14, "b4_l1024_h128", 4, 1024, 128),
        Case(15, "b16_l1024_h128", 16, 1024, 128),
        Case(16, "b64_l1024_h128", 64, 1024, 128),
        Case(17, "b1_l4096_h128", 1, 4096, 128),
        Case(18, "b16_l4096_h128", 16, 4096, 128),
        Case(19, "b1_l16384_h128", 1, 16384, 128),
        Case(20, "b4_l16384_h128", 4, 16384, 128),
        Case(21, "odd_b3_l1023_h64", 3, 1023, 64),
        Case(22, "odd_b5_l257_h128", 5, 257, 128),
    ]


def online_cases() -> list[Case]:
    """Published online ordering, kept separate from the local smoke suite."""
    return [
        Case(1, "b1_l1024_h64", 1, 1024, 64),
        Case(2, "b1_l4096_h64", 1, 4096, 64),
        Case(3, "b1_l8192_h64", 1, 8192, 64),
        Case(4, "b1_l16384_h64", 1, 16384, 64),
        Case(5, "b4_l1024_h64", 4, 1024, 64),
        Case(6, "b4_l4096_h64", 4, 4096, 64),
        Case(7, "b4_l8192_h64", 4, 8192, 64),
        Case(8, "b4_l16384_h64", 4, 16384, 64),
        Case(9, "b16_l1024_h64", 16, 1024, 64),
        Case(10, "b16_l4096_h64", 16, 4096, 64),
        Case(11, "b16_l8192_h64", 16, 8192, 64),
        Case(12, "b16_l16384_h64", 16, 16384, 64),
        Case(13, "b1_l1024_h128", 1, 1024, 128),
        Case(14, "b1_l4096_h128", 1, 4096, 128),
        Case(15, "b1_l8192_h128", 1, 8192, 128),
        Case(16, "b1_l16384_h128", 1, 16384, 128),
        Case(17, "b4_l1024_h128", 4, 1024, 128),
        Case(18, "b4_l4096_h128", 4, 4096, 128),
        Case(19, "b4_l8192_h128", 4, 8192, 128),
        Case(20, "b4_l16384_h128", 4, 16384, 128),
        Case(21, "b16_l1024_h128", 16, 1024, 128),
        Case(22, "b16_l4096_h128", 16, 4096, 128),
        Case(23, "b16_l8192_h128", 16, 8192, 128),
        Case(24, "b16_l16384_h128", 16, 16384, 128),
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
    run.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int64] * 7
    run.restype = None
    run._library = library  # type: ignore[attr-defined]
    return run


def launch(
    run: ctypes._CFuncPtr,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv: torch.Tensor,
    kpe: torch.Tensor,
    output: torch.Tensor,
    q_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_lens: torch.Tensor,
    case: Case,
) -> None:
    run(
        ctypes.c_void_p(q_nope.data_ptr()),
        ctypes.c_void_p(q_pe.data_ptr()),
        ctypes.c_void_p(ckv.data_ptr()),
        ctypes.c_void_p(kpe.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(q_indptr.data_ptr()),
        ctypes.c_void_p(kv_indptr.data_ptr()),
        ctypes.c_void_p(kv_indices.data_ptr()),
        ctypes.c_void_p(kv_lens.data_ptr()),
        case.batch_size,
        case.seq_len,
        case.num_heads,
        HEAD_DIM_CKV,
        HEAD_DIM_KPE,
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


def make_metadata(case: Case, device: torch.device, seed: int, permute_pages: bool):
    """Build the MLA decode metadata: one query row per request, seq_len pages each."""
    total_tokens = case.batch_size * case.seq_len
    q_indptr = torch.arange(0, case.batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(
        0, (case.batch_size + 1) * case.seq_len, case.seq_len,
        dtype=torch.int32, device=device,
    )
    kv_lens = torch.full(
        (case.batch_size,), case.seq_len, dtype=torch.int32, device=device,
    )
    if permute_pages:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 1000 + case.case_id)
        permutation = torch.randperm(total_tokens, generator=generator, dtype=torch.int32)
        kv_indices = permutation.to(device)
    else:
        kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=device)
    return q_indptr, kv_indptr, kv_indices, kv_lens


def benchmark_case(
    run: ctypes._CFuncPtr,
    case: Case,
    workspace: torch.Tensor,
    seed: int,
    max_repeats: int,
    permute_pages: bool,
    zero_pe: bool,
) -> dict[str, object]:
    device = torch.device("cuda")
    torch.manual_seed(seed + case.case_id)
    q_indptr, kv_indptr, kv_indices, kv_lens = make_metadata(
        case, device, seed, permute_pages
    )
    q_nope = torch.randn(
        (case.batch_size, case.num_heads, HEAD_DIM_CKV),
        dtype=torch.bfloat16, device=device,
    )
    q_pe = torch.zeros(
        (case.batch_size, case.num_heads, HEAD_DIM_KPE),
        dtype=torch.bfloat16, device=device,
    ) if zero_pe else torch.randn(
        (case.batch_size, case.num_heads, HEAD_DIM_KPE),
        dtype=torch.bfloat16, device=device,
    )
    ckv = torch.randn(
        (case.batch_size * case.seq_len, 1, HEAD_DIM_CKV),
        dtype=torch.bfloat16, device=device,
    )
    kpe = torch.zeros(
        (case.batch_size * case.seq_len, 1, HEAD_DIM_KPE),
        dtype=torch.bfloat16, device=device,
    ) if zero_pe else torch.randn(
        (case.batch_size * case.seq_len, 1, HEAD_DIM_KPE),
        dtype=torch.bfloat16, device=device,
    )
    output = torch.empty_like(q_nope)
    reference = torch.empty_like(q_nope)

    sm_scale = 1.0 / ((HEAD_DIM_CKV + HEAD_DIM_KPE) ** 0.5)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="auto")
    wrapper.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        case.num_heads, HEAD_DIM_CKV, HEAD_DIM_KPE,
        PAGE_SIZE, False, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    wrapper.run(q_nope, q_pe, ckv, kpe, out=reference, return_lse=False)
    launch(run, q_nope, q_pe, ckv, kpe, output,
           q_indptr, kv_indptr, kv_indices, kv_lens, case)
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
        lambda: launch(run, q_nope, q_pe, ckv, kpe, output,
                       q_indptr, kv_indptr, kv_indices, kv_lens, case),
        1,
    )
    candidate_repeats = choose_repeats(first_candidate_ms, max_repeats)
    candidate_ms = first_candidate_ms if candidate_repeats == 1 else elapsed_ms(
        lambda: launch(run, q_nope, q_pe, ckv, kpe, output,
                       q_indptr, kv_indptr, kv_indices, kv_lens, case),
        candidate_repeats,
    )
    first_reference_ms = elapsed_ms(
        lambda: wrapper.run(q_nope, q_pe, ckv, kpe, out=reference, return_lse=False), 1
    )
    reference_repeats = choose_repeats(first_reference_ms, max(20, max_repeats))
    reference_ms = first_reference_ms if reference_repeats == 1 else elapsed_ms(
        lambda: wrapper.run(q_nope, q_pe, ckv, kpe, out=reference, return_lse=False),
        reference_repeats,
    )

    # QK over ckv+kpe plus PV over ckv, one multiply and one add each.
    flops = (2.0 * case.batch_size * case.num_heads * case.seq_len
             * (2 * HEAD_DIM_CKV + HEAD_DIM_KPE))
    # KV cache dominates the traffic: every request streams its own ckv+kpe rows.
    kv_bytes = 2.0 * case.batch_size * case.seq_len * (HEAD_DIM_CKV + HEAD_DIM_KPE)
    return {
        "case_id": case.case_id,
        "name": case.name,
        "batch_size": case.batch_size,
        "seq_len": case.seq_len,
        "num_heads": case.num_heads,
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
    parser.add_argument("--suite", choices=("smoke", "online"), default="smoke",
                        help="case table: local smoke coverage or published online order")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--max-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--permute-pages", action="store_true",
                        help="Shuffle kv_indices instead of the official arange page table")
    parser.add_argument("--zero-pe", action="store_true",
                        help="Use exact-zero q_pe/kpe to reproduce the historical online anchor")
    parser.add_argument("--disable-pointer-replay", action="store_true",
                        help="Disable a source's optional replay probe before benchmarking")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/MACA GPU is required")
    selected_ids = parse_case_filter(args.cases)
    suite = online_cases() if args.suite == "online" else cases()
    selected = [case for case in suite if selected_ids is None or case.case_id in selected_ids]
    if not selected:
        raise ValueError("No cases selected")

    source = args.source.resolve()
    library = args.library.resolve()
    compile_library(source, library, force=args.force_build)
    run = load_kernel(library)
    if args.disable_pointer_replay:
        try:
            configure_replay = run._library.configure_pointer_replay_probe
        except AttributeError as exc:
            raise RuntimeError(
                "--disable-pointer-replay requested but the library has no probe export"
            ) from exc
        configure_replay.argtypes = [ctypes.c_int32]
        configure_replay.restype = ctypes.c_int
        if configure_replay(0) != 0:
            raise RuntimeError("configure_pointer_replay_probe(0) failed")
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
            f"L={case.seq_len}, H={case.num_heads}", flush=True,
        )
        try:
            row = benchmark_case(
                run, case, workspace, args.seed, args.max_repeats,
                args.permute_pages, args.zero_pe
            )
        except Exception as exc:
            rows.append({
                "case_id": case.case_id, "name": case.name,
                "batch_size": case.batch_size, "seq_len": case.seq_len,
                "num_heads": case.num_heads, "passed": False, "error": repr(exc),
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
            f"kv_indices={'random permutation' if args.permute_pages else 'identity'}; "
            f"pe_inputs={'zero' if args.zero_pe else 'random-normal'}; "
            f"causal=0; page_size={PAGE_SIZE}; "
            f"head_dim_ckv={HEAD_DIM_CKV}; head_dim_kpe={HEAD_DIM_KPE}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"[result] {args.output}", flush=True)
    print(f"[meta]   {metadata_path}", flush=True)
    return 0 if passed_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
