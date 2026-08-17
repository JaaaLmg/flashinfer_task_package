#!/usr/bin/env python3
"""Project OJ display score from a bracketing, like-for-like reference run.

This deliberately does not use FlashInfer timing as a transfer function.  The
candidate and the historical 0f3400b reference must be benchmarked with the
same inputs, replay disabled, the same repeat cap, and adjacent in time.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 24:
        raise ValueError(f"{path}: expected 24 cases, found {len(rows)}")
    for row in rows:
        if row.get("passed") != "True":
            raise ValueError(f"{path}: case {row.get('case_id')} did not pass")
    return rows


def parse_online_report(path: Path) -> tuple[list[float], list[float], list[int]]:
    text = path.read_text(encoding="utf-8")
    baseline = [float(x) for x in re.findall(r"Baseline:\s+([0-9.]+)", text)]
    user = [float(x) for x in re.findall(r"User kernel:\s+([0-9.]+)", text)]
    display = [int(x) for x in re.findall(r"Display score:\s+([0-9]+)", text)]
    if not (len(baseline) == len(user) == len(display) == 24):
        raise ValueError(f"{path}: incomplete 24-case SPJ report")
    return baseline, user, display


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-report", type=Path, required=True)
    parser.add_argument("--reference-pre", type=Path, required=True)
    parser.add_argument("--reference-post", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline, online_reference, reference_display = parse_online_report(
        args.online_report
    )
    pre = load_csv(args.reference_pre)
    post = load_csv(args.reference_post)
    candidate = load_csv(args.candidate)

    projected_rows: list[dict[str, object]] = []
    for index, (pre_row, post_row, candidate_row) in enumerate(
        zip(pre, post, candidate)
    ):
        identity = (
            pre_row["case_id"], pre_row["batch_size"],
            pre_row["seq_len"], pre_row["num_heads"],
        )
        if identity != (
            post_row["case_id"], post_row["batch_size"],
            post_row["seq_len"], post_row["num_heads"],
        ) or identity != (
            candidate_row["case_id"], candidate_row["batch_size"],
            candidate_row["seq_len"], candidate_row["num_heads"],
        ):
            raise ValueError(f"case ordering mismatch at row {index + 1}")

        pre_ms = float(pre_row["candidate_ms"])
        post_ms = float(post_row["candidate_ms"])
        candidate_ms = float(candidate_row["candidate_ms"])
        # A smaller reference denominator makes the candidate ratio worse, so
        # min(pre, post) is the conservative end of the bracket.
        conservative_reference_ms = min(pre_ms, post_ms)
        local_ratio = candidate_ms / conservative_reference_ms
        projected_online_ms = online_reference[index] * local_ratio
        raw_score = 100.0 * baseline[index] / (
            baseline[index] + projected_online_ms
        )
        display_score = math.floor(raw_score)
        projected_rows.append({
            "case_id": identity[0],
            "name": candidate_row["name"],
            "batch_size": identity[1],
            "seq_len": identity[2],
            "num_heads": identity[3],
            "online_baseline_ms": baseline[index],
            "online_reference_ms": online_reference[index],
            "online_reference_display": reference_display[index],
            "local_reference_pre_ms": pre_ms,
            "local_reference_post_ms": post_ms,
            "local_reference_conservative_ms": conservative_reference_ms,
            "local_candidate_ms": candidate_ms,
            "candidate_over_reference": local_ratio,
            "projected_online_ms": projected_online_ms,
            "projected_raw_score": raw_score,
            "projected_display_score": display_score,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=projected_rows[0].keys())
        writer.writeheader()
        writer.writerows(projected_rows)

    reference_mean = sum(reference_display) / len(reference_display)
    raw_mean = sum(float(row["projected_raw_score"]) for row in projected_rows) / 24
    display_mean = sum(int(row["projected_display_score"]) for row in projected_rows) / 24
    print(f"online_reference_display_mean={reference_mean:.6f}")
    print(f"candidate_projected_raw_mean={raw_mean:.6f}")
    print(f"candidate_projected_display_mean={display_mean:.6f}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
