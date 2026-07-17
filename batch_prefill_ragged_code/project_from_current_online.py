#!/usr/bin/env python3
"""Project a candidate from the latest online per-case times and local A/B ratios."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from project_online_score import CASE_RE, infer_hardware_time, read_local_times


def read_online(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            int(row["case_id"]): {
                "online_ms": float(row["online_ms"]),
                "display_score": float(row["display_score"]),
            }
            for row in csv.DictReader(handle)
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-spj", type=Path, required=True)
    parser.add_argument("--online-current", type=Path, required=True)
    parser.add_argument("--local-before", type=Path, required=True)
    parser.add_argument("--local-after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    before = read_local_times(args.local_before)
    after = read_local_times(args.local_after)
    current = read_online(args.online_current)
    reports = list(CASE_RE.finditer(args.anchor_spj.read_text(encoding="utf-8")))
    if not reports:
        raise ValueError(f"No testcase reports found in {args.anchor_spj}")

    rows = []
    for match in reports:
        case_id = int(match.group("case_id"))
        baseline = float(match.group("baseline"))
        online_before = current[case_id]["online_ms"]
        display_score = current[case_id]["display_score"]
        hardware = infer_hardware_time(baseline, online_before, display_score)
        local_ratio = after[case_id] / before[case_id]
        online_after = online_before * local_ratio
        score_before = 100.0 / (
            1.0 + (online_before - hardware) / (baseline - hardware)
        )
        score_after = 100.0 / (
            1.0 + (online_after - hardware) / (baseline - hardware)
        )
        rows.append(
            {
                "case_id": case_id,
                "online_baseline_ms": baseline,
                "online_before_ms": online_before,
                "inferred_hardware_ms": hardware,
                "local_before_ms": before[case_id],
                "local_after_ms": after[case_id],
                "local_ratio": local_ratio,
                "projected_online_ms": online_after,
                "formula_score_before": score_before,
                "display_score_before": current[case_id]["display_score"],
                "projected_score_after": score_after,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    display_before = sum(row["display_score_before"] for row in rows) / len(rows)
    formula_before = sum(row["formula_score_before"] for row in rows) / len(rows)
    projected = sum(row["projected_score_after"] for row in rows) / len(rows)
    calibrated = display_before + projected - formula_before
    print(f"display_score_before={display_before:.6f}")
    print(f"formula_score_before={formula_before:.6f}")
    print(f"formula_score_after={projected:.6f}")
    print(f"calibrated_score_after={calibrated:.6f}")
    print(f"online_time_before_ms={sum(row['online_before_ms'] for row in rows):.6f}")
    print(f"projected_online_time_ms={sum(row['projected_online_ms'] for row in rows):.6f}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
