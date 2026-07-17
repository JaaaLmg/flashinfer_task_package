#!/usr/bin/env python3
"""Project an online score from a prior SPJ report and local A/B timings."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


CASE_RE = re.compile(
    r"Testcase\s+#(?P<case_id>\d+).*?"
    r"Baseline:\s+(?P<baseline>[0-9.]+).*?"
    r"User kernel:\s+(?P<user>[0-9.]+).*?"
    r"Score ratio:\s+(?P<score>[0-9.]+)",
    re.S,
)


def read_local_times(path: Path) -> dict[int, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            int(row["case_id"]): float(row["candidate_ms"])
            for row in csv.DictReader(handle)
        }


def infer_hardware_time(baseline: float, user: float, score: float) -> float:
    # score = 100 / (1 + (user - hardware) / (baseline - hardware))
    ratio = 100.0 / score - 1.0
    return (user - ratio * baseline) / (1.0 - ratio)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spj", type=Path, required=True)
    parser.add_argument("--local-before", type=Path, required=True)
    parser.add_argument("--local-after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    before = read_local_times(args.local_before)
    after = read_local_times(args.local_after)
    reports = list(CASE_RE.finditer(args.spj.read_text(encoding="utf-8")))
    if not reports:
        raise ValueError(f"No testcase reports found in {args.spj}")

    rows = []
    for match in reports:
        case_id = int(match.group("case_id"))
        baseline = float(match.group("baseline"))
        online_before = float(match.group("user"))
        score_before = float(match.group("score")) * 100.0
        hardware = infer_hardware_time(baseline, online_before, score_before)
        local_ratio = after[case_id] / before[case_id]
        online_after = online_before * local_ratio
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
                "online_score_before": score_before,
                "projected_score_after": score_after,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    old_score = sum(row["online_score_before"] for row in rows) / len(rows)
    new_score = sum(row["projected_score_after"] for row in rows) / len(rows)
    old_time = sum(row["online_before_ms"] for row in rows)
    new_time = sum(row["projected_online_ms"] for row in rows)
    print(f"cases={len(rows)}")
    print(f"online_score_before={old_score:.6f}")
    print(f"projected_score_after={new_score:.6f}")
    print(f"online_time_before_ms={old_time:.6f}")
    print(f"projected_online_time_ms={new_time:.6f}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
