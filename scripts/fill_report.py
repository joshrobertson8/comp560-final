"""
After training completes, read checkpoints/history.json and the evaluate.py
summary CSV, then replace the TBD rows in REPORT.md with real numbers.

Run:  python scripts/fill_report.py --student_id joshrobertson
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def load_history() -> list[dict]:
    p = ROOT / "checkpoints" / "history.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def find_latest_summary(student_id: str) -> dict | None:
    """Pick the most recent results/<student_id>_*_summary.csv."""
    results_dir = ROOT / "results"
    if not results_dir.exists():
        return None
    summaries = sorted(results_dir.glob(f"{student_id}_*_summary.csv"))
    if not summaries:
        return None
    with summaries[-1].open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["dataset"] == "dataset_a":
                return row
    return None


def best_dev(history: list[dict]) -> dict | None:
    """Return the history record with the highest dev_mean_map."""
    best = None
    for rec in history:
        if rec.get("dev_mean_map") is None:
            continue
        if best is None or rec["dev_mean_map"] > best["dev_mean_map"]:
            best = rec
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_id", default="joshrobertson")
    parser.add_argument("--report", default="REPORT.md")
    args = parser.parse_args()

    history = load_history()
    summary = find_latest_summary(args.student_id)
    best = best_dev(history)

    report_path = ROOT / args.report
    report = report_path.read_text()

    # Fill Dataset A table
    if summary is not None:
        ours_row = (
            f"| **Ours (fine-tuned)**          |  {float(summary['Rank-1']):.2f} |  "
            f"{float(summary['Rank-5']):.2f} |    {float(summary['Rank-10']):.2f} | "
            f"{float(summary['mAP']):.2f} | {float(summary['mINP']):.2f} |"
        )
        report = re.sub(
            r"\| \*\*Ours \(fine-tuned\)\*\*.*\|",
            ours_row.replace("\\", "\\\\"),
            report,
            count=1,
        )
        print("Filled Dataset A metrics.")
    else:
        print(f"No summary CSV found under results/ for student_id={args.student_id}")

    # Fill cross-domain dev table
    if best is not None and best.get("dev"):
        dev = best["dev"]

        def pct(sub: str, key: str) -> str:
            return f"{dev.get(sub, {}).get(key, 0):.1f}"

        best_row = (
            f"| **Best checkpoint**       |             {pct('AAUZebraFish', 'mAP')} |"
            f"               {pct('PolarBearVidID', 'mAP')} |       {pct('SMALST', 'mAP')} |"
            f"     {best['dev_mean_map']:.1f} |"
        )
        report = re.sub(
            r"\| \*\*Best checkpoint\*\*.*\|",
            best_row.replace("\\", "\\\\"),
            report,
            count=1,
        )
        print(f"Filled cross-domain dev metrics (best at epoch {best.get('epoch')}).")
    else:
        print("No dev history available yet.")

    report_path.write_text(report)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
