#!/usr/bin/env python3
"""Recompute emissions_kg / energy_kwh / cc_duration_s for an existing summary.csv.

Fix for bug where original aggregator globbed only <log_dir>/codecarbon/*.csv and
missed heavy-action CSVs at <log_dir>/env_log/codecarbon/*.csv, undercounting
emissions by 25-70%.

Usage:
    python scripts/recompute_emissions.py <summary.csv> [--inplace]
"""
import argparse
import csv
import glob
import os
import sys


def totals_for(log_dir: str) -> tuple[float, float, float, bool]:
    em = en = du = 0.0
    found = False
    paths = (
        glob.glob(os.path.join(log_dir, "codecarbon", "*.csv"))
        + glob.glob(os.path.join(log_dir, "env_log", "codecarbon", "*.csv"))
    )
    for p in paths:
        try:
            with open(p, newline="") as f:
                for row in csv.DictReader(f):
                    found = True
                    try: em += float(row.get("emissions") or 0)
                    except (TypeError, ValueError): pass
                    try: en += float(row.get("energy_consumed") or 0)
                    except (TypeError, ValueError): pass
                    try: du += float(row.get("duration") or 0)
                    except (TypeError, ValueError): pass
        except Exception:
            continue
    return em, en, du, found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary_csv")
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--log-root", default=None,
                    help="Override log_dir prefix when the recorded path is not reachable from here.")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.summary_csv)))
    if not rows:
        print("empty summary", file=sys.stderr)
        return 1

    out_rows = []
    for r in rows:
        log_dir = r["log_dir"]
        if args.log_root:
            log_dir = os.path.join(args.log_root, os.path.basename(log_dir))
        em, en, du, ok = totals_for(log_dir)
        if ok:
            r["emissions_kg"] = f"{em:.9f}"
            r["energy_kwh"]   = f"{en:.9f}"
            r["cc_duration_s"] = f"{du:.3f}"
        out_rows.append(r)

    fieldnames = list(rows[0].keys())
    dest = args.summary_csv if args.inplace else args.summary_csv + ".recomputed.csv"
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {dest} ({len(out_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
