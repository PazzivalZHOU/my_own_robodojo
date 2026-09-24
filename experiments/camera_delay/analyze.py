#!/usr/bin/env python3
"""Aggregate a camera_delay sweep into a comparison table (and a plot).

Reads the ``_result.json`` files produced by ``run_eval.py`` (one per delay, under
``results/<stamp>/delay_<d>/``), optionally combines several sweeps/seeds, and
writes:

  * ``analysis.md``   -- human-readable comparison table (+ deltas vs baseline)
  * ``analysis.csv``  -- same numbers, machine-readable
  * ``delay_vs_performance.png`` -- success rate & score vs delay (needs matplotlib)

Usage::

    # latest sweep under experiments/camera_delay/results/
    python experiments/camera_delay/analyze.py

    # a specific sweep (or several sweeps / the results root)
    python experiments/camera_delay/analyze.py --results experiments/camera_delay/results/20260924_130437

    # several sweeps (e.g. multiple seeds) -> mean +/- std per delay
    python experiments/camera_delay/analyze.py --results <sweepA> <sweepB>

All inputs are read-only; outputs go to ``--out`` (default: the sweep dir, or a
fresh ``_analysis_<stamp>`` dir when several inputs are combined).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import glob
import json
import os
import re
import statistics
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_RESULTS = os.path.join(_HERE, "results")

_DELAY_RE = re.compile(r"delay_(\d+)")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[analyze] WARNING: cannot read {path}: {exc}", file=sys.stderr)
        return {}


def _delay_from_result_path(path: str) -> Optional[int]:
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    match = _DELAY_RE.search(parent)
    return int(match.group(1)) if match else None


def _record_from_result_file(path: str, delay: int) -> Dict[str, Any]:
    data = _load_json(path)
    details = data.get("details") or {}
    n_success = None
    n_total = None
    if isinstance(details, dict) and details:
        n_total = len(details)
        n_success = sum(
            1 for v in details.values() if isinstance(v, dict) and v.get("success")
        )
    return {
        "delay": delay,
        "success_rate": data.get("success_rate"),
        "score": data.get("score"),
        "eval_time": data.get("eval_time"),
        "n_success": n_success,
        "n_total": n_total,
        "source": path,
    }


def collect_records(paths: List[str]) -> List[Dict[str, Any]]:
    """Find every ``delay_*/_result.json`` under the given paths and load it."""
    records: List[Dict[str, Any]] = []
    for raw in paths:
        path = os.path.abspath(raw)
        if os.path.isfile(path):
            delay = _delay_from_result_path(path)
            if delay is not None:
                records.append(_record_from_result_file(path, delay))
            continue
        if not os.path.isdir(path):
            print(f"[analyze] WARNING: not found: {path}", file=sys.stderr)
            continue
        found = sorted(glob.glob(os.path.join(path, "**", "delay_*", "_result.json"), recursive=True))
        for result_file in found:
            delay = _delay_from_result_path(result_file)
            if delay is not None:
                records.append(_record_from_result_file(result_file, delay))

    if records:
        return records

    # Fallback: sweeps that only have summary.json (e.g. _result.json moved away)
    for raw in paths:
        path = os.path.abspath(raw)
        summary_path = path if path.endswith(".json") else os.path.join(path, "summary.json")
        if not os.path.isfile(summary_path):
            continue
        for run in (_load_json(summary_path).get("runs") or []):
            if run.get("complete") is False:
                continue
            if "delay_frames" not in run:
                continue
            records.append(
                {
                    "delay": int(run["delay_frames"]),
                    "success_rate": run.get("success_rate"),
                    "score": run.get("score"),
                    "eval_time": run.get("eval_time"),
                    "n_success": None,
                    "n_total": None,
                    "source": summary_path,
                }
            )
    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _mean_std(values: List[float]) -> tuple:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None
    mean = statistics.fmean(clean)
    std = statistics.pstdev(clean) if len(clean) > 1 else 0.0
    return mean, std


def aggregate(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_delay: Dict[int, List[Dict[str, Any]]] = {}
    for rec in records:
        by_delay.setdefault(int(rec["delay"]), []).append(rec)

    rows: List[Dict[str, Any]] = []
    for delay in sorted(by_delay):
        group = by_delay[delay]
        sr_mean, sr_std = _mean_std([r["success_rate"] for r in group])
        sc_mean, sc_std = _mean_std([r["score"] for r in group])
        eval_time = sum(int(r["eval_time"]) for r in group if r.get("eval_time") is not None)
        rows.append(
            {
                "delay": delay,
                "n": len(group),
                "success_rate_mean": sr_mean,
                "success_rate_std": sr_std,
                "score_mean": sc_mean,
                "score_std": sc_std,
                "eval_time": eval_time,
            }
        )

    baseline = next((r for r in rows if r["delay"] == 0), None)
    for row in rows:
        if baseline and row["success_rate_mean"] is not None and baseline["success_rate_mean"] is not None:
            row["success_rate_delta"] = row["success_rate_mean"] - baseline["success_rate_mean"]
        else:
            row["success_rate_delta"] = None
        if baseline and row["score_mean"] is not None and baseline["score_mean"] is not None:
            row["score_delta"] = row["score_mean"] - baseline["score_mean"]
        else:
            row["score_delta"] = None
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(value: Optional[float], std: Optional[float] = None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if std is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ± {std:.{digits}f}"


def write_markdown(rows: List[Dict[str, Any]], out_path: str, title: str, sources: List[str]) -> None:
    lines = [f"# {title}", ""]
    lines.append(f"- generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- input(s): {', '.join(sources)}")
    lines.append("")
    lines.append("| delay (frames) | runs | success_rate | score | eval_time | Δ success vs d=0 | Δ score vs d=0 |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        d_sr = None if row["success_rate_delta"] is None else abs(row["success_rate_delta"])
        d_sc = None if row["score_delta"] is None else abs(row["score_delta"])
        sign_sr = "" if row["success_rate_delta"] is None else ("+" if row["success_rate_delta"] >= 0 else "-")
        sign_sc = "" if row["score_delta"] is None else ("+" if row["score_delta"] >= 0 else "-")
        lines.append(
            "| {d} | {n} | {sr} | {sc} | {et} | {dsr} | {dsc} |".format(
                d=row["delay"],
                n=row["n"],
                sr=_fmt(row["success_rate_mean"], row["success_rate_std"]),
                sc=_fmt(row["score_mean"], row["score_std"], digits=1),
                et=row["eval_time"],
                dsr="n/a" if d_sr is None else f"{sign_sr}{d_sr:.3f}",
                dsc="n/a" if d_sc is None else f"{sign_sc}{d_sc:.1f}",
            )
        )
    lines.append("")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    fields = [
        "delay",
        "n",
        "success_rate_mean",
        "success_rate_std",
        "score_mean",
        "score_std",
        "eval_time",
        "success_rate_delta",
        "score_delta",
    ]

    def _round(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return round(value, 6)
        return value

    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _round(row.get(key)) for key in fields})


def write_plot(rows: List[Dict[str, Any]], out_path: str, title: str) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib optional
        print(f"[analyze] matplotlib unavailable ({exc}); skipping plot.", file=sys.stderr)
        return None

    xs = [r["delay"] for r in rows]
    sr = [r["success_rate_mean"] for r in rows]
    sr_err = [r["success_rate_std"] or 0.0 for r in rows]
    sc = [r["score_mean"] for r in rows]
    sc_err = [r["score_std"] or 0.0 for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].errorbar(xs, sr, yerr=sr_err, marker="o", capsize=4, color="#1f77b4")
    axes[0].set_xlabel("camera delay (frames)")
    axes[0].set_ylabel("success rate")
    axes[0].set_title("Success rate vs camera delay")
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(xs, sc, yerr=sc_err, marker="s", capsize=4, color="#d62728")
    axes[1].set_xlabel("camera delay (frames)")
    axes[1].set_ylabel("score")
    axes[1].set_title("Score vs camera delay")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _latest_sweep(results_root: str) -> Optional[str]:
    if not os.path.isdir(results_root):
        return None
    candidates = [p for p in glob.glob(os.path.join(results_root, "*")) if os.path.isdir(p)]
    return sorted(candidates)[-1] if candidates else None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate camera_delay sweep results.")
    parser.add_argument(
        "--results",
        nargs="*",
        default=None,
        help="One or more sweep dirs (or the results root). Default: latest sweep under results/.",
    )
    parser.add_argument("--out", default=None, help="Output directory. Default: the input sweep dir.")
    parser.add_argument("--title", default="camera_delay: RGB frame-delay sweep")
    parser.add_argument("--no-plot", action="store_true", help="Skip the matplotlib figure.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.results:
        paths = list(args.results)
    else:
        latest = _latest_sweep(_DEFAULT_RESULTS)
        if latest is None:
            print(f"[analyze] no results under {_DEFAULT_RESULTS}; pass --results <dir>.", file=sys.stderr)
            return 2
        paths = [latest]

    records = collect_records(paths)
    if not records:
        print(f"[analyze] no _result.json/summary.json records found under: {paths}", file=sys.stderr)
        return 1

    rows = aggregate(records)

    if args.out:
        out_dir = args.out
    elif len(paths) == 1 and os.path.isdir(paths[0]):
        out_dir = paths[0]
    else:
        out_dir = os.path.join(_DEFAULT_RESULTS, f"_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    md_path = os.path.join(out_dir, "analysis.md")
    csv_path = os.path.join(out_dir, "analysis.csv")
    write_markdown(rows, md_path, args.title, paths)
    write_csv(rows, csv_path)

    print(f"[analyze] delays: {[r['delay'] for r in rows]}")
    print(f"[analyze] wrote {md_path}")
    print(f"[analyze] wrote {csv_path}")

    if not args.no_plot:
        plot_path = write_plot(rows, os.path.join(out_dir, "delay_vs_performance.png"), args.title)
        if plot_path:
            print(f"[analyze] wrote {plot_path}")

    # Console table
    print()
    print(f"{'delay':>5} {'runs':>5} {'success_rate':>16} {'score':>15} {'eval_time':>10}")
    for row in rows:
        sr = _fmt(row["success_rate_mean"], row["success_rate_std"])
        sc = _fmt(row["score_mean"], row["score_std"], digits=1)
        print(f"{row['delay']:>5} {row['n']:>5} {sr:>16} {sc:>15} {row['eval_time']:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
