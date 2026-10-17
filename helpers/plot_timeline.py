#!/usr/bin/env python3
"""ASCII-plot PM sampling timeseries and warp-state timelines from ncu reports.

PM sampling metrics have per-instance values that form a time-ordered series
across the kernel's execution. Plotting the series reveals tail effects,
pipeline bubbles, and sawtooth patterns that are invisible in aggregate metrics.

The series come from the PmSampling section (part of `--set full`), named
`<UNIT>.<group>.<metric>` (e.g. `TPC.TriageCompute.sm__cycles_active...`);
`--list` prints the series a report has and its timed warp-sample availability.
When the reader exposes timeline alignment, series use aligned workload
timestamps; older readers fall back to report-provided sample times.

ncu collects the series over replays of the kernel, and each replay has its own
time base. Series of one replay share a time axis: they are plotted together,
over the window where any of them is non-zero, so their shapes line up. Series
of different replays do not line up sample for sample — compare their shapes,
not positions.

The stall reasons over time come from the optional timed warp samples the same
section holds. Some architectures and report API versions do not provide these
samples. PM sampling timelines are unsupported on Volta/SM70.

Produces `<run-dir>/analysis/pm_timeline_<tag>.txt` per profiled kernel.

Usage:
    python3 plot_timeline.py --run-dir "$PROFILE_RUN_DIR" \\
            --report "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" --tag <tag> [--list]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ncu_utils import (  # noqa: E402
    action_architecture, action_tags, kernel_name, pm_series_names, pm_series_result,
    select_actions, timed_warp_sample_info, warp_state_timeline_result,
)


# Default series, as patterns: SM activity, issue rate, tensor pipe, memory
# throughputs, and the stall-reason timelines. Each pattern plots every series it
# matches; `--metric` replaces the list.
DEFAULT_PATTERNS = [
    r"\.sm__cycles_active\.avg",
    r"\.sm__inst_executed_realtime\.avg\.per_cycle_active$",
    r"\.sm__pipe_tensor_cycles_active_realtime\.",
    r"\.dram__throughput\.",
    r"\.lts__throughput\.",
    r"\.l1tex__throughput\.",
    r"^pmsampling:smsp__warps_issue_stalled_(long_scoreboard|short_scoreboard|wait|barrier|"
    r"math_pipe_throttle|mio_throttle|lg_throttle|sleeping)\.",
]


def ascii_plot(vals, label, max_rows=20, max_cols=80):
    """Return list of ASCII strings rendering the timeseries (already windowed)."""
    vals = [v if v is not None else 0.0 for v in vals]
    n = len(vals)
    if n == 0 or not any(vals):
        return [f"\n{label}: all zero"]

    # Bucket into max_cols columns
    ncols = min(max_cols, n)
    buckets = []
    for c in range(ncols):
        chunk = vals[c * n // ncols:(c + 1) * n // ncols]
        buckets.append(sum(chunk) / len(chunk) if chunk else 0.0)
    mx = max(buckets) or 1.0

    lines = [f"\n{label}", f"  (max={mx:.3g})"]
    for r in range(max_rows, 0, -1):
        threshold = mx * r / max_rows
        row = "".join("#" if b >= threshold else " " for b in buckets)
        lines.append(f"  {threshold:8.2g} | {row}")
    lines.append("  " + " " * 10 + "-" * len(buckets))
    lines.append("  " + " " * 10 + " (time →)")
    return lines


def _active_samples(group, n):
    """Sample indices where the kernel runs: SM activity above 1% of its peak when the group has
    it (memory series carry background noise for the whole capture), else where any series is
    above 1% of its own peak."""
    sm = [v for name, _, v in group if ".sm__cycles_active." in name]
    if sm:
        peak = max(x or 0 for x in sm[0]) or 1
        return [i for i in range(n) if (sm[0][i] or 0) > 0.01 * peak]
    peaks = [max(x or 0 for x in v) for _, _, v in group]
    return [i for i in range(n) if any((v[i] or 0) > 0.01 * pk for (_, _, v), pk in zip(group, peaks) if pk)]


def plot_replays(action, names, rows, cols):
    """Plot `names`, grouped by the replay (time base) each series was sampled in."""
    replays = {}
    alignment_statuses = set()
    alignment_errors = []
    for name in names:
        series, status, error = pm_series_result(action, name)
        alignment_statuses.add(status)
        if error:
            alignment_errors.append(error)
        if series is None:
            continue
        ts, vals = series
        key = (ts[0], ts[-1], len(ts)) if ts else name
        replays.setdefault(key, []).append((name, ts, vals))
    out = []
    if "alignment_error" in alignment_statuses:
        out.append("[note] timeline alignment failed; raw PM sample series are shown where available: "
                   + alignment_errors[0])
    elif alignment_statuses and alignment_statuses <= {"raw"}:
        out.append("[note] using report-provided sample times; this API exposes no aligned timeline view")
    elif "raw" in alignment_statuses:
        out.append("[note] one or more PM series use report-provided sample times without alignment")
    for k, group in enumerate(replays.values()):
        ts = group[0][1]
        n = len(group[0][2])
        active = _active_samples(group, n)
        lo, hi = (active[0], active[-1] + 1) if active else (0, n)
        span = f", {(ts[hi - 1] - ts[lo]) / 1e3:.1f} us" if ts and hi - lo > 1 else ""
        interval = f", ~{(ts[-1] - ts[0]) / (len(ts) - 1) / 1e3:.3g} us/sample" if ts and len(ts) > 1 else ""
        out.append(f"\n----- replay {k}: samples {lo}..{hi} of {n}{span}{interval}; "
                   f"these {len(group)} series share this time axis -----")
        for name, _, vals in group:
            out.extend(ascii_plot(vals[lo:hi], name, rows, cols))
    return out


def plot_warp_states(action, rows, cols, top, timeline_result=None):
    """The `top` most sampled stall reasons over the kernel's run, from its timed warp samples."""
    if timeline_result is None:
        timeline_result = warp_state_timeline_result(action, cols)
    result, status, error = timeline_result
    if result is None:
        if status == "error":
            return [f"\n[note] timed warp-sample processing failed: {error}"]
        return []
    span, shares = result
    order = sorted(shares, key=lambda r: -sum(shares[r]))[:top]
    out = [f"\n----- warp states over the kernel: share of the timed warp samples per reason, "
           f"{span / 1e3:.1f} us (its own time axis) -----"]
    for reason in order:
        out.extend(ascii_plot(shares[reason], f"stall reason: {reason}", rows, cols))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, action="append", required=True)
    ap.add_argument("--tag", type=str, action="append", required=True)
    ap.add_argument("--metric", type=str, action="append", default=None,
                    help="Regex on series names (repeatable); replaces the default list.")
    ap.add_argument("--kernel", type=str, default=None,
                    help="Regex on the kernel's function or demangled name: keep only matching kernels.")
    ap.add_argument("--list", action="store_true", help="Print the series and timed warp sample count of each report and exit.")
    ap.add_argument("--stall-reasons", type=int, default=6,
                    help="How many of the most sampled stall reasons to plot over time (0: none).")
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--cols", type=int, default=80)
    args = ap.parse_args()

    if len(args.report) != len(args.tag):
        ap.error("--report and --tag counts must match")
    if len(set(args.tag)) != len(args.tag):
        ap.error("each --tag must be different: outputs are named after it")

    patterns = [re.compile(p) for p in (args.metric or DEFAULT_PATTERNS)]
    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for rep, tag in zip(args.report, args.tag):
        if not rep.exists():
            print(f"[skip] {rep} not found", file=sys.stderr)
            continue
        _report, acts = select_actions(rep, args.kernel)
        if not acts:
            print(f"[skip] {rep}: no kernel matches {args.kernel!r}", file=sys.stderr)
            continue
        for action, t in zip(acts, action_tags(tag, acts)):
            names = pm_series_names(action)
            arch = action_architecture(action)
            warp_status, warp_count, warp_error = timed_warp_sample_info(action)
            if args.list:
                print(f"[{t}] {len(names)} PM series, timed warp samples: "
                      f"{warp_count if warp_count is not None else warp_status}; architecture: {arch}")
                print("\n".join(f"  {n}" for n in names))
                if arch == "sm_70":
                    print("  PM sampling timelines are unsupported on Volta/SM70.")
                continue
            if arch == "sm_70":
                print(f"[{t}] PM sampling timelines are unsupported on Volta/SM70; "
                      "use aggregate metrics and source counters instead.", file=sys.stderr)
                continue
            warp_timeline = warp_state_timeline_result(action, args.cols) \
                if args.stall_reasons else (None, "disabled", None)
            warp_states = plot_warp_states(
                action, args.rows, args.cols, args.stall_reasons, warp_timeline
            ) if args.stall_reasons else []
            if not names and not warp_states:
                if warp_timeline[1] == "error":
                    reason = f"timed warp-sample processing failed: {warp_timeline[2]}"
                elif warp_status == "unsupported":
                    reason = "the reader API has no timed warp-sample method and the report has no PM series"
                elif warp_status == "error":
                    reason = f"timed warp-sample processing failed: {warp_error}"
                elif warp_status == "no_data":
                    reason = "the report has no timed warp samples or PM series"
                else:
                    reason = "no PM sampling measurements were found"
                print(f"[{t}] timeline unavailable: {reason}; aggregate analysis remains available",
                      file=sys.stderr)
                continue
            dur = action["gpu__time_duration.sum"].value() if "gpu__time_duration.sum" in action.metric_names() else None
            dur = f" ({dur / 1e3:.1f} us, gpu__time_duration)" if dur else ""
            out_lines = [f"{'=' * 60}\n{t}: {rep.name}\n{kernel_name(action, 'function')}{dur}\n"
                         f"Architecture: {arch}\nTimed warp samples: {warp_status}"
                         f"{f' ({warp_count})' if warp_count is not None else ''}\n"
                         f"Warp-state timeline: {warp_timeline[1]}"
                         f"{f' ({warp_timeline[2]})' if warp_timeline[2] else ''}\n{'=' * 60}"]
            chosen = [n for p in patterns for n in names if p.search(n)]
            out_lines.extend(plot_replays(action, list(dict.fromkeys(chosen)), args.rows, args.cols))
            out_lines.extend(warp_states)
            out_path = analysis_dir / f"pm_timeline_{t}.txt"
            out_path.write_text("\n".join(out_lines) + "\n")
            print(f"[{t}] -> {out_path}")


if __name__ == "__main__":
    main()
