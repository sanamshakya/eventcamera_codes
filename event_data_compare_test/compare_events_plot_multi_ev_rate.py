#!/usr/bin/env python3
"""
compare_events.py - compares any number (>= 2) of captured event runs on
exactly two measures, with every run drawn together on one figure:

  1. Event rate: event count in fixed-width time bins (default 100 ms)
  2. Per-pixel inter-event interval (tau): time since the last event at
     the same pixel (DVS-Voltmeter paper, Fig. 5 style), plotted on a
     LINEAR axis from 0 to --tau-max-ms (default 40 ms). Intervals longer
     than --tau-cutoff-ms (default 30 ms) are excluded: those are revisits
     that span more than one frame, not events generated between two
     successive frames.

The first --skip-s seconds (default 5) of every run are ignored.

Reads the CSV pairs written by EventLogger (<label>.events.csv,
<label>.frames.csv). Each label on the command line is one run.

Usage:
    python3 compare_events.py raw clahe
    python3 compare_events.py prophesee raw clahe clahe_custom sim5 sim6
    python3 compare_events.py a b c d --bin-ms 50 --skip-s 5
    python3 compare_events.py a b c --tau-max-ms 40 --tau-cutoff-ms 30 --tau-bins 80
    python3 compare_events.py a b c --out-prefix myexp   # myexp_rate.png, myexp_tau.png
"""

import argparse
import sys

import numpy as np
import pandas as pd

# Large, clear defaults for every plot this script makes.
FONT_TITLE = 20
FONT_LABEL = 17
FONT_TICK = 14
FONT_LEGEND = 14
LINEWIDTH = 2.5
DPI = 180


def get_colors(n: int):
    """n distinct colours: tab10 for up to 10 runs, tab20 beyond that."""
    import matplotlib
    cmap = matplotlib.colormaps["tab10" if n <= 10 else "tab20"]
    return [cmap(i % cmap.N) for i in range(n)]


def load_run(label: str):
    events = pd.read_csv(f"{label}.events.csv")
    frames = pd.read_csv(f"{label}.frames.csv")
    return events, frames


def trim_run(events: pd.DataFrame, frames: pd.DataFrame, skip_s: float):
    """Drop everything in the first skip_s seconds of a run (warm-up).

    Time zero is the earliest timestamp in that run's own logs, so runs
    that were not started at the same absolute clock are trimmed the same
    way. Remaining timestamps are shifted so the kept data starts at t=0.
    Because the trimmed events are removed before tau is computed, the
    first kept event at each pixel has no predecessor and is dropped
    there; warm-up events are never used as the "last event".
    """
    if skip_s <= 0:
        return events, frames
    t0 = min(events["t"].min() if len(events) else np.inf,
             frames["t"].min() if len(frames) else np.inf)
    if not np.isfinite(t0):
        return events, frames
    cut = t0 + skip_s
    events = events[events["t"] >= cut].copy()
    frames = frames[frames["t"] >= cut].copy()
    events["t"] -= cut
    frames["t"] -= cut
    return events, frames


def bin_counts(frames: pd.DataFrame, bin_ms: float) -> np.ndarray:
    """Event count per fixed-width time bin. Missing bins count as 0."""
    bin_s = bin_ms / 1000.0
    frames = frames.copy()
    frames["bin"] = (frames["t"] // bin_s).astype(int)
    per_bin = frames.groupby("bin")["event_count"].sum()
    full_range = np.arange(per_bin.index.min(), per_bin.index.max() + 1)
    return per_bin.reindex(full_range, fill_value=0).to_numpy()


def rate_stats(frames: pd.DataFrame, counts: np.ndarray, bin_ms: float):
    """Average and peak event rate in Mega events/s.

    avg = total events / duration, where duration is the span between the
          first and last frame timestamp of the (trimmed) run.
    max = highest event count in any one time bin / bin width, so it
          depends on --bin-ms (a smaller bin gives a higher, spikier peak).
    """
    total = float(frames["event_count"].sum())
    span = float(frames["t"].max() - frames["t"].min())
    avg = total / span / 1e6 if span > 0 else float("nan")
    peak = float(counts.max()) / (bin_ms / 1000.0) / 1e6
    return total, span, avg, peak


def compute_tau(events: pd.DataFrame) -> np.ndarray:
    """Per-pixel inter-event interval: for every event, time since the
    last event at that same pixel. A pixel's first event has no
    predecessor and is dropped. Returned in seconds."""
    df = events.sort_values(["x", "y", "t"])
    tau = df.groupby(["x", "y"])["t"].diff()
    return tau.dropna().to_numpy()


def plot_rate(labels, bin_ms, counts, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 8))
    colors = get_colors(len(labels))

    for label, c, color in zip(labels, counts, colors):
        t_axis = np.arange(len(c)) * bin_ms / 1000.0
        ax.plot(t_axis, c, label=label, linewidth=LINEWIDTH, color=color)

    ax.set_title(f"Events per {bin_ms:.0f} ms bin", fontsize=FONT_TITLE, pad=14)
    ax.set_xlabel("time (s)", fontsize=FONT_LABEL)
    ax.set_ylabel("event count", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND, loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout(pad=2.5)
    plt.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"\n  [plot saved: {out_path}]")


def plot_tau(labels, taus, out_path,
             tau_max_ms=40.0, tau_cutoff_ms=30.0, n_bins=80):
    """Linear-axis tau PDF over [0, tau_max_ms] for every run.

    tau values (seconds) above tau_cutoff_ms are discarded before the
    histogram is built, so each density is normalised over that run's
    kept intervals only (they are not counted in the denominator).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 8))
    colors = get_colors(len(labels))

    # Never let the cutoff exceed the plotted range.
    cutoff_us = min(tau_cutoff_ms, tau_max_ms) * 1e3
    bins_us = np.linspace(0.0, tau_max_ms * 1e3, n_bins + 1)

    print(f"\n[tau plot: linear 0-{tau_max_ms:g} ms, "
          f"keeping tau <= {min(tau_cutoff_ms, tau_max_ms):g} ms]")

    for label, tau, color in zip(labels, taus, colors):
        tau_us = tau[tau > 0] * 1e6
        if len(tau_us) == 0:
            print(f"  {label}: no repeat events, not plotted")
            continue
        kept = tau_us[tau_us <= cutoff_us]
        dropped = len(tau_us) - len(kept)
        print(f"  {label}: kept {len(kept)} of {len(tau_us)} intervals "
              f"({100.0 * len(kept) / len(tau_us):.1f}%), "
              f"excluded {dropped} above {cutoff_us / 1e3:g} ms")
        if len(kept) == 0:
            continue
        ax.hist(kept, bins=bins_us, density=True, histtype="step",
                linewidth=LINEWIDTH, color=color,
                label=f"{label} (n={len(kept)})")

    ax.set_xlim(0, tau_max_ms * 1e3)
    ax.set_title(r"Per-pixel inter-event interval $\tau$ "
                 r"(time since last event, same pixel)",
                 fontsize=FONT_TITLE, pad=14)
    ax.set_xlabel(r"$\tau$ (µs)", fontsize=FONT_LABEL)
    ax.set_ylabel("probability density", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND, loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout(pad=2.5)
    plt.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  [plot saved: {out_path}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="+",
                    help="run labels (any number >= 2), e.g. raw clahe clahe_custom")
    ap.add_argument("--bin-ms", type=float, default=100.0,
                    help="time bin width (ms) for the event-rate comparison (default 100)")
    ap.add_argument("--tau-max-ms", type=float, default=40.0,
                    help="upper x-limit of the linear tau plot in ms (default 40)")
    ap.add_argument("--tau-cutoff-ms", type=float, default=30.0,
                    help="exclude tau above this many ms from the tau plot (default 30)")
    ap.add_argument("--tau-bins", type=int, default=80,
                    help="number of linear tau histogram bins (default 80)")
    ap.add_argument("--skip-s", type=float, default=5.0,
                    help="ignore the first N seconds of each run (default 5; 0 = use everything)")
    ap.add_argument("--out-prefix", default="compare",
                    help="output file prefix: <prefix>_rate.png, <prefix>_tau.png (default 'compare')")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    labels = args.labels
    if len(labels) < 2:
        ap.error("need at least two runs to compare")
    if len(set(labels)) != len(labels):
        ap.error("duplicate run labels given")

    # ---- load + trim every run once ----
    runs = {}
    for label in labels:
        try:
            events, frames = load_run(label)
        except FileNotFoundError as e:
            print(f"error: could not load '{label}' - {e}", file=sys.stderr)
            sys.exit(1)
        n_ev, n_fr = len(events), len(frames)
        events, frames = trim_run(events, frames, args.skip_s)
        print(f"{label}: {n_ev} events / {n_fr} frames -> "
              f"{len(events)} / {len(frames)} after skipping first {args.skip_s:g} s")
        if len(events) == 0 or len(frames) == 0:
            print(f"error: '{label}' has no data after skipping {args.skip_s:g} s "
                  f"(run shorter than the skip?)", file=sys.stderr)
            sys.exit(1)
        runs[label] = (events, frames)

    print(f"\n{'='*70}\ncomparing {len(labels)} runs: {', '.join(labels)}\n{'='*70}")

    # ---- 1. Events per time bin ----
    counts = [bin_counts(runs[l][1], args.bin_ms) for l in labels]
    print(f"\n[events per {args.bin_ms:.0f} ms bin]")
    for label, c in zip(labels, counts):
        std = c.std(ddof=1) if len(c) > 1 else float("nan")
        print(f"  {label}: {len(c)} bins  mean={c.mean():.2f}  std={std:.2f}")

    # ---- 1b. Average / peak event rate in Mev/s ----
    print(f"\n[event rate, Mev/s  (peak uses {args.bin_ms:.0f} ms bins)]")
    print(f"  {'run':<24}{'events':>14}{'duration s':>12}{'avg Mev/s':>12}{'max Mev/s':>12}")
    for label, c in zip(labels, counts):
        total, span, avg, peak = rate_stats(runs[label][1], c, args.bin_ms)
        print(f"  {label:<24}{total:>14.0f}{span:>12.2f}{avg:>12.4f}{peak:>12.4f}")

    # ---- 2. Per-pixel inter-event interval (tau) ----
    taus = [compute_tau(runs[l][0]) for l in labels]
    print(f"\n[per-pixel inter-event interval (tau), all revisits]")
    for label, tau in zip(labels, taus):
        if len(tau):
            print(f"  {label}: n={len(tau)} pixel-revisits  "
                  f"median={np.median(tau)*1e6:.1f} us  mean={tau.mean()*1e6:.1f} us")
        else:
            print(f"  {label}: no repeat events at any pixel")

    if not args.no_plots:
        plot_rate(labels, args.bin_ms, counts, f"{args.out_prefix}_rate.png")
        plot_tau(labels, taus, f"{args.out_prefix}_tau.png",
                 tau_max_ms=args.tau_max_ms, tau_cutoff_ms=args.tau_cutoff_ms,
                 n_bins=args.tau_bins)


if __name__ == "__main__":
    main()
