#!/usr/bin/env python3
"""
compare_events.py - compares two or more captured event runs on exactly
two measures:

  1. Event rate: event count in fixed-width time bins (default 100 ms)
  2. Per-pixel inter-event interval (tau): time since the last event at
     the same pixel (DVS-Voltmeter paper, Fig. 5 style), plotted on a
     LINEAR axis from 0 to --tau-max-ms (default 40 ms). Intervals longer
     than --tau-cutoff-ms (default 30 ms) are excluded: those are revisits
     that span more than one frame, not events generated between two
     successive frames.

Reads the CSV pairs written by EventLogger (<label>.events.csv,
<label>.frames.csv).

Usage:
    python3 compare_events.py raw clahe
    python3 compare_events.py raw clahe --bin-ms 50
    python3 compare_events.py raw clahe --tau-max-ms 40 --tau-cutoff-ms 30 --tau-bins 80
"""

import argparse
import sys
from itertools import combinations

import numpy as np
import pandas as pd

# Large, clear defaults for every plot this script makes.
FONT_TITLE = 20
FONT_LABEL = 17
FONT_TICK = 14
FONT_LEGEND = 14
LINEWIDTH = 2.5
DPI = 180


def load_run(label: str):
    events = pd.read_csv(f"{label}.events.csv")
    frames = pd.read_csv(f"{label}.frames.csv")
    return events, frames


def bin_counts(frames: pd.DataFrame, bin_ms: float) -> np.ndarray:
    """Event count per fixed-width time bin. Missing bins count as 0."""
    bin_s = bin_ms / 1000.0
    frames = frames.copy()
    frames["bin"] = (frames["t"] // bin_s).astype(int)
    per_bin = frames.groupby("bin")["event_count"].sum()
    full_range = np.arange(per_bin.index.min(), per_bin.index.max() + 1)
    return per_bin.reindex(full_range, fill_value=0).to_numpy()


def compute_tau(events: pd.DataFrame) -> np.ndarray:
    """Per-pixel inter-event interval: for every event, time since the
    last event at that same pixel. A pixel's first event has no
    predecessor and is dropped. Returned in seconds."""
    df = events.sort_values(["x", "y", "t"])
    tau = df.groupby(["x", "y"])["t"].diff()
    return tau.dropna().to_numpy()


def plot_rate(label_a, bin_ms, ba, label_b, bb, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 8))

    t_axis_a = np.arange(len(ba)) * bin_ms / 1000.0
    t_axis_b = np.arange(len(bb)) * bin_ms / 1000.0
    ax.plot(t_axis_a, ba, label=label_a, linewidth=LINEWIDTH, color="tab:blue")
    ax.plot(t_axis_b, bb, label=label_b, linewidth=LINEWIDTH, color="tab:orange")
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


def plot_tau(label_a, tau_a, label_b, tau_b, out_path,
             tau_max_ms=40.0, tau_cutoff_ms=30.0, n_bins=80):
    """Linear-axis tau PDF over [0, tau_max_ms].

    tau values (seconds) above tau_cutoff_ms are discarded before the
    histogram is built, so the density is normalised over the kept
    intervals only (they are not counted in the denominator).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 8))

    # Never let the cutoff exceed the plotted range.
    cutoff_us = min(tau_cutoff_ms, tau_max_ms) * 1e3
    bins_us = np.linspace(0.0, tau_max_ms * 1e3, n_bins + 1)

    print(f"\n  [tau plot: linear 0-{tau_max_ms:g} ms, "
          f"keeping tau <= {min(tau_cutoff_ms, tau_max_ms):g} ms]")

    for label, tau, color in ((label_a, tau_a, "tab:blue"),
                              (label_b, tau_b, "tab:orange")):
        tau_us = tau[tau > 0] * 1e6
        if len(tau_us) == 0:
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


def compare_pair(label_a, events_a, frames_a, label_b, events_b, frames_b,
                 bin_ms, make_plots, tau_max_ms, tau_cutoff_ms, tau_bins):
    print(f"\n{'='*70}\n{label_a}  vs  {label_b}\n{'='*70}")

    # ---- 1. Events per time bin ----
    ba = bin_counts(frames_a, bin_ms)
    bb = bin_counts(frames_b, bin_ms)

    print(f"\n[events per {bin_ms:.0f} ms bin]")
    print(f"  {label_a}: {len(ba)} bins  mean={ba.mean():.2f}  std={ba.std(ddof=1):.2f}")
    print(f"  {label_b}: {len(bb)} bins  mean={bb.mean():.2f}  std={bb.std(ddof=1):.2f}")

    # ---- 2. Per-pixel inter-event interval (tau) ----
    tau_a = compute_tau(events_a)
    tau_b = compute_tau(events_b)

    print(f"\n[per-pixel inter-event interval (tau), all revisits]")
    for label, tau in ((label_a, tau_a), (label_b, tau_b)):
        if len(tau):
            print(f"  {label}: n={len(tau)} pixel-revisits  "
                  f"median={np.median(tau)*1e6:.1f} us  mean={tau.mean()*1e6:.1f} us")
        else:
            print(f"  {label}: no repeat events at any pixel")

    if make_plots:
        plot_rate(label_a, bin_ms, ba, label_b, bb,
                  f"rate_{label_a}_vs_{label_b}.png")
        plot_tau(label_a, tau_a, label_b, tau_b,
                 f"tau_{label_a}_vs_{label_b}.png",
                 tau_max_ms=tau_max_ms, tau_cutoff_ms=tau_cutoff_ms,
                 n_bins=tau_bins)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="+", help="run labels, e.g. raw clahe clahe_custom")
    ap.add_argument("--bin-ms", type=float, default=100.0,
                    help="time bin width (ms) for the event-rate comparison (default 100)")
    ap.add_argument("--tau-max-ms", type=float, default=40.0,
                    help="upper x-limit of the linear tau plot in ms (default 40)")
    ap.add_argument("--tau-cutoff-ms", type=float, default=30.0,
                    help="exclude tau above this many ms from the tau plot (default 30)")
    ap.add_argument("--tau-bins", type=int, default=80,
                    help="number of linear tau histogram bins (default 80)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    runs = {}
    for label in args.labels:
        try:
            runs[label] = load_run(label)
        except FileNotFoundError as e:
            print(f"error: could not load '{label}' - {e}", file=sys.stderr)
            sys.exit(1)

    for label_a, label_b in combinations(args.labels, 2):
        events_a, frames_a = runs[label_a]
        events_b, frames_b = runs[label_b]
        compare_pair(
            label_a, events_a, frames_a,
            label_b, events_b, frames_b,
            args.bin_ms,
            make_plots=not args.no_plots,
            tau_max_ms=args.tau_max_ms,
            tau_cutoff_ms=args.tau_cutoff_ms,
            tau_bins=args.tau_bins,
        )


if __name__ == "__main__":
    main()
