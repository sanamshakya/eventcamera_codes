#!/usr/bin/env python3
"""
compare_events.py - compares two or more captured event runs on exactly
two measures:

  1. Event rate: event count in fixed-width time bins (default 100 ms)
  2. Per-pixel inter-event interval (tau): time since the last event at
     the same pixel (DVS-Voltmeter paper, Fig. 5 style)

No other statistics are computed. Reads the CSV pairs written by
EventLogger (<label>.events.csv, <label>.frames.csv).

Usage:
    python3 compare_events.py raw clahe
    python3 compare_events.py raw clahe clahe_custom --bin-ms 50
"""

import argparse
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

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


def plot_tau(label_a, tau_a, label_b, tau_b, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 8))

    all_tau = np.concatenate([tau_a[tau_a > 0], tau_b[tau_b > 0]]) if len(tau_a) and len(tau_b) else np.array([])
    if len(all_tau) > 0:
        lo, hi = np.percentile(all_tau, [0.1, 99.9])
        lo = max(lo, all_tau.min())
        bins = np.logspace(np.log10(lo), np.log10(hi), 60)

        for label, tau, color in ((label_a, tau_a, "tab:blue"), (label_b, tau_b, "tab:orange")):
            tau_pos = tau[tau > 0]
            if len(tau_pos) == 0:
                continue
            ax.hist(tau_pos * 1e6, bins=bins * 1e6, density=True, histtype="step",
                    linewidth=LINEWIDTH, label=f"{label} (n={len(tau_pos)})", color=color)
        ax.set_xscale("log")

    ax.set_title(r"Per-pixel inter-event interval $\tau$ (time since last event, same pixel)",
                 fontsize=FONT_TITLE, pad=14)
    ax.set_xlabel(r"$\tau$ (µs, log scale)", fontsize=FONT_LABEL)
    ax.set_ylabel("probability density", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND, loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout(pad=2.5)
    plt.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"\n  [plot saved: {out_path}]")


def compare_pair(label_a, events_a, frames_a, label_b, events_b, frames_b,
                  bin_ms, alpha, make_plots):
    print(f"\n{'='*70}\n{label_a}  vs  {label_b}\n{'='*70}")

    # ---- 1. Events per time bin ----
    ba = bin_counts(frames_a, bin_ms)
    bb = bin_counts(frames_b, bin_ms)

    print(f"\n[events per {bin_ms:.0f} ms bin]")
    print(f"  {label_a}: {len(ba)} bins  mean={ba.mean():.2f}  std={ba.std(ddof=1):.2f}")
    print(f"  {label_b}: {len(bb)} bins  mean={bb.mean():.2f}  std={bb.std(ddof=1):.2f}")

    #ks_rate = stats.ks_2samp(ba, bb)
    #print(f"  KS test: D={ks_rate.statistic:.4f}  p={ks_rate.pvalue:.4g}"
    #      f"  {'-> SIGNIFICANT' if ks_rate.pvalue < alpha else '(not significant)'}")

    # ---- 2. Per-pixel inter-event interval (tau) ----
    tau_a = compute_tau(events_a)
    tau_b = compute_tau(events_b)

    print(f"\n[per-pixel inter-event interval (tau)]")
    if len(tau_a):
        print(f"  {label_a}: n={len(tau_a)} pixel-revisits  "
              f"median={np.median(tau_a)*1e6:.1f} us  mean={tau_a.mean()*1e6:.1f} us")
    else:
        print(f"  {label_a}: no repeat events at any pixel")
    if len(tau_b):
        print(f"  {label_b}: n={len(tau_b)} pixel-revisits  "
              f"median={np.median(tau_b)*1e6:.1f} us  mean={tau_b.mean()*1e6:.1f} us")
    else:
        print(f"  {label_b}: no repeat events at any pixel")

    #ks_tau = None
    #if len(tau_a) > 1 and len(tau_b) > 1:
    #    ks_tau = stats.ks_2samp(tau_a, tau_b)
    #    print(f"  KS test: D={ks_tau.statistic:.4f}  p={ks_tau.pvalue:.4g}"
    #          f"  {'-> SIGNIFICANT' if ks_tau.pvalue < alpha else '(not significant)'}")
    #else:
    #    print("  (not enough pixel-revisits in one or both runs to compare)")

    if make_plots:
        plot_rate(label_a, bin_ms, ba, label_b, bb,
                  f"rate_{label_a}_vs_{label_b}.png")
        plot_tau(label_a, tau_a, label_b, tau_b,
                 f"tau_{label_a}_vs_{label_b}.png")

    return {
        "pair": (label_a, label_b),
        #"rate_ks_p": ks_rate.pvalue,
        #"tau_ks_p": ks_tau.pvalue if ks_tau is not None else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="+", help="run labels, e.g. raw clahe clahe_custom")
    ap.add_argument("--bin-ms", type=float, default=100.0,
                     help="time bin width (ms) for the event-rate comparison (default 100)")
    ap.add_argument("--alpha", type=float, default=0.05, help="significance threshold")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    runs = {}
    for label in args.labels:
        try:
            runs[label] = load_run(label)
        except FileNotFoundError as e:
            print(f"error: could not load '{label}' - {e}", file=sys.stderr)
            sys.exit(1)

    results = []
    for label_a, label_b in combinations(args.labels, 2):
        events_a, frames_a = runs[label_a]
        events_b, frames_b = runs[label_b]
        results.append(compare_pair(
            label_a, events_a, frames_a,
            label_b, events_b, frames_b,
            args.bin_ms, args.alpha,
            make_plots=not args.no_plots,
        ))

    if len(args.labels) > 2:
        print(f"\n{'='*70}\nsummary (p-values)\n{'='*70}")
        summary = pd.DataFrame(results).set_index("pair")
        print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
