#!/usr/bin/env python3
"""analyze_intensity_csv.py

Reads the per-frame raw-intensity CSV files written by
CIntensityCaptureWriter::WriteFrame() (see CIntensityCaptureWriter.hpp /
CNvSIPLConsumer::EnableIntensityCSVCapture() in the NVSIPL consumer) -
files named "<prefix>_frame_<N>.csv", each a plain width x height grid of
uint16_t values, comma-separated, no header row - and:

  1. prints per-frame and overall min/max intensity
  2. computes dynamic range from those min/max values, two ways:
       - in dB:    20 * log10(max / min)
       - in stops: log2(max / min)       (doublings of signal, as in photography)
     plus how many of the container's bits that span actually uses
     (ceil(log2(max - min + 1))), so you can see e.g. "using 11 of 16 bits."
  3. saves a histogram plot (log-scaled y-axis by default, since real
     sensor histograms are usually sharply peaked around a baseline with
     a long thin tail) as a PNG.

USAGE
    python3 analyze_intensity_csv.py --prefix /path/to/run1
    python3 analyze_intensity_csv.py --prefix /path/to/run1 --bins 512 --linear-y
    python3 analyze_intensity_csv.py --prefix /path/to/run1 --frames 0,1,2
    python3 analyze_intensity_csv.py --prefix /path/to/run1 --out histogram.png

This looks for files matching "<prefix>_frame_*.csv" (glob), so --prefix
should be the exact same string passed to EnableIntensityCSVCapture() on
the C++ side, including any directory part.

DYNAMIC RANGE CAVEAT: min/max-based dynamic range is a coarse, single-frame
estimate (signal span / noise floor, using the frame's darkest and
brightest pixel values as stand-ins for noise floor and full-scale signal).
It is NOT a substitute for a proper sensor dynamic-range characterization
(which needs a controlled light source and multiple exposures to separate
read noise from the signal itself) - treat this as a quick sanity check on
captured data, not a sensor datasheet number.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe (embedded/SSH boxes have no display)
    import matplotlib.pyplot as plt
except ImportError:
    print("error: matplotlib is required (pip install matplotlib)", file=sys.stderr)
    sys.exit(1)


# ----------------------------------------------------------------------
# A small, print-friendly single-hue palette (sequential: this is one
# quantitative series, a pixel-intensity histogram, not categorical data -
# no need for a multi-hue categorical palette here).
# ----------------------------------------------------------------------
HIST_FILL = "#3B6FA0"     # muted blue, readable on white, prints fine in grayscale
HIST_EDGE = "#1F4066"
GRID_COLOR = "#D8DCE1"
TEXT_COLOR = "#2B2F36"
ANNOT_BG = "#F4F6F8"


def find_frame_files(prefix, frame_list=None):
    """Return sorted (frame_index, path) pairs for <prefix>_frame_<N>.csv."""
    pattern = f"{prefix}_frame_*.csv"
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(
            f"no files matched '{pattern}' - check --prefix (it should be the "
            f"exact same string passed to EnableIntensityCSVCapture() in C++, "
            f"directory included)"
        )

    rx = re.compile(r"_frame_(\d+)\.csv$")
    indexed = []
    for p in paths:
        m = rx.search(os.path.basename(p))
        if not m:
            continue
        indexed.append((int(m.group(1)), p))
    indexed.sort(key=lambda t: t[0])

    if frame_list is not None:
        wanted = set(frame_list)
        indexed = [(i, p) for i, p in indexed if i in wanted]
        missing = wanted - {i for i, _ in indexed}
        if missing:
            print(f"warning: requested frame(s) not found: {sorted(missing)}", file=sys.stderr)

    if not indexed:
        raise FileNotFoundError("no frame files left after filtering by --frames")

    return indexed


def load_frame(path):
    """Load one <prefix>_frame_<N>.csv as a uint16 2D array."""
    # genfromtxt is dependency-free (no pandas requirement) and plenty fast
    # for the handful of frames this capture is capped at (see
    # CIntensityCaptureWriter.hpp's numFrames cap).
    arr = np.genfromtxt(path, delimiter=",", dtype=np.uint32)
    return arr.astype(np.uint16)


def compute_dynamic_range(min_val, max_val, container_bits=16):
    """Returns dict with ratio/dB/stops/bits_used, handling min==0 safely.

    min_val == 0 (a genuinely black pixel) makes max/min undefined, so the
    noise-floor stand-in falls back to 1 LSB for the ratio-based figures
    only - the printed min/max themselves are never altered.
    """
    floor = max(int(min_val), 1)
    ratio = float(max_val) / float(floor)
    span = max(int(max_val) - int(min_val), 0)
    bits_used = int(np.ceil(np.log2(span + 1))) if span > 0 else 0

    return {
        "min": int(min_val),
        "max": int(max_val),
        "floor_used_for_ratio": floor,
        "ratio": ratio,
        "db": 20.0 * np.log10(ratio),
        "stops": np.log2(ratio),
        "bits_used": bits_used,
        "container_bits": container_bits,
        "container_max": (1 << container_bits) - 1,
    }


def print_report(per_frame_stats, overall, container_bits):
    print("\n=== Per-frame intensity stats ===")
    print(f"{'frame':>6}  {'min':>6}  {'max':>6}  {'mean':>9}  {'std':>9}")
    for idx, mn, mx, mean, std in per_frame_stats:
        print(f"{idx:>6}  {mn:>6}  {mx:>6}  {mean:>9.2f}  {std:>9.2f}")

    print("\n=== Overall (all loaded frames combined) ===")
    print(f"min intensity : {overall['min']}")
    print(f"max intensity : {overall['max']}")
    print(f"mean          : {overall['mean']:.2f}")
    print(f"std           : {overall['std']:.2f}")

    
    print()


def plot_histogram(frames, overall, bins, log_y, out_path, container_bits, title):
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    all_values = np.concatenate([f.ravel() for f in frames])

    counts, edges, _ = ax.hist(
        all_values, bins=bins,
        color=HIST_FILL, edgecolor=HIST_EDGE, linewidth=0.4,
        alpha=0.95,
    )

    if log_y:
        ax.set_yscale("log")
        ax.set_ylabel("Pixel count (log scale)", color=TEXT_COLOR, fontsize=10)
    else:
        ax.set_ylabel("Pixel count", color=TEXT_COLOR, fontsize=10)

    ax.set_xlabel("Intensity (raw sensor value)", color=TEXT_COLOR, fontsize=10)
    ax.set_title(title, color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=12)

    # Recessive grid/axes - data carries the weight, chrome stays quiet.
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)

    # min/max markers
    ax.axvline(overall["min"], color="#B0392B", linestyle="--", linewidth=1.2, zorder=3)
    ax.axvline(overall["max"], color="#B0392B", linestyle="--", linewidth=1.2, zorder=3)

    dr = compute_dynamic_range(overall["min"], overall["max"], container_bits)
    annot = (
        f"min = {overall['min']}\n"
        f"max = {overall['max']}\n"
        f"dynamic range ≈ {dr['db']:.1f} dB ({dr['stops']:.1f} stops)"
    )
    ax.text(
        0.98, 0.95, annot,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color=TEXT_COLOR,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=ANNOT_BG, edgecolor=GRID_COLOR, linewidth=0.7),
    )

    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"histogram saved to {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Histogram + min/max/dynamic-range report for captured raw-intensity CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--prefix", required=True,
                     help="path prefix passed to EnableIntensityCSVCapture() in C++ "
                          "(files matched as '<prefix>_frame_*.csv')")
    ap.add_argument("--frames", default=None,
                     help="comma-separated frame indices to include (default: all found)")
    ap.add_argument("--bins", type=int, default=256,
                     help="histogram bin count (default: 256)")
    ap.add_argument("--linear-y", action="store_true",
                     help="use a linear y-axis instead of the default log scale")
    ap.add_argument("--container-bits", type=int, default=16,
                     help="bit depth of the storage container (default: 16, matching "
                          "the uint16_t buffer the consumer writes - NOT necessarily "
                          "the sensor's native bit depth, e.g. 10/12-bit data "
                          "left-/right-justified in a 16-bit container)")
    ap.add_argument("--out", default=None,
                     help="output PNG path (default: '<prefix>_intensity_histogram.png')")
    args = ap.parse_args()

    frame_list = None
    if args.frames:
        frame_list = [int(x) for x in args.frames.split(",")]

    indexed = find_frame_files(args.prefix, frame_list)
    print(f"loading {len(indexed)} frame file(s)...")

    frames = []
    per_frame_stats = []
    for idx, path in indexed:
        arr = load_frame(path)
        frames.append(arr)
        per_frame_stats.append((idx, int(arr.min()), int(arr.max()), float(arr.mean()), float(arr.std())))

    all_values = np.concatenate([f.ravel() for f in frames])
    overall = {
        "min": int(all_values.min()),
        "max": int(all_values.max()),
        "mean": float(all_values.mean()),
        "std": float(all_values.std()),
    }

    print_report(per_frame_stats, overall, args.container_bits)

    out_path = args.out or f"{args.prefix}_intensity_histogram.png"
    frame_word = "frame" if len(frames) == 1 else "frames"
    title = f"Raw intensity histogram - {len(frames)} {frame_word} ({os.path.basename(args.prefix)})"
    plot_histogram(frames, overall, args.bins, not args.linear_y, out_path, args.container_bits, title)


if __name__ == "__main__":
    main()
