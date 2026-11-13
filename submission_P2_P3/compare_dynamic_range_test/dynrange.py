#!/usr/bin/env python3
"""
dynrange.py - dynamic-range profiling of the input frames written by the
DriveWorks camera sample's `--dump-input` option.

Input (one capture per lighting condition, as written by the app):
    <prefix>.raw.u16          N frames of raw uint16 (little endian), tightly packed
    <prefix>.rgba.u8          N frames of RGBA uint8, tightly packed
    <prefix>.meta.json        width / height / frame counts of the files above
    <prefix>.timestamps.csv   index,frame,timestamp_us

Example:
    python3 dynrange.py \
        --run bright=cap/bright --run indoor=cap/indoor --run lowlight=cap/low \
        --dark cap/dark --bits 12 --out report

What is computed, per lighting condition and per stream (raw, RGBA luma):

  * Pooled histogram over all frames -> p0.1 / p1 / p50 / p99 / p99.9, min, max, mean.
  * saturation % (pixels at full scale) and black % (pixels at / below the black level).
  * scene DR  = 20*log10( (p99.9 - black) / max(p0.1 - black, noise floor, 1) )
        The range of intensities this lighting condition actually produced.
  * used DR   = 20*log10( (p99.9 - black) / noise floor )
        How much of the sensor's usable range the condition used.
  * sensor DR = 20*log10( (full scale - black) / noise floor )
        The sensor's own limit. Exact if you pass --dark (lens-capped frames with the
        SAME exposure / gain); otherwise an ESTIMATE from the temporal noise of the
        darkest 5 % of pixels.
  * bits used = ceil(log2(p99.9 + 1)) and the --event-raw-shift that brings the
    p99.9 level into 0..255 (the intensity scale the DVS sensor presets expect).

Assumptions - read before trusting a number:
  * Temporal noise is the per-pixel standard deviation over the frames of one run. It is
    only meaningful for a STATIC scene and a fixed camera; motion or flicker inflates it.
    Needs >= 8 frames, 30+ is better.
  * Black level: from --dark if given, else --black-level (default 0). With a sensor
    pedestal (e.g. 64 DN) and no dark frames, scene DR is understated.
  * `--bits` is the effective bit depth AFTER `--rshift` (12 -> full scale 4095).
    If the data is left-aligned in the 16-bit container, use --rshift (e.g. 4).
  * RGBA is the camera's ISP output (tone mapping / gamma applied), so its DR is not
    comparable to the raw DR in dB; compare raw with raw, RGBA with RGBA.
  * Percentile-based DR ignores the extreme 0.1 % tails on purpose (hot pixels,
    specular highlights).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

STOPS_PER_DB = 1.0 / 6.0206


def db(x: float) -> float:
    return 20.0 * math.log10(x)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_stream(prefix: str, kind: str, size_override):
    """Returns (memmap or None, info dict). kind is 'raw' or 'rgba'."""
    ext, dtype, ch = ("raw.u16", "<u2", 1) if kind == "raw" else ("rgba.u8", np.uint8, 4)
    path = f"{prefix}.{ext}"
    if not os.path.isfile(path):
        return None, {"reason": f"{path} not found"}

    w = h = None
    meta_path = f"{prefix}.meta.json"
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        s = meta.get(kind)
        if s:
            w, h = int(s["width"]), int(s["height"])
    if size_override:
        w, h = size_override
    if not w or not h:
        return None, {"reason": f"no size for {kind}: {meta_path} missing, pass --{kind}-size WxH"}

    itemsize = np.dtype(dtype).itemsize
    frame_bytes = w * h * ch * itemsize
    nbytes = os.path.getsize(path)
    n = nbytes // frame_bytes
    if nbytes % frame_bytes:
        print(f"  warning: {path}: {nbytes % frame_bytes} trailing bytes (truncated last frame dropped)")
    if n == 0:
        return None, {"reason": f"{path} holds no complete {w}x{h} frame"}
    shape = (n, h, w) if ch == 1 else (n, h, w, ch)
    return np.memmap(path, dtype=dtype, mode="r", shape=shape), {"w": w, "h": h, "n": n, "path": path}


# --------------------------------------------------------------------------
# Accumulation (one pass over the frames: histogram, per-pixel sum / sum of squares)
# --------------------------------------------------------------------------
class Acc:
    def __init__(self, h, w, nbins):
        self.nbins = nbins
        self.hist = np.zeros(nbins, np.int64)
        self.s = np.zeros((h, w), np.float64)
        self.s2 = np.zeros((h, w), np.float64)
        self.n = 0
        self.per_frame = []  # (min, max, mean) per frame
        self.clipped_high = 0  # values that did not fit the histogram (should stay 0)
        self.ch_sat = np.zeros(3, np.int64)   # RGBA only: R,G,B == 255
        self.ch_zero = np.zeros(3, np.int64)  # RGBA only: R,G,B == 0
        self.px = 0

    def add(self, f: np.ndarray):
        over = int((f >= self.nbins).sum())
        if over:
            self.clipped_high += over
            f = np.minimum(f, self.nbins - 1)
        self.hist += np.bincount(f.ravel(), minlength=self.nbins)
        ff = f.astype(np.float64)
        self.s += ff
        self.s2 += ff * ff
        self.per_frame.append((int(f.min()), int(f.max()), float(ff.mean())))
        self.n += 1
        self.px += f.size

    def temporal(self):
        """Per-pixel temporal mean and std over the frames (needs n >= 2)."""
        m = self.s / self.n
        var = np.maximum(self.s2 / self.n - m * m, 0.0) * (self.n / max(self.n - 1, 1))
        return m, np.sqrt(var)


def pct(hist: np.ndarray, q: float) -> int:
    c = np.cumsum(hist)
    return int(min(np.searchsorted(c, q / 100.0 * c[-1], side="left"), len(hist) - 1))


def analyse_raw(frames, rshift, max_frames):
    n, h, w = frames.shape
    n = min(n, max_frames) if max_frames else n
    acc = Acc(h, w, 65536)
    for i in range(n):
        f = np.asarray(frames[i])
        acc.add(f >> rshift if rshift else f)
    return acc


def analyse_rgba(frames, max_frames):
    n, h, w, _ = frames.shape
    n = min(n, max_frames) if max_frames else n
    acc = Acc(h, w, 256)
    for i in range(n):
        f = np.asarray(frames[i])
        r, g, b = (f[..., c].astype(np.uint16) for c in range(3))
        acc.add(((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint16))  # same luma as the app
        for c in range(3):
            acc.ch_sat[c] += int((f[..., c] == 255).sum())
            acc.ch_zero[c] += int((f[..., c] == 0).sum())
    return acc


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def dark_reference(acc: Acc):
    """Black level and read noise from lens-capped frames."""
    if acc.n < 8:
        print("  warning: dark run has < 8 frames, read noise is unreliable")
    m, sd = acc.temporal()
    return {"black": float(m.mean()), "noise": float(np.median(sd)), "fpn": float(m.std())}


def estimate_noise(acc: Acc):
    """Noise floor estimate: median temporal std of the darkest 5 % of pixels."""
    if acc.n < 8:
        return None
    m, sd = acc.temporal()
    dark = m <= np.percentile(m, 5)
    return float(np.median(sd[dark]))


def metrics(label, acc: Acc, bits, black, black_src, dark, rgba=False):
    full = (1 << bits) - 1
    total = int(acc.hist.sum())
    p = {q: pct(acc.hist, q) for q in (0.1, 1, 50, 99, 99.9)}
    nz = np.nonzero(acc.hist)[0]

    if dark:
        noise, noise_src = dark["noise"], "dark"
    else:
        noise, noise_src = estimate_noise(acc), "est"
    noise_dn = max(noise, 0.5) if noise is not None else None  # floor: sub-DN noise is quantisation

    hi = p[99.9]
    sig_hi = max(hi - black, 1.0)
    lo_eff = max(p[0.1] - black, noise_dn or 1.0, 1.0)
    bits_used = max(1, math.ceil(math.log2(hi + 1)))

    m = {
        "run": label,
        "frames": acc.n,
        "min": int(nz[0]), "max": int(nz[-1]),
        "mean": float((np.arange(len(acc.hist)) * acc.hist).sum() / total),
        "p0.1": p[0.1], "p1": p[1], "p50": p[50], "p99": p[99], "p99.9": hi,
        "sat_pct": 100.0 * acc.hist[full:].sum() / total,
        "black_pct": 100.0 * acc.hist[: int(black) + 1].sum() / total,
        "black_level": black, "black_src": black_src,
        "noise_dn": noise_dn, "noise_src": noise_src if noise_dn is not None else "n/a",
        "scene_dr_db": db(sig_hi / lo_eff),
        "used_dr_db": db(sig_hi / noise_dn) if noise_dn else None,
        "sensor_dr_db": db(max(full - black, 1.0) / noise_dn) if noise_dn else None,
        "bits_used": bits_used,
        "suggest_rshift": max(0, bits_used - 8),
    }
    m["scene_dr_stops"] = m["scene_dr_db"] * STOPS_PER_DB
    if rgba:
        # The RGBA stream is ISP output (tone mapping / gamma): sensor-level numbers and the
        # event-shift suggestion do not apply. Only the luma scene DR and clipping are kept.
        m["used_dr_db"] = m["sensor_dr_db"] = m["suggest_rshift"] = None
        t = max(acc.px, 1)
        for i, c in enumerate("RGB"):
            m[f"{c}_sat_pct"] = 100.0 * acc.ch_sat[i] / t
            m[f"{c}_zero_pct"] = 100.0 * acc.ch_zero[i] / t
    if acc.clipped_high:
        print(f"  warning [{label}]: {acc.clipped_high} values exceeded the histogram range")
    if not rgba and acc.hist[full + 1:].sum() > 0:
        print(f"  warning [{label}]: values above {full} found (max {m['max']}): "
              f"--bits {bits} is too small for this data, or use --rshift")
    return m


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
COLS = [("run", "{:<10}"), ("frames", "{:>6}"), ("p0.1", "{:>6}"), ("p50", "{:>6}"), ("p99.9", "{:>6}"),
        ("max", "{:>6}"), ("sat_pct", "{:>7.2f}"), ("black_pct", "{:>8.2f}"), ("noise_dn", "{:>8.2f}"),
        ("scene_dr_db", "{:>8.1f}"), ("scene_dr_stops", "{:>7.1f}"), ("used_dr_db", "{:>8.1f}"),
        ("sensor_dr_db", "{:>9.1f}"), ("bits_used", "{:>5}"), ("suggest_rshift", "{:>6}")]
HEAD = {"sat_pct": "sat%", "black_pct": "black%", "noise_dn": "noise", "scene_dr_db": "scene dB",
        "scene_dr_stops": "stops", "used_dr_db": "used dB", "sensor_dr_db": "sensor dB",
        "bits_used": "bits", "suggest_rshift": "shift"}


def print_table(title, rows):
    print(f"\n=== {title} ===")
    print(" ".join(f"{HEAD.get(c, c):>{max(len(HEAD.get(c, c)), 6)}}" if c != "run" else f"{'run':<10}" for c, _ in COLS))
    for r in rows:
        out = []
        for c, fmt in COLS:
            v = r.get(c)
            out.append(fmt.format(v) if v is not None else f"{'n/a':>{len(fmt.format(0)) if c != 'run' else 10}}")
        print(" ".join(out))
    srcs = {(r["black_src"], r["noise_src"]) for r in rows}
    print(f"black level source / noise source: {sorted(srcs)}   (dark = measured, est = estimated, arg = --black-level)")
    if any(r["noise_src"] == "est" for r in rows):
        print("note: 'est' noise = temporal std of the darkest 5 % of pixels IN THAT SCENE. It includes shot noise,\n"
              "      so it is overstated in bright scenes and the used / sensor DR are understated.\n"
              "      Capture a lens-capped run with the same exposure/gain and pass --dark for exact values.")


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def make_plots(out, kind, accs, rows, bits):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed: plots skipped (pip install matplotlib)")
        return
    full = (1 << bits) - 1 if kind == "raw" else 255
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # 1) histograms (linear x and log x: log x shows the shadow detail)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for i, (label, acc) in enumerate(accs.items()):
        h = acc.hist[: full + 2].astype(np.float64)
        h /= max(h.sum(), 1)
        x = np.arange(len(h))
        c = colors[i % len(colors)]
        for a in ax:
            a.step(x, np.where(h > 0, h, np.nan), where="mid", color=c, lw=1, label=label)
            a.axvline(rows[i]["p99.9"], color=c, ls=":", lw=1)
    for a in ax:
        a.set_yscale("log")
        a.axvline(full, color="k", ls="--", lw=1)
        a.set_ylabel("fraction of pixels")
        a.grid(alpha=0.3)
    ax[0].set_xlabel("code value (dotted = p99.9, dashed = full scale)")
    ax[1].set_xscale("log")
    ax[1].set_xlim(1, full * 1.1)
    ax[1].set_xlabel("code value (log)")
    ax[0].legend()
    fig.suptitle(f"{kind.upper()} histograms per lighting condition")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"hist_{kind}.png"), dpi=130)
    plt.close(fig)

    # 2) dynamic range per condition
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = [r["run"] for r in rows]
    ax.bar(labels, [r["scene_dr_db"] for r in rows], color=colors[: len(rows)])
    sens = [r["sensor_dr_db"] for r in rows if r["sensor_dr_db"] is not None]
    if sens:
        ax.axhline(sens[0], color="k", ls="--", lw=1, label=f"sensor DR {sens[0]:.1f} dB ({rows[0]['noise_src']})")
        ax.legend()
    ax.set_ylabel("scene dynamic range [dB]")
    ax.set_title(f"{kind.upper()} scene dynamic range")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"dr_{kind}.png"), dpi=130)
    plt.close(fig)

    # 3) stability over the capture
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (label, acc) in enumerate(accs.items()):
        pf = np.array(acc.per_frame)
        c = colors[i % len(colors)]
        ax.plot(pf[:, 2], color=c, label=f"{label} mean")
        ax.plot(pf[:, 1], color=c, ls=":", label=f"{label} max")
    ax.set_xlabel("saved frame")
    ax.set_ylabel("code value")
    ax.set_title(f"{kind.upper()} per-frame mean / max (exposure or flicker drift shows here)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"timeline_{kind}.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
def parse_size(s):
    if not s:
        return None
    w, h = s.lower().split("x")
    return int(w), int(h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=PREFIX",
                    help="one capture per lighting condition (repeatable)")
    ap.add_argument("--dark", metavar="PREFIX", help="lens-capped raw capture, same exposure/gain: gives black level and read noise")
    ap.add_argument("--stream", choices=["raw", "rgba", "both"], default="both")
    ap.add_argument("--bits", type=int, default=12, help="effective raw bit depth after --rshift (default 12)")
    ap.add_argument("--rshift", type=int, default=0, help="right shift applied to raw before analysis (default 0)")
    ap.add_argument("--black-level", type=float, default=None, help="raw black level in DN when no --dark is given (default 0)")
    ap.add_argument("--max-frames", type=int, default=0, help="analyse at most N frames per run (default all)")
    ap.add_argument("--raw-size", help="WxH, only if meta.json is missing")
    ap.add_argument("--rgba-size", help="WxH, only if meta.json is missing")
    ap.add_argument("--out", default="dynrange_out")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = []
    for spec in a.run:
        label, sep, prefix = spec.partition("=")
        runs.append((label, prefix) if sep else (os.path.basename(spec), spec))

    # ---- dark reference (raw only) ----
    dark = None
    if a.dark and a.stream in ("raw", "both"):
        fr, info = load_stream(a.dark, "raw", parse_size(a.raw_size))
        if fr is None:
            sys.exit(f"dark run: {info['reason']}")
        print(f"dark: {info['n']} frames {info['w']}x{info['h']}")
        dark = dark_reference(analyse_raw(fr, a.rshift, a.max_frames))
        print(f"  black level {dark['black']:.2f} DN, read noise {dark['noise']:.2f} DN, "
              f"fixed-pattern std {dark['fpn']:.2f} DN")

    summary = {}
    for kind in ("raw", "rgba"):
        if a.stream not in (kind, "both"):
            continue
        accs, rows = {}, []
        for label, prefix in runs:
            fr, info = load_stream(prefix, kind, parse_size(a.raw_size if kind == "raw" else a.rgba_size))
            if fr is None:
                print(f"[{label}] {kind}: skipped ({info['reason']})")
                continue
            print(f"[{label}] {kind}: {info['n']} frames {info['w']}x{info['h']}")
            if kind == "raw":
                acc = analyse_raw(fr, a.rshift, a.max_frames)
                black = dark["black"] if dark else (a.black_level or 0.0)
                src = "dark" if dark else ("arg" if a.black_level is not None else "default 0")
                rows.append(metrics(label, acc, a.bits, black, src, dark))
            else:
                acc = analyse_rgba(fr, a.max_frames)
                rows.append(metrics(label, acc, 8, 0.0, "default 0", None, rgba=True))
            accs[label] = acc
        if not rows:
            continue
        print_table(f"{kind.upper()} (bits={a.bits if kind == 'raw' else 8}"
                    f"{', rshift=' + str(a.rshift) if kind == 'raw' and a.rshift else ''})", rows)
        write_csv(os.path.join(a.out, f"summary_{kind}.csv"), rows)
        summary[kind] = rows
        if not a.no_plots:
            make_plots(a.out, kind, accs, rows, a.bits)

    if "raw" in summary:
        need = max(r["suggest_rshift"] for r in summary["raw"])
        print(f"\nSuggested --event-raw-shift to keep the BRIGHTEST condition within 0..255: {need}")
        print("  (dim conditions lose resolution after the shift: check p99.9 >> shift in the dimmest run)")
    if dark:
        summary["dark"] = dark
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nwrote {a.out}/summary_*.csv, summary.json" + ("" if a.no_plots else ", *.png"))


if __name__ == "__main__":
    main()
