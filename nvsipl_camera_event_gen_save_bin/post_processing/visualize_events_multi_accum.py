#!/usr/bin/env python3
"""
Parse the CEventFileWriter binary format and visualize events.

Usage:
    # single accumulation over the whole file (or a --slice of it)
    python3 visualize_events.py events.bin [--slice START END] [--out accum.png]

    # time-windowed event frames (e.g. 20 ms bins) written to a directory
    python3 visualize_events.py events.bin --dt 20 --outdir frames/

Verification checklist this helps with:
  1. Header width/height match your sensor resolution.
  2. Accumulated ON (red) / OFF (blue) counts form a recognizable image
     (edges of scene objects), not noise scattered uniformly or
     concentrated in one corner (would suggest x/y or pitch bug).
  3. Timestamps are monotonically increasing and spaced sensibly
     (matches step 3's frame-interval check).
  4. Per-window (--dt) frames look like a coherent moving scene rather
     than jittering noise from window to window.
"""

import argparse
import os
import struct
import numpy as np

HEADER_FMT = "<IIII"          # magic, version, width, height
HEADER_SIZE = struct.calcsize(HEADER_FMT)

RECORD_FMT = "<HHdb"          # x, y, timestamp, polarity (int8)
RECORD_SIZE = struct.calcsize(RECORD_FMT)


def load_events(path):
    with open(path, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)
        magic, version, width, height = struct.unpack(HEADER_FMT, header_bytes)

        if magic != 0x45565342:
            raise ValueError(f"Bad magic 0x{magic:08X} - not an EVSB file "
                              f"(or struct packing mismatch between C++ and Python)")

        print(f"Header: version={version} width={width} height={height}")

        data = f.read()

    n_records = len(data) // RECORD_SIZE
    remainder = len(data) % RECORD_SIZE
    if remainder != 0:
        print(f"WARNING: {remainder} trailing bytes after last full record "
              f"- file may be truncated or RECORD_FMT doesn't match C++ struct packing")

    dtype = np.dtype([
        ("x", "<u2"),
        ("y", "<u2"),
        ("timestamp", "<f8"),
        ("polarity", "i1"),
    ])
    events = np.frombuffer(data[:n_records * RECORD_SIZE], dtype=dtype)

    return width, height, events


def print_event_sample(events, n=10):
    """Print the first and last n event records for a quick sanity check."""
    total = len(events)
    n = min(n, total)

    def dump(idx_range, label):
        print(f"-- {label} {n} events --")
        for i in idx_range:
            e = events[i]
            print(f"  [{i:>8}] x={e['x']:>4} y={e['y']:>4} "
                  f"t={e['timestamp']:.6f} pol={int(e['polarity']):+d}")

    dump(range(0, n), "first")
    if total > n:
        dump(range(total - n, total), "last")
    else:
        print("(file has <= n events, first/last sets overlap)")


def accumulate_frame(events, width, height):
    """Build one RGB accumulation image (red=ON, blue=OFF) from a set of events."""
    accum = np.zeros((height, width, 3), dtype=np.float32)
    if len(events) == 0:
        return accum.astype(np.uint8)

    on_mask = events["polarity"] > 0
    off_mask = ~on_mask

    np.add.at(accum[:, :, 0], (events["y"][on_mask], events["x"][on_mask]), 1)
    np.add.at(accum[:, :, 2], (events["y"][off_mask], events["x"][off_mask]), 1)

    maxval = accum.max()
    if maxval > 0:
        accum = (accum / maxval * 255.0).astype(np.uint8)
    else:
        accum = accum.astype(np.uint8)
    return accum


def save_image(img, path):
    try:
        import cv2
        cv2.imwrite(path, img)
    except ImportError:
        import matplotlib.pyplot as plt
        plt.imsave(path, img)


def write_time_windowed_frames(events, width, height, dt_ms, outdir):
    """Bin events into fixed dt_ms windows (by timestamp) and write one
    accumulation frame per window into outdir."""
    os.makedirs(outdir, exist_ok=True)

    ts = events["timestamp"]
    t0, t1 = ts.min(), ts.max()
    dt = dt_ms / 1000.0

    n_frames = int(np.ceil((t1 - t0) / dt)) if t1 > t0 else 1
    print(f"Building {n_frames} frame(s) at dt={dt_ms} ms "
          f"over {t1 - t0:.4f} s span")

    # bin index per event, then group without re-sorting the array
    bin_idx = np.floor((ts - t0) / dt).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_frames - 1)

    order = np.argsort(bin_idx, kind="stable")
    sorted_bins = bin_idx[order]
    sorted_events = events[order]
    boundaries = np.searchsorted(sorted_bins, np.arange(n_frames + 1))

    empty_frames = 0
    for f in range(n_frames):
        lo, hi = boundaries[f], boundaries[f + 1]
        frame_events = sorted_events[lo:hi]
        if len(frame_events) == 0:
            empty_frames += 1
        img = accumulate_frame(frame_events, width, height)
        path = os.path.join(outdir, f"frame_{f:05d}.png")
        save_image(img, path)

    if empty_frames:
        print(f"WARNING: {empty_frames}/{n_frames} frames had zero events "
              f"(check event rate vs. --dt)")
    print(f"Wrote {n_frames} frame(s) to {outdir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--slice", nargs=2, type=int, metavar=("START", "END"),
                     help="only use events[START:END] (by index, not time)")
    ap.add_argument("--out", default="accum.png",
                     help="output path for single whole-file accumulation image")
    ap.add_argument("--dt", type=float, default=None,
                     help="accumulation window in ms (e.g. 20) to build a "
                          "sequence of time-windowed frames instead of one image")
    ap.add_argument("--outdir", default="frames",
                     help="output directory for --dt frame sequence")
    ap.add_argument("--n-sample", type=int, default=10,
                     help="number of events to print at start and end (default 10)")
    args = ap.parse_args()

    width, height, events = load_events(args.path)
    print(f"Total events: {len(events)}")

    if len(events) == 0:
        print("No events found - nothing to visualize.")
        return

    if args.slice:
        s, e = args.slice
        events = events[s:e]

    print_event_sample(events, n=args.n_sample)

    ts = events["timestamp"]
    print(f"Timestamp range: {ts.min():.4f} -> {ts.max():.4f} "
          f"({ts.max() - ts.min():.4f} s span)")
    non_monotonic = np.sum(np.diff(ts) < 0)
    if non_monotonic > 0:
        print(f"WARNING: {non_monotonic} out-of-order timestamps found "
              f"(check packet ordering / timestamp source)")

    n_on = np.sum(events["polarity"] > 0)
    n_off = np.sum(events["polarity"] < 0)
    print(f"ON events: {n_on}   OFF events: {n_off}")

    if args.dt is not None:
        write_time_windowed_frames(events, width, height, args.dt, args.outdir)
    else:
        img = accumulate_frame(events, width, height)
        save_image(img, args.out)
        print(f"Wrote accumulated image to {args.out}")


if __name__ == "__main__":
    main()
