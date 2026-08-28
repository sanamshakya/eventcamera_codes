#!/usr/bin/env python3
"""
Parse the CEventFileWriter binary format and visualize events.

Usage:
    python3 visualize_events.py events.bin [--slice START END] [--out accum.png]

Verification checklist this helps with:
  1. Header width/height match your sensor resolution.
  2. Accumulated ON (red) / OFF (blue) counts form a recognizable image
     (edges of scene objects), not noise scattered uniformly or
     concentrated in one corner (would suggest x/y or pitch bug).
  3. Timestamps are monotonically increasing and spaced sensibly
     (matches step 3's frame-interval check).
"""

import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--slice", nargs=2, type=int, metavar=("START", "END"),
                     help="only use events[START:END] (by index, not time)")
    ap.add_argument("--out", default="accum.png")
    args = ap.parse_args()

    width, height, events = load_events(args.path)
    print(f"Total events: {len(events)}")

    if len(events) == 0:
        print("No events found - nothing to visualize.")
        return

    if args.slice:
        s, e = args.slice
        events = events[s:e]

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

    # Accumulate into an RGB image: red = ON count, blue = OFF count
    accum = np.zeros((height, width, 3), dtype=np.float32)
    on_mask = events["polarity"] > 0
    off_mask = ~on_mask

    np.add.at(accum[:, :, 0], (events["y"][on_mask], events["x"][on_mask]), 1)
    np.add.at(accum[:, :, 2], (events["y"][off_mask], events["x"][off_mask]), 1)

    # Normalize for display
    maxval = accum.max()
    if maxval > 0:
        accum = (accum / maxval * 255.0).astype(np.uint8)
    else:
        accum = accum.astype(np.uint8)

    try:
        import cv2
        cv2.imwrite(args.out, accum)
        print(f"Wrote accumulated image to {args.out}")
    except ImportError:
        import matplotlib.pyplot as plt
        plt.imsave(args.out, accum)
        print(f"Wrote accumulated image to {args.out} (via matplotlib)")


if __name__ == "__main__":
    main()
