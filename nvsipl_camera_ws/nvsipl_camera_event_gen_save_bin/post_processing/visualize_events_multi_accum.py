#!/usr/bin/env python3
"""
Parse the CEventFileWriter binary format and visualize events.

Handles all three capture layouts CEventFileWriter can now produce:
  - v1: a single file, one contiguous block of event records (old format,
    no per-frame headers).
  - v2, single continuous file (SetEventFramesPerFile(0), the default):
    one FrameHeader (frameNumber, startTime, endTime, numEvents) followed
    by that frame's records, repeated for every frame.
  - v2, multi-file (SetEventFramesPerFile(N>=1)): the same FrameHeader +
    records layout, just split across files named
    "<base>_frame<startFrameNumber>.<ext>". Point this script at the
    directory or a glob pattern and it stitches them back together in
    frame order.

Usage:
    # single accumulation image, from one file, a directory, or a glob
    python3 visualize_events_multi_accum.py events.evsb
    python3 visualize_events_multi_accum.py captures/                      # directory of split files
    python3 visualize_events_multi_accum.py "captures/events_frame*.evsb"  # explicit glob

    # time-windowed accumulation frames (fixed dt bins, re-derived from timestamps)
    python3 visualize_events_multi_accum.py events.evsb --dt 20 --outdir frames/

    # one image per ACTUAL capture frame (exact FrameHeader boundaries,
    # not re-binned) - filenames line up with "frame %llu produced ..." logs
    python3 visualize_events_multi_accum.py captures/ --by-capture-frame --outdir frames/

    # just validate (frame-number gaps/dupes, magic, record counts) - no images
    python3 visualize_events_multi_accum.py captures/ --check-only

Verification checklist this helps with:
  1. Header width/height match your sensor resolution.
  2. Accumulated ON (red) / OFF (blue) counts form a recognizable image
     (edges of scene objects), not noise scattered uniformly or
     concentrated in one corner (would suggest x/y or pitch bug).
  3. Timestamps are monotonically increasing and spaced sensibly.
  4. Frame numbers have no gaps or duplicates (a gap usually means the
     capture-side queue dropped a frame under backpressure - see
     check_frame_sequence()).
  5. Per-window (--dt) or per-capture-frame (--by-capture-frame) images
     look like a coherent moving scene rather than jittering noise.
"""

import argparse
import glob
import os
import re
import struct
import numpy as np

HEADER_FMT = "<IIII"           # magic, version, width, height
HEADER_SIZE = struct.calcsize(HEADER_FMT)
HEADER_MAGIC = 0x45565342      # 'EVSB'

FRAME_HEADER_FMT = "<IQddQ"    # magic, frameNumber, startTime, endTime, numEvents
FRAME_HEADER_SIZE = struct.calcsize(FRAME_HEADER_FMT)
FRAME_MAGIC = 0x46524D48       # 'FRMH'

RECORD_FMT = "<HHdb"           # x, y, timestamp, polarity (int8)
RECORD_SIZE = struct.calcsize(RECORD_FMT)

RECORD_DTYPE = np.dtype([
    ("x", "<u2"),
    ("y", "<u2"),
    ("timestamp", "<f8"),
    ("polarity", "i1"),
])

FRAME_NUM_RE = re.compile(r"_frame(\d+)")


# ---------------------------------------------------------------------------
# File discovery / low-level parsing
# ---------------------------------------------------------------------------

def find_capture_files(path):
    """
    Resolve `path` to a sorted list of capture files (by embedded frame
    number, so files sort correctly even with mixed digit counts):
      - a single file            -> [path]
      - a glob pattern (* ? [)   -> glob.glob(path)
      - a directory              -> every "*_frameNNNNNN.*" file inside;
                                     falls back to the single file inside
                                     if there's exactly one and it has no
                                     frame-number suffix (single-file mode)
    """
    if os.path.isdir(path):
        all_files = sorted(glob.glob(os.path.join(path, "*")))
        numbered = [p for p in all_files if FRAME_NUM_RE.search(os.path.basename(p))]
        if numbered:
            candidates = numbered
        elif len(all_files) == 1:
            candidates = all_files
        else:
            raise FileNotFoundError(
                f"{path} contains {len(all_files)} file(s) with no "
                f"'_frameNNNNNN' suffix - point at a single file explicitly "
                f"instead of a directory")
    elif any(c in path for c in "*?["):
        candidates = glob.glob(path)
        if not candidates:
            raise FileNotFoundError(f"No files matched pattern: {path}")
    else:
        return [path]

    def sort_key(p):
        m = FRAME_NUM_RE.search(os.path.basename(p))
        return (0, int(m.group(1))) if m else (1, os.path.basename(p))

    return sorted(candidates, key=sort_key)


def read_file_header(f):
    header_bytes = f.read(HEADER_SIZE)
    if len(header_bytes) < HEADER_SIZE:
        raise ValueError("File too short for a header")
    magic, version, width, height = struct.unpack(HEADER_FMT, header_bytes)
    if magic != HEADER_MAGIC:
        raise ValueError(f"Bad magic 0x{magic:08X} - not an EVSB file "
                          f"(or struct packing mismatch between C++ and Python)")
    return version, width, height


def parse_frames_v2(f, source_label):
    """Yield one dict per FrameHeader block: frameNumber, startTime, endTime, events."""
    while True:
        fh_bytes = f.read(FRAME_HEADER_SIZE)
        if len(fh_bytes) == 0:
            return  # clean EOF between frames
        if len(fh_bytes) < FRAME_HEADER_SIZE:
            print(f"WARNING: {source_label}: {len(fh_bytes)} trailing bytes - "
                  f"truncated frame header, stopping")
            return

        magic, frame_number, start_time, end_time, num_events = \
            struct.unpack(FRAME_HEADER_FMT, fh_bytes)

        if magic != FRAME_MAGIC:
            print(f"WARNING: {source_label}: bad frame magic 0x{magic:08X} "
                  f"- file may be corrupt or desynced, stopping")
            return

        record_bytes = f.read(num_events * RECORD_SIZE)
        if len(record_bytes) < num_events * RECORD_SIZE:
            actual = len(record_bytes) // RECORD_SIZE
            print(f"WARNING: {source_label}: frame {frame_number} claims "
                  f"{num_events} events but file ended early ({actual} read) "
                  f"- truncated capture")
            record_bytes = record_bytes[:actual * RECORD_SIZE]

        events = np.frombuffer(record_bytes, dtype=RECORD_DTYPE)

        yield {
            "frameNumber": frame_number,
            "startTime": start_time,
            "endTime": end_time,
            "events": events,
        }


def parse_v1_whole_file(f):
    """Old format: no per-frame headers, just one contiguous block of records."""
    data = f.read()
    n_records = len(data) // RECORD_SIZE
    remainder = len(data) % RECORD_SIZE
    if remainder != 0:
        print(f"WARNING: {remainder} trailing bytes after last full record "
              f"- file may be truncated or RECORD_FMT doesn't match C++ struct packing")
    events = np.frombuffer(data[:n_records * RECORD_SIZE], dtype=RECORD_DTYPE)
    return [{"frameNumber": None, "startTime": None, "endTime": None, "events": events}]


def load_capture(path):
    """
    Load one or more capture files and return (width, height, frames),
    frames being a list of {frameNumber, startTime, endTime, events} dicts
    in file/frame order. v1 files come back as a single frame with
    frameNumber=None (no per-frame boundaries available).
    """
    files = find_capture_files(path)
    print(f"Found {len(files)} capture file(s)")

    width = height = None
    frames = []

    for fp in files:
        with open(fp, "rb") as f:
            version, w, h = read_file_header(f)

            if width is None:
                width, height = w, h
            elif (w, h) != (width, height):
                print(f"WARNING: {fp} has header {w}x{h}, expected "
                      f"{width}x{height} - skipping this file")
                continue

            if version == 1:
                file_frames = parse_v1_whole_file(f)
            elif version >= 2:
                file_frames = list(parse_frames_v2(f, os.path.basename(fp)))
            else:
                print(f"WARNING: unknown version {version} in {fp}, skipping")
                continue

            frames.extend(file_frames)

    return width, height, frames


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def check_frame_sequence(frames):
    """Look for gaps/duplicates/out-of-order frame numbers - a gap usually
    means the capture-side queue dropped a frame under backpressure."""
    numbered = [fr for fr in frames if fr["frameNumber"] is not None]
    if not numbered:
        print("No per-frame headers present (v1 file) - frame-sequence check skipped")
        return

    nums = [fr["frameNumber"] for fr in numbered]
    gaps = []
    for prev, cur in zip(nums, nums[1:]):
        if cur == prev:
            print(f"WARNING: duplicate frameNumber {cur}")
        elif cur < prev:
            print(f"WARNING: out-of-order frameNumber {cur} after {prev}")
        elif cur != prev + 1:
            gaps.append((prev, cur))

    if gaps:
        total_missing = sum(cur - prev - 1 for prev, cur in gaps)
        preview = ", ".join(f"{p}->{c}" for p, c in gaps[:5])
        print(f"WARNING: {len(gaps)} gap(s) in frameNumber sequence, "
              f"{total_missing} frame(s) missing total (likely dropped under "
              f"queue backpressure): {preview}"
              f"{' ...' if len(gaps) > 5 else ''}")
    else:
        print(f"Frame sequence OK: {nums[0]}..{nums[-1]}, no gaps or duplicates")


# ---------------------------------------------------------------------------
# Accumulation / image output
# ---------------------------------------------------------------------------

def concat_events(frames):
    arrays = [fr["events"] for fr in frames if len(fr["events"]) > 0]
    if not arrays:
        return np.array([], dtype=RECORD_DTYPE)
    return np.concatenate(arrays)


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
    accumulation frame per window into outdir. Windows are re-derived from
    timestamps, independent of the capture-side frame boundaries."""
    os.makedirs(outdir, exist_ok=True)

    ts = events["timestamp"]
    t0, t1 = ts.min(), ts.max()
    dt = dt_ms / 1000.0

    n_frames = int(np.ceil((t1 - t0) / dt)) if t1 > t0 else 1
    print(f"Building {n_frames} frame(s) at dt={dt_ms} ms "
          f"over {t1 - t0:.4f} s span")

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


def write_capture_frames(frames, width, height, outdir, frame_range=None):
    """One image per ACTUAL capture frame (exact FrameHeader boundaries),
    instead of re-binning by a fixed --dt window. Filenames use the real
    frameNumber so they line up with the 'frame %llu produced ...' logs and
    with dropped-frame gaps reported by check_frame_sequence()."""
    numbered = [fr for fr in frames if fr["frameNumber"] is not None]
    if not numbered:
        print("No per-frame headers found (v1 file?) - can't use --by-capture-frame")
        return

    if frame_range:
        lo, hi = frame_range
        numbered = [fr for fr in numbered if lo <= fr["frameNumber"] <= hi]

    os.makedirs(outdir, exist_ok=True)

    empty = 0
    for fr in numbered:
        if len(fr["events"]) == 0:
            empty += 1
        img = accumulate_frame(fr["events"], width, height)
        path = os.path.join(outdir, f"frame_{fr['frameNumber']:06d}.png")
        save_image(img, path)

    if empty:
        print(f"WARNING: {empty}/{len(numbered)} capture frames had zero events")
    print(f"Wrote {len(numbered)} capture-frame image(s) to {outdir}/")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a single .evsb file, a directory of split "
                                  "files, or a glob pattern like "
                                  "'events_frame*.evsb'")
    ap.add_argument("--slice", nargs=2, type=int, metavar=("START", "END"),
                     help="only use events[START:END] (by index, not time) "
                          "from the concatenated event stream - ignored with "
                          "--by-capture-frame")
    ap.add_argument("--out", default="accum.png",
                     help="output path for single whole-capture accumulation image")
    ap.add_argument("--dt", type=float, default=None,
                     help="accumulation window in ms (e.g. 20) to build a "
                          "sequence of time-rebinned frames instead of one image")
    ap.add_argument("--by-capture-frame", action="store_true",
                     help="write one image per actual capture frame using the "
                          "exact FrameHeader boundaries, instead of re-binning "
                          "by time; requires v2 (header-per-frame) files")
    ap.add_argument("--frame-range", nargs=2, type=int, metavar=("START", "END"),
                     help="with --by-capture-frame, only write frames whose "
                          "frameNumber is in [START, END]")
    ap.add_argument("--outdir", default="frames",
                     help="output directory for --dt or --by-capture-frame frame sequences")
    ap.add_argument("--n-sample", type=int, default=10,
                     help="number of events to print at start and end (default 10)")
    ap.add_argument("--check-only", action="store_true",
                     help="parse and validate only (frame-number gaps/dupes, "
                          "magic, record counts) - no images written")
    args = ap.parse_args()

    width, height, frames = load_capture(args.path)
    print(f"Resolution: {width}x{height}, {len(frames)} frame(s) loaded")

    check_frame_sequence(frames)

    events_all = concat_events(frames)
    print(f"Total events: {len(events_all)}")

    if len(events_all) == 0:
        print("No events found - nothing further to check/visualize.")
        return

    ts = events_all["timestamp"]
    print(f"Timestamp range: {ts.min():.4f} -> {ts.max():.4f} "
          f"({ts.max() - ts.min():.4f} s span)")
    non_monotonic = np.sum(np.diff(ts) < 0)
    if non_monotonic > 0:
        print(f"WARNING: {non_monotonic} out-of-order timestamps found within "
              f"the concatenated stream")

    n_on = np.sum(events_all["polarity"] > 0)
    n_off = np.sum(events_all["polarity"] < 0)
    print(f"ON events: {n_on}   OFF events: {n_off}")

    if args.check_only:
        return

    if args.by_capture_frame:
        write_capture_frames(frames, width, height, args.outdir,
                              frame_range=args.frame_range)
        return

    events = events_all
    if args.slice:
        s, e = args.slice
        events = events[s:e]

    print_event_sample(events, n=args.n_sample)

    if args.dt is not None:
        write_time_windowed_frames(events, width, height, args.dt, args.outdir)
    else:
        img = accumulate_frame(events, width, height)
        save_image(img, args.out)
        print(f"Wrote accumulated image to {args.out}")


if __name__ == "__main__":
    main()
