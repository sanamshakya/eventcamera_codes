#!/usr/bin/env python3
"""
Analyze raw Bayer16 frames captured by CRawCaptureWriter, BEFORE wiring
raw data into EventGenerator.

Usage:
    python3 visualize_raw_bayer.py raw_capture.bin [--frame N] [--pattern RGGB]

Checks performed:
  1. Header sanity (width/height match sensor resolution).
  2. Bit-depth check: is data right-justified (raw10/12 values fit in
     0-1023/0-4095) or left-justified (shifted up near 65535)? This
     determines what scaling EventGenerator's thresholds need.
  3. CFA (color filter array) pattern check: computes mean of each of
     the 4 quadrant positions in a 2x2 Bayer block. If your sensor is
     RGGB, expect quadrant[0,0] (R) and quadrant[1,1] (B) means to
     differ from the two G positions, with G roughly similar to each
     other (real scenes vary, but a clear R/G/B color image content
     should show DIFFERENT means per channel, not near-identical
     values, which would suggest the pattern assumption or byte
     order is wrong).
  4. Frame-to-frame difference stats (temporal noise floor / motion
     level) - useful to sanity check EventGenerator threshold choice
     later, since thresholds should be well above the noise floor
     seen here.
  5. Quick-look demosaiced preview image (for a human sanity check
     that this is actually a recognizable scene, not garbage).
"""

import argparse
import struct
import numpy as np

HEADER_FMT = "<IIII"  # magic, version, width, height
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MAGIC = 0x42574152  # 'RAWB'


def load_frames(path):
    with open(path, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)
        magic, version, width, height = struct.unpack(HEADER_FMT, header_bytes)
        if magic != MAGIC:
            raise ValueError(f"Bad magic 0x{magic:08X} - not a RAWB file")
        print(f"Header: version={version} width={width} height={height}")

        data = f.read()

    frame_bytes = width * height * 2
    n_frames = len(data) // frame_bytes
    remainder = len(data) % frame_bytes
    if remainder != 0:
        print(f"WARNING: {remainder} trailing bytes - file may be truncated "
              f"or width/height mismatch")

    frames = np.frombuffer(data[:n_frames * frame_bytes], dtype="<u2")
    frames = frames.reshape(n_frames, height, width)
    print(f"Loaded {n_frames} frames of {width}x{height}")
    return frames, width, height


def check_bit_depth(frame):
    maxval = frame.max()
    minval = frame.min()
    print(f"\n--- Bit depth check ---")
    print(f"min={minval} max={maxval}")
    if maxval <= 1023:
        print("-> Looks like RAW10, right-justified in 16-bit container "
              "(values fit 0-1023)")
    elif maxval <= 4095:
        print("-> Looks like RAW12, right-justified (values fit 0-4095)")
    elif maxval <= 65535 and minval > 60000:
        print("-> Values clustered near 65535 - possibly LEFT-justified "
              "RAW10/12, or saturated/overexposed frame")
    elif maxval > 4095:
        print(f"-> max={maxval} exceeds 4095 - check if this is genuinely "
              f"16-bit sensor data, or left-justified 10/12-bit (would show "
              f"as multiples of 64 or 16 - check histogram below)")

    # Check if values are multiples of a power of 2 (sign of left-justification)
    nonzero = frame[frame > 0]
    if len(nonzero) > 0:
        sample = nonzero[:10000]
        for shift, label in [(6, "x64 (left-just. RAW10)"), (4, "x16 (left-just. RAW12)")]:
            frac_aligned = np.mean(sample % (1 << shift) == 0)
            if frac_aligned > 0.95:
                print(f"-> {frac_aligned*100:.1f}% of sampled values are multiples of "
                      f"{1<<shift} -> strongly suggests {label}")


def check_cfa_pattern(frame):
    print(f"\n--- CFA quadrant means (2x2 Bayer block positions) ---")
    tl = frame[0::2, 0::2].mean()  # top-left
    tr = frame[0::2, 1::2].mean()  # top-right
    bl = frame[1::2, 0::2].mean()  # bottom-left
    br = frame[1::2, 1::2].mean()  # bottom-right
    print(f"  [0,0] (top-left):     {tl:.1f}")
    print(f"  [0,1] (top-right):    {tr:.1f}")
    print(f"  [1,0] (bottom-left):  {bl:.1f}")
    print(f"  [1,1] (bottom-right): {br:.1f}")
    print("If this is RGGB: [0,0]=R, [0,1]=G, [1,0]=G, [1,1]=B - the two G "
          "positions ([0,1],[1,0]) should be closer to each other than to "
          "R or B, for typical scene content. If all four are nearly "
          "identical, this may not actually be Bayer-mosaiced data "
          "(check pixel format assumption).")


def check_temporal_diff(frames):
    if frames.shape[0] < 2:
        return
    print(f"\n--- Frame-to-frame difference (frame 0 vs frame 1) ---")
    diff = frames[1].astype(np.int32) - frames[0].astype(np.int32)
    print(f"  mean abs diff: {np.mean(np.abs(diff)):.2f}")
    print(f"  std: {np.std(diff):.2f}")
    print(f"  max abs diff: {np.max(np.abs(diff))}")
    print("Use this as a rough noise floor - EventGenerator's "
          "positiveThreshold/negativeThreshold should sit well above "
          "the static-scene noise level, or you'll get spurious events.")


def save_preview(frame, width, height, out_path, pattern):
    try:
        import cv2
        # Normalize to 8-bit for demosaic preview (assumes right-justified
        # data; if bit-depth check above says otherwise, this preview
        # will look wrong until that's corrected).
        norm = (frame.astype(np.float32) / max(frame.max(), 1) * 255.0).astype(np.uint8)
        code_map = {
            "RGGB": cv2.COLOR_BayerRG2BGR,
            "BGGR": cv2.COLOR_BayerBG2BGR,
            "GRBG": cv2.COLOR_BayerGR2BGR,
            "GBRG": cv2.COLOR_BayerGB2BGR,
        }
        code = code_map.get(pattern.upper())
        if code is None:
            print(f"Unknown pattern {pattern}, skipping preview")
            return
        color = cv2.cvtColor(norm, code)
        cv2.imwrite(out_path, color)
        print(f"\nWrote demosaiced preview to {out_path} (assumed pattern={pattern})")
        print("If colors look wrong/swapped, try a different --pattern "
              "(RGGB, BGGR, GRBG, GBRG).")
    except ImportError:
        print("opencv-python not available, skipping preview image")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--frame", type=int, default=0, help="which frame index to analyze")
    ap.add_argument("--pattern", default="RGGB",
                     help="assumed CFA pattern for preview: RGGB/BGGR/GRBG/GBRG")
    ap.add_argument("--out", default="raw_preview.png")
    args = ap.parse_args()

    frames, width, height = load_frames(args.path)
    if frames.shape[0] == 0:
        print("No frames loaded.")
        return

    frame = frames[args.frame]

    check_bit_depth(frame)
    check_cfa_pattern(frame)
    check_temporal_diff(frames)
    save_preview(frame, width, height, args.out, args.pattern)


if __name__ == "__main__":
    main()
