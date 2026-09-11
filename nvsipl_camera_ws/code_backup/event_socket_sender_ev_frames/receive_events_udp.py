#!/usr/bin/env python3
"""
Receives chunked raw event-accumulation frames sent by EventRawUdpSender (on
the embedded target, no OpenCV needed there) over UDP, reassembles each
frame from its chunks, colorizes it, and displays it live.

Wire format per UDP datagram (all little-endian, matches
EventRawUdpSender's packed RawFrameChunkHeader exactly):

    uint32 magic        ('EFRH' = 0x45465248)
    uint32 frameSeq
    uint32 width
    uint32 height
    uint32 totalChunks
    uint32 chunkIndex
    uint32 chunkBytes   (payload bytes following this header in THIS datagram)
    <chunkBytes> bytes of frame payload

The full frame is width*height bytes, 1 byte/pixel, centered at 128 ("no
activity"): brighter = net ON events, darker = net OFF events. Chunks are
reassembled by concatenating payloads in chunkIndex order (not by assuming
a fixed chunk size), so this works regardless of the sender's configured
chunkPayloadBytes.

Run this on the host PC:
    python3 receive_events_udp.py --port 5005

UDP is unordered/lossy - a frame missing even one chunk is simply dropped
(this is a live monitoring feed, not the .evsb file capture path, which is
unaffected by anything here).
"""

import argparse
import socket
import struct
import time

import cv2
import numpy as np

HEADER_FMT = "<IIIIIII"  # magic, frameSeq, width, height, totalChunks, chunkIndex, chunkBytes
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MAGIC = 0x45465248

# How long (seconds) to keep an incomplete frame around waiting for its
# remaining chunks before giving up on it, to bound memory if chunks are
# lost outright rather than just arriving late.
STALE_FRAME_TIMEOUT_S = 2.0


def colorize(gray):
    """gray: (H, W) uint8, centered at 128. Returns a BGR uint8 image:
    red for net-ON pixels, blue for net-OFF pixels, black for no activity -
    matching the convention used everywhere else in this pipeline."""
    signed = gray.astype(np.int16) - 128
    img = np.zeros((*gray.shape, 3), dtype=np.uint8)

    pos = signed > 0
    neg = signed < 0

    img[..., 2][pos] = np.clip(signed[pos] * 2, 0, 255).astype(np.uint8)   # red = ON
    img[..., 0][neg] = np.clip(-signed[neg] * 2, 0, 255).astype(np.uint8)  # blue = OFF

    return img


class FrameReassembler:
    """Tracks in-progress frames by frameSeq and yields completed ones."""

    def __init__(self):
        self._pending = {}  # frameSeq -> {"chunks": {idx: bytes}, "total": int, "w": int, "h": int, "t0": float}

    def add_chunk(self, frame_seq, width, height, total_chunks, chunk_index, payload):
        entry = self._pending.get(frame_seq)
        if entry is None:
            entry = {"chunks": {}, "total": total_chunks, "w": width, "h": height, "t0": time.time()}
            self._pending[frame_seq] = entry

        entry["chunks"][chunk_index] = payload

        self._evict_stale(except_seq=frame_seq)

        if len(entry["chunks"]) == entry["total"]:
            del self._pending[frame_seq]
            buf = b"".join(entry["chunks"][i] for i in range(entry["total"]))
            expected = entry["w"] * entry["h"]
            if len(buf) != expected:
                print(f"WARNING: frame {frame_seq} reassembled to {len(buf)} bytes, "
                      f"expected {expected} ({entry['w']}x{entry['h']}) - dropping")
                return None
            return np.frombuffer(buf, dtype=np.uint8).reshape(entry["h"], entry["w"])

        return None

    def _evict_stale(self, except_seq):
        now = time.time()
        stale = [seq for seq, e in self._pending.items()
                 if seq != except_seq and now - e["t0"] > STALE_FRAME_TIMEOUT_S]
        for seq in stale:
            entry = self._pending.pop(seq)
            print(f"WARNING: frame {seq} incomplete after {STALE_FRAME_TIMEOUT_S}s "
                  f"({len(entry['chunks'])}/{entry['total']} chunks) - discarding")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--recvbuf", type=int, default=1 << 22,
                     help="socket receive buffer size in bytes (default 4MB - "
                          "raw frames are much bigger than JPEGs, needs more headroom)")
    ap.add_argument("--save-dir", default=None,
                     help="optional: also save each completed frame as a PNG here")
    ap.add_argument("--no-display", action="store_true",
                     help="don't open a window - just log stats (and save, if --save-dir set)")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.recvbuf)
    sock.bind((args.bind, args.port))
    print(f"Listening for raw event frame chunks on {args.bind}:{args.port} ...")

    if args.save_dir:
        import os
        os.makedirs(args.save_dir, exist_ok=True)

    reassembler = FrameReassembler()

    n_frames = 0
    n_bad_magic = 0
    fps_count = 0
    last_fps_t = time.time()

    try:
        while True:
            data, addr = sock.recvfrom(65536)

            if len(data) < HEADER_SIZE:
                print(f"WARNING: datagram from {addr} too short ({len(data)} bytes) for header")
                continue

            magic, frame_seq, width, height, total_chunks, chunk_index, chunk_bytes = \
                struct.unpack(HEADER_FMT, data[:HEADER_SIZE])

            if magic != MAGIC:
                n_bad_magic += 1
                print(f"WARNING: bad magic 0x{magic:08X} from {addr} - ignoring datagram")
                continue

            payload = data[HEADER_SIZE:HEADER_SIZE + chunk_bytes]
            if len(payload) != chunk_bytes:
                print(f"WARNING: frame {frame_seq} chunk {chunk_index}: expected "
                      f"{chunk_bytes} payload bytes, got {len(payload)} - truncated datagram, dropping chunk")
                continue

            frame = reassembler.add_chunk(frame_seq, width, height, total_chunks, chunk_index, payload)
            if frame is None:
                continue

            n_frames += 1
            fps_count += 1

            now = time.time()
            if now - last_fps_t >= 1.0:
                print(f"{fps_count} frame(s)/s -- total {n_frames} complete, "
                      f"{n_bad_magic} bad-magic datagram(s)")
                fps_count = 0
                last_fps_t = now

            img = colorize(frame)

            if args.save_dir:
                cv2.imwrite(os.path.join(args.save_dir, f"frame_{n_frames:06d}.png"), img)

            if not args.no_display:
                cv2.imshow("Events (raw UDP)", img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()
        sock.close()
        print(f"Done - {n_frames} complete frame(s), {n_bad_magic} bad-magic datagram(s)")


if __name__ == "__main__":
    main()
