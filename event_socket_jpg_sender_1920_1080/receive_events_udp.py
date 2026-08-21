#!/usr/bin/env python3
"""
Receives single-datagram JPEG-encoded event-accumulation frames sent by
EventUdpSender (on the embedded target) over UDP and displays them live.

Run this on the host PC:
    python3 receive_events_udp.py --port 5005

Then on the target, enable streaming to this host's IP before capture starts
(see CNvSIPLConsumer::SetEventUdpStreaming).

Each datagram is one complete JPEG frame - no reassembly needed. UDP is
unordered/lossy by nature; a dropped or out-of-order frame just means one
frame's worth of the live view is skipped, which is fine for a monitoring
feed (this is not the same event data path as the on-target .evsb file
writer - that one goes through TCP-like reliable disk I/O, not this
socket, so file capture isn't affected by anything here).
"""

import argparse
import os
import socket
import time

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--recvbuf", type=int, default=1 << 20,
                     help="socket receive buffer size in bytes (default 1MB)")
    ap.add_argument("--save-dir", default=None,
                     help="optional: also save each received frame as a PNG here")
    ap.add_argument("--no-display", action="store_true",
                     help="don't open a window - just log stats (and save, if --save-dir set)")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.recvbuf)
    sock.bind((args.bind, args.port))
    print(f"Listening for event frames on {args.bind}:{args.port} ...")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    n_frames = 0
    n_decode_fail = 0
    fps_count = 0
    last_fps_t = time.time()

    try:
        while True:
            data, addr = sock.recvfrom(65536)

            buf = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

            if img is None:
                n_decode_fail += 1
                print(f"WARNING: failed to decode {len(data)} bytes from {addr} "
                      f"(corrupt/partial datagram?)")
                continue

            n_frames += 1
            fps_count += 1

            now = time.time()
            if now - last_fps_t >= 1.0:
                print(f"{fps_count} frame(s)/s -- total {n_frames} received, "
                      f"{n_decode_fail} decode failures, latest {len(data)} bytes from {addr}")
                fps_count = 0
                last_fps_t = now

            if args.save_dir:
                cv2.imwrite(os.path.join(args.save_dir, f"frame_{n_frames:06d}.png"), img)

            if not args.no_display:
                cv2.imshow("Events (UDP)", img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()
        sock.close()
        print(f"Done - {n_frames} frame(s) received, {n_decode_fail} decode failure(s)")


if __name__ == "__main__":
    main()
