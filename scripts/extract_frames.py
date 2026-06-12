#!/usr/bin/env python3
"""
Step 1 of the crowd-counting pipeline: turn a (possibly screen-recorded)
aerial video into clean, cropped keyframes.

What it does:
  1. Auto-detects the actual video band inside the frame (crops away static
     phone/app UI by finding the region that changes over time).
     For original drone footage the band is simply the whole frame.
  2. Detects scene cuts (hard transitions between different drone shots).
  3. Saves evenly spaced keyframes per scene, cropped to the video band.

The video to analyze is expected in the input_video/ folder. If you run the
script without a video argument and input_video/ holds exactly one video,
that one is used automatically.

Usage:
  python extract_frames.py [VIDEO] [--out DIR] [--fps 2] [--no-autocrop]
"""
import argparse
import os

import cv2
import numpy as np

INPUT_DIR = "input_video"
VIDEO_EXTS = (".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm")


def resolve_video(arg):
    """Find the input video: explicit path, name inside input_video/, or the
    single video sitting in input_video/."""
    if arg:
        if os.path.isfile(arg):
            return arg
        cand = os.path.join(INPUT_DIR, arg)
        if os.path.isfile(cand):
            return cand
        raise SystemExit(f"video not found: {arg} (also tried {cand})")
    vids = []
    if os.path.isdir(INPUT_DIR):
        vids = [os.path.join(INPUT_DIR, f) for f in sorted(os.listdir(INPUT_DIR))
                if f.lower().endswith(VIDEO_EXTS)]
    if len(vids) == 1:
        return vids[0]
    if not vids:
        raise SystemExit(f"no video argument given and no video found in "
                         f"{INPUT_DIR}/ — put your video there")
    raise SystemExit(f"{INPUT_DIR}/ holds several videos, name one:\n  "
                     + "\n  ".join(vids))


def detect_video_band(cap, n_samples=24):
    """Find bounding box of the temporally-changing region (the real video)."""
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(n - 2, 0), n_samples).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    stack = np.stack(frames)
    var = stack.std(axis=0)
    mask = var > 12  # pixels that actually change over time

    def longest_run(flags):
        """Start/end of the longest contiguous True run."""
        best, cur, best_span = None, None, 0
        for i, f in enumerate(list(flags) + [False]):
            if f and cur is None:
                cur = i
            elif not f and cur is not None:
                if i - cur > best_span:
                    best, best_span = (cur, i), i - cur
                cur = None
        return best

    rows = longest_run(mask.mean(axis=1) > 0.25)
    if rows is None:
        h, w = frames[0].shape
        return 0, 0, w, h
    y0, y1 = rows
    cols = longest_run(mask[y0:y1].mean(axis=0) > 0.25)
    x0, x1 = cols if cols else (0, frames[0].shape[1])
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def detect_cuts(cap, band, thresh=0.5):
    """Return frame indices where a hard cut happens (histogram distance)."""
    x, y, w, h = band
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cuts, prev_hist, i = [], None, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        roi = f[y:y + h, x:x + w]
        roi = cv2.resize(roi, (160, 90))
        hist = cv2.calcHist([cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)],
                            [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            d = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if d > thresh:
                cuts.append(i)
        prev_hist = hist
        i += 1
    return cuts, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=None,
                    help="video file (default: the one video in input_video/)")
    ap.add_argument("--out", default="keyframes")
    ap.add_argument("--fps", type=float, default=2.0,
                    help="keyframes per second of video to save")
    ap.add_argument("--no-autocrop", action="store_true",
                    help="use full frame (original drone footage)")
    args = ap.parse_args()
    args.video = resolve_video(args.video)
    print(f"input video: {args.video}")

    cap = cv2.VideoCapture(args.video)
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if args.no_autocrop:
        band = (0, 0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    else:
        band = detect_video_band(cap)
    print(f"video band (x,y,w,h) = {band}")

    cuts, total = detect_cuts(cap, band)
    # merge cuts closer than 0.5 s (flashes etc.)
    merged = []
    for c in cuts:
        if not merged or c - merged[-1] > vid_fps / 2:
            merged.append(c)
    scenes = list(zip([0] + merged, merged + [total]))
    print(f"{total} frames, {len(scenes)} scene(s): {scenes}")

    os.makedirs(args.out, exist_ok=True)
    x, y, w, h = band
    step = max(int(round(vid_fps / args.fps)), 1)
    for s, (a, b) in enumerate(scenes):
        for i in range(a + 2, b - 1, step):  # skip frames right at the cut
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if not ok:
                continue
            crop = f[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(args.out, f"scene{s:02d}_f{i:05d}.jpg"),
                        crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cap.release()
    n_saved = len(os.listdir(args.out))
    print(f"saved {n_saved} keyframes to {args.out}/")


if __name__ == "__main__":
    main()
