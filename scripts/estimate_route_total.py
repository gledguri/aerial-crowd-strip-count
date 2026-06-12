#!/usr/bin/env python3
"""
Step 3 of the crowd-counting pipeline: combine per-frame density maps into a
single whole-route estimate for a video where the drone flies ALONG the
street in one continuous pass.

Idea ("strip summing"): as the drone flies forward, the ground slides toward
the bottom edge of the frame; everyone the drone overflies exits the frame
through the bottom edge exactly once. So:

    total = sum over keyframe intervals( density mass in the strip of ground
            that exits the bottom edge during the interval )
          + density mass still visible in the final keyframe

Each person is counted once, at their closest (= most reliably counted)
position. The ground advance is measured on the RAW consecutive video frames
(motion per frame is small, so tracking is reliable even in sped-up reels)
with Lucas-Kanade feature tracking, then accumulated between keyframes.

Usage:
  python estimate_route_total.py [VIDEO] keyframes/ counts_dir/
        [--no-autocrop] [--window 0.45]
counts_dir must contain *_density.npy from count_crowd.py --save-density.
If VIDEO is omitted, the single video in input_video/ is used.

Outputs (written into counts_dir):
  route_strips.csv  - per-keyframe ground advance + people counted in each
                      exiting strip + running cumulative total
  route_total.txt   - the final numbers (exited / last frame / TOTAL)
"""
import argparse
import csv
import glob
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_frames import detect_video_band, resolve_video  # noqa: E402


def per_frame_motion(video, band, window_frac, scale=0.5):
    """Median (dy, dx) of tracked ground features for every consecutive raw
    frame pair, measured in the bottom `window_frac` of the cropped band.
    Returns arrays indexed by frame."""
    x, y, w, h = band
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dys, dxs = np.zeros(n), np.zeros(n)
    prev_gray = None
    win_y = int(h * (1 - window_frac))
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=scale, fy=scale)
        if prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(prev_gray[int(win_y * scale):],
                                          maxCorners=300, qualityLevel=0.01,
                                          minDistance=7)
            if pts is not None and len(pts) >= 10:
                pts += np.array([[0, win_y * scale]], dtype=np.float32)
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, g, pts, None, **lk)
                good = st.ravel() == 1
                if good.sum() >= 10:
                    v = (nxt[good] - pts[good]).reshape(-1, 2)
                    dxs[i] = np.median(v[:, 0]) / scale
                    dys[i] = np.median(v[:, 1]) / scale
        prev_gray = g
        i += 1
    cap.release()
    return dys[:i], dxs[:i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=None,
                    help="video file (default: the one video in input_video/)")
    ap.add_argument("keyframes")
    ap.add_argument("counts", help="dir with *_density.npy from count_crowd.py")
    ap.add_argument("--window", type=float, default=0.45,
                    help="fraction of band height (from bottom) used for "
                         "motion tracking")
    ap.add_argument("--no-autocrop", action="store_true")
    args = ap.parse_args()
    args.video = resolve_video(args.video)
    print(f"input video: {args.video}")

    frames = sorted(glob.glob(os.path.join(args.keyframes, "*.jpg")))
    if len(frames) < 2:
        raise SystemExit("need at least 2 keyframes")
    kf_idx = [int(re.search(r"_f(\d+)", os.path.basename(p)).group(1))
              for p in frames]

    cap = cv2.VideoCapture(args.video)
    if args.no_autocrop:
        band = (0, 0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    else:
        band = detect_video_band(cap)
    cap.release()
    print(f"video band {band}; tracking ground motion on raw frames ...")
    dys, dxs = per_frame_motion(args.video, band, args.window)
    print(f"cumulative ground advance: {dys.sum():.0f} px "
          f"({dys.sum() / band[3]:.1f} frame-heights), "
          f"lateral drift {dxs.sum():.0f} px")

    total_exit = 0.0
    strip_rows = []
    print(f"\n{'keyframe':28s} {'advance_px':>10s} {'strip_count':>11s}  note")
    for i in range(len(frames) - 1):
        adv = float(dys[kf_idx[i]:kf_idx[i + 1]].sum())
        stem = os.path.splitext(os.path.basename(frames[i]))[0]
        density = np.load(os.path.join(args.counts, stem + "_density.npy"))
        h = density.shape[0]
        note, strip = "", 0.0
        if adv > 0:
            strip = float(density[h - min(int(round(adv)), h):, :].sum())
            total_exit += strip
        else:
            note = "no forward motion, strip skipped"
        strip_rows.append([stem, round(adv), round(strip, 1),
                           round(total_exit, 1), note])
        print(f"{stem:28s} {adv:10.0f} {strip:11.0f}  {note}")

    last_stem = os.path.splitext(os.path.basename(frames[-1]))[0]
    last_mass = float(np.load(os.path.join(args.counts,
                                           last_stem + "_density.npy")).sum())
    total = total_exit + last_mass

    strips_csv = os.path.join(args.counts, "route_strips.csv")
    with open(strips_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["keyframe", "ground_advance_px", "people_in_exiting_strip",
                    "cumulative_people", "note"])
        w.writerows(strip_rows)
        w.writerow([last_stem, "", round(last_mass, 1), round(total, 1),
                    "people still visible in last keyframe"])

    summary = (
        f"video                                         : {args.video}\n"
        f"keyframes                                     : {len(frames)}\n"
        f"cumulative ground advance (px / frame-heights): "
        f"{dys.sum():.0f} / {dys.sum() / band[3]:.1f}\n"
        f"people who exited the frame during the flight : {total_exit:.0f}\n"
        f"people still visible in the last keyframe     : {last_mass:.0f}\n"
        f"ESTIMATED TOTAL ALONG THE ROUTE               : {total:.0f}\n"
        "\nCaveats: far-field people in the final frame are under-counted; "
        "crowd walking relative to the drone biases strips by roughly "
        "(walking speed / drone ground speed); night footage and "
        "compression push everything toward UNDER-counting.\n")
    total_txt = os.path.join(args.counts, "route_total.txt")
    with open(total_txt, "w") as fh:
        fh.write(summary)

    print("-" * 75)
    print(summary)
    print("saved:", strips_csv)
    print("saved:", total_txt)


if __name__ == "__main__":
    main()
