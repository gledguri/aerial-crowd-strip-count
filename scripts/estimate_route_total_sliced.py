#!/usr/bin/env python3
"""
Step 3 (alternative): whole-route estimate by SLICE AVERAGING instead of
strip summing.

Strip summing (estimate_route_total.py) counts every person exactly once,
but from a single observation — the keyframe where their strip of ground
exits the bottom edge. Each number therefore inherits the full noise of one
model reading, and any error in the per-interval motion estimate makes
strips overlap (double count) or gap (undercount).

Slice averaging instead maps every keyframe onto a shared 1-D route axis:
the ground advance gives each keyframe an offset along the route, so the
same physical slice of street is observed in many keyframes, at a different
image row each time. Per route position we AVERAGE all its observations,
then integrate the averaged profile over the whole route:

    keyframe 1: [100, 180, 350]            (route slices A, B, C)
    keyframe 2:      [220, 390, 470]       (route slices B, C, D)
    keyframe 3:           [240, 570, 610]  (route slices C, D, E)
    total = 100 + (180+220)/2 + (350+390+240)/3 + (470+570)/2 + 610

Each slice's estimate is the mean of ~(window height / advance per
keyframe) independent readings, and motion-estimate noise only blurs the
profile instead of double counting.

Because the model systematically under-counts the far field, by default
only observations from the bottom --use-frac of the frame (nearest the
camera) enter the average. Route positions never seen there — essentially
the stretch ahead of the drone in the final keyframe — fall back to their
single nearest-camera observation.

Usage:
  python estimate_route_total_sliced.py [VIDEO] keyframes/ counts_dir/
        [--use-frac 0.45] [--slice-px 60] [--window 0.45] [--no-autocrop]
counts_dir must contain *_density.npy from count_crowd.py --save-density.
If VIDEO is omitted, the single video in input_video/ is used.

Outputs (written into counts_dir):
  route_slices.csv        - per --slice-px slice of route: observations,
                            averaged people count, running cumulative total
  route_total_sliced.txt  - the final numbers
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
from estimate_route_total import per_frame_motion  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=None,
                    help="video file (default: the one video in input_video/)")
    ap.add_argument("keyframes")
    ap.add_argument("counts", help="dir with *_density.npy from count_crowd.py")
    ap.add_argument("--use-frac", type=float, default=0.45,
                    help="fraction of frame height (from the bottom) whose "
                         "rows enter the average; 1.0 = average over the "
                         "whole frame including the unreliable far field")
    ap.add_argument("--slice-px", type=int, default=60,
                    help="slice size along the route for route_slices.csv "
                         "(reporting only; the math runs per pixel row)")
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
    stems = [os.path.splitext(os.path.basename(p))[0] for p in frames]
    densities = [np.load(os.path.join(args.counts, s + "_density.npy"))
                 for s in stems]

    cap = cv2.VideoCapture(args.video)
    if args.no_autocrop:
        band = (0, 0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    else:
        band = detect_video_band(cap)
    cap.release()
    print(f"video band {band}; tracking ground motion on raw frames ...")
    dys, dxs = per_frame_motion(args.video, band, args.window)

    H = densities[0].shape[0]
    if abs(H - band[3]) > 2:  # density maps saved at keyframe (=band) size
        print(f"WARNING: density height {H} != band height {band[3]}, "
              f"rescaling motion accordingly")
        dys = dys * (H / band[3])
    print(f"cumulative ground advance: {dys.sum():.0f} px "
          f"({dys.sum() / H:.1f} frame-heights), "
          f"lateral drift {dxs.sum():.0f} px")

    # route offset of each keyframe = ground advance accumulated so far;
    # image row y of keyframe k sits at route position offset[k] + (H-1-y)
    offsets = np.array([dys[:i].sum() for i in kf_idx])
    offsets -= offsets.min()
    n_bins = int(round(offsets.max())) + H
    row_profiles = [d.sum(axis=1) for d in densities]

    y_min = int(round(H * (1.0 - args.use_frac)))
    sum_near = np.zeros(n_bins)      # near-field observations -> averaged
    cnt_near = np.zeros(n_bins)
    far_val = np.zeros(n_bins)       # far-field fallback: keep the reading
    far_y = np.full(n_bins, -1)      # taken nearest the camera (largest y)
    for k, prof in enumerate(row_profiles):
        base = int(round(offsets[k]))
        rows = np.arange(H)
        bins = base + (H - 1 - rows)
        near = rows >= y_min
        np.add.at(sum_near, bins[near], prof[near])
        np.add.at(cnt_near, bins[near], 1)
        for y in range(y_min - 1, -1, -1):  # far rows, nearest first
            b = base + (H - 1 - y)
            if y > far_y[b]:
                far_y[b] = y
                far_val[b] = prof[y]

    covered = cnt_near > 0
    avg = np.where(covered, sum_near / np.maximum(cnt_near, 1), far_val)
    total = float(avg.sum())
    near_total = float(avg[covered].sum())
    far_total = total - near_total
    mean_obs = float(cnt_near[covered].mean()) if covered.any() else 0.0

    slices_csv = os.path.join(args.counts, "route_slices.csv")
    with open(slices_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["route_start_px", "route_end_px", "mean_observations",
                    "people_in_slice", "cumulative_people"])
        cum = 0.0
        for s in range(0, n_bins, args.slice_px):
            e = min(s + args.slice_px, n_bins)
            people = float(avg[s:e].sum())
            cum += people
            w.writerow([s, e, round(float(cnt_near[s:e].mean()), 1),
                        round(people, 1), round(cum, 1)])

    summary = (
        f"video                                         : {args.video}\n"
        f"keyframes                                     : {len(frames)}\n"
        f"cumulative ground advance (px / frame-heights): "
        f"{dys.sum():.0f} / {dys.sum() / H:.1f}\n"
        f"route length (px) / observations per slice    : "
        f"{n_bins} / {mean_obs:.1f}\n"
        f"people on slices seen in the near field       : {near_total:.0f}\n"
        f"people on the far-field-only final stretch    : {far_total:.0f}\n"
        f"ESTIMATED TOTAL ALONG THE ROUTE               : {total:.0f}\n"
        "\nCaveats: each slice is the mean of its near-field observations, "
        "so single-frame noise is averaged out, but the model's systematic "
        "under-counting (night footage, compression, tree canopies) is NOT "
        "fixed by averaging; the final stretch ahead of the drone is only "
        "seen far away and stays under-counted.\n")
    total_txt = os.path.join(args.counts, "route_total_sliced.txt")
    with open(total_txt, "w") as fh:
        fh.write(summary)

    print("-" * 75)
    print(summary)
    print("saved:", slices_csv)
    print("saved:", total_txt)


if __name__ == "__main__":
    main()
