#!/usr/bin/env python3
"""
Step 3c: stitch the whole flight into ONE route mosaic and count on it.

Every keyframe is placed into a shared route canvas using the tracked
ground motion — vertical advance AND lateral drift, so frames are aligned
in 2-D, not just along the route axis. Two things are stitched:

  1. The keyframe images themselves -> route_mosaic.jpg, one picture of
     the entire flight (later frames overwrite, so every spot shows its
     nearest = sharpest view).
  2. The density maps -> wherever N frames overlap the same ground, their
     N density readings are AVERAGED (near-field rows only, where the
     model is reliable; ground never seen in the near field falls back to
     its nearest available reading). The averaged mosaic is summed for the
     route total and rendered as route_mosaic_density.jpg with the same
     cluster/grid count labels as the per-frame overlays.

This is the 2-D version of estimate_route_total_sliced.py: same averaging
idea, but overlapping observations are aligned per pixel instead of per
row, so lateral drift no longer smears the average.

The mosaic jpg can also be fed straight back into the model as a single
huge "frame" (the stitched-picture cross-check):
  count_crowd.py counts/route_mosaic.jpg --out counts_mosaic ...

Usage:
  python stitch_route.py [VIDEO] keyframes/ counts_dir/
        [--use-frac 0.45] [--window 0.45] [--min-cluster 3] [--no-autocrop]
counts_dir must contain *_density.npy from count_crowd.py --save-density.

Outputs (written into counts_dir):
  route_mosaic.jpg          - the stitched flight
  route_mosaic_density.jpg  - averaged density over the mosaic, labeled
  route_total_mosaic.txt    - the final numbers
"""
import argparse
import glob
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_frames import detect_video_band, resolve_video  # noqa: E402
from estimate_route_total import per_frame_motion  # noqa: E402
from count_crowd import label_clusters  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=None,
                    help="video file (default: the one video in input_video/)")
    ap.add_argument("keyframes")
    ap.add_argument("counts", help="dir with *_density.npy from count_crowd.py")
    ap.add_argument("--use-frac", type=float, default=0.45,
                    help="fraction of frame height (from the bottom) whose "
                         "density rows enter the average")
    ap.add_argument("--window", type=float, default=0.45,
                    help="fraction of band height (from bottom) used for "
                         "motion tracking")
    ap.add_argument("--min-cluster", type=float, default=3.0,
                    help="only label clusters of at least this many people")
    ap.add_argument("--no-autocrop", action="store_true")
    args = ap.parse_args()
    args.video = resolve_video(args.video)
    print(f"input video: {args.video}")

    frames = sorted(glob.glob(os.path.join(args.keyframes, "*.jpg")) +
                    glob.glob(os.path.join(args.keyframes, "*.png")))
    if len(frames) < 2:
        raise SystemExit("need at least 2 keyframes")
    kf_idx = [int(re.search(r"_f(\d+)", os.path.basename(p)).group(1))
              for p in frames]
    stems = [os.path.splitext(os.path.basename(p))[0] for p in frames]
    densities = [np.load(os.path.join(args.counts, s + "_density.npy"))
                 for s in stems]
    H, W = densities[0].shape

    cap = cv2.VideoCapture(args.video)
    if args.no_autocrop:
        band = (0, 0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    else:
        band = detect_video_band(cap)
    cap.release()
    print(f"video band {band}; tracking ground motion on raw frames ...")
    dys, dxs = per_frame_motion(args.video, band, args.window)
    if abs(H - band[3]) > 2:
        print(f"WARNING: density height {H} != band height {band[3]}, "
              f"rescaling motion accordingly")
        dys, dxs = dys * (H / band[3]), dxs * (H / band[3])
    print(f"cumulative ground advance: {dys.sum():.0f} px "
          f"({dys.sum() / H:.1f} frame-heights), "
          f"lateral drift {dxs.sum():.0f} px")

    # ground slides down/right by (dy,dx) per raw frame, so each keyframe's
    # paste offset is the NEGATIVE cumulative motion; route start ends up at
    # the canvas bottom, flight direction points up
    offy = np.array([-dys[:i].sum() for i in kf_idx])
    offx = np.array([-dxs[:i].sum() for i in kf_idx])
    offy -= offy.min()
    offx -= offx.min()
    CH, CW = int(round(offy.max())) + H, int(round(offx.max())) + W

    mosaic = np.zeros((CH, CW, 3), np.uint8)
    seen = np.zeros((CH, CW), bool)
    sum_near = np.zeros((CH, CW), np.float32)
    cnt_near = np.zeros((CH, CW), np.uint16)
    far_val = np.zeros((CH, CW), np.float32)
    far_y = np.full((CH, CW), -1, np.int16)
    y_min = int(round(H * (1.0 - args.use_frac)))
    rows = np.arange(y_min, dtype=np.int16)[:, None]  # far-field row index

    for k, (path, d) in enumerate(zip(frames, densities)):
        oy, ox = int(round(offy[k])), int(round(offx[k]))
        img = cv2.imread(path)
        mosaic[oy:oy + H, ox:ox + W] = img  # later = nearer = sharper view
        seen[oy:oy + H, ox:ox + W] = True
        sum_near[oy + y_min:oy + H, ox:ox + W] += d[y_min:]
        cnt_near[oy + y_min:oy + H, ox:ox + W] += 1
        fy = far_y[oy:oy + y_min, ox:ox + W]
        nearer = rows > fy  # keep the reading taken nearest the camera
        far_val[oy:oy + y_min, ox:ox + W][nearer] = d[:y_min][nearer]
        fy[nearer] = np.broadcast_to(rows, nearer.shape)[nearer]

    covered = cnt_near > 0
    avg = np.where(covered, sum_near / np.maximum(cnt_near, 1), far_val)
    total = float(avg.sum())
    near_total = float(avg[covered].sum())
    mean_obs = float(cnt_near[covered].mean())

    mosaic_path = os.path.join(args.counts, "route_mosaic.jpg")
    cv2.imwrite(mosaic_path, mosaic, [cv2.IMWRITE_JPEG_QUALITY, 92])

    vis = avg / avg.max() if avg.max() > 0 else avg
    heat = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat[~seen] = 0
    overlay = cv2.addWeighted(mosaic, 0.55, heat, 0.45, 0)
    labeled, n = label_clusters(overlay, avg, args.min_cluster)
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, CW / 900.0
    thick = max(2, int(round(scale * 2)))
    text = f"route total ~{total:,.0f} | {labeled:,.0f} in {n} clusters"
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    org = ((CW - tw) // 2, th + max(12, CH // 80))
    cv2.putText(overlay, text, org, font, scale, (0, 0, 0), thick + 4,
                cv2.LINE_AA)
    cv2.putText(overlay, text, org, font, scale, (255, 255, 255), thick,
                cv2.LINE_AA)
    overlay_path = os.path.join(args.counts, "route_mosaic_density.jpg")
    cv2.imwrite(overlay_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

    summary = (
        f"video                                         : {args.video}\n"
        f"keyframes                                     : {len(frames)}\n"
        f"mosaic size (route px x width px)             : {CH} x {CW}\n"
        f"cumulative ground advance / lateral drift (px): "
        f"{dys.sum():.0f} / {dxs.sum():.0f}\n"
        f"mean overlapping observations per ground px   : {mean_obs:.1f}\n"
        f"people on ground seen in the near field       : {near_total:.0f}\n"
        f"people on the far-field-only final stretch    : "
        f"{total - near_total:.0f}\n"
        f"ESTIMATED TOTAL ALONG THE ROUTE               : {total:.0f}\n"
        "\nSame averaging as estimate_route_total_sliced.py but aligned in "
        "2-D (lateral drift compensated). The cluster labels on "
        "route_mosaic_density.jpg decompose this exact total.\n")
    total_txt = os.path.join(args.counts, "route_total_mosaic.txt")
    with open(total_txt, "w") as fh:
        fh.write(summary)

    print("-" * 75)
    print(summary)
    print("saved:", mosaic_path)
    print("saved:", overlay_path)
    print("saved:", total_txt)


if __name__ == "__main__":
    main()
