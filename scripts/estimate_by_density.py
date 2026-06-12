#!/usr/bin/env python3
"""
Method 5: people/m² x measured route length (model-assisted Jacobs).

The crowd model under-counts on bad footage, but its MAP of where the
crowd is stays good. So instead of trusting its absolute numbers, work in
physical units: you measure the real-world length of the strip the drone
covered (Google Maps / Earth, mapchecking.com), and this script converts
the route mosaic into square meters:

  meters per pixel  = route length / mosaic height
  crowd area (m²)   = pixels inside the crowd mask x (m/px)²
  model density     = model people inside the mask / area    [people/m²]

Then the total is anchored two independent ways:
  - reference totals: area x standard Jacobs density classes
    (0.5 loose / 1 walking / 2 slow shuffle / 4 packed people per m²)
  - implied calibration: how much the model must be multiplied to reach
    each class. Pick the class that matches what you SEE per segment in
    the footage (density_slices.csv breaks it down along the route), and
    you have a measured --calibrate factor for steps 3b/3c instead of a
    guessed one.

Usage:
  python estimate_by_density.py --route-length-m L [VIDEO] keyframes/ counts/
        [--use-frac 0.45] [--window 0.45] [--slice-px 60] [--no-autocrop]
L is the ground distance in meters from the BOTTOM edge of route_mosaic.jpg
to its TOP edge (= take-off point of the pass to the far edge of the last
frame), not the whole street if the drone covered less.

Outputs (written into counts_dir):
  route_area.jpg      - mosaic with the measured crowd area outlined (CHECK
                        THIS: area drives everything)
  density_slices.csv  - per route slice: width, area, model people, p/m²
  density_report.txt  - areas, densities, reference totals, calibration
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_frames import resolve_video  # noqa: E402
from stitch_route import build_route_mosaic  # noqa: E402
from count_crowd import crowd_mask  # noqa: E402

CLASSES = [(0.5, "loose crowd"), (1.0, "steady walking crowd"),
           (2.0, "slow shuffle"), (4.0, "packed tight (rare outdoors)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=None,
                    help="video file (default: the one video in input_video/)")
    ap.add_argument("keyframes")
    ap.add_argument("counts", help="dir with *_density.npy from count_crowd.py")
    ap.add_argument("--route-length-m", type=float, required=True,
                    help="measured ground length (m) covered by the mosaic, "
                         "bottom edge to top edge")
    ap.add_argument("--use-frac", type=float, default=0.45)
    ap.add_argument("--window", type=float, default=0.45)
    ap.add_argument("--slice-px", type=int, default=60,
                    help="slice size along the route for density_slices.csv")
    ap.add_argument("--no-autocrop", action="store_true")
    args = ap.parse_args()
    args.video = resolve_video(args.video)
    print(f"input video: {args.video}")

    mosaic, seen, avg, cnt_near, meta = build_route_mosaic(
        args.video, args.keyframes, args.counts, args.use_frac,
        args.window, args.no_autocrop)
    CH, CW = avg.shape
    s = args.route_length_m / CH  # meters per pixel (assumed isotropic)

    mask = crowd_mask(avg, CW).astype(bool) & seen
    area_m2 = float(mask.sum()) * s * s
    masked_total = float(avg[mask].sum())
    model_total = float(avg.sum())
    mean_density = masked_total / area_m2 if area_m2 > 0 else 0.0

    # per-slice breakdown along the route (canvas bottom = route start)
    slices_csv = os.path.join(args.counts, "density_slices.csv")
    with open(slices_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["route_start_m", "route_end_m", "crowd_width_m",
                    "area_m2", "model_people", "model_people_per_m2"])
        for top in range(0, CH, args.slice_px):
            bot = min(top + args.slice_px, CH)
            rows = slice(CH - bot, CH - top)  # canvas rows for this stretch
            a = float(mask[rows].sum()) * s * s
            people = float(avg[rows].sum())
            width = float(mask[rows].sum(axis=1).mean()) * s
            w.writerow([round(top * s, 1), round(bot * s, 1),
                        round(width, 1), round(a, 1), round(people, 1),
                        round(people / a, 2) if a > 0 else ""])

    # QC image: the area everything is based on
    overlay = mosaic.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, CW / 900.0
    thick = max(2, int(round(scale * 2)))
    text = (f"crowd area ~{area_m2:,.0f} m2 | scale {s * 100:.1f} cm/px | "
            f"model {mean_density:.2f} p/m2")
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    org = ((CW - tw) // 2, th + max(12, CH // 80))
    cv2.putText(overlay, text, org, font, scale, (0, 0, 0), thick + 4,
                cv2.LINE_AA)
    cv2.putText(overlay, text, org, font, scale, (255, 255, 255), thick,
                cv2.LINE_AA)
    area_jpg = os.path.join(args.counts, "route_area.jpg")
    cv2.imwrite(area_jpg, overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

    ref_lines = ""
    for c, name in CLASSES:
        ref_lines += (f"  {c:3.1f} p/m2 {name:28s}: "
                      f"{area_m2 * c:8,.0f} people  "
                      f"(model would need x{c / mean_density:.1f})\n"
                      if mean_density > 0 else "")
    summary = (
        f"video                                  : {args.video}\n"
        f"route length (YOUR input)              : "
        f"{args.route_length_m:.0f} m\n"
        f"scale                                  : {s * 100:.1f} cm/px\n"
        f"crowd area (inside green outline)      : {area_m2:,.0f} m2\n"
        f"model people inside area / route total : {masked_total:,.0f} / "
        f"{model_total:,.0f}\n"
        f"model implied mean density             : {mean_density:.2f} "
        f"people/m2\n"
        f"\nreference totals at standard density classes:\n{ref_lines}"
        "\nHow to use this: open route_area.jpg and check the green outline "
        "hugs the crowd (the area drives everything). Then look at the "
        "footage segment by segment (density_slices.csv) and decide which "
        "density class each stretch really is. The matching reference "
        "total is your Jacobs estimate; the 'model would need xF' of your "
        "chosen class is a MEASURED --calibrate factor for steps 3b/3c.\n")
    report = os.path.join(args.counts, "density_report.txt")
    with open(report, "w") as fh:
        fh.write(summary)

    print("-" * 75)
    print(summary)
    print("saved:", area_jpg)
    print("saved:", slices_csv)
    print("saved:", report)


if __name__ == "__main__":
    main()
