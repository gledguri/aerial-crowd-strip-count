#!/usr/bin/env python3
"""
Step 3: density estimate in physical units (people/m²).

The crowd model under-counts on bad footage, but its MAP of where the
crowd is stays good. So this step anchors everything in real-world
geometry. Give it ONE measurement you trust:

  --area-m2 A         the real occupied area in m² (measure on Google
                      Maps/Earth or mapchecking.com): the px->m scale is
                      calibrated so the crowd outline equals YOUR area;
  --route-length-m L  ground length covered by the mosaic, bottom edge to
                      top edge.

Give BOTH for an anisotropic scale — the right choice for oblique drone
footage, where ground meters per pixel differ along vs across the flight:
L anchors the along-route scale, A the across-route scale, and strip
lengths/widths then match the real street geometry.

It then reports the crowd's density (model-implied people/m²): one single
number if the density is roughly uniform along the route, or split into
strips if it varies a lot. Strip geometry (length, width, area) is always
written to jacobs_segments.csv — that file is the input of step 4, where
you assign each strip the density class you SEE in the footage.

NOTE on the densities: the model's absolute level runs LOW on night or
compressed footage — treat its p/m² as a lower bound and the per-strip
VARIATION as trustworthy.

Usage:
  python estimate_by_density.py --area-m2 15000 [VIDEO] keyframes/ counts/
        [--segment-m 50] [--weights QNRF] [--upscale 2]
        [--use-frac 0.45] [--window 0.45] [--no-autocrop]
The density model runs automatically on any keyframe that has no
*_density.npy in counts_dir yet (a few seconds per frame, once per video).

Outputs (written into counts_dir):
  route_area.jpg       - mosaic with the crowd area outlined (CHECK THIS:
                         the outline must hug the crowd)
  jacobs_segments.csv  - per strip: length, width, area, model density,
                         empty assumed_density column for you -> step 4
  density_report.txt   - geometry, density estimate, reference classes
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
from count_crowd import crowd_mask, generate_density_maps  # noqa: E402


def ensure_density_maps(keyframes, counts, weights, upscale):
    """Run the density model on any keyframe that has no *_density.npy yet
    (a few seconds per frame on CPU, once per video)."""
    import glob as g
    frames = sorted(g.glob(os.path.join(keyframes, "*.jpg")) +
                    g.glob(os.path.join(keyframes, "*.png")))
    missing = [p for p in frames if not os.path.exists(os.path.join(
        counts, os.path.splitext(os.path.basename(p))[0] + "_density.npy"))]
    if missing:
        print(f"estimating density maps for {len(missing)} keyframe(s) "
              f"(model: DM-Count {weights}, upscale x{upscale}) ...")
        generate_density_maps(missing, counts, weights=weights,
                              upscale=upscale)

CLASSES = [(0.5, "loose crowd"), (1.0, "steady walking crowd"),
           (2.0, "slow shuffle"), (4.0, "packed tight (rare outdoors)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=None,
                    help="video file (default: the one video in input_video/)")
    ap.add_argument("keyframes")
    ap.add_argument("counts", help="dir with *_density.npy from count_crowd.py")
    ap.add_argument("--area-m2", type=float, default=None,
                    help="YOUR measured occupied area in m²; calibrates the "
                         "px->m scale")
    ap.add_argument("--route-length-m", type=float, default=None,
                    help="measured ground length (m) covered by the mosaic, "
                         "bottom to top edge. Give BOTH this and --area-m2 "
                         "for an anisotropic scale (best for oblique "
                         "footage): length anchors the along-route scale, "
                         "area the across-route scale")
    ap.add_argument("--segment-m", type=float, default=50.0,
                    help="strip length (m) for the per-strip breakdown")
    ap.add_argument("--weights", default="QNRF",
                    choices=["SHA", "SHB", "QNRF"],
                    help="model weights when density maps must be computed")
    ap.add_argument("--upscale", type=int, default=2,
                    help="model upscale factor when density maps must be "
                         "computed (2 recovers small people in low-res "
                         "footage)")
    ap.add_argument("--use-frac", type=float, default=0.45)
    ap.add_argument("--window", type=float, default=0.45)
    ap.add_argument("--no-autocrop", action="store_true")
    args = ap.parse_args()
    if args.area_m2 is None and args.route_length_m is None:
        ap.error("give --area-m2 (preferred) or --route-length-m")
    args.video = resolve_video(args.video)
    print(f"input video: {args.video}")

    ensure_density_maps(args.keyframes, args.counts, args.weights,
                        args.upscale)
    mosaic, seen, avg, cnt_near, meta = build_route_mosaic(
        args.video, args.keyframes, args.counts, args.use_frac,
        args.window, args.no_autocrop)
    CH, CW = avg.shape

    mask = crowd_mask(avg, CW).astype(bool) & seen
    mask_px = float(mask.sum())
    # pixel->meter scale. With BOTH measurements the scale is anisotropic:
    # the route length fixes the along-route scale (sy) and the area fixes
    # the across-route scale (sx). Oblique drone footage genuinely has
    # different ground resolution along vs across the flight direction, so
    # the two scales can differ a lot — that is information, not an error.
    if args.area_m2 and args.route_length_m:
        sy = args.route_length_m / CH
        sx = args.area_m2 / (mask_px * sy)
        area_m2 = args.area_m2
        anchor = "YOUR area + route length (anisotropic)"
    elif args.area_m2:
        sx = sy = (args.area_m2 / mask_px) ** 0.5
        area_m2, anchor = args.area_m2, "YOUR area measurement"
    else:
        sx = sy = args.route_length_m / CH
        area_m2, anchor = mask_px * sx * sy, "YOUR route length"
    route_len_m = CH * sy
    masked_total = float(avg[mask].sum())
    mean_density = masked_total / area_m2 if area_m2 > 0 else 0.0

    # independent density estimate from people-as-units: count local peaks
    # of the density map (one peak = one localized person) instead of
    # integrating mass. Only physically meaningful when one person spans
    # enough pixels for neighbors to stay separable.
    person_px = 0.45 / max(sx, sy)  # worst-direction person size
    sm = cv2.GaussianBlur(avg, (0, 0), 1.0)
    kx = max(3, int(round(0.35 / sx)) | 1)  # min spacing between people
    ky = max(3, int(round(0.35 / sy)) | 1)
    maxf = cv2.dilate(sm, np.ones((ky, kx), np.float32))
    vals = sm[mask & (sm > 0)]
    thr = float(np.percentile(vals, 60)) if vals.size else 0.0
    n_peaks = int(((sm >= maxf - 1e-12) & mask & (sm > thr)).sum())
    peak_density = n_peaks / area_m2 if area_m2 > 0 else 0.0
    peaks_valid = person_px >= 5.0

    # per-strip breakdown along the route (canvas bottom = route start)
    seg_px = max(1, int(round(args.segment_m / sy)))
    segs = []
    for start in range(0, CH, seg_px):
        end = min(start + seg_px, CH)
        rows = slice(CH - end, CH - start)
        a = float(mask[rows].sum()) * sx * sy
        people = float(avg[rows][mask[rows]].sum())
        width = float(mask[rows].sum(axis=1).mean()) * sx
        segs.append(dict(start_m=start * sy, end_m=end * sy,
                         length_m=(end - start) * sy, width_m=width,
                         area_m2=a, people=people,
                         density=people / a if a > 0 else 0.0))

    seg_csv = os.path.join(args.counts, "jacobs_segments.csv")
    with open(seg_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["segment", "start_m", "end_m", "length_m",
                    "mean_width_m", "area_m2", "model_density_p_m2",
                    "assumed_density_p_m2"])
        for i, g in enumerate(segs):
            w.writerow([f"strip{i:02d}", round(g["start_m"], 1),
                        round(g["end_m"], 1), round(g["length_m"], 1),
                        round(g["width_m"], 1), round(g["area_m2"], 1),
                        round(g["density"], 2), ""])

    # uniform or not? judge on strips that actually hold crowd
    dens = [g["density"] for g in segs if g["width_m"] >= 2.0]
    cv = float(np.std(dens) / np.mean(dens)) if dens else 0.0
    strip_mean = float(np.mean(dens)) if dens else 0.0
    if cv <= 0.30:
        verdict = (f"density is roughly UNIFORM along the route "
                   f"(spread {cv:.0%}):\n"
                   f"  -> use ONE density: {mean_density:.2f} people/m2 "
                   f"over {area_m2:,.0f} m2\n")
    else:
        lines = "".join(
            f"  {g['start_m']:5.0f}-{g['end_m']:5.0f} m  "
            f"width {g['width_m']:5.1f} m  area {g['area_m2']:7,.0f} m2  "
            f"{g['density']:.2f} p/m2\n" for g in segs)
        verdict = (f"density VARIES along the route (spread {cv:.0%}) — "
                   f"use the per-strip values:\n{lines}")
    verdict += (f"average density of all strips: {mean_density:.2f} "
                f"people/m2 (area-weighted)\n"
                f"                               {strip_mean:.2f} "
                f"people/m2 (simple mean of the strips)\n")

    # QC image: the area everything is based on
    overlay = mosaic.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, CW / 900.0
    thick = max(2, int(round(scale * 2)))
    text = (f"crowd area {area_m2:,.0f} m2 | scale {sy * 100:.0f}x"
            f"{sx * 100:.0f} cm/px | model {mean_density:.2f} p/m2")
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    org = ((CW - tw) // 2, th + max(12, CH // 80))
    cv2.putText(overlay, text, org, font, scale, (0, 0, 0), thick + 4,
                cv2.LINE_AA)
    cv2.putText(overlay, text, org, font, scale, (255, 255, 255), thick,
                cv2.LINE_AA)
    area_jpg = os.path.join(args.counts, "route_area.jpg")
    cv2.imwrite(area_jpg, overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

    ref_lines = "".join(
        f"  {c:3.1f} p/m2 {name:28s}: {area_m2 * c:8,.0f} people\n"
        for c, name in CLASSES)
    summary = (
        f"video                                  : {args.video}\n"
        f"scale anchored on                      : {anchor}\n"
        f"scale along route / across route       : {sy * 100:.1f} / "
        f"{sx * 100:.1f} cm/px\n"
        f"route length covered by the mosaic     : {route_len_m:.0f} m\n"
        f"crowd area (inside green outline)      : {area_m2:,.0f} m2\n"
        f"one person at this scale               : ~{person_px:.1f} px "
        f"in the coarsest direction\n"
        f"\nDENSITY ESTIMATE (model-implied; absolute level is a LOWER "
        f"bound on bad footage,\nthe per-strip variation is the "
        f"trustworthy part):\n{verdict}"
        f"\ndensity via counting people as units (blob peaks): "
        f"{peak_density:.2f} people/m2\n")
    if peaks_valid:
        summary += ("  individuals are resolvable at this scale — this is "
                    "a usable independent estimate.\n")
    else:
        summary += (
            f"  UNRELIABLE here: a person is only ~{person_px:.1f} px, "
            f"below the ~5 px needed to\n  separate neighbors, so blob "
            f"counting under-reads. On this footage the absolute\n"
            f"  density can only come from your eyes (step 4) or from "
            f"higher-resolution source\n  video.\n")
    summary += (
        f"\nfor orientation, the standard density classes over this area:\n"
        f"{ref_lines}"
        "\nNext (step 4): open route_area.jpg and check the green outline "
        "hugs the crowd. Then fill the assumed_density_p_m2 column of "
        "jacobs_segments.csv with the class you SEE per strip in the "
        "footage and run jacobs_estimate.py on it.\n")
    report = os.path.join(args.counts, "density_report.txt")
    with open(report, "w") as fh:
        fh.write(summary)

    print("-" * 75)
    print(summary)
    print("saved:", area_jpg)
    print("saved:", seg_csv)
    print("saved:", report)


if __name__ == "__main__":
    main()
