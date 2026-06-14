#!/usr/bin/env python3
"""
Step 6: final report — calibrate the model totals against the Jacobs
total (step 4) and re-stamp every keyframe overlay with the ACCUMULATED
calibrated count.

The calibration factor F is the honest version of "the model runs ~3x
low": it is MEASURED as

    F = Jacobs total (step 4)  /  mean of the step-5 model totals

and applied to both step-5 methods. Every counts/scene*_density.jpg is
then re-rendered so its banner shows the running total of people counted
from the start of the flight up to that frame (calibrated) — frame 1
shows everyone it sees, frame 2 adds only the newly revealed ground, and
the last frame shows the final number. Rebuild the gif afterwards and you
get a counter that climbs as the drone flies.

Requires: step 5 already run (route_total_sliced.txt and
route_total_mosaic.txt present in counts_dir).

Usage:
  python report_route.py --jacobs-total N [VIDEO] keyframes/ counts/
        [--use-frac 0.45] [--window 0.45] [--no-autocrop]
  (or pass --calibrate F directly instead of --jacobs-total)

Outputs:
  counts/scene*_density.jpg  - re-stamped with the accumulated total
  counts/report.txt          - all numbers + the range to quote
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_frames import resolve_video  # noqa: E402
from stitch_route import build_route_mosaic  # noqa: E402
from count_crowd import save_overlay  # noqa: E402


def read_model_total(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — run step 5 first "
                         "(estimate_route_total_sliced.py and "
                         "stitch_route.py)")
    m = re.search(r"model route total\s*:\s*([\d.]+)", open(path).read())
    if not m:
        raise SystemExit(f"could not find 'model route total' in {path} — "
                         "re-run step 5 with the current scripts")
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=None,
                    help="video file (default: the one video in input_video/)")
    ap.add_argument("keyframes")
    ap.add_argument("counts")
    ap.add_argument("--jacobs-total", type=float, default=None,
                    help="the step-4 Jacobs total; calibration factor is "
                         "derived from it")
    ap.add_argument("--calibrate", type=float, default=None,
                    help="use this factor directly instead of deriving it "
                         "from --jacobs-total")
    ap.add_argument("--use-frac", type=float, default=0.45)
    ap.add_argument("--window", type=float, default=0.45)
    ap.add_argument("--no-autocrop", action="store_true")
    args = ap.parse_args()
    if args.jacobs_total is None and args.calibrate is None:
        ap.error("give --jacobs-total (from step 4) or --calibrate")
    args.video = resolve_video(args.video)
    print(f"input video: {args.video}")

    total_5a = read_model_total(
        os.path.join(args.counts, "route_total_sliced.txt"))
    total_5b = read_model_total(
        os.path.join(args.counts, "route_total_mosaic.txt"))
    model_mean = (total_5a + total_5b) / 2
    if args.calibrate is not None:
        F = args.calibrate
        jacobs = args.jacobs_total
    else:
        F = args.jacobs_total / model_mean
        jacobs = args.jacobs_total

    mosaic, seen, avg, cnt_near, meta = build_route_mosaic(
        args.video, args.keyframes, args.counts, args.use_frac,
        args.window, args.no_autocrop)

    # accumulated calibrated count at keyframe k = everything from the
    # route start (canvas bottom) up to that keyframe's top edge
    print()
    for k, path in enumerate(meta["frames"]):
        cum = float(avg[int(round(meta["offy"][k])):, :].sum()) * F
        stem = os.path.splitext(os.path.basename(path))[0]
        npy = os.path.join(args.counts, stem + "_density.npy")
        out = os.path.join(args.counts, stem + "_density.jpg")
        save_overlay(path, np.load(npy), out,
                     label=f"Counted ~ {cum:,.0f}")
        print(f"{stem:28s} Counted ~ {cum:9,.0f}")
    final = float(avg.sum()) * F

    lo, hi = sorted([total_5a * F, total_5b * F])
    summary = (
        f"video                                   : {args.video}\n"
        f"5a slice-averaging model total          : {total_5a:,.0f}\n"
        f"5b mosaic-averaging model total         : {total_5b:,.0f}\n")
    if jacobs is not None:
        summary += (f"step-4 Jacobs total                     : "
                    f"{jacobs:,.0f}\n")
    summary += (
        f"calibration factor F                    : x{F:.2f}\n"
        f"calibrated 5a / 5b                      : {total_5a * F:,.0f} / "
        f"{total_5b * F:,.0f}\n"
        f"FINAL ESTIMATE                          : {final:,.0f}  "
        f"(quote {lo * 0.7:,.0f} .. {hi * 1.3:,.0f} as the honest range)\n"
        "\nEvery counts/scene*_density.jpg now shows the accumulated "
        "calibrated count up to that frame; rebuild the gif to get the "
        "climbing counter (see README Extras).\n")
    report = os.path.join(args.counts, "report.txt")
    with open(report, "w") as fh:
        fh.write(summary)
    print("-" * 75)
    print(summary)
    print("saved:", report)


if __name__ == "__main__":
    main()
