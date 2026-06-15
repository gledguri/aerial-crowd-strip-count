#!/usr/bin/env python3
"""
Stamp a manual label (e.g. "Day 10") centered just below the "Counted ~ N"
banner on every *density.jpg in a folder.

Run it ONCE, after step 6 (report_route.py re-writes the frames fresh, so
running this twice would stamp the label twice). Rebuild the gif afterwards
to carry the label into the animation.

Usage:
  python stamp_day.py "Day 10" counts/
  python stamp_day.py "Day 10" counts_mosaic/    # the stitched mosaic too
"""
import argparse
import glob
import os

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", help='label to stamp, e.g. "Day 10"')
    ap.add_argument("folder", help="folder holding the *density.jpg images")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.folder, "*density.jpg")))
    if not paths:
        raise SystemExit(f"no *density.jpg found in {args.folder}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    for path in paths:
        img = cv2.imread(path)
        h, w = img.shape[:2]
        scale = w / 900.0
        thick = max(2, int(round(scale * 2)))
        # same geometry as the "Counted ~" banner, so the day sits below it
        (_, bh), _ = cv2.getTextSize("Counted ~ 0", font, scale, thick)
        banner_baseline = bh + max(12, h // 40)
        dscale = scale * 0.8
        dthick = max(1, int(round(dscale * 2)))
        (dw, dh), _ = cv2.getTextSize(args.text, font, dscale, dthick)
        org = ((w - dw) // 2, banner_baseline + dh + max(8, h // 80))
        cv2.putText(img, args.text, org, font, dscale, (0, 0, 0),
                    dthick + 4, cv2.LINE_AA)
        cv2.putText(img, args.text, org, font, dscale, (255, 255, 255),
                    dthick, cv2.LINE_AA)
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("stamped", os.path.basename(path))
    print(f"\nstamped {len(paths)} image(s) with '{args.text}'")


if __name__ == "__main__":
    main()
