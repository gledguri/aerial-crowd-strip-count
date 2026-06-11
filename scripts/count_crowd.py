#!/usr/bin/env python3
"""
Step 2 of the crowd-counting pipeline: estimate people per keyframe with a
pretrained crowd-density model (via the `lwcc` package).

For dense crowds seen from the air, density-map models (DM-Count, CSRNet,
SFANet trained on ShanghaiTech/UCF-QNRF) are far more reliable than person
detectors, which fail once people are only a few pixels tall.

Outputs, per image:
  - predicted count
  - a density-heatmap overlay JPG (so you can verify the model is firing on
    the crowd and not on trees/lights)
  - one CSV with all counts

Usage:
  python count_crowd.py keyframes/ [--out counts] [--model DM-Count]
                        [--weights SHA] [--no-resize]

Weights: SHA  = ShanghaiTech A (dense crowds)      <- default
         SHB  = ShanghaiTech B (sparser crowds)
         QNRF = UCF-QNRF (very large dense crowds; good cross-check)
"""
import argparse
import csv
import glob
import os
import re


def _fix_lwcc_weights_path():
    """lwcc 0.0.x bug: os.path.join(home, '/.lwcc/...') drops `home` and
    points at the filesystem root. Rewrite the installed file in place
    (idempotent) so weights land in ~/.lwcc/weights."""
    import lwcc.util.functions as f

    path = f.__file__
    with open(path) as fh:
        src = fh.read()
    fixed = src.replace('Path("/.lwcc/weights")',
                        'Path(os.path.join(str(Path.home()), ".lwcc", "weights"))')
    fixed = fixed.replace('os.path.join(home, "/.lwcc/weights/", file_name)',
                          'os.path.join(home, ".lwcc", "weights", file_name)')
    if fixed != src:
        with open(path, "w") as fh:
            fh.write(fixed)
        print("patched lwcc weights path ->", path)
        import importlib
        importlib.reload(f)


def save_overlay(img_path, density, out_path):
    import cv2
    import numpy as np

    img = cv2.imread(img_path)
    d = density.astype("float32")
    d = cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    if d.max() > 0:
        d = d / d.max()
    heat = cv2.applyColorMap((d * 255).astype("uint8"), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
    cv2.imwrite(out_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", help="image file, directory, or glob")
    ap.add_argument("--out", default="counts")
    ap.add_argument("--model", default="DM-Count",
                    choices=["DM-Count", "CSRNet", "SFANet", "Bay"])
    ap.add_argument("--weights", default="SHA", choices=["SHA", "SHB", "QNRF"])
    ap.add_argument("--no-resize", action="store_true",
                    help="skip downscaling to 1000px (use for very dense, "
                         "high-res originals)")
    ap.add_argument("--upscale", type=int, default=1,
                    help="upscale factor before inference; 2 helps recover "
                         "small far-away people in low-res footage "
                         "(implies --no-resize)")
    ap.add_argument("--save-density", action="store_true",
                    help="save per-frame density maps as .npy (rescaled to "
                         "original frame size, mass-preserving) for "
                         "downstream route-total estimation")
    args = ap.parse_args()
    if args.upscale > 1:
        args.no_resize = True

    if os.path.isdir(args.images):
        paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")) +
                       glob.glob(os.path.join(args.images, "*.png")))
    else:
        paths = sorted(glob.glob(args.images))
    if not paths:
        raise SystemExit(f"no images found at {args.images}")

    _fix_lwcc_weights_path()
    from lwcc import LWCC

    os.makedirs(args.out, exist_ok=True)
    model = LWCC.load_model(model_name=args.model, model_weights=args.weights)

    rows = []
    tmpdir = None
    if args.upscale > 1:
        import tempfile
        import cv2
        tmpdir = tempfile.mkdtemp(prefix="upscaled_")

    for p in paths:
        infer_path = p
        if tmpdir:
            img = cv2.imread(p)
            img = cv2.resize(img, None, fx=args.upscale, fy=args.upscale,
                             interpolation=cv2.INTER_CUBIC)
            infer_path = os.path.join(tmpdir, os.path.basename(p))
            cv2.imwrite(infer_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count, density = LWCC.get_count(
            infer_path, model=model, return_density=True,
            resize_img=not args.no_resize)
        name = os.path.basename(p)
        stem = re.sub(r"\.(jpg|png)$", "", name)
        save_overlay(p, density, os.path.join(args.out, stem + "_density.jpg"))
        if args.save_density:
            import cv2
            import numpy as np
            orig = cv2.imread(p)
            h, w = orig.shape[:2]
            d = density.astype("float32")
            mass = d.sum()
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
            if d.sum() > 0:
                d *= mass / d.sum()  # keep total count unchanged
            np.save(os.path.join(args.out, stem + "_density.npy"), d)
        rows.append((name, round(float(count))))
        print(f"{name:35s} {count:9.0f}")

    csv_path = os.path.join(args.out, f"counts_{args.model}_{args.weights}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "count"])
        w.writerows(rows)

    counts = [c for _, c in rows]
    print("-" * 47)
    print(f"frames: {len(counts)}  min: {min(counts)}  "
          f"median: {sorted(counts)[len(counts)//2]}  max: {max(counts)}")
    print("NOTE: frames overlap — do NOT sum frames blindly. Sum only "
          "frames chosen to tile the street without overlap.")
    print("saved:", csv_path)


if __name__ == "__main__":
    main()
