#!/usr/bin/env python3
"""
Internal model engine of the pipeline — not a pipeline step.

Runs a pretrained crowd-density model (DM-Count via the `lwcc` package)
on images and saves, per image, a mass-preserving density map
(*_density.npy) plus a heatmap overlay (*_density.jpg). For dense crowds
seen from the air, density-map models are far more reliable than person
detectors, which fail once people are only a few pixels tall.

estimate_by_density.py (step 3) calls generate_density_maps() automatically
whenever density maps are missing, so you normally never run this file.

Standalone use (QC / experiments):
  python count_crowd.py keyframes/ [--out counts] [--model DM-Count]
        [--weights QNRF] [--upscale 2] [--no-resize]
        [--label-clusters] [--min-cluster 3]

Weights: SHA  = ShanghaiTech A (dense crowds)
         SHB  = ShanghaiTech B (sparser crowds)
         QNRF = UCF-QNRF (very large dense crowds)   <- pipeline default
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


def crowd_mask(d, w):
    """Binary mask of where the density map says 'crowd': threshold at 6%
    of peak, then close gaps so per-person blobs merge into crowd regions
    (kernel scales with image width w)."""
    import cv2

    mask = (d > 0.06 * d.max()).astype("uint8")
    k = (max(5, w // 60)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def label_clusters(overlay, d, min_cluster, tile_above=60.0):
    """Outline each density cluster and stamp it with its people count
    (= density mass inside the blob). Clusters bigger than `tile_above`
    people are subdivided into grid cells, each labeled with its own count,
    so every printed number is small enough to eyeball-check against the
    image. Returns (labeled_mass, n_labeled)."""
    import cv2
    import numpy as np

    h, w = overlay.shape[:2]
    n, labels = cv2.connectedComponents(crowd_mask(d, w))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale = max(0.45, w / 1500.0)
    fthick = max(1, int(round(fscale * 2)))

    def stamp(text, cx, cy, color=(255, 255, 255), scale=fscale):
        thick = max(1, int(round(scale * 2)))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        org = (int(np.clip(cx - tw / 2, 2, w - tw - 2)),
               int(np.clip(cy + th / 2, th + 2, h - 2)))
        cv2.putText(overlay, text, org, font, scale, (0, 0, 0),
                    thick + 3, cv2.LINE_AA)
        cv2.putText(overlay, text, org, font, scale, color, thick,
                    cv2.LINE_AA)

    labeled_mass, n_labeled = 0.0, 0
    for i in range(1, n):
        comp = labels == i
        people = float(d[comp].sum())
        if people < min_cluster:
            continue
        labeled_mass += people
        n_labeled += 1
        contours, _ = cv2.findContours(comp.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
        ys, xs = np.nonzero(comp)
        if people <= tile_above:
            stamp(f"{people:.0f}", xs.mean(), ys.mean())
            continue
        # big cluster: grid cells with per-cell counts, total above the box
        t = max(48, w // 8)
        y0, x0 = int(ys.min()), int(xs.min())
        for ty in range(y0, int(ys.max()) + 1, t):
            for tx in range(x0, int(xs.max()) + 1, t):
                sub = comp[ty:ty + t, tx:tx + t]
                if not sub.any():
                    continue
                cell = float(d[ty:ty + t, tx:tx + t][sub].sum())
                if cell < min_cluster:
                    continue
                cv2.rectangle(overlay, (tx, ty),
                              (min(tx + t, w - 1), min(ty + t, h - 1)),
                              (180, 180, 180), 1)
                cys, cxs = np.nonzero(sub)
                stamp(f"{cell:.0f}", tx + cxs.mean(), ty + cys.mean(),
                      scale=fscale * 0.8)
        stamp(f"cluster: {people:.0f}", xs.mean(), y0 - 14,
              color=(0, 255, 255))
    return labeled_mass, n_labeled


def save_overlay(img_path, density, out_path, count=None, clusters=False,
                 min_cluster=3.0, label=None):
    import cv2
    import numpy as np

    img = cv2.imread(img_path)
    d = density.astype("float32")
    mass = float(d.sum())
    d = cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    d = np.clip(d, 0, None)
    if d.sum() > 0:
        d *= mass / d.sum()  # mass-preserving: blob sums stay people counts
    if d.max() > 0:
        d = d / d.max()
    heat = cv2.applyColorMap((d * 255).astype("uint8"), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
    if label is None and count is not None:
        label = f"~{count:,.0f} people"
    if clusters and mass > 0:
        d_people = d / d.sum() * mass if d.sum() > 0 else d
        labeled, n = label_clusters(overlay, d_people, min_cluster)
        if label:
            label += f" | {labeled:,.0f} in {n} clusters"
    if label:
        h, w = overlay.shape[:2]
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, w / 900.0
        thick = max(2, int(round(scale * 2)))
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        org = ((w - tw) // 2, th + max(12, h // 40))
        cv2.putText(overlay, label, org, font, scale, (0, 0, 0),
                    thick + 4, cv2.LINE_AA)
        cv2.putText(overlay, label, org, font, scale, (255, 255, 255),
                    thick, cv2.LINE_AA)
    cv2.imwrite(out_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])


def generate_density_maps(paths, out_dir, model_name="DM-Count",
                          weights="QNRF", upscale=2, no_resize=None,
                          overlay_counts=False, label_clusters_qc=False,
                          min_cluster=3.0):
    """Run the density model on every image in `paths`; write per image a
    mass-preserving *_density.npy (at original image size) and a heatmap
    *_density.jpg into out_dir. The overlay carries no numbers unless
    overlay_counts/label_clusters_qc — step 6 stamps the real (calibrated)
    ones. Returns [(image_name, model_count), ...]."""
    import cv2
    import numpy as np

    if no_resize is None:
        no_resize = upscale > 1
    _fix_lwcc_weights_path()
    from lwcc import LWCC

    os.makedirs(out_dir, exist_ok=True)
    model = LWCC.load_model(model_name=model_name, model_weights=weights)

    tmpdir = None
    if upscale > 1:
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="upscaled_")

    rows = []
    for p in paths:
        infer_path = p
        if tmpdir:
            img = cv2.imread(p)
            img = cv2.resize(img, None, fx=upscale, fy=upscale,
                             interpolation=cv2.INTER_CUBIC)
            stem_up = os.path.splitext(os.path.basename(p))[0]
            infer_path = os.path.join(tmpdir, stem_up + ".png")  # lossless
            cv2.imwrite(infer_path, img)
        count, density = LWCC.get_count(
            infer_path, model=model, return_density=True,
            resize_img=not no_resize)
        name = os.path.basename(p)
        stem = re.sub(r"\.(jpg|png)$", "", name)
        save_overlay(p, density, os.path.join(out_dir, stem + "_density.jpg"),
                     count=float(count) if overlay_counts else None,
                     clusters=label_clusters_qc, min_cluster=min_cluster)
        orig = cv2.imread(p)
        h, w = orig.shape[:2]
        d = density.astype("float32")
        mass = d.sum()
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        if d.sum() > 0:
            d *= mass / d.sum()  # keep total density mass unchanged
        np.save(os.path.join(out_dir, stem + "_density.npy"), d)
        rows.append((name, round(float(count))))
        print(f"{name:35s} {count:9.0f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", help="image file, directory, or glob")
    ap.add_argument("--out", default="counts")
    ap.add_argument("--model", default="DM-Count",
                    choices=["DM-Count", "CSRNet", "SFANet", "Bay"])
    ap.add_argument("--weights", default="QNRF",
                    choices=["SHA", "SHB", "QNRF"])
    ap.add_argument("--no-resize", action="store_true",
                    help="skip downscaling to 1000px (use for very dense, "
                         "high-res originals)")
    ap.add_argument("--upscale", type=int, default=2,
                    help="upscale factor before inference; 2 helps recover "
                         "small far-away people in low-res footage "
                         "(implies --no-resize)")
    ap.add_argument("--label-clusters", action="store_true",
                    help="outline each density cluster on the overlay and "
                         "stamp it with its people count, for visual QC")
    ap.add_argument("--min-cluster", type=float, default=3.0,
                    help="only label clusters of at least this many people")
    args = ap.parse_args()

    if os.path.isdir(args.images):
        paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")) +
                       glob.glob(os.path.join(args.images, "*.png")))
    else:
        paths = sorted(glob.glob(args.images))
    if not paths:
        raise SystemExit(f"no images found at {args.images}")

    rows = generate_density_maps(
        paths, args.out, args.model, args.weights, args.upscale,
        args.no_resize or args.upscale > 1, overlay_counts=True,
        label_clusters_qc=args.label_clusters, min_cluster=args.min_cluster)

    csv_path = os.path.join(args.out, f"counts_{args.model}_{args.weights}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "count"])
        w.writerows(rows)

    counts = [c for _, c in rows]
    print("-" * 47)
    print(f"frames: {len(counts)}  min: {min(counts)}  "
          f"median: {sorted(counts)[len(counts)//2]}  max: {max(counts)}")
    print("NOTE: frames overlap — do NOT sum frames blindly. The pipeline "
          "(steps 5a/5b) does this correctly.")
    print("saved:", csv_path)


if __name__ == "__main__":
    main()
