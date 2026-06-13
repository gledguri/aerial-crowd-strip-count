#!/usr/bin/env python3
"""
Alternative model engine: P2PNet (Tencent YouTu, ICCV'21) — counts people
as individual POINTS instead of density mass.

P2PNet predicts one (x, y) point per person with a confidence score, i.e.
exactly the "count people as units" idea. This script runs it on every
keyframe and writes pipeline-compatible outputs into a SEPARATE counts
dir (default counts_p2p/):

  *_density.npy  - a map with exactly 1.0 at each detected person, so
                   every downstream step (3, 4, 5a, 5b, 6) works
                   unchanged: just point them at counts_p2p/
  *_density.jpg  - the keyframe with one red dot per detected person
                   (your QC: dots must sit on heads, not lights/trees)

Setup (once):
  git clone --depth 1 \
    https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet \
    third_party/p2pnet
  (pretrained SHTechA weights ship inside the repo)

Usage:
  python p2pnet_maps.py keyframes/ [--out counts_p2p] [--upscale 2]
        [--threshold 0.5]

Then e.g.:
  python estimate_by_density.py --area-m2 15000 keyframes/ counts_p2p/
  python estimate_route_total_sliced.py keyframes/ counts_p2p/
  python stitch_route.py keyframes/ counts_p2p/

Same physics caveat as everywhere: if one person is only ~3 px wide,
P2PNet misses people too — compare its dots overlay against the footage
before trusting it.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    os.pardir, "third_party", "p2pnet")


def _patch_repo():
    """Make the 2021 research code run on a current setup (idempotent):
    1. util/misc.py parses torchvision.__version__[:3] as a float, so
       '0.27' reads as 0.2 < 0.7 and imports an API removed years ago.
    2. models/backbone.py loads ImageNet VGG weights from a hardcoded
       Tencent cluster path; unnecessary — the SHTechA checkpoint we load
       afterwards contains every weight."""
    fixes = [
        (os.path.join(REPO, "util", "misc.py"),
         "if float(torchvision.__version__[:3]) < 0.7:",
         "if tuple(int(x) for x in "
         "torchvision.__version__.split('.')[:2]) < (0, 7):"),
        (os.path.join(REPO, "models", "backbone.py"),
         "models.vgg16_bn(pretrained=True)",
         "models.vgg16_bn(pretrained=False)"),
        (os.path.join(REPO, "models", "backbone.py"),
         "models.vgg16(pretrained=True)",
         "models.vgg16(pretrained=False)"),
        # 3. anchor points are pinned to cuda-or-cpu; follow the input
        #    tensor's device instead so Apple MPS works too
        (os.path.join(REPO, "models", "p2pnet.py"),
         "        if torch.cuda.is_available():\n"
         "            return torch.from_numpy("
         "all_anchor_points.astype(np.float32)).cuda()",
         "        if True:\n"
         "            return torch.from_numpy("
         "all_anchor_points.astype(np.float32)).to(image.device)"),
    ]
    for path, old, new in fixes:
        with open(path) as fh:
            src = fh.read()
        fixed = src.replace(old, new)
        if fixed != src:
            with open(path, "w") as fh:
                fh.write(fixed)
            print(f"patched {os.path.basename(path)}: {old[:40]}...")


def load_model():
    if not os.path.isdir(REPO):
        raise SystemExit(
            "third_party/p2pnet not found — clone it first:\n"
            "  git clone --depth 1 "
            "https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet "
            "third_party/p2pnet")
    _patch_repo()
    sys.path.insert(0, REPO)
    import torch
    from models import build_model

    class A:  # the arg fields build_model actually reads
        backbone, row, line = "vgg16_bn", 2, 2

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    model = build_model(A()).to(device)
    ckpt = torch.load(os.path.join(REPO, "weights", "SHTechA.pth"),
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, device


def detect_points(model, device, img_bgr, upscale, threshold):
    """Run P2PNet on one BGR image; return Nx2 array of (x, y) in the
    image's own pixel coordinates."""
    import torch

    h, w = img_bgr.shape[:2]
    up = cv2.resize(img_bgr, None, fx=upscale, fy=upscale,
                    interpolation=cv2.INTER_CUBIC) if upscale > 1 else img_bgr
    uh, uw = up.shape[:2]
    nw, nh = max(uw // 128, 1) * 128, max(uh // 128, 1) * 128
    rs = cv2.resize(up, (nw, nh), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(rs, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    t = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    scores = torch.nn.functional.softmax(out["pred_logits"], -1)[0, :, 1]
    pts = out["pred_points"][0][scores > threshold].cpu().numpy()
    if len(pts) == 0:
        return np.zeros((0, 2))
    pts[:, 0] *= w / nw  # back to original frame coordinates
    pts[:, 1] *= h / nh
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", help="keyframes dir (or glob)")
    ap.add_argument("--out", default="counts_p2p")
    ap.add_argument("--upscale", type=int, default=2,
                    help="upscale before inference; helps small people")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="confidence threshold for a point to count")
    args = ap.parse_args()

    if os.path.isdir(args.images):
        paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")) +
                       glob.glob(os.path.join(args.images, "*.png")))
    else:
        paths = sorted(glob.glob(args.images))
    if not paths:
        raise SystemExit(f"no images found at {args.images}")

    model, device = load_model()
    print(f"P2PNet (SHTechA) on {device.type}, upscale x{args.upscale}, "
          f"threshold {args.threshold}")
    os.makedirs(args.out, exist_ok=True)

    for p in paths:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        pts = detect_points(model, device, img, args.upscale, args.threshold)
        d = np.zeros((h, w), np.float32)
        dots = img.copy()
        for x, y in pts:
            xi = min(max(int(round(x)), 0), w - 1)
            yi = min(max(int(round(y)), 0), h - 1)
            d[yi, xi] += 1.0
            cv2.circle(dots, (xi, yi), 2, (0, 0, 255), -1)
        stem = os.path.splitext(os.path.basename(p))[0]
        np.save(os.path.join(args.out, stem + "_density.npy"), d)
        cv2.imwrite(os.path.join(args.out, stem + "_density.jpg"), dots,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"{os.path.basename(p):35s} {len(pts):6d} points")
    print(f"\nsaved point maps + dot overlays to {args.out}/ — run the "
          f"pipeline steps against that dir to compare engines.")


if __name__ == "__main__":
    main()
