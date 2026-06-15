# Counting protest crowds from aerial video

A pipeline that estimates how many people are on a street from a video taken
by a drone flying along the route. It works on two kinds of input:

- **A. An original drone video** (file straight from the drone / exported
  from its app) — best case.
- **B. A screen-recorded Instagram reel / TikTok / story** of drone footage —
  works, but expect a rougher, low-biased estimate (compression destroys
  far-away people; the pipeline auto-crops the app UI away).

The method in one paragraph: person-detectors (YOLO-style) fail on aerial
crowds because each person is only a few pixels, so we use a crowd-density
model (DM-Count, via the `lwcc` package) that predicts a density map whose
sum is the people count. The model's absolute numbers run LOW on bad
footage, but its *map* of where the crowd is stays good — so the pipeline
splits the problem: first anchor the crowd's geometry in real-world units
with one measurement you make yourself (step 3) and turn it into an
area-x-density estimate (step 4, the Jacobs method); independently compute
the model's own route totals by tracking how far the ground slides between
overlapping frames and averaging every reading of the same ground (step
5); then calibrate the model totals against the Jacobs total — a measured
factor, not a guess — and produce the final report and visuals (step 6).
On high-resolution footage an alternative point-based engine (P2PNet, one
detected point per person) can drive the same pipeline — see "Alternative
engine" near the end.

Pipeline at a glance:

| step | script | what it does | key outputs |
|---|---|---|---|
| 1 | `extract_frames.py` | video → cropped keyframes | `keyframes/*.jpg` |
| 3 | `estimate_by_density.py` | crowd geometry + people/m², anchored on YOUR area measurement (estimates density maps automatically) | `route_area.jpg`, `jacobs_segments.csv`, `density_report.txt` |
| 4 | `jacobs_estimate.py` | area x density total from the strips you assessed | printed Jacobs total |
| 5a | `estimate_route_total_sliced.py` | model route total, slice averaging (1-D) | `route_slices.csv`, `route_total_sliced.txt` |
| 5b | `stitch_route.py` | model route total, mosaic averaging (2-D) + the whole flight stitched into one picture | `counts_mosaic/route_mosaic*.jpg`, `route_total_mosaic.txt` |
| 6 | `report_route.py` | calibrate step 5 against step 4, re-stamp overlays with a climbing counter, final range | `report.txt`, re-stamped `*_density.jpg` |
| alt | `p2pnet_maps.py` | optional second engine: P2PNet, one point per person (see "Alternative engine" below) | `counts_p2p/*_density.npy/.jpg` |

---

## 0. Setup (once)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Model weights (~86 MB) download automatically to `~/.lwcc/weights/` the
first time the density model runs (inside step 3). A known path bug in
the `lwcc` package is auto-patched on that first run — the "patched lwcc
weights path" message is normal.

## 1. Extract keyframes

Put the video you want to analyze in the **`input_video/`** folder. If that
folder holds exactly one video, the scripts find it on their own; if it
holds several, pass the filename as the first argument.

```bash
# B. screen-recorded reel (default: auto-crops phone/app UI, detects cuts):
.venv/bin/python scripts/extract_frames.py --out keyframes --fps 1.5

# A. original drone file (no UI to crop):
.venv/bin/python scripts/extract_frames.py --out keyframes --fps 1.5 --no-autocrop

# several videos in input_video/ — name the one you mean:
.venv/bin/python scripts/extract_frames.py YOUR_VIDEO.mov --out keyframes --fps 1.5
```

- It prints the detected video band `(x, y, w, h)` and the scene list.
  **Open 2-3 of the saved keyframes and check** the crop contains only video,
  no app buttons/text.
- `--fps` is keyframes per second of video. 1.5 is fine for slow flights;
  use 2-3 for fast or sped-up footage.
- Keyframes are saved as JPEG q95 (near-lossless); add `--png` to save
  lossless PNGs instead — every downstream step accepts both. Understand
  what this can and cannot do: it removes the pipeline's only
  re-compression, but **keyframes can never be better than the source
  video**. Check the printed band size — if the cropped video is only
  ~600 px wide, people are a few pixels tall and the estimate will run
  low no matter the format. The fix for that is a better source file
  (original export, higher resolution — see the filming checklist), not
  extraction settings.
- If the video has hard cuts (different shots stitched together), frames are
  named `scene00_*, scene01_*, ...`. Treat each scene as a separate flight:
  put each scene's frames in their own folder and run steps 3-6 per scene —
  all the route math assumes one continuous pass.

*(There is no step 2 — the density model runs automatically inside
step 3. `count_crowd.py` still exists as the internal model engine; run
it standalone only for QC experiments, e.g. `--label-clusters` overlays
with per-cluster people counts.)*

## 3. Crowd geometry & density (people/m²)

Anchors the pipeline in physical units. The script estimates a **density
map** for every keyframe (people per area, predicted directly by the
model — a few seconds per frame on CPU the first time, cached in
`counts/` afterwards), stitches the maps onto the route, and converts to
real units with ONE measurement you make yourself on Google Maps/Earth
("Measure distance") or https://www.mapchecking.com — preferably the
occupied **area in m²**. The pixel→meter scale is calibrated so the crowd
outline matches your measurement:

```bash
.venv/bin/python scripts/estimate_by_density.py --area-m2 14000 \
      --route-length-m 700 keyframes/ counts/
```

- Best: give **both** `--area-m2` and `--route-length-m` (the ground
  length covered by the mosaic, bottom to top edge). Oblique drone footage
  has different ground resolution along vs across the flight, so the
  length anchors the along-route scale and the area the across-route
  scale — strip lengths and street widths then match reality.
- With only one of the two, an isotropic scale is derived from it (fine
  for near-nadir footage, distorted geometry for oblique footage).
- `--segment-m 50` sets the strip length for the per-strip breakdown.

Model flags (only matter the first time, when the density maps are
computed):

| footage | recommended flags |
|---|---|
| low-res / night / reel (B) | default (= `--weights QNRF --upscale 2`) |
| 4K original, dense crowd (A) | `--weights QNRF --upscale 1` (try 2 as well; keep whichever heatmap looks cleaner) |
| sparse crowd, people clearly separated | `--weights SHA` or `SHB` |

**Outputs in `counts/`:**

- `route_area.jpg` — the whole flight stitched into one picture, with the
  crowd area outlined in green. **Check this first: the outline must hug
  the crowd**, because the area drives everything downstream.
- `density_report.txt` — the geometry (scale, implied route length, area,
  and **how many pixels wide one person is** at this scale) and the
  **density estimate**: if the density is roughly uniform along the route
  (spread ≤ 30%), one single people/m² figure; if it varies, a per-strip
  table (each strip with its length, width, area and density). A second,
  independent density estimate counts people as *units* (blob peaks in
  the density map, sized by the person's pixel footprint) — it is
  reported with an automatic validity check: it needs one person to span
  ≥ ~5 px; below that (low-res source) it under-reads and says so, and
  the absolute density must come from your eyes in step 4. Reference
  totals at the standard density classes are listed for orientation.
- `jacobs_segments.csv` — every strip's measured geometry plus an empty
  `assumed_density_p_m2` column. **This is the input of step 4.**
- `*_density.npy` + `*_density.jpg` — the per-keyframe density maps and
  their heatmap overlays (no numbers on them; step 6 stamps the
  calibrated climbing counter). The heatmaps are your QC: color must sit
  on the crowd, not on trees, streetlights or parked cars.

Note on the densities: the model's absolute people/m² is a **lower
bound** on night/compressed footage; the per-strip *variation* is the
trustworthy part. Use your eyes for the absolute level — that's step 4.

## 4. Jacobs total (area x density)

The standard method researchers and press use to size protests. Look at
the footage strip by strip, decide which density class each strip
*visibly* is, and put it in the `assumed_density_p_m2` column of
`jacobs_segments.csv` (use 0 for genuinely empty stretches):

| class | people/m² | what it looks like |
|---|---|---|
| loose | 0.5 | people walking freely, big gaps |
| moderate | 1.0 | steady walking crowd |
| dense | 2.0 | slow shuffle, shoulders close |
| packed | 4.0 | barely moving; rare outdoors |

```bash
.venv/bin/python scripts/jacobs_estimate.py counts/jacobs_segments.csv
```

It prints the per-strip people counts, the **JACOBS TOTAL**, and the
honest x0.7..x1.3 range. For parts of the event the drone never filmed
(side streets, the stretch behind the take-off point), edit the manual
`SEGMENTS` list in the script and run it without arguments, then add the
two totals.

## 5. Model route totals (5a slice averaging, 5b mosaic averaging)

Both methods make the model count each person exactly once despite the
overlapping frames: the tracked ground motion places every keyframe on a
shared route axis, so each piece of ground is observed in ~8-15 frames,
and all its readings are **averaged**. With three keyframes sliding over
route slices A-E:

```
keyframe 1: [100, 180, 350]           (sees slices A, B, C)
keyframe 2:      [220, 390, 470]      (sees slices B, C, D)
keyframe 3:           [240, 570, 610] (sees slices C, D, E)
total = 100 + (180+220)/2 + (350+390+240)/3 + (470+570)/2 + 610
```

Only rows in the bottom `--use-frac` of each frame enter the average (the
model under-counts the far field; averaging it in would drag the total
down), and the final stretch ahead of the drone falls back to its single
nearest reading. 5a does this per route slice (1-D); 5b aligns the
overlaps per pixel in 2-D, so the drone's sideways drift no longer smears
the average, and also outputs the stitched flight as one picture:

```bash
.venv/bin/python scripts/estimate_route_total_sliced.py keyframes/ counts/
.venv/bin/python scripts/stitch_route.py keyframes/ counts/
# --use-frac 0.45   rows (from the bottom) entering the average
# --calibrate F     apply a measured calibration factor (step 6 does this
#                   for you; only use directly if you skip step 6)
# stitch_route.py also takes --label-clusters for a QC mosaic with
# per-cluster/cell counts
```

**Outputs:** 5a writes into `counts/`: `route_total_sliced.txt` and
`route_slices.csv` (the latter has a `cumulative_people_calibrated`
column = the running total multiplied by `--calibrate F`). 5b writes into
`counts_mosaic/` (override with `--out`): `route_total_mosaic.txt`,
`route_mosaic.jpg`, and `route_mosaic_density.jpg`, whose banner shows the
counted total `Counted ~ N` (calibrated when `--calibrate` is given).

Sanity checks:

- The two totals should agree within a few %. If not, the motion track is
  suspect — open `counts_mosaic/route_mosaic.jpg`, misalignment is obvious
  to the eye.
- "cumulative ground advance" should be several frame-heights for a
  flight along a long street. If it's ~0, motion tracking failed
  (hovering drone, footage too dark) — fall back to picking
  non-overlapping keyframes by hand and summing their model counts
  (run `count_crowd.py` standalone to get a per-frame CSV).
- Advances should grow/shrink smoothly; wild sign flips mean the drone
  yawed/reversed — trim the keyframes to the clean forward stretch.
- Whichever method you use, anyone behind the take-off point, beyond the
  last frame, or under tree canopies is **not counted**.
- (Legacy: `estimate_route_total.py`, strip summing, still exists. It
  counts each person from a single reading at the clipped bottom edge and
  runs ~30% low — only useful as a lower-bound cross-check.)

## 6. Report (calibrated totals + climbing counter)

The model totals (step 5) are precise about *relative* structure but low
in absolute level; the Jacobs total (step 4) is anchored in physical
reality. Step 6 bridges them with a **measured** calibration factor

    F = Jacobs total / mean(5a, 5b)

— this is the honest version of "the model runs ~2-4x low": derived from
your area measurement and your density assessment, not invented.

```bash
.venv/bin/python scripts/report_route.py --jacobs-total 7500 keyframes/ counts/
# or apply a factor you already know directly:  --calibrate 2.06
```

**What it does:**

- writes `counts/report.txt`: both model totals, the Jacobs total, F, the
  calibrated totals, and the **final estimate with the honest range to
  quote**;
- re-stamps every `counts/scene*_density.jpg` with `Counted ~ N`, the
  **accumulated calibrated count up to that frame** — frame 1 shows
  everyone visible at the start, each next frame adds only the newly
  revealed ground, the last frame shows the final number. Rebuild the gif
  (Extras) and you get a counter that climbs as the drone flies.

Reporting guidance: quote the **range, not a number**. For multiple
videos/passes of the same event, run the pipeline per pass and report the
median and spread; note the time of each pass — crowds turn over during
an event.

Worked example (this repo's `dita_10.mp4`, night reel, 45 keyframes,
area measured at 15,000 m²): model totals 3,631 (5a) / 3,657 (5b); Jacobs
at a conservative 0.5 p/m² everywhere = 7,500 → F = x2.06 → final
estimate ~7,500, quoted range ~5,200..9,800. A denser (and visually
defensible) 1.0 p/m² would double the Jacobs side — which is exactly why
the density class assessment, not the model, deserves your attention.

![Output](output/side_by_side.gif)

---

## Alternative engine: P2PNet (people as points)

[P2PNet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet)
(Tencent YouTu, ICCV'21) predicts one **(x, y) point per person** with a
confidence score instead of a density map — the "count people as units"
approach. Setup (once):

```bash
git clone --depth 1 \
  https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet \
  third_party/p2pnet
```

(pretrained SHTechA weights ship inside the repo; the adapter auto-patches
three small incompatibilities with current PyTorch on first run)

```bash
.venv/bin/python scripts/p2pnet_maps.py keyframes/ --out counts_p2p \
      --upscale 2 --threshold 0.35
```

It writes pipeline-compatible `*_density.npy` maps (exactly 1.0 per
detected person) and `*_density.jpg` dot overlays into `counts_p2p/` —
**check the dots sit on heads**. Every pipeline step then works against
that folder unchanged, e.g.:

```bash
.venv/bin/python scripts/stitch_route.py keyframes/ counts_p2p/
```

Reality check: point localization needs *more* pixels per person than
density regression, not fewer. On low-res night footage (person ≈ 3 px)
P2PNet finds ~2-3x fewer people than DM-Count — use it as a second
opinion there, not as the main engine. On footage where one person is
≥ 5-8 px it becomes the better engine: unit counts, directly usable
densities, and dots you can verify by eye.

## Filming checklist (for whoever flies the drone)

1. Export the **original file** — never screen-record an app. Every
   re-compression erases the far-field crowd.
2. **Daylight or dusk** beats night by a wide margin.
3. **One slow, steady, continuous pass** over the whole route (2-5 m/s),
   no cuts, no yaw spins, constant altitude.
4. Camera **pointed steeply down** (nadir or close to it) — oblique views
   hide people behind people and under trees.
5. Altitude: high enough to see building-to-building across the street, low
   enough that a person is at least ~8-10 px tall in the frame (with 4K,
   roughly 60-100 m works).
6. Fly at the **peak moment**, and note the time.
7. If possible, fly **two or three passes** — independent estimates.

## Troubleshooting

- `lwcc` errors about `/.lwcc/weights` on a fresh machine: run step 3
  once; its first model run patches the package in place.
- Keyframes contain app UI: re-run step 1 and check the printed band; you
  can also crop manually with ffmpeg and use `--no-autocrop`.
- Keyframes look blurry / low quality: it's the source video, not the
  extraction. Compare file sizes and resolutions of the videos you have
  (e.g. `ffprobe` or just QuickTime's inspector) and run the pipeline on
  the largest/highest-resolution version of the SAME flight; `--png`
  (step 1) removes the last bit of re-compression but cannot add detail.
- Heatmaps light up only the nearest rows of people: increase `--upscale`
  (step 3); if they light up trees/lights instead, decrease it. The
  density maps are cached — **delete `counts/*_density.npy` first**, or
  changed model flags will silently have no effect.
- "I re-ran it and nothing got produced": if the inputs didn't change, a
  re-run reproduces byte-identical outputs — check the file timestamps in
  `counts/`. Also note the gif is NOT one of the script outputs: it only
  changes when you rebuild it yourself (see Extras), so after any re-run
  it shows the previous results until you do. Finder/Quick Look
  thumbnails can lag too — open the `.jpg` files themselves.
- GPU is not required; CPU inference takes a few seconds per frame.

## Extras
### Stamping a day label under the count

Adds a manual label (e.g. `Day 10`) centered just below the `Counted ~ N`
banner on every `*density.jpg`. Run it **once, after step 6** (step 6
re-writes the frames fresh; running this twice would stamp twice). Set
`DAY` and the target glob, then run with `.venv/bin/python`:

```python
import cv2, glob, os

DAY = "Day 10"                      # <- set this manually
IMAGES = "counts/*density.jpg"      # which images to stamp

for path in sorted(glob.glob(IMAGES)):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = w / 900.0
    thick = max(2, int(round(scale * 2)))
    # same geometry as the "Counted ~" banner, so the day sits just below it
    (_, bh), _ = cv2.getTextSize("Counted ~ 0", font, scale, thick)
    banner_baseline = bh + max(12, h // 40)
    dscale = scale * 0.8
    dthick = max(1, int(round(dscale * 2)))
    (dw, dh), _ = cv2.getTextSize(DAY, font, dscale, dthick)
    org = ((w - dw) // 2, banner_baseline + dh + max(8, h // 80))
    cv2.putText(img, DAY, org, font, dscale, (0, 0, 0), dthick + 4, cv2.LINE_AA)
    cv2.putText(img, DAY, org, font, dscale, (255, 255, 255), dthick, cv2.LINE_AA)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("stamped", os.path.basename(path))
```

Rebuild the gif afterwards (below) to carry the label into the animation.
To also stamp the stitched mosaic, set `IMAGES = "counts_mosaic/*density.jpg"`.

### Converting output images to .gif

Run this **from inside `counts/`** (the gif is written to the folder you
are standing in), and re-run it after every step-3 or step-6 run — it
does not update itself:

```bash
brew install imagemagick
cd counts/
magick scene*_density.jpg -delay 10 -loop 0 output.gif
```
(`scene*` keeps the route mosaic images out of the animation.)

Converting output to video
```bash
brew install ffmpeg
cd counts/

ffmpeg -i output.gif \
-vf "scale=1206:718" \
-pix_fmt yuv420p \
density_video.mp4
```
