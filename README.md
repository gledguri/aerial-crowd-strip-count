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
sum is the people count. Because the drone flies along the street, every
frame overlaps the next one, so summing frames would count people many
times over. Instead the pipeline tracks how far the ground slides between
frames (forward advance + sideways drift) and uses that alignment to count
each person exactly once. Three estimators share that alignment: **strip
summing** (each person counted once, from the single frame where they exit
the bottom edge), **slice averaging** (each piece of ground counted as the
average of the ~8-15 frames that saw it — recommended), and the **route
mosaic** (slice averaging done in 2-D, plus the whole flight stitched into
one picture). An independent area-x-density check (Jacobs method) is
included to validate whatever number the model gives you.

Pipeline at a glance:

| step | script | what it does | key outputs |
|---|---|---|---|
| 1 | `extract_frames.py` | video → cropped keyframes | `keyframes/*.jpg` |
| 2 | `count_crowd.py` | density model per keyframe | `counts/*.csv`, `*_density.jpg/.npy` |
| 3 | `estimate_route_total.py` | route total, strip summing (single reading per person) | `route_strips.csv`, `route_total.txt` |
| 3b | `estimate_route_total_sliced.py` | route total, slice averaging — **recommended** | `route_slices.csv`, `route_total_sliced.txt` |
| 3c | `stitch_route.py` | 2-D averaging + whole-flight mosaic | `route_mosaic*.jpg`, `route_total_mosaic.txt` |
| 4 | `jacobs_estimate.py` | independent area x density cross-check | printed estimate |
| 5 | `estimate_by_density.py` | people/m² x measured route length → measured calibration factor | `route_area.jpg`, `density_report.txt` |

---

## 0. Setup (once)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Model weights (~86 MB) download automatically to `~/.lwcc/weights/` on first
run. `count_crowd.py` auto-patches a known path bug in the `lwcc` package on
first run — the "patched lwcc weights path" message is normal.

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
  put each scene's frames in their own folder and run steps 2-3 per scene —
  all the step-3 route math assumes one continuous pass.

## 2. Count people per keyframe

```bash
.venv/bin/python scripts/count_crowd.py keyframes/ --out counts \
      --weights QNRF --upscale 2 --save-density
# add --label-clusters for QC overlays with per-cluster/cell counts
```

Recipe guidance:

| footage | recommended flags |
|---|---|
| low-res / night / reel (B) | `--weights QNRF --upscale 2` |
| 4K original, dense crowd (A) | `--weights QNRF --no-resize` (try `--upscale 2` too, keep the higher one if its overlay looks cleaner) |
| sparse crowd, people clearly separated | `--weights SHA`, or `SHB` |

**Outputs in `counts/`:**

- `counts_DM-Count_QNRF.csv` — **the per-keyframe people counts** (how many
  the model sees in that whole frame).
- `*_density.jpg` — heatmap overlays, each stamped top-center with that
  frame's estimated count. With `--label-clusters`, every density cluster
  is outlined and stamped with its own people count (= the density mass
  inside the blob, so the cluster numbers genuinely decompose the frame
  total); clusters above ~60 people are subdivided into grid cells with
  per-cell counts, small enough to verify by eye (`--min-cluster N` hides
  labels below N people, default 3). The banner also shows how much of the
  frame total the labels cover, e.g. `~978 people | 957 in 5 clusters` —
  the gap is sub-threshold haze. **This is your quality control: look at
  several.** Color must sit on the crowd, not on trees, streetlights,
  parked cars, or app icons. Hand-count one or two near-field cells
  against their printed number — if the model runs low there, the whole
  total runs at least that much low. If the far half of the street shows
  crowd but no color, the model is missing it — try `--upscale 2` (or 3 on
  4K input). Note: the cluster/grid labels are for *your eyes only* — the
  route math in step 3 works on the raw density maps, not on the squares.
- `*_density.npy` — raw density maps (needed by step 3; written only with
  `--save-density`, so don't drop that flag).

Remember: keyframes overlap heavily — **never sum this CSV across frames**.
That's what step 3 is for.

![Output](counts/output.gif)

## 3. Whole-route total (strip summing) — kept as a cross-check

Requires: one continuous pass along the street, and step 2 run with
`--save-density`.

How it counts: everyone the drone overflies exits through the bottom edge
of the frame exactly once, so each interval's total is the density mass in
the strip of ground that exited since the previous keyframe. Each person
is counted **once, from a single reading**, taken right at the bottom edge
where bodies are clipped — so this method runs systematically low (~30%
below 3b/3c in practice) and motion-estimate errors accumulate as strip
overlaps or gaps. Prefer 3b/3c for the headline number and use this one as
a lower-bound sanity check.

```bash
.venv/bin/python scripts/estimate_route_total.py keyframes/ counts/
# add --no-autocrop for an original drone file (A)
# (uses the video in input_video/; with several videos there, put the
#  filename first: estimate_route_total.py YOUR_VIDEO.mov keyframes/ counts/)
```

**Outputs in `counts/`:**

- `route_strips.csv` — per keyframe: ground advance (px), **people counted
  in the strip that exited the frame**, and the running cumulative total.
- `route_total.txt` — the bottom line: people who exited during the flight,
  people still visible in the last keyframe, and **ESTIMATED TOTAL ALONG THE
  ROUTE** (= the number you're after).

Sanity checks before trusting it:

- "cumulative ground advance" should be several frame-heights for a flight
  along a long street. If it's ~0, motion tracking failed (hovering drone,
  or footage too dark) — fall back to picking non-overlapping keyframes by
  hand and summing their CSV counts.
- Advances should grow/shrink smoothly. Wild sign flips mean the drone
  yawed/reversed; trim the keyframes to the clean forward stretch.
- Whichever step-3 variant you run, anyone behind the take-off point,
  beyond the last frame, or under tree canopies is **not counted**.

## 3b. Whole-route total (slice averaging) — recommended

Same inputs as step 3, different math: every keyframe is mapped onto a
shared route axis using the ground advance, so each physical slice of
street is observed in many keyframes. Per slice, all near-field
observations are averaged, then the averaged profile is summed over the
route. With three keyframes sliding over route slices A-E, where the model
reads the same slice slightly differently each time:

```
keyframe 1: [100, 180, 350]            (sees slices A, B, C)
keyframe 2:      [220, 390, 470]      (sees slices B, C, D)
keyframe 3:           [240, 570, 610] (sees slices C, D, E)
total = 100 + (180+220)/2 + (350+390+240)/3 + (470+570)/2 + 610
```

(Internally the "slices" are 1 px deep, so this happens per pixel row.)
Compared to strip summing, each slice gets ~8-15 independent readings
instead of one, motion-estimate noise blurs instead of double-counting,
and nothing rests on the frame's bottom edge (where people are clipped
and counts run low). Two deliberate choices to know about:

- Only rows in the bottom `--use-frac` of each frame enter the average,
  because the model systematically under-counts the far field — averaging
  it in would drag the total down, not sharpen it.
- The final stretch of route (ahead of the drone in the last keyframe) is
  never seen up close; it falls back to its single nearest reading and is
  reported separately in the summary.

```bash
.venv/bin/python scripts/estimate_route_total_sliced.py keyframes/ counts/
# --use-frac 0.45  rows (from the bottom) entering the average; 1.0 = whole
#                  frame, lower = trust only the nearest, sharpest rows
# --slice-px 60    slice size in route_slices.csv (reporting only)
# --calibrate F    multiply the total by a MEASURED factor (see step 5)
```

**Outputs in `counts/`:** `route_slices.csv` (per slice of route: mean
observation count, averaged people, cumulative) and
`route_total_sliced.txt` (the bottom line). The same sanity checks as in
step 3 apply. Expect this total to land ~30-45% **above** strip summing
(that gap is the averaging recovering people, see 3d) and within a few %
of the mosaic total (3c).

## 3c. Route mosaic — see the whole flight in one picture

Stitches every keyframe into one route-long image using the tracked ground
motion (vertical advance + lateral drift), and aligns the density maps the
same way: wherever several frames overlap the same ground, their readings
are **averaged per pixel**. This is the 2-D version of step 3b — same
math, but sideways drift no longer smears the average — and unlike the
per-frame overlays, the cluster/grid labels drawn on the mosaic decompose
the actual route total used in the calculation.

```bash
.venv/bin/python scripts/stitch_route.py keyframes/ counts/
# --label-clusters  draw per-cluster/cell QC counts (default: banner only)
# --calibrate F     multiply the total by a MEASURED factor (see step 5)
```

**Outputs in `counts/`:** `route_mosaic.jpg` (the stitched flight — also
handy for measuring the route for steps 4/5), `route_mosaic_density.jpg`
(averaged density over the whole route, stamped with the route total),
and `route_total_mosaic.txt`. Expect the total to land within a few % of
step 3b; large disagreement means the motion tracking failed — look at
the mosaic, misalignment is obvious to the eye.

You can also feed the stitched picture back through the model in one pass
(`count_crowd.py counts/route_mosaic.jpg --out counts_mosaic ...`) as a
cross-check, but treat it as a floor: each spot then gets a single reading
instead of ~8 averaged ones, and seams/scale variation make the model run
low on giant stitched images.

## 3d. Which number do I quote?

All step-3 variants read the same density maps and the same motion track —
they differ only in how many readings each person gets. Real example
(`dita_10.mp4`, night reel, 45 keyframes):

| method | readings per person | total |
|---|---|---|
| 3 — strip summing | 1 (at the clipped bottom edge) | 2,530 |
| model on the stitched mosaic, one pass | 1 | 2,457 |
| 3b — slice averaging (1-D) | ~8 | 3,631 |
| 3c — mosaic averaging (2-D) | ~8 | **3,657** |

The pattern to expect: the two single-reading methods agree with each
other, the two averaging methods agree with each other (within a few %),
and the averaging pair lands noticeably higher because single readings
miss people that other views of the same spot catch. **Quote the averaging
result (3b/3c) as the method's answer** — still a floor on bad footage,
since averaging fixes noise but not the model's systematic under-counting
on night/compressed video. If 3b and 3c disagree by much more than a few
%, the motion track is suspect: open `route_mosaic.jpg`, misalignment is
obvious to the eye.

If the floor is clearly far below what the footage shows (dense crowd,
tiny number), do NOT invent a multiplier — measure one with step 5 (or
hand counts) and pass it to `--calibrate`.

## 4. Cross-check with area x density (Jacobs method)

Never publish a model number alone. Measure the occupied street on Google
Maps/Earth ("Measure distance") or https://www.mapchecking.com, split it
into segments by how packed each looks in your footage, edit `SEGMENTS` in
the script, then:

```bash
.venv/bin/python scripts/jacobs_estimate.py
```

Density classes: 0.5/m² loose, 1/m² steady walking crowd, 2/m² slow shuffle,
4/m² packed-tight (rare outdoors). If the model total and the Jacobs total
disagree by more than ~2x, find out why before quoting either (usual
culprits: model missing the far field — check overlays; or your width/length
measurement including empty sidewalk).

## 5. Method 5 — people/m² x measured route length

The crowd model under-counts on bad footage, but its *map* of where the
crowd is stays good. Method 5 works in physical units instead of trusting
the model's absolute numbers: measure the real length of the strip the
drone covered (Google Maps/Earth "Measure distance", or directly on
`route_mosaic.jpg` against a known landmark distance), then:

```bash
.venv/bin/python scripts/estimate_by_density.py --route-length-m 540 \
      keyframes/ counts/
```

The length must be the ground covered by the mosaic **bottom edge to top
edge**, not the whole street if the drone covered less. Reference totals
scale with the *square* of this length — measure it carefully.

**Outputs in `counts/`:**

- `route_area.jpg` — the mosaic with the measured crowd area outlined in
  green. **Check this first: the area drives everything.**
- `density_report.txt` — crowd area in m², the model's implied people/m²,
  reference totals at the standard density classes (0.5 loose / 1 walking
  / 2 slow shuffle / 4 packed), and the calibration factor the model
  would need to reach each class.
- `density_slices.csv` — width, area, model people and people/m² per
  stretch of route, so you can assign a density class per segment
  instead of one for the whole route.

How to turn this into a defensible number: look at the footage segment by
segment and decide which density class each stretch *visibly* is. The
matching reference total is your Jacobs estimate built on the model's own
crowd outline; the "model would need xF" line of your chosen class is a
**measured** `--calibrate` factor for steps 3b/3c — quote both and they
should now agree.

## 6. Reporting

Quote a **range, not a number** (e.g. Jacobs x0.7 .. x1.3, with the
averaged route total from 3b/3c as the floor). For multiple videos/passes
of the same event, run the pipeline per pass and report the median and
spread. Note the time of each pass — crowds turn over during an event.

---

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

- `lwcc` errors about `/.lwcc/weights` on a fresh machine: run
  `count_crowd.py` once; it patches the package in place.
- Keyframes contain app UI: re-run step 1 and check the printed band; you
  can also crop manually with ffmpeg and use `--no-autocrop`.
- Keyframes look blurry / low quality: it's the source video, not the
  extraction. Compare file sizes and resolutions of the videos you have
  (e.g. `ffprobe` or just QuickTime's inspector) and run the pipeline on
  the largest/highest-resolution version of the SAME flight; `--png`
  (step 1) removes the last bit of re-compression but cannot add detail.
- Counts look absurdly low and overlays light up only the nearest rows of
  people: increase `--upscale`; if overlays light up trees/lights instead,
  decrease it.
- "I re-ran it and nothing got produced": if the inputs didn't change, a
  re-run reproduces byte-identical outputs — check the file timestamps in
  `counts/`. Also note the gif is NOT one of the script outputs: it only
  changes when you rebuild it yourself (see Extras), so after any re-run
  it shows the previous results until you do. Finder/Quick Look
  thumbnails can lag too — open the `.jpg` files themselves.
- GPU is not required; CPU inference takes a few seconds per frame.

## Extras
### Converting output images to .gif

Run this **from inside `counts/`** (the gif is written to the folder you
are standing in), and re-run it after every `count_crowd.py` run — it does
not update itself:

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

