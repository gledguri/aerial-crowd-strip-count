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
frame overlaps the next one; to count each person exactly once, the pipeline
measures how far the ground slides between frames and only counts the strip
of people that exits the bottom edge of the image each step ("strip
summing"). An independent area-x-density check (Jacobs method) is included
to validate whatever number the model gives you.

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
- If the video has hard cuts (different shots stitched together), frames are
  named `scene00_*, scene01_*, ...`. Treat each scene as a separate flight:
  put each scene's frames in their own folder and run steps 2-3 per scene —
  the strip-summing math assumes one continuous pass.

## 2. Count people per keyframe

```bash
.venv/bin/python scripts/count_crowd.py keyframes/ --out counts \
      --weights QNRF --upscale 2 --save-density
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
  frame's estimated count. **This is your quality control: look at
  several.** Color must sit on the crowd, not on trees, streetlights,
  parked cars, or app icons. If the far half of the street shows crowd but
  no color, the model is missing it — try `--upscale 2` (or 3 on 4K input).
- `*_density.npy` — raw density maps (needed by step 3).

Remember: keyframes overlap heavily — **never sum this CSV across frames**.
That's what step 3 is for.

![Output](counts_dmcount_sha/output.gif)

## 3. Whole-route total (strip summing)

Requires: one continuous pass along the street, and step 2 run with
`--save-density`.

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
- The strip method counts people where they are nearest the camera, which is
  where the model is most reliable — but anyone behind the take-off point,
  beyond the last frame, or under tree canopies is **not counted**.

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

## 5. Reporting

Quote a **range, not a number** (e.g. Jacobs x0.7 .. x1.3, and the strip
total as a floor). For multiple videos/passes of the same event, run the
pipeline per pass and report the median and spread. Note the time of each
pass — crowds turn over during an event.

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
- Counts look absurdly low and overlays light up only the nearest rows of
  people: increase `--upscale`; if overlays light up trees/lights instead,
  decrease it.
- GPU is not required; CPU inference takes a few seconds per frame.

## Extras
### Converting output images to .gif
```bash
brew install imagemagick
cd /path/to/images
magick *.jpg -delay 10 -loop 0 output.gif
```

Converting output to video
```bash
brew install ffmpeg
cd counts/

ffmpeg -i output.gif \
-vf "scale=1206:718" \
-pix_fmt yuv420p \
density_video.mp4
```

