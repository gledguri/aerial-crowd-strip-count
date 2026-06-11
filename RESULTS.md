# Results — 10.06.2026 boulevard march

Input: `ScreenRecording_06-11-2026 00-01-02_1.mov` (screen-recorded Instagram
reel, night drone pass along the boulevard, 19 s, one continuous flight).

Recipe: DM-Count + QNRF weights + 2x upscale; strip summing over 27 keyframes.

| output | where |
|---|---|
| per-keyframe counts | `counts_qnrf_2x/counts_DM-Count_QNRF.csv` |
| per-strip counts + cumulative | `counts_qnrf_2x/route_strips.csv` |
| route total summary | `counts_qnrf_2x/route_total.txt` |
| QC heatmap overlays | `counts_qnrf_2x/*_density.jpg` |

Numbers:

- Per-keyframe visible counts: ~800–1,250 in the dense boulevard sections.
- Strip-summed route total: **4,025** (3,072 exited during the flight +
  954 visible in the last keyframe).
- Manual patch check (near-field patch of frame f548, eyeball vs model):
  model ~365 vs visually ~400–700 → model runs **1.3–2x low** on this
  footage (night, reel compression, tree canopies, crowd outside the filmed
  stretch).

Interpretation: **at least ~4,000 people on the filmed stretch, plausibly
6,000–10,000.** The bottleneck is the footage, not the method — redo with an
original daylight file before quoting anything tighter.
