#!/usr/bin/env python3
"""
Step 4: the Jacobs method (area x density), the standard used by
researchers and press to size protests.

Preferred input: the jacobs_segments.csv written by step 3
(estimate_by_density.py). It already holds each strip's measured length,
width and area; you fill the assumed_density_p_m2 column with the class
you SEE per strip in the footage (the model_density_p_m2 column is the
model's lower-bound suggestion):

  loose       0.5 people/m2   (people walking freely, big gaps)
  moderate    1.0 people/m2   (steady walking crowd)
  dense       2.0 people/m2   (slow shuffle, shoulders close)
  packed      4.0 people/m2   (barely moving, mosh-pit; rare outdoors)

  python jacobs_estimate.py counts/jacobs_segments.csv

Strips with an empty assumed_density_p_m2 are skipped with a warning, so
fill them all (use 0 for genuinely empty stretches).

Manual fallback (no CSV): edit SEGMENTS below and run without arguments —
useful for parts of the event the drone never filmed.
"""
import csv
import sys

# (name, length_m, width_m, people_per_m2)
SEGMENTS = [
    ("example: main boulevard", 500, 30, 0.5),
    ("example: not captured by drone", 100, 30, 0.5),
    ("example: spill-over",    60, 10, 0.25),
]


def rows_from_csv(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            name = (f"{r['segment']} ({r['start_m']}-{r['end_m']} m, "
                    f"w {r['mean_width_m']} m)")
            dens = r.get("assumed_density_p_m2", "").strip()
            if not dens:
                print(f"WARNING: {r['segment']} has no assumed density — "
                      f"skipped (model suggested "
                      f">={r['model_density_p_m2']})")
                continue
            rows.append((name, float(r["area_m2"]), float(dens)))
    return rows


def main():
    if len(sys.argv) > 1:
        rows = rows_from_csv(sys.argv[1])
    else:
        rows = [(name, length * width, dens)
                for name, length, width, dens in SEGMENTS]

    total = 0.0
    print(f"\n{'segment':42s} {'area_m2':>9s} {'p/m2':>5s} {'people':>8s}")
    for name, area, dens in rows:
        n = area * dens
        total += n
        print(f"{name:42s} {area:9.0f} {dens:5.2f} {n:8.0f}")
    print("-" * 68)
    print(f"{'JACOBS TOTAL':42s} {'':>9s} {'':>5s} {total:8.0f}")
    print(f"\nReport a range: x0.7 .. x1.3 of this number is honest "
          f"({total * 0.7:.0f} .. {total * 1.3:.0f}).")
    print("Next (step 5+6): pass this total to report_route.py "
          f"--jacobs-total {total:.0f}")


if __name__ == "__main__":
    main()
