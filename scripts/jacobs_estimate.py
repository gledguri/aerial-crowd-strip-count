#!/usr/bin/env python3
"""
Independent cross-check: the Jacobs method (area x density), the standard
used by researchers and press to size protests.

Measure the occupied street area on a map (Google Maps / Earth: right-click
-> "Measure distance"; or https://www.mapchecking.com), split the route into
segments by how packed they look in your footage, and assign each segment a
density class:

  loose       0.5 people/m2   (people walking freely, big gaps)
  moderate    1.0 people/m2   (steady walking crowd)
  dense       2.0 people/m2   (slow shuffle, shoulders close)
  packed      4.0 people/m2   (barely moving, mosh-pit; rare outdoors)

Edit SEGMENTS below for your event and run:  python jacobs_estimate.py
"""

# (name, length_m, width_m, people_per_m2)
SEGMENTS = [
    ("example: boulevard south half", 400, 30, 2.0),
    ("example: boulevard north half", 400, 30, 1.0),
    ("example: square spill-over",    150, 40, 0.5),
]

total = 0.0
print(f"{'segment':35s} {'area_m2':>9s} {'p/m2':>5s} {'people':>8s}")
for name, length, width, dens in SEGMENTS:
    n = length * width * dens
    total += n
    print(f"{name:35s} {length * width:9.0f} {dens:5.1f} {n:8.0f}")
print("-" * 60)
print(f"{'TOTAL':35s} {'':>9s} {'':>5s} {total:8.0f}")
print("\nReport a range: x0.7 .. x1.3 of this number is honest "
      f"({total * 0.7:.0f} .. {total * 1.3:.0f}).")
