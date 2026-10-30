#!/usr/bin/env python3
"""Summarize each set's energy for the set-picking spreadsheet.

Per file in the feature index (from index_features.py), predicts energy with
the fitted model, smooths it, and reports:

  progression  one colored square per 10 minutes showing that stretch's
               75th-percentile energy (how it feels while the music is going,
               not dragged down by brief transitions), rounded onto the
               Energy Scale: 🟦 0-2 chill, 🟩 3-4 low,
               🟨 5-6 mid, 🟧 7-8 high, 🟥 9-10 (or --blocks: ▁ <1 ... █ 7+)
  p25...p95    percentiles of energy over the whole set

Usage:
  .venv/bin/python set_stats.py [--index feature_index.csv] [--out set_stats.tsv]
"""

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy.ndimage import uniform_filter1d

from energy_curve import add_derived, load_model, predict

BLOCKS = "▁▂▃▄▅▆▇█"
# Squares carry their own color, so they survive pasting into a spreadsheet;
# hues follow the sheet's Energy dropdown (Chill blue, Low green, Mid yellow, High orange).
SQUARES = ["🟦", "🟦", "🟦", "🟩", "🟩", "🟨", "🟨", "🟧", "🟧", "🟥", "🟥"]


def square(energy):
    return SQUARES[min(max(int(round(energy)), 0), 10)]


def block(energy):
    return BLOCKS[min(max(int(energy), 0), len(BLOCKS) - 1)]


def progression(t, e, segment_s, glyph=square, pct=75.0):
    """Per-segment energy percentile as characters. A trailing partial
    segment gets its own character only if it's at least half a segment."""
    n = int(np.ceil((t.max() + 1e-9) / segment_s)) or 1
    if n > 1 and t.max() - (n - 1) * segment_s < segment_s / 2:
        n -= 1
    seg = np.minimum((t // segment_s).astype(int), n - 1)
    return "".join(glyph(np.percentile(e[seg == k], pct) if (seg == k).any() else 0.0)
                   for k in range(n))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", default="feature_index.csv")
    p.add_argument("--model", default="energy_model.json")
    p.add_argument("--smooth", type=float, default=60.0, help="smoothing in seconds (default 60)")
    p.add_argument("--segment", type=float, default=600.0,
                   help="seconds per progression character (default 600)")
    p.add_argument("--window", type=float, default=20.0, help="index window length (default 20)")
    p.add_argument("--segment-pct", type=float, default=75.0,
                   help="percentile of energy each progression character shows (default 75)")
    p.add_argument("--blocks", action="store_true",
                   help="progression as block heights instead of colored squares")
    p.add_argument("--out", help="write TSV here instead of stdout")
    args = p.parse_args()

    model = load_model(args.model)
    by_file = defaultdict(list)
    for r in csv.DictReader(open(args.index)):
        by_file[r["file"]].append(r)

    rows = []
    for name, recs in sorted(by_file.items(), key=lambda kv: kv[0].lower()):
        cols = {k: np.array([float(r[k]) for r in recs]) for k in recs[0] if k != "file"}
        cols = add_derived(cols)
        t = cols["start_s"] + args.window / 2
        e = predict(model, cols)
        e = np.where(np.isfinite(e), e, np.nanmedian(e))
        hop = np.median(np.diff(t)) if len(t) > 1 else args.window
        e = np.clip(uniform_filter1d(e, max(1, int(round(args.smooth / hop))), mode="nearest"), 0, 10)
        pct = np.percentile(e, [25, 50, 90, 95])
        rows.append([name, f"{(cols['start_s'].max() + args.window) / 60:.0f}",
                     progression(t, e, args.segment, block if args.blocks else square,
                                 args.segment_pct)] + [f"{v:.1f}" for v in pct])

    out = open(args.out, "w", newline="") if args.out else sys.stdout
    w = csv.writer(out, delimiter="\t")
    w.writerow(["file", "minutes", "progression", "p25", "p50", "p90", "p95"])
    w.writerows(rows)


if __name__ == "__main__":
    main()
