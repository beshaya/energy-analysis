#!/usr/bin/env python3
"""Precompute raw features for sliding windows across every set in a folder.

The index lets rate_gui.py pick clips by predicted energy (--focus) without
decoding audio on the fly.

Usage:
  .venv/bin/python index_features.py .. [--out feature_index.csv]
"""

import argparse
import csv
import os

from energy_curve import FEATURES, window_features
from rate_gui import AUDIO_EXTS


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio_dir")
    p.add_argument("--out", default="feature_index.csv")
    p.add_argument("--window", type=float, default=20.0)
    p.add_argument("--hop", type=float, default=10.0)
    args = p.parse_args()

    names = sorted(n for n in os.listdir(args.audio_dir)
                   if os.path.splitext(n)[1].lower() in AUDIO_EXTS)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "start_s"] + FEATURES)
        for i, name in enumerate(names, 1):
            print(f"[{i}/{len(names)}] {name}", flush=True)
            d = window_features(os.path.join(args.audio_dir, name),
                                args.window, args.hop)
            for j in range(len(d["time_s"])):
                start = d["time_s"][j] - args.window / 2
                w.writerow([name, f"{start:.2f}"] + [f"{d[k][j]:.4f}" for k in FEATURES])


if __name__ == "__main__":
    main()
