#!/usr/bin/env python3
"""Precompute raw features for sliding windows across every set in a folder.

The index lets rate_gui.py pick clips by predicted energy (--focus) and
set_stats.py summarize whole sets, without decoding audio on the fly.

Usage:
  .venv/bin/python index_features.py .. [--out feature_index.csv]
  .venv/bin/python index_features.py .. --new             # only files not yet indexed
  .venv/bin/python index_features.py .. --only a.m4a b.mp3  # (re)index just these
"""

import argparse
import csv
import os

from energy_curve import FEATURES, add_derived, window_features
from rate_gui import AUDIO_EXTS


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio_dir")
    p.add_argument("--out", default="feature_index.csv")
    p.add_argument("--window", type=float, default=20.0)
    p.add_argument("--hop", type=float, default=10.0)
    p.add_argument("--new", action="store_true", help="only index files missing from --out")
    p.add_argument("--only", nargs="+", metavar="FILE", help="(re)index just these file names")
    args = p.parse_args()

    names = sorted(n for n in os.listdir(args.audio_dir)
                   if os.path.splitext(n)[1].lower() in AUDIO_EXTS)
    cols = ["file", "start_s"] + FEATURES
    kept = []
    if (args.new or args.only) and os.path.exists(args.out):
        with open(args.out) as f:
            reader = csv.DictReader(f)
            kept = list(reader)
        missing = set(cols) - set(reader.fieldnames)
        if missing and kept:
            # Derived features (e.g. dance_tempo) can be filled in without re-decoding.
            for r in kept:
                derived = add_derived({k: float(v) for k, v in r.items() if k != "file"})
                r.update({k: f"{float(derived[k]):.4f}" for k in missing if k in derived})
            missing = set(cols) - set(kept[0])
        if missing:
            p.error(f"{args.out} lacks {sorted(missing)}; rebuild it without --new/--only")
    if args.only:
        missing = set(args.only) - set(names)
        if missing:
            p.error(f"not found in {args.audio_dir}: {sorted(missing)}")
        todo = sorted(args.only)
    elif args.new:
        todo = [n for n in names if n not in {r["file"] for r in kept}]
    else:
        todo = names
    kept = [r for r in kept if r["file"] not in todo and r["file"] in names]

    rows = []
    for i, name in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {name}", flush=True)
        d = window_features(os.path.join(args.audio_dir, name), args.window, args.hop)
        for j in range(len(d["time_s"])):
            start = d["time_s"][j] - args.window / 2
            rows.append({"file": name, "start_s": f"{start:.2f}",
                         **{k: f"{d[k][j]:.4f}" for k in FEATURES}})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(kept + rows, key=lambda r: (r["file"], float(r["start_s"]))))
    print(f"indexed {len(todo)} file(s); {args.out} now covers "
          f"{len({r['file'] for r in kept + rows})} files")


if __name__ == "__main__":
    main()
