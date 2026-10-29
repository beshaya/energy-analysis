#!/usr/bin/env python3
"""Fit a ridge-regression energy model to human ratings.

Reads ratings.csv (from rate_gui.py), recomputes the current feature set for
each rated clip from the audio (cached in rating_features.csv), standardizes,
picks the ridge penalty by leave-one-out cross-validation, and writes
energy_model.json.

Usage:
  .venv/bin/python fit_energy.py [AUDIO_DIR] [--ratings ratings.csv] [--clip 20]
  .venv/bin/python fit_energy.py .. --features loudness,brightness,hi_onset_rate
"""

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from energy_curve import FEATURES, clip_features, predict

# Raw tempo stays available for plots; the model uses the danced tempo instead.
MODEL_FEATURES = [f for f in FEATURES if f != "tempo"]

ALPHAS = [0.1, 0.3, 1, 3, 10, 30, 100]


def _clip_features(args):
    path, start, dur = args
    return clip_features(path, start, dur)


def load_ratings(ratings, audio_dir, clip_s, cache="rating_features.csv"):
    """Ratings for clips of length clip_s, with current features (cached)."""
    rows = [r for r in csv.DictReader(open(ratings)) if float(r["duration_s"]) == clip_s]
    key = lambda r: (r["file"], f"{float(r['start_s']):.2f}", f"{float(r['duration_s']):.1f}")

    cached = {}
    if os.path.exists(cache):
        for c in csv.DictReader(open(cache)):
            if all(f in c for f in FEATURES):
                cached[(c["file"], c["start_s"], c["duration_s"])] = c
    todo = sorted({key(r) for r in rows} - cached.keys())
    if todo:
        print(f"computing features for {len(todo)} clips ...", flush=True)
        jobs = [(os.path.join(audio_dir, f), float(s), float(d)) for f, s, d in todo]
        with ProcessPoolExecutor() as ex:
            for k, feats in zip(todo, ex.map(_clip_features, jobs)):
                cached[k] = dict(zip(("file", "start_s", "duration_s"), k), **feats)
        with open(cache, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "start_s", "duration_s"] + FEATURES,
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(cached.values())

    X = np.array([[float(cached[key(r)][f]) for f in FEATURES] for r in rows])
    y = np.array([float(r["rating"]) for r in rows])
    ok = np.isfinite(X).all(axis=1)
    if not ok.all():
        print(f"dropping {np.sum(~ok)} clips with missing features (silence?)")
    return X[ok], y[ok]


def fit_ridge(X, y, alpha, features):
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = (X - mean) / std
    coef = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return {"features": features, "mean": mean.tolist(), "std": std.tolist(),
            "coef": coef.tolist(), "intercept": float(y.mean()), "alpha": alpha}


def as_dict(X, features):
    return {f: X[:, i] for i, f in enumerate(features)}


def loo_predictions(X, y, alpha, features):
    pred = np.empty_like(y)
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        model = fit_ridge(X[keep], y[keep], alpha, features)
        pred[i] = predict(model, as_dict(X[i:i + 1], features))[0]
    return pred


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio_dir", nargs="?", default="..")
    p.add_argument("--ratings", default="ratings.csv")
    p.add_argument("--clip", type=float, default=20.0, help="only use clips of this length")
    p.add_argument("--features", help="comma-separated subset of features "
                                      "(default: all except raw tempo)")
    p.add_argument("--out", default="energy_model.json")
    p.add_argument("--plot", default="fit.png")
    args = p.parse_args()

    features = args.features.split(",") if args.features else MODEL_FEATURES
    unknown = set(features) - set(FEATURES)
    if unknown:
        p.error(f"unknown features {sorted(unknown)}; choose from {FEATURES}")
    X_all, y = load_ratings(args.ratings, args.audio_dir, args.clip)
    X = X_all[:, [FEATURES.index(f) for f in features]]
    print(f"{len(y)} ratings of {args.clip:.0f}s clips, "
          f"range {y.min():g}–{y.max():g}, mean {y.mean():.2f}; {len(features)} features")

    print("\ncorrelation of each feature with your ratings:")
    for i, f in sorted(enumerate(features), key=lambda t: -abs(np.corrcoef(X[:, t[0]], y)[0, 1])):
        print(f"  {f:14s} r={np.corrcoef(X[:, i], y)[0, 1]:+.2f}")

    base = rmse(y, np.array([np.delete(y, i).mean() for i in range(len(y))]))
    print(f"\nLOO RMSE, predict-the-mean baseline: {base:.2f}")
    scores = {}
    for a in ALPHAS:
        scores[a] = rmse(y, loo_predictions(X, y, a, features))
        print(f"LOO RMSE, ridge alpha={a:<5g}: {scores[a]:.2f}")
    alpha = min(scores, key=scores.get)
    pred = loo_predictions(X, y, alpha, features)
    r = np.corrcoef(pred, y)[0, 1]
    print(f"\nchosen alpha={alpha}: LOO RMSE {scores[alpha]:.2f} (baseline {base:.2f}), "
          f"LOO r={r:.2f}, R²≈{1 - scores[alpha] ** 2 / base ** 2:.2f}")

    model = fit_ridge(X, y, alpha, features)
    model.update({"n": int(len(y)), "loo_rmse": scores[alpha], "clip_s": args.clip})
    print("\nrating points per +1 SD of feature (all features held fixed):")
    for f, c, s in sorted(zip(features, model["coef"], model["std"]), key=lambda t: -abs(t[1])):
        print(f"  {f:14s} {c:+.2f}   (1 SD = {s:.3g})")
    with open(args.out, "w") as f:
        json.dump(model, f, indent=2)
    print(f"\nwrote {args.out}")

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 10], [0, 10], color="#d0d7de", linewidth=1, zorder=0)
    ax.scatter(y, pred, s=36, color="#2563eb", edgecolor="white", linewidth=1)
    ax.set(xlim=(0, 10), ylim=(0, 10), xlabel="Your rating",
           ylabel="Model prediction (leave-one-out)")
    ax.set_title(f"n={len(y)}  LOO RMSE {scores[alpha]:.2f}  r={r:.2f}", loc="left", fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(color="#e6e8eb", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.savefig(args.plot, dpi=120, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
