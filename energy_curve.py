#!/usr/bin/env python3
"""Plot a perceived-"energy" curve over time for long DJ sets.

Energy is a weighted blend of audio features that the music-perception
literature links to how fast / intense music feels:

  loudness     RMS level in dB                 (strongest single arousal cue)
  onset_rate   detected note/hit onsets per s  (event density -> perceived speed)
  flux         mean spectral flux              (how much the spectrum is changing)
  tempo        beat rate in BPM                (autocorrelation of onset envelope)
  brightness   spectral centroid in Hz         (brighter timbre reads as more energetic)
  pulse        beat clarity, 0..1              (how strong/obvious the beat is)

Each feature is robust-z-scored (median / IQR) across all files given in one
run, so curves from the same run are comparable with each other.

Usage:
  .venv/bin/python energy_curve.py SET.mp3 [SET2.m4a ...] [-o energy_plots]
"""

import argparse
import csv
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks

SR = 22050
N_FFT = 2048
HOP = 512
FPS = SR / HOP
N_MELS = 64

FEATURES = ["loudness", "onset_rate", "flux", "tempo", "brightness", "pulse"]
DEFAULT_WEIGHTS = {"loudness": 0.30, "onset_rate": 0.25, "flux": 0.15,
                   "tempo": 0.20, "brightness": 0.10, "pulse": 0.0}
LABELS = {"energy": "Energy (0–100)", "loudness": "Loudness (dB)",
          "onset_rate": "Onsets / s", "flux": "Spectral flux",
          "tempo": "Tempo (BPM)", "brightness": "Brightness (Hz)",
          "pulse": "Pulse clarity"}


def mel_filterbank(n_mels=N_MELS, fmin=30.0, fmax=SR / 2):
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = edges[i:i + 3]
        up = (freqs - lo) / (mid - lo)
        down = (hi - freqs) / (hi - mid)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


def frame_features(path):
    """Stream-decode with ffmpeg and return per-frame rms, centroid, flux."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR),
           "-f", "f32le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    window = np.hanning(N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR).astype(np.float32)
    fb = mel_filterbank()
    block_bytes = SR * 60 * 4  # one minute of audio per read

    rms, cent, flux = [], [], []
    buf = np.zeros(0, dtype=np.float32)
    prev_mel = None
    while True:
        raw = proc.stdout.read(block_bytes)
        if raw:
            buf = np.concatenate([buf, np.frombuffer(raw, dtype=np.float32)])
        n = 1 + (len(buf) - N_FFT) // HOP if len(buf) >= N_FFT else 0
        if n > 0:
            idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
            frames = buf[idx]
            rms.append(np.sqrt(np.mean(frames ** 2, axis=1)))
            mag = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)
            cent.append((mag @ freqs) / (mag.sum(axis=1) + 1e-9))
            mel_db = 10 * np.log10(mag ** 2 @ fb.T + 1e-10)
            prev = mel_db[:1] if prev_mel is None else prev_mel
            diff = np.diff(np.vstack([prev, mel_db]), axis=0)
            flux.append(np.maximum(diff, 0).mean(axis=1))
            prev_mel = mel_db[-1:]
            buf = buf[n * HOP:]
        if not raw:
            break
    proc.wait()
    if proc.returncode != 0 or not rms:
        sys.exit(f"ffmpeg failed to decode {path}")
    return np.concatenate(rms), np.concatenate(cent), np.concatenate(flux)


def estimate_tempo(env, bpm_center, bpm_min=60.0, bpm_max=200.0):
    """Autocorrelation tempo with a log-normal prior; returns (bpm, clarity)."""
    env = env - env.mean()
    n = len(env)
    spec = np.fft.rfft(env, 2 * n)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:n]
    if ac[0] <= 0:
        return np.nan, 0.0
    ac /= ac[0]
    lags = np.arange(int(60 * FPS / bpm_max), int(60 * FPS / bpm_min) + 2)
    lags = lags[lags < n - 1]
    bpms = 60 * FPS / lags
    prior = np.exp(-0.5 * (np.log2(bpms / bpm_center) / 1.0) ** 2)
    # Reward lags whose multiples also correlate (true beat period, not a subdivision).
    acp = np.clip(ac, 0, None)
    harm = acp[lags] + 0.5 * acp[np.minimum(2 * lags, n - 1)]
    k = int(np.argmax(harm * prior))
    lag = lags[k]
    # Parabolic interpolation for sub-frame lag resolution.
    a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
    denom = a - 2 * b + c
    shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
    return 60 * FPS / (lag + np.clip(shift, -0.5, 0.5)), float(max(ac[lags].max(), 0.0))


def window_features(path, win_s, hop_s, bpm_center, fold):
    rms, cent, flux = frame_features(path)
    # Onsets: peaks of flux above a local adaptive threshold.
    thresh = median_filter(flux, size=int(FPS)) + 0.5 * np.std(flux)
    onsets, _ = find_peaks(flux, height=thresh, distance=max(1, int(0.05 * FPS)))
    onset_mask = np.zeros(len(flux), dtype=bool)
    onset_mask[onsets] = True

    win, hop = int(win_s * FPS), int(hop_s * FPS)
    rows = []
    for start in range(0, max(1, len(rms) - win + 1), hop):
        sl = slice(start, start + win)
        r = rms[sl]
        loud = 10 * np.log10(np.mean(r ** 2) + 1e-10)
        voiced = r > np.max(r) * 0.05
        bright = float(np.median(cent[sl][voiced])) if voiced.any() else np.nan
        bpm, clarity = estimate_tempo(flux[sl], bpm_center)
        rows.append({"time_s": (start + win / 2) / FPS,
                     "loudness": max(loud, -80.0),
                     "onset_rate": onset_mask[sl].sum() / (len(r) / FPS),
                     "flux": float(flux[sl].mean()),
                     "tempo": bpm,
                     "brightness": bright,
                     "pulse": clarity})
    data = {k: np.array([row[k] for row in rows], dtype=float) for k in rows[0]}
    if fold:
        # Fold into one octave around the prior so double/half-time detections
        # don't masquerade as speed changes (event density already covers that).
        t = data["tempo"]
        lo, hi = bpm_center / np.sqrt(2), bpm_center * np.sqrt(2)
        while np.any(t > hi) or np.any(t < lo):
            t = np.where(t > hi, t / 2, np.where(t < lo, t * 2, t))
        data["tempo"] = t
    # Remaining per-window jitter: median-smooth (~45 s).
    data["tempo"] = median_filter(data["tempo"], size=9, mode="nearest")
    return data


def robust_z(x, ref):
    med = np.nanmedian(ref)
    iqr = np.nanpercentile(ref, 75) - np.nanpercentile(ref, 25)
    return np.clip((x - med) / (iqr / 1.349 + 1e-9), -3, 3)


def add_energy(all_data, weights, smooth_s, hop_s):
    pooled = {k: np.concatenate([d[k] for d in all_data]) for k in FEATURES}
    total_w = sum(weights.values())
    raw_all = []
    for d in all_data:
        e = sum(w * np.nan_to_num(robust_z(d[k], pooled[k])) for k, w in weights.items())
        d["energy_raw"] = e / total_w
        raw_all.append(d["energy_raw"])
    lo, hi = np.percentile(np.concatenate(raw_all), [1, 99])
    size = max(1, int(round(smooth_s / hop_s)))
    for d in all_data:
        scaled = np.clip((d["energy_raw"] - lo) / (hi - lo) * 100, 0, 100)
        d["energy"] = uniform_filter1d(scaled, size=size, mode="nearest")


def fmt_hms(x, _pos=None):
    m = int(round(x))
    return f"{m // 60}:{m % 60:02d}"


def plot(d, title, out_png, show_features):
    ink, muted, grid = "#1f2328", "#656d76", "#e6e8eb"
    accent, series = "#c2410c", "#2563eb"
    panels = ["energy"] + (FEATURES if show_features else [])
    heights = [3] + [1] * (len(panels) - 1)
    fig, axes = plt.subplots(len(panels), 1, sharex=True,
                             figsize=(14, 3.2 + 1.15 * (len(panels) - 1)),
                             gridspec_kw={"height_ratios": heights, "hspace": 0.18})
    axes = np.atleast_1d(axes)
    t_min = d["time_s"] / 60
    for ax, key in zip(axes, panels):
        y = d[key]
        if key == "energy":
            ax.fill_between(t_min, 0, y, color=accent, alpha=0.15, linewidth=0)
            ax.plot(t_min, y, color=accent, linewidth=2)
            ax.set_ylim(0, 100)
        else:
            ax.plot(t_min, y, color=series, linewidth=0.8, alpha=0.35)
            sm = uniform_filter1d(np.nan_to_num(y, nan=np.nanmedian(y)), size=7, mode="nearest")
            ax.plot(t_min, sm, color=series, linewidth=1.6)
        ax.set_ylabel(LABELS[key], color=muted, fontsize=9, rotation=0,
                      ha="right", va="center", labelpad=8)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.grid(axis="x", color=grid, linewidth=0.5)
        ax.tick_params(colors=muted, labelsize=8, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_axisbelow(True)
    span = t_min[-1] if len(t_min) else 1
    step = 5 if span <= 75 else 10 if span <= 150 else 15
    axes[-1].xaxis.set_major_locator(MultipleLocator(step))
    axes[-1].xaxis.set_major_formatter(FuncFormatter(fmt_hms))
    axes[-1].set_xlabel("Time (h:mm)", color=muted, fontsize=9)
    axes[0].set_xlim(0, span)
    axes[0].set_title(title, loc="left", color=ink, fontsize=12, pad=10)
    fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_csv(d, out_csv):
    cols = ["time_s", "energy"] + FEATURES
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(d["time_s"])):
            w.writerow([f"{d[c][i]:.4f}" for c in cols])


def parse_weights(s):
    weights = dict(DEFAULT_WEIGHTS)
    for part in filter(None, (s or "").split(",")):
        k, v = part.split("=")
        if k not in FEATURES:
            sys.exit(f"unknown feature in --weights: {k} (choose from {FEATURES})")
        weights[k] = float(v)
    return {k: v for k, v in weights.items() if v != 0}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("-o", "--outdir", default="energy_plots")
    p.add_argument("--window", type=float, default=20.0,
                   help="analysis window in seconds (default 20)")
    p.add_argument("--hop", type=float, default=5.0,
                   help="step between windows in seconds (default 5)")
    p.add_argument("--smooth", type=float, default=60.0,
                   help="smoothing of the energy curve in seconds (default 60)")
    p.add_argument("--bpm-center", type=float, default=90.0,
                   help="tempo prior center; steers octave choice (default 90)")
    p.add_argument("--no-fold", action="store_true",
                   help="don't fold tempo into one octave around --bpm-center")
    p.add_argument("--weights", help="override weights, e.g. tempo=0.4,brightness=0")
    p.add_argument("--energy-only", action="store_true",
                   help="plot only the energy curve, not the feature panels")
    args = p.parse_args()

    weights = parse_weights(args.weights)
    os.makedirs(args.outdir, exist_ok=True)
    all_data = []
    for path in args.files:
        print(f"analyzing {path} ...", flush=True)
        all_data.append(window_features(path, args.window, args.hop, args.bpm_center,
                                        not args.no_fold))
    add_energy(all_data, weights, args.smooth, args.hop)

    for path, d in zip(args.files, all_data):
        stem = os.path.splitext(os.path.basename(path))[0]
        base = os.path.join(args.outdir, stem)
        plot(d, stem, base + ".png", not args.energy_only)
        write_csv(d, base + ".csv")
        print(f"  wrote {base}.png and {base}.csv  "
              f"(median tempo {np.nanmedian(d['tempo']):.0f} BPM)")


if __name__ == "__main__":
    main()
