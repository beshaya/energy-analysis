#!/usr/bin/env python3
"""Plot a perceived-"energy" curve over time for long DJ sets.

Features are computed in sliding windows (see FEATURE_HELP). If a fitted model
(energy_model.json, from fit_energy.py) exists, energy is the model's predicted
0-10 rating; otherwise it falls back to a hand-weighted blend of robust-z-scored
features, normalized across all files given in one run.

Usage:
  .venv/bin/python energy_curve.py SET.mp3 [SET2.m4a ...] [-o energy_plots]
"""

import argparse
import csv
import json
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
MEL_FMIN = 30.0
BANDS = {"lo": (0, 400), "mid": (400, 2000), "hi": (2000, SR / 2)}
BPM_GRID = 2 ** np.arange(np.log2(60), np.log2(140), 1 / 96)  # ~0.7% steps
MAX_BPM_PER_MIN = 10.0  # DJs rarely ramp tempo faster than this
DANCE_TEMPO_MAX = 95.0  # above this, dancers usually move at half time

FEATURE_HELP = {
    "loudness": "RMS level (dB)",
    "brightness": "spectral centroid (Hz)",
    "flux": "mean spectral flux, all bands",
    "onset_rate": "onsets per second, all bands",
    "tempo": "beat rate (BPM), tracked smoothly across the whole file",
    "dance_tempo": "tempo as danced: halved above DANCE_TEMPO_MAX (half-time)",
    "pulse": "beat clarity 0-1 (autocorrelation peak height)",
    "hi_onset_rate": "onsets per second above 2 kHz (hats, shakers)",
    "lo_onset_rate": "onsets per second below 400 Hz (kick, bass)",
    "lo_flux": "spectral flux below 400 Hz",
    "mid_flux": "spectral flux 400 Hz-2 kHz",
    "hi_flux": "spectral flux above 2 kHz",
    "bal_lo": "level below 400 Hz relative to total (dB)",
    "bal_mid": "level 400 Hz-2 kHz relative to total (dB)",
    "bal_hi": "level above 2 kHz relative to total (dB)",
    "perc_ratio": "percussive share of energy (HPSS), 0-1",
    "attack": "onset peakiness: 95th pct / mean flux",
    "choppiness": "loudness fluctuation faster than 1 Hz (dB SD)",
    "hat_rate": "fastest strong periodicity above 2 kHz (events/s)",
}
FEATURES = list(FEATURE_HELP)
DEFAULT_WEIGHTS = {"loudness": 0.30, "onset_rate": 0.25, "flux": 0.15,
                   "tempo": 0.20, "brightness": 0.10}
LABELS = {"loudness": "Loudness (dB)", "onset_rate": "Onsets / s", "flux": "Spectral flux",
          "tempo": "Tempo (BPM)", "dance_tempo": "Danced tempo (BPM)", "brightness": "Brightness (Hz)", "pulse": "Pulse clarity",
          "hi_onset_rate": "Hi onsets / s", "lo_onset_rate": "Lo onsets / s",
          "lo_flux": "Lo flux", "mid_flux": "Mid flux", "hi_flux": "Hi flux",
          "bal_lo": "Lo balance (dB)", "bal_mid": "Mid balance (dB)",
          "bal_hi": "Hi balance (dB)", "perc_ratio": "Percussive share",
          "attack": "Attack", "choppiness": "Choppiness (dB)",
          "hat_rate": "Hat rate (/s)"}
PLAIN_FEATURES = ["loudness", "onset_rate", "flux", "tempo", "brightness", "pulse"]


def mel_filterbank(n_mels=N_MELS, fmin=MEL_FMIN, fmax=SR / 2):
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
    return fb, edges[1:-1]


def frame_features(path, start=None, duration=None):
    """Stream-decode with ffmpeg; return per-frame rms, centroid and mel power (dB)."""
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    window = np.hanning(N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR).astype(np.float32)
    fb, _ = mel_filterbank()
    block_bytes = SR * 60 * 4  # one minute of audio per read

    rms, cent, mel = [], [], []
    buf = np.zeros(0, dtype=np.float32)
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
            mel.append((10 * np.log10(mag ** 2 @ fb.T + 1e-10)).astype(np.float32))
            buf = buf[n * HOP:]
        if not raw:
            break
    proc.wait()
    if proc.returncode != 0 or not rms:
        sys.exit(f"ffmpeg failed to decode {path}")
    return np.concatenate(rms), np.concatenate(cent), np.vstack(mel)


def band_flux(mel_db, cols=slice(None)):
    diff = np.diff(mel_db[:, cols], axis=0, prepend=mel_db[:1, cols])
    return np.maximum(diff, 0).mean(axis=1)


def onset_mask(env):
    """Peaks of an onset envelope above a local (~3 s) adaptive threshold."""
    k = int(3 * FPS)
    mean = uniform_filter1d(env, k, mode="nearest")
    sd = np.sqrt(np.maximum(uniform_filter1d(env ** 2, k, mode="nearest") - mean ** 2, 0))
    thresh = median_filter(env, size=int(FPS), mode="nearest") + 0.5 * sd
    peaks, _ = find_peaks(env, height=thresh, distance=max(1, int(0.05 * FPS)))
    mask = np.zeros(len(env), dtype=bool)
    mask[peaks] = True
    return mask


def autocorr(env):
    env = env - env.mean()
    n = len(env)
    ac = np.fft.irfft(np.abs(np.fft.rfft(env, 2 * n)) ** 2)[:n]
    return ac / ac[0] if ac[0] > 0 else None


def refine_lag(ac, lag):
    """Parabolic interpolation for sub-frame lag resolution."""
    a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
    denom = a - 2 * b + c
    return lag + (np.clip(0.5 * (a - c) / denom, -0.5, 0.5) if denom != 0 else 0.0)


def tempo_salience(env, bpm_center):
    """Score each BPM in BPM_GRID for one window; also return pulse clarity.

    A beat period T scores by the onset autocorrelation at T, 2T and 4T. The
    4T (one bar) term matters for zouk: in a 3-3-2 rhythm the dotted-quarter
    period also correlates at T and 2T, but only the true beat repeats per bar.
    """
    ac = autocorr(env)
    if ac is None:
        return np.zeros(len(BPM_GRID)), 0.0
    acp = np.clip(ac, 0, None)
    lags = np.arange(len(acp))
    period = 60 * FPS / BPM_GRID
    score = sum(w * np.interp(k * period, lags, acp, right=0.0)
                for k, w in ((1, 1.0), (2, 0.5), (4, 0.5)))
    prior = np.exp(-0.5 * (np.log2(BPM_GRID / bpm_center) / 1.0) ** 2)
    in_range = (lags >= period.min()) & (lags <= period.max())
    return score * prior, float(acp[in_range].max()) if in_range.any() else 0.0


def track_tempo(salience, hop_s, max_bpm_per_min=MAX_BPM_PER_MIN, stiffness=0.05):
    """Most salient smooth tempo path through per-window salience (Viterbi).

    Each step pays stiffness * (change / allowed change)^2, where the allowed
    change is max_bpm_per_min scaled to the hop, so a sudden 75 <-> 100 flip
    costs as much as minutes of contrary evidence while slow ramps are cheap.
    """
    S = np.asarray(salience, dtype=float)
    peak = S.max(axis=1, keepdims=True)
    S = np.where(peak > 0, S / np.where(peak > 0, peak, 1), 1.0 / S.shape[1])
    allowed = max(max_bpm_per_min * hop_s / 60, 1e-3)
    trans = stiffness * ((BPM_GRID[:, None] - BPM_GRID[None, :]) / allowed) ** 2
    score = S[0].copy()
    back = np.zeros(S.shape, dtype=np.int32)
    for t in range(1, len(S)):
        cand = score[:, None] - trans          # [from, to]
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(len(BPM_GRID))] + S[t]
    path = np.empty(len(S), dtype=np.int32)
    path[-1] = int(np.argmax(score))
    for t in range(len(S) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return BPM_GRID[path]


def fastest_periodicity(env, min_hz=2.0, max_hz=10.0, floor=0.2):
    """Highest-rate autocorrelation peak (events/s) whose height clears `floor`,
    else the strongest peak in range."""
    ac = autocorr(env)
    if ac is None:
        return np.nan
    lo, hi = int(FPS / max_hz), int(FPS / min_hz) + 1
    peaks, _ = find_peaks(ac[:hi + 1])
    peaks = peaks[peaks >= lo]
    if len(peaks) == 0:
        return np.nan
    strong = peaks[ac[peaks] >= floor]
    lag = strong[0] if len(strong) else peaks[np.argmax(ac[peaks])]
    return FPS / refine_lag(ac, lag)


def window_features(path, win_s, hop_s, bpm_center=90.0, start=None, duration=None,
                    max_bpm_per_min=MAX_BPM_PER_MIN):
    rms, cent, mel_db = frame_features(path, start, duration)
    _, centers = mel_filterbank()
    cols = {b: (centers >= lo) & (centers < hi) for b, (lo, hi) in BANDS.items()}

    flux = band_flux(mel_db)
    bflux = {b: band_flux(mel_db, c) for b, c in cols.items()}
    power = 10 ** (mel_db / 10)
    total_db = 10 * np.log10(power.sum(axis=1) + 1e-10)
    bal = {b: 10 * np.log10(power[:, c].sum(axis=1) + 1e-10) - total_db
           for b, c in cols.items()}
    # Harmonic/percussive split: sustained energy is smooth along time,
    # percussive energy is smooth along frequency.
    harm = median_filter(power, size=(17, 1), mode="nearest")
    perc = median_filter(power, size=(1, 9), mode="nearest")
    perc_frame = (perc ** 2 / (harm ** 2 + perc ** 2 + 1e-20) * power).sum(axis=1)
    del harm, perc
    power_sum = power.sum(axis=1)
    del power
    chop = total_db - uniform_filter1d(total_db, int(FPS), mode="nearest")
    on_all, on_hi, on_lo = onset_mask(flux), onset_mask(bflux["hi"]), onset_mask(bflux["lo"])

    win, hop = int(win_s * FPS), int(hop_s * FPS)
    rows, salience = [], []
    for s in range(0, max(1, len(rms) - win + 1), hop):
        sl = slice(s, s + win)
        r = rms[sl]
        secs = len(r) / FPS
        voiced = r > np.max(r) * 0.05
        # Tempo evidence from all bands plus the kick/bass band.
        sal_all, clarity = tempo_salience(flux[sl], bpm_center)
        sal_lo, _ = tempo_salience(bflux["lo"][sl], bpm_center)
        salience.append(sal_all / (sal_all.max() + 1e-9) + sal_lo / (sal_lo.max() + 1e-9))
        rows.append({
            "time_s": (s + win / 2) / FPS,
            "loudness": max(10 * np.log10(np.mean(r ** 2) + 1e-10), -80.0),
            "brightness": float(np.median(cent[sl][voiced])) if voiced.any() else np.nan,
            "flux": float(flux[sl].mean()),
            "onset_rate": on_all[sl].sum() / secs,
            "pulse": clarity,
            "hi_onset_rate": on_hi[sl].sum() / secs,
            "lo_onset_rate": on_lo[sl].sum() / secs,
            "lo_flux": float(bflux["lo"][sl].mean()),
            "mid_flux": float(bflux["mid"][sl].mean()),
            "hi_flux": float(bflux["hi"][sl].mean()),
            "bal_lo": float(bal["lo"][sl].mean()),
            "bal_mid": float(bal["mid"][sl].mean()),
            "bal_hi": float(bal["hi"][sl].mean()),
            "perc_ratio": float(perc_frame[sl].sum() / (power_sum[sl].sum() + 1e-20)),
            "attack": float(np.percentile(flux[sl], 95) / (flux[sl].mean() + 1e-9)),
            "choppiness": float(np.std(chop[sl])),
            "hat_rate": fastest_periodicity(bflux["hi"][sl]),
        })
    data = {k: np.array([row[k] for row in rows], dtype=float) for k in rows[0]}
    data["tempo"] = track_tempo(salience, hop / FPS, max_bpm_per_min)
    return add_derived(data)


def add_derived(data):
    """Features computed from other features (also usable on index rows)."""
    t = np.asarray(data["tempo"], dtype=float)
    data["dance_tempo"] = np.where(t > DANCE_TEMPO_MAX, t / 2, t)
    return data


def summarize(path, bpm_center=90.0, start=None, duration=None):
    """Raw (unnormalized) features of a whole short clip, or a segment of a file."""
    # A window longer than any clip collapses to one window spanning all of it.
    d = window_features(path, 1e6, 1e6, bpm_center, start, duration)
    return {k: float(d[k][0]) for k in FEATURES}


def clip_features(path, start, duration, context_s=60.0, hop_s=5.0, bpm_center=90.0):
    """Features of one clip of a long file, with tempo tracked through
    `context_s` of surrounding audio so it matches what the full-file curve sees."""
    s0 = max(0.0, start - context_s)
    d = window_features(path, duration, hop_s, bpm_center, s0, start - s0 + duration + context_s)
    i = min(int(round((start - s0) * FPS / int(hop_s * FPS))), len(d["time_s"]) - 1)
    return {k: float(d[k][i]) for k in FEATURES}


def load_model(path):
    with open(path) as f:
        return json.load(f)


def predict(model, data):
    """Predicted rating; `data` maps feature name -> value or array."""
    X = np.column_stack([np.asarray(data[f], dtype=float) for f in model["features"]])
    Z = (X - np.asarray(model["mean"])) / np.asarray(model["std"])
    return Z @ np.asarray(model["coef"]) + model["intercept"]


def robust_z(x, ref):
    med = np.nanmedian(ref)
    iqr = np.nanpercentile(ref, 75) - np.nanpercentile(ref, 25)
    return np.clip((x - med) / (iqr / 1.349 + 1e-9), -3, 3)


def add_energy(all_data, weights, smooth_s, hop_s, model=None):
    size = max(1, int(round(smooth_s / hop_s)))
    if model:
        for d in all_data:
            pred = predict(model, d)
            pred = np.where(np.isfinite(pred), pred, np.nanmedian(pred))
            d["energy"] = uniform_filter1d(np.clip(pred, 0, 10), size=size, mode="nearest")
        return
    pooled = {k: np.concatenate([d[k] for d in all_data]) for k in FEATURES}
    total_w = sum(weights.values())
    raw_all = []
    for d in all_data:
        e = sum(w * np.nan_to_num(robust_z(d[k], pooled[k])) for k, w in weights.items())
        d["energy_raw"] = e / total_w
        raw_all.append(d["energy_raw"])
    lo, hi = np.percentile(np.concatenate(raw_all), [1, 99])
    for d in all_data:
        scaled = np.clip((d["energy_raw"] - lo) / (hi - lo) * 10, 0, 10)
        d["energy"] = uniform_filter1d(scaled, size=size, mode="nearest")


def fmt_hms(x, _pos=None):
    m = int(round(x))
    return f"{m // 60}:{m % 60:02d}"


def plot(d, title, out_png, panels, energy_label):
    ink, muted, grid = "#1f2328", "#656d76", "#e6e8eb"
    accent, series = "#c2410c", "#2563eb"
    panels = ["energy"] + panels
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
            ax.set_ylim(0, 10)
        else:
            ax.plot(t_min, y, color=series, linewidth=0.8, alpha=0.35)
            sm = uniform_filter1d(np.nan_to_num(y, nan=np.nanmedian(y)), size=7, mode="nearest")
            ax.plot(t_min, sm, color=series, linewidth=1.6)
        ax.set_ylabel(energy_label if key == "energy" else LABELS[key], color=muted,
                      fontsize=9, rotation=0, ha="right", va="center", labelpad=8)
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
    p.add_argument("--max-bpm-per-min", type=float, default=MAX_BPM_PER_MIN,
                   help="how fast tempo may plausibly ramp (default 10 BPM/min)")
    p.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "energy_model.json"),
                   help="fitted model from fit_energy.py (default: energy_model.json "
                        "next to this script, if present)")
    p.add_argument("--weights", help="ignore the model and use hand weights, "
                                     "e.g. tempo=0.4,brightness=0")
    p.add_argument("--energy-only", action="store_true",
                   help="plot only the energy curve, not the feature panels")
    args = p.parse_args()

    model = None
    if not args.weights and os.path.exists(args.model):
        model = load_model(args.model)
        print(f"using fitted model {args.model} (n={model.get('n')})")
    weights = parse_weights(args.weights)
    if model:
        top = sorted(zip(model["features"], model["coef"]), key=lambda t: -abs(t[1]))
        panels = [f for f, _ in top[:6]]
        energy_label = "Energy (predicted rating)"
    else:
        panels, energy_label = PLAIN_FEATURES, "Energy (0–10, relative)"

    os.makedirs(args.outdir, exist_ok=True)
    all_data = []
    for path in args.files:
        print(f"analyzing {path} ...", flush=True)
        all_data.append(window_features(path, args.window, args.hop, args.bpm_center,
                                        max_bpm_per_min=args.max_bpm_per_min))
    add_energy(all_data, weights, args.smooth, args.hop, model)

    for path, d in zip(args.files, all_data):
        stem = os.path.splitext(os.path.basename(path))[0]
        base = os.path.join(args.outdir, stem)
        plot(d, stem, base + ".png", [] if args.energy_only else panels, energy_label)
        write_csv(d, base + ".csv")
        print(f"  wrote {base}.png and {base}.csv  "
              f"(median tempo {np.nanmedian(d['tempo']):.0f} BPM)")


if __name__ == "__main__":
    main()
