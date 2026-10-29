# energy-analysis

Plot a perceived-"energy" curve over time for long DJ sets — a proxy for how
fast/intense the music feels, not just its BPM.

## Setup

Requires Python 3.10+ and `ffmpeg` on your PATH (used to decode mp3/m4a/opus/etc.).

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```
.venv/bin/python energy_curve.py SET.mp3 [SET2.m4a ...] [-o energy_plots]
```

Writes `<name>.png` (energy curve + per-feature panels) and `<name>.csv` per file.
Files analyzed in the same run share one normalization, so their curves are
comparable.

Useful options:

- `--weights tempo=0.4,brightness=0` — change the blend
- `--smooth 180` — smoothing of the energy curve in seconds (default 60)
- `--bpm-center 90` / `--no-fold` — tempo prior and octave folding
- `--energy-only` — plot only the energy curve

## Features

| Feature | Measure | Default weight |
|---|---|---|
| loudness | RMS level (dB) | 0.30 |
| onset_rate | onsets per second (event density) | 0.25 |
| tempo | BPM from onset-envelope autocorrelation | 0.20 |
| flux | mean spectral flux | 0.15 |
| brightness | spectral centroid (Hz) | 0.10 |
| pulse | beat clarity, 0–1 | 0 (plotted only) |

Each feature is robust-z-scored (median/IQR), weighted, and rescaled to 0–100.
The weights are a heuristic informed by music-perception research, not fitted
to listener ratings.

**Known limitation:** on zouk the tempo estimate can flip between ~75 and
~100 BPM (a 3:4 ambiguity from the syncopated rhythm). Onset rate and loudness
are the more reliable components.
