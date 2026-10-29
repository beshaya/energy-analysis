# energy-analysis

Plot a perceived-"energy" curve over time for long DJ sets: a proxy for how
fast and intense the music feels to a dancer, learned from your own ratings.

The loop:

1. **Rate** random 20 s clips (`rate_gui.py`) → `ratings.csv`
2. **Fit** a model to the ratings (`fit_energy.py`) → `energy_model.json`
3. **Use** it: chart whole sets (`energy_curve.py`) or play a set with a live
   energy cursor (`player.py`)

All tools take the folder containing your audio files (here, `..`).

## Setup

Requires Python 3.10+ and `ffmpeg` on your PATH (used to decode mp3/m4a/opus/etc.).

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Rating clips

```
.venv/bin/python rate_gui.py ..
.venv/bin/python rate_gui.py .. --focus high   # oversample predicted high-energy clips
```

Opens a page that plays a random 20 s clip (any moment of any set is equally
likely; `--clip` changes the length) and asks for a 0–10 energy rating. Each
rating is appended to `ratings.csv` with the file name and offset, so features
can be recomputed for old ratings whenever the feature set changes.

`--focus high|low` draws 70% of clips (`--focus-frac`) from the top/bottom 15%
(`--focus-top`) of windows by predicted energy, using the index built by
`index_features.py ..` (slow, run once). The rest stay uniformly random, and
the page doesn't say which pool a clip came from.

## Fitting

```
.venv/bin/python fit_energy.py ..
```

Recomputes the current features for every rated clip (cached in
`rating_features.csv`), fits ridge regression on standardized features with
the penalty chosen by leave-one-out cross-validation, prints per-feature
correlations and weights, and writes `energy_model.json` plus a
predicted-vs-rated plot `fit.png`. `--features a,b,c` fits a subset.

## Charting and playing sets

```
.venv/bin/python energy_curve.py ../SET.mp3 [../SET2.m4a ...] [-o energy_plots]
.venv/bin/python player.py ..
```

`energy_curve.py` writes `<name>.png` (energy curve plus the model's six most
influential features) and `<name>.csv` per file. Options: `--smooth` (seconds,
default 60), `--energy-only`, `--max-bpm-per-min`, and `--weights` to ignore
the model and use a hand-weighted blend instead.

`player.py` opens a page to pick a set, play it, and watch the playhead move
along its energy curve. Click or drag the chart to seek; Space plays/pauses,
←/→ skip 10 s; "Show features" adds the model's top features. Curves are
computed on first open (about a minute per hour of audio) and cached in
`.player_cache/` until the model or analysis code changes.

## Features

Computed per 20 s window (`FEATURE_HELP` in `energy_curve.py`):

| Feature | Measure |
|---|---|
| `loudness` | RMS level (dB) |
| `brightness` | spectral centroid (Hz) |
| `flux` | mean spectral flux, all bands |
| `onset_rate` | onsets per second, all bands |
| `tempo` | beat rate (BPM), tracked smoothly across the whole file |
| `dance_tempo` | tempo as danced: halved above DANCE_TEMPO_MAX (half-time) |
| `pulse` | beat clarity 0-1 (autocorrelation peak height) |
| `hi_onset_rate` | onsets per second above 2 kHz (hats, shakers) |
| `lo_onset_rate` | onsets per second below 400 Hz (kick, bass) |
| `lo_flux` | spectral flux below 400 Hz |
| `mid_flux` | spectral flux 400 Hz-2 kHz |
| `hi_flux` | spectral flux above 2 kHz |
| `bal_lo` | level below 400 Hz relative to total (dB) |
| `bal_mid` | level 400 Hz-2 kHz relative to total (dB) |
| `bal_hi` | level above 2 kHz relative to total (dB) |
| `perc_ratio` | percussive share of energy (HPSS), 0-1 |
| `attack` | onset peakiness: 95th pct / mean flux |
| `choppiness` | loudness fluctuation faster than 1 Hz (dB SD) |
| `hat_rate` | fastest strong periodicity above 2 kHz (events/s) |

The model uses every feature except raw `tempo`, which is replaced by
`dance_tempo`; raw tempo is still plotted. Without `energy_model.json`, energy
falls back to a hand-weighted blend of loudness, onset rate, flux, tempo and
brightness, scaled 0–10 relative to the files in that run.

**Tempo tracking.** Each window scores every BPM from 60 to 140 by onset
autocorrelation at one beat, two beats and one bar (the bar term separates the
true beat from the dotted-quarter pulse in zouk's 3-3-2 rhythm). A Viterbi pass
then picks the best path that ramps at most about 10 BPM/min, which removes
the "square wave" a per-window estimate produces. Above 95 BPM
(`DANCE_TEMPO_MAX`) dancers usually move at half time, so `dance_tempo` halves
it there.
