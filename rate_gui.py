#!/usr/bin/env python3
"""Collect human energy ratings for random short clips (20 s by default).

Serves a small page in your browser: press Start, listen to a random clip
from a random set in AUDIO_DIR, rate its energy 0-10, Submit. Each rating is
appended to a CSV together with the clip's raw feature stats from
energy_curve.py, plus the file and offset so features can be recomputed later.

Usage:
  .venv/bin/python rate_gui.py ..            # audio files live one level up
  .venv/bin/python rate_gui.py .. --out ratings.csv --port 8765
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from energy_curve import FEATURES, summarize
from webutil import send_html, send_json, serve_file

AUDIO_EXTS = {".mp3", ".m4a", ".opus", ".ogg", ".wav", ".flac", ".aac"}
CSV_COLS = ["timestamp", "file", "start_s", "duration_s", "rating"] + FEATURES


def probe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def focus_pool(audio_dir, clip_s, model, index, direction, top, frac):
    """Index windows whose predicted energy is in the top (or bottom) fraction."""
    import numpy as np
    from energy_curve import add_derived, load_model, predict

    model = load_model(model)
    rows = [r for r in csv.DictReader(open(index))
            if os.path.exists(os.path.join(audio_dir, r["file"]))]
    if not rows:
        sys.exit(f"no rows in {index} match files in {audio_dir}")
    cols = {k: [float(r[k]) for r in rows] for k in rows[0] if k not in ("file", "start_s")}
    cols = add_derived(cols)
    missing = set(model["features"]) - set(cols)
    if missing:
        sys.exit(f"{index} lacks {sorted(missing)}; rebuild it with index_features.py")
    pred = predict(model, cols)
    if direction == "low":
        pred = -pred
    ok = np.isfinite(pred)  # silent windows have no brightness/tempo
    cut = np.quantile(pred[ok], 1 - top)
    return [(os.path.join(audio_dir, r["file"]), float(r["start_s"]))
            for r, p, good in zip(rows, pred, ok) if good and p >= cut]


class Rater:
    def __init__(self, audio_dir, out_csv, clip_s, focus=None):
        self.out_csv = out_csv
        self.clip_s = clip_s
        self.tmp = tempfile.mkdtemp(prefix="rate_gui_")
        self.clips = {}
        self.lock = threading.Lock()
        self.sources = []
        for name in sorted(os.listdir(audio_dir)):
            path = os.path.join(audio_dir, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in AUDIO_EXTS:
                dur = probe_duration(path)
                if dur > clip_s:
                    self.sources.append((path, dur))
        if not self.sources:
            sys.exit(f"no audio files longer than {clip_s:.0f}s in {audio_dir}")
        print(f"found {len(self.sources)} audio files "
              f"({sum(d for _, d in self.sources) / 3600:.1f} h total)")
        self.focus_pool, self.focus_frac = [], 0.0
        if focus:
            self.focus_pool = focus_pool(audio_dir, clip_s, **focus)
            self.focus_frac = focus["frac"]
            print(f"focus: {self.focus_frac:.0%} of clips from {len(self.focus_pool)} windows "
                  f"predicted {focus['direction']}-energy")

    def n_ratings(self):
        if not os.path.exists(self.out_csv):
            return 0
        with open(self.out_csv) as f:
            return max(0, sum(1 for _ in f) - 1)

    def next_clip(self, discard=None):
        self.drop(discard)
        if self.focus_pool and random.random() < self.focus_frac:
            path, start = random.choice(self.focus_pool)
        else:
            # Weight by duration so every second of audio is equally likely.
            path, dur = random.choices(self.sources, weights=[d for _, d in self.sources])[0]
            start = random.uniform(0, dur - self.clip_s)
        cid = uuid.uuid4().hex[:12]
        wav = os.path.join(self.tmp, cid + ".wav")
        subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-t", str(self.clip_s),
                        "-i", path, "-vn", "-ac", "2", "-ar", "44100", "-y", wav], check=True)
        clip = {"path": path, "start": start, "wav": wav, "stats": None}
        clip["thread"] = threading.Thread(target=self._stats, args=(clip,), daemon=True)
        clip["thread"].start()
        with self.lock:
            self.clips[cid] = clip
        return cid

    def _stats(self, clip):
        clip["stats"] = summarize(clip["wav"])

    def submit(self, cid, rating):
        with self.lock:
            clip = self.clips.get(cid)
        if clip is None:
            raise KeyError("unknown clip")
        clip["thread"].join()
        row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
               "file": os.path.basename(clip["path"]),
               "start_s": f"{clip['start']:.2f}",
               "duration_s": self.clip_s,
               "rating": rating}
        row.update({k: f"{v:.4f}" for k, v in clip["stats"].items()})
        with self.lock:
            new = not os.path.exists(self.out_csv)
            # Keep an existing file's columns; fit_energy.py recomputes features anyway.
            cols = CSV_COLS if new else next(csv.reader(open(self.out_csv)))
            with open(self.out_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(row)
        self.drop(cid)
        return row

    def drop(self, cid):
        with self.lock:
            clip = self.clips.pop(cid, None) if cid else None
        if clip:
            clip["thread"].join()
            try:
                os.remove(clip["wav"])
            except OSError:
                pass

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Energy Rater</title>
<style>
:root { --bg:#f6f7f9; --card:#fff; --ink:#1f2328; --muted:#656d76; --line:#d0d7de;
        --accent:#c2410c; --accent-ink:#fff; --err:#b42318; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d1117; --card:#161b22; --ink:#e6edf3; --muted:#8d96a0; --line:#30363d;
          --accent:#f0883e; --accent-ink:#0d1117; --err:#ff7b72; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width:520px; margin:12vh auto; padding:0 16px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:28px; }
h1 { font-size:20px; margin:0 0 4px; }
.sub { color:var(--muted); font-size:14px; margin:0 0 24px; }
audio { width:100%; margin-bottom:20px; }
label { display:block; font-size:14px; color:var(--muted); margin-bottom:6px; }
.row { display:flex; gap:8px; }
input { flex:1; font:inherit; font-size:20px; padding:10px 12px; border-radius:8px;
        border:1px solid var(--line); background:var(--bg); color:var(--ink); min-width:0; }
button { font:inherit; padding:10px 18px; border-radius:8px; cursor:pointer;
         border:1px solid var(--line); background:var(--card); color:var(--ink); }
button.primary { background:var(--accent); color:var(--accent-ink); border-color:var(--accent); font-weight:600; }
button:disabled { opacity:.5; cursor:default; }
#start { width:100%; font-size:18px; padding:14px; }
.secondary { display:flex; gap:8px; margin-top:12px; }
.secondary button { flex:1; font-size:14px; padding:8px; }
#status { margin-top:18px; font-size:14px; color:var(--muted); min-height:1.5em; }
#status.err { color:var(--err); }
.hidden { display:none; }
</style></head>
<body><main><div class="card">
  <h1>Energy rater</h1>
  <p class="sub">Rate how fast or intense each clip feels, from 0 (lowest) to 10 (highest).
     <span id="count"></span></p>
  <button id="start" class="primary">Start</button>
  <div id="rater" class="hidden">
    <audio id="player" controls autoplay></audio>
    <label for="rating">Energy rating (0–10)</label>
    <div class="row">
      <input id="rating" type="number" min="0" max="10" step="any" inputmode="decimal" autocomplete="off">
      <button id="submit" class="primary">Submit</button>
    </div>
    <div class="secondary">
      <button id="replay">Replay</button>
      <button id="skip">Skip clip</button>
    </div>
  </div>
  <div id="status" aria-live="polite"></div>
</div></main>
<script>
const $ = id => document.getElementById(id);
let clipId = null, busy = false;

function status(msg, err=false) { $("status").textContent = msg; $("status").className = err ? "err" : ""; }
function setCount(n) { $("count").textContent = n + " rating" + (n === 1 ? "" : "s") + " so far."; }
function setBusy(b) { busy = b; for (const id of ["submit","skip","replay","start"]) $(id).disabled = b; }

async function api(path, body) {
  const r = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"},
                               body: JSON.stringify(body || {})});
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

async function loadNext(prefix="") {
  setBusy(true);
  status(prefix + "Loading a random clip…");
  try {
    const j = await api("/api/next", {discard: clipId});
    clipId = j.clip_id;
    $("player").src = "/clip/" + clipId + ".wav";
    $("player").play().catch(() => {});
    $("rating").value = "";
    $("rating").focus();
    status(prefix);
  } catch (e) { status("Couldn't load a clip: " + e.message, true); }
  setBusy(false);
}

async function submit() {
  if (busy || !clipId) return;
  const v = $("rating").value.trim(), x = Number(v);
  if (v === "" || !Number.isFinite(x) || x < 0 || x > 10) {
    status("Enter a number from 0 to 10.", true); $("rating").focus(); return;
  }
  setBusy(true);
  status("Saving…");
  try {
    const j = await api("/api/submit", {clip_id: clipId, rating: x});
    clipId = null;
    $("player").pause();
    setCount(j.n_ratings);
    await loadNext(`Saved ${x} for ${j.file} @ ${j.at}. `);
  } catch (e) { status("Couldn't save: " + e.message, true); setBusy(false); }
}

$("start").onclick = () => { $("start").classList.add("hidden"); $("rater").classList.remove("hidden"); loadNext(); };
$("submit").onclick = submit;
$("rating").addEventListener("keydown", e => { if (e.key === "Enter") submit(); });
$("skip").onclick = () => { if (!busy) { $("player").pause(); loadNext("Skipped. "); } };
$("replay").onclick = () => { $("player").currentTime = 0; $("player").play(); };
fetch("/api/count").then(r => r.json()).then(j => setCount(j.n_ratings));
</script></body></html>
"""


def make_handler(rater):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            if self.path == "/":
                send_html(self, PAGE)
            elif self.path == "/api/count":
                send_json(self, {"n_ratings": rater.n_ratings()})
            elif m := re.fullmatch(r"/clip/([0-9a-f]+)\.wav", self.path):
                clip = rater.clips.get(m[1])
                if clip is None:
                    self.send_error(404)
                    return
                serve_file(self, clip["wav"], "audio/wav")
            else:
                self.send_error(404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/api/next":
                    send_json(self, {"clip_id": rater.next_clip(body.get("discard"))})
                elif self.path == "/api/submit":
                    rating = float(body["rating"])
                    if not 0 <= rating <= 10:
                        raise ValueError("rating must be between 0 and 10")
                    row = rater.submit(body["clip_id"], rating)
                    s = float(row["start_s"])
                    send_json(self, {"n_ratings": rater.n_ratings(), "file": row["file"],
                                    "at": f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}"})
                else:
                    self.send_error(404)
            except (KeyError, ValueError, subprocess.CalledProcessError) as e:
                send_json(self, {"error": str(e)}, 400)

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio_dir", nargs="?", default=".")
    p.add_argument("--out", default="ratings.csv")
    p.add_argument("--clip", type=float, default=20.0, help="clip length in seconds (default 20)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--focus", choices=["high", "low"],
                   help="oversample clips the model predicts are high/low energy")
    p.add_argument("--focus-top", type=float, default=0.15,
                   help="focus pool = this fraction of index windows (default 0.15)")
    p.add_argument("--focus-frac", type=float, default=0.7,
                   help="share of clips drawn from the focus pool; rest are random (default 0.7)")
    p.add_argument("--model", default="energy_model.json")
    p.add_argument("--index", default="feature_index.csv")
    args = p.parse_args()

    focus = None
    if args.focus:
        focus = {"model": args.model, "index": args.index, "direction": args.focus,
                 "top": args.focus_top, "frac": args.focus_frac}
    rater = Rater(args.audio_dir, args.out, args.clip, focus)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(rater))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"rating server at {url}  (writing to {args.out}; Ctrl-C to quit)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        rater.cleanup()


if __name__ == "__main__":
    main()
