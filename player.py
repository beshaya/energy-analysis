#!/usr/bin/env python3
"""Play a set in the browser with its energy curve and a moving playhead.

Pick any audio file in AUDIO_DIR; its energy curve is computed on first use
(about a minute per hour of audio) and cached in .player_cache/. Click or drag
on the chart to seek.

Usage:
  .venv/bin/python player.py ..            # audio files live one level up
  .venv/bin/python player.py .. --smooth 60 --port 8766
"""

import argparse
import json
import os
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import energy_curve
from energy_curve import (DEFAULT_WEIGHTS, LABELS, PLAIN_FEATURES, add_energy,
                          load_model, window_features)
from rate_gui import AUDIO_EXTS
from webutil import send_html, send_json, serve_file

HOP_S = 5.0
CONTENT_TYPES = {".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".aac": "audio/aac",
                 ".opus": "audio/ogg", ".ogg": "audio/ogg", ".wav": "audio/wav",
                 ".flac": "audio/flac"}


class Curves:
    """Computes energy curves in the background and caches them on disk."""

    def __init__(self, audio_dir, model_path, smooth_s, cache_dir):
        self.audio_dir = audio_dir
        self.model_path = model_path
        self.smooth_s = smooth_s
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.files = sorted(n for n in os.listdir(audio_dir)
                            if os.path.splitext(n)[1].lower() in AUDIO_EXTS)
        self.jobs = {}
        self.lock = threading.Lock()

    def path(self, i):
        return os.path.join(self.audio_dir, self.files[i])

    def _cache_file(self, i):
        return os.path.join(self.cache_dir, self.files[i] + ".json")

    def _model_stamp(self):
        # Invalidate on a refit and on any change to the analysis code.
        model = os.path.getmtime(self.model_path) if os.path.exists(self.model_path) else None
        return [model, os.path.getmtime(energy_curve.__file__)]

    def cached(self, i):
        try:
            with open(self._cache_file(i)) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if data.get("model_mtime") != self._model_stamp() or data.get("smooth_s") != self.smooth_s:
            return None
        return data

    def get(self, i):
        """Return ("ready", data) or ("running", None) or ("error", message)."""
        data = self.cached(i)
        if data:
            return "ready", data
        with self.lock:
            job = self.jobs.get(i)
            if job is None or (job["status"] == "error" and not job["thread"].is_alive()):
                job = {"status": "running", "error": None}
                job["thread"] = threading.Thread(target=self._compute, args=(i, job), daemon=True)
                self.jobs[i] = job
                job["thread"].start()
        if job["status"] == "error":
            return "error", job["error"]
        return "running", None

    def _compute(self, i, job):
        try:
            model = load_model(self.model_path) if os.path.exists(self.model_path) else None
            d = window_features(self.path(i), 20.0, HOP_S)
            add_energy([d], DEFAULT_WEIGHTS, self.smooth_s, HOP_S, model)
            if model:
                top = sorted(zip(model["features"], model["coef"]), key=lambda t: -abs(t[1]))
                feats = [f for f, _ in top[:6]]
            else:
                feats = PLAIN_FEATURES

            def clean(a):
                return [None if not np.isfinite(v) else round(float(v), 3) for v in a]

            data = {"model_mtime": self._model_stamp(), "smooth_s": self.smooth_s,
                    "model": bool(model), "time_s": clean(d["time_s"]),
                    "energy": clean(d["energy"]),
                    "features": [{"name": f, "label": LABELS[f], "values": clean(d[f])}
                                 for f in feats]}
            with open(self._cache_file(i), "w") as f:
                json.dump(data, f)
            job["status"] = "done"
        except BaseException as e:  # ffmpeg failure calls sys.exit
            job["status"], job["error"] = "error", str(e) or type(e).__name__


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Energy Player</title>
<style>
:root { --bg:#f6f7f9; --card:#fff; --ink:#1f2328; --ink2:#424a53; --muted:#656d76;
        --line:#d0d7de; --grid:#e6e8eb; --accent:#c2410c; --series:#2563eb;
        --cursor:#1f2328; --err:#b42318; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d1117; --card:#161b22; --ink:#e6edf3; --ink2:#c9d1d9; --muted:#8d96a0;
          --line:#30363d; --grid:#262c34; --accent:#f0883e; --series:#58a6ff;
          --cursor:#e6edf3; --err:#ff7b72; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width:1200px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:20px; margin:0 0 16px; }
.controls { display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin-bottom:16px; }
select { font:inherit; padding:8px 10px; border-radius:8px; border:1px solid var(--line);
         background:var(--card); color:var(--ink); max-width:100%; flex:1 1 320px; }
label.toggle { display:flex; gap:6px; align-items:center; color:var(--ink2); font-size:14px; white-space:nowrap; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; }
audio { width:100%; margin-bottom:12px; }
.readout { display:flex; gap:24px; align-items:baseline; margin:0 0 8px; flex-wrap:wrap; }
.readout .big { font-size:28px; font-weight:600; font-variant-numeric:tabular-nums; }
.readout .small { color:var(--muted); font-size:14px; font-variant-numeric:tabular-nums; }
#chart { width:100%; display:block; cursor:pointer; touch-action:none; user-select:none; }
#chart text { fill:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }
#chart .grid { stroke:var(--grid); stroke-width:1; }
#chart .energy { stroke:var(--accent); stroke-width:2; fill:none; }
#chart .area { fill:var(--accent); opacity:.12; }
#chart .feat { stroke:var(--series); stroke-width:1.5; fill:none; }
#chart .cursor { stroke:var(--cursor); stroke-width:1.5; }
#chart .hover { stroke:var(--muted); stroke-width:1; stroke-dasharray:3 3; }
#chart .dot { fill:var(--accent); stroke:var(--card); stroke-width:2; }
#chart .plabel { fill:var(--ink2); font-size:12px; }
#tip { position:fixed; pointer-events:none; background:var(--card); color:var(--ink);
       border:1px solid var(--line); border-radius:6px; padding:4px 8px; font-size:12px;
       font-variant-numeric:tabular-nums; box-shadow:0 2px 8px rgba(0,0,0,.12); display:none; }
#status { color:var(--muted); font-size:14px; min-height:1.5em; margin:8px 0 0; }
#status.err { color:var(--err); }
.hint { color:var(--muted); font-size:13px; margin-top:8px; }
</style></head>
<body><main>
  <h1>Energy player</h1>
  <div class="controls">
    <select id="file" aria-label="Set"><option value="">Choose a set…</option></select>
    <label class="toggle"><input type="checkbox" id="showFeat"> Show features</label>
  </div>
  <div class="card">
    <audio id="audio" controls preload="metadata"></audio>
    <div class="readout">
      <div><span class="big" id="nowEnergy">–</span> <span class="small">energy</span></div>
      <div class="small" id="nowTime">0:00:00</div>
    </div>
    <svg id="chart" role="img" aria-label="Energy over time"></svg>
    <div id="status" aria-live="polite"></div>
    <div class="hint">Click or drag on the chart to seek · Space play/pause · ←/→ 10 s</div>
  </div>
  <div id="tip"></div>
</main>
<script>
const $ = id => document.getElementById(id);
const audio = $("audio"), svg = $("chart"), tip = $("tip");
const NS = "http://www.w3.org/2000/svg";
let data = null, fileIdx = null, geom = null, pollTimer = null, dragging = false;

const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} },
};
function status(msg, err=false) { $("status").textContent = msg; $("status").className = err ? "err" : ""; }
function hms(s) {
  s = Math.max(0, Math.floor(s));
  return `${Math.floor(s / 3600)}:${String(Math.floor(s % 3600 / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}
function hm(s) { const m = Math.round(s / 60); return `${Math.floor(m / 60)}:${String(m % 60).padStart(2, "0")}`; }
function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  (parent || svg).appendChild(e);
  return e;
}
function valueAt(arr, t) {
  const ts = data.time_s;
  if (!ts.length) return null;
  let lo = 0, hi = ts.length - 1;
  if (t <= ts[0]) return arr[0];
  if (t >= ts[hi]) return arr[hi];
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; (ts[mid] <= t ? lo = mid : hi = mid); }
  const a = arr[lo], b = arr[hi];
  if (a == null || b == null) return a ?? b;
  return a + (b - a) * (t - ts[lo]) / (ts[hi] - ts[lo]);
}

function pathFor(xs, ys, x, y) {
  let d = "", pen = false;
  for (let i = 0; i < xs.length; i++) {
    if (ys[i] == null) { pen = false; continue; }
    d += (pen ? "L" : "M") + x(xs[i]).toFixed(1) + "," + y(ys[i]).toFixed(1);
    pen = true;
  }
  return d;
}

function draw() {
  svg.replaceChildren();
  if (!data) { svg.setAttribute("height", 0); return; }
  const W = svg.clientWidth || 800, padL = 56, padR = 12, mainH = 260, featH = 64, gap = 14;
  const feats = $("showFeat").checked ? data.features : [];
  const H = 8 + mainH + feats.length * (featH + gap) + 28;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("height", H);
  const dur = audio.duration && isFinite(audio.duration) ? audio.duration
            : (data.time_s[data.time_s.length - 1] || 1) + 10;
  const x = t => padL + (W - padL - padR) * t / dur;
  geom = { padL, padR, W, H, dur, x, top: 8, mainH };

  // Main energy panel, fixed 0-10 scale.
  const y = v => 8 + mainH - mainH * v / 10;
  for (let v = 0; v <= 10; v += 2) {
    el("line", {x1: padL, x2: W - padR, y1: y(v), y2: y(v), class: "grid"});
    el("text", {x: padL - 8, y: y(v) + 4, "text-anchor": "end"}).textContent = v;
  }
  const line = pathFor(data.time_s, data.energy, x, y);
  const first = data.time_s[0], last = data.time_s[data.time_s.length - 1];
  el("path", {d: line + `L${x(last)},${y(0)}L${x(first)},${y(0)}Z`, class: "area"});
  el("path", {d: line, class: "energy"});

  // Feature small multiples, each on its own scale.
  let top = 8 + mainH + gap;
  for (const f of feats) {
    const vals = f.values.filter(v => v != null);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (hi === lo) { hi += 1; lo -= 1; }
    const fy = v => top + featH - featH * (v - lo) / (hi - lo);
    el("line", {x1: padL, x2: W - padR, y1: top + featH, y2: top + featH, class: "grid"});
    el("path", {d: pathFor(data.time_s, f.values, x, fy), class: "feat"});
    el("text", {x: padL + 4, y: top + 11, class: "plabel"}).textContent = f.label;
    el("text", {x: padL - 8, y: top + 10, "text-anchor": "end"}).textContent = +hi.toPrecision(3);
    el("text", {x: padL - 8, y: top + featH, "text-anchor": "end"}).textContent = +lo.toPrecision(3);
    top += featH + gap;
  }

  // Time axis.
  const steps = [60, 120, 300, 600, 900, 1800, 3600];
  const step = steps.find(s => (W - padL - padR) / (dur / s) >= 70) || 3600;
  for (let t = 0; t <= dur; t += step) {
    el("line", {x1: x(t), x2: x(t), y1: 8, y2: H - 24, class: "grid"});
    el("text", {x: x(t), y: H - 8, "text-anchor": "middle"}).textContent = hm(t);
  }

  geom.hover = el("line", {y1: 8, y2: H - 24, class: "hover", visibility: "hidden"});
  geom.cursor = el("line", {y1: 8, y2: H - 24, class: "cursor"});
  geom.dot = el("circle", {r: 5, class: "dot"});
  geom.y = y;
  updateCursor();
}

function updateCursor() {
  if (!geom || !data) return;
  const t = audio.currentTime || 0, cx = geom.x(t);
  geom.cursor.setAttribute("x1", cx); geom.cursor.setAttribute("x2", cx);
  const e = valueAt(data.energy, t);
  if (e != null) {
    geom.dot.setAttribute("cx", cx); geom.dot.setAttribute("cy", geom.y(e));
    $("nowEnergy").textContent = e.toFixed(1);
  }
  $("nowTime").textContent = hms(t) + (isFinite(audio.duration) ? " / " + hms(audio.duration) : "");
}

function tick() { updateCursor(); if (!audio.paused) requestAnimationFrame(tick); }
audio.addEventListener("play", () => requestAnimationFrame(tick));
audio.addEventListener("seeked", updateCursor);
audio.addEventListener("loadedmetadata", () => {
  const saved = parseFloat(store.get("pos:" + fileIdx));
  if (saved > 0 && saved < audio.duration) audio.currentTime = saved;
  draw();
});
audio.addEventListener("error", () => status("This browser can't play this file's format.", true));
setInterval(() => { if (fileIdx != null && !audio.paused) store.set("pos:" + fileIdx, audio.currentTime); }, 2000);

function timeAtEvent(ev) {
  const r = svg.getBoundingClientRect();
  const px = (ev.clientX - r.left) * geom.W / r.width;
  return Math.min(geom.dur, Math.max(0, geom.dur * (px - geom.padL) / (geom.W - geom.padL - geom.padR)));
}
svg.addEventListener("pointerdown", ev => {
  if (!geom) return;
  dragging = true; svg.setPointerCapture(ev.pointerId);
  audio.currentTime = timeAtEvent(ev); updateCursor();
});
svg.addEventListener("pointermove", ev => {
  if (!geom) return;
  const t = timeAtEvent(ev), hx = geom.x(t);
  if (dragging) { audio.currentTime = t; updateCursor(); }
  geom.hover.setAttribute("x1", hx); geom.hover.setAttribute("x2", hx);
  geom.hover.setAttribute("visibility", "visible");
  const e = valueAt(data.energy, t);
  tip.textContent = `${hms(t)} · energy ${e == null ? "–" : e.toFixed(1)}`;
  tip.style.display = "block";
  const tw = tip.offsetWidth;
  tip.style.left = Math.min(window.innerWidth - tw - 8, ev.clientX + 12) + "px";
  tip.style.top = (ev.clientY - 32) + "px";
});
svg.addEventListener("pointerup", () => { dragging = false; });
svg.addEventListener("pointerleave", () => {
  tip.style.display = "none";
  if (geom) geom.hover.setAttribute("visibility", "hidden");
});

document.addEventListener("keydown", ev => {
  if (ev.target.tagName === "SELECT" || ev.target.tagName === "INPUT") return;
  if (ev.key === " ") { ev.preventDefault(); audio.paused ? audio.play() : audio.pause(); }
  else if (ev.key === "ArrowRight") { audio.currentTime = Math.min(audio.duration || 0, audio.currentTime + 10); }
  else if (ev.key === "ArrowLeft") { audio.currentTime = Math.max(0, audio.currentTime - 10); }
});

async function loadCurve(i) {
  clearTimeout(pollTimer);
  try {
    const r = await fetch("/api/curve/" + i);
    const j = await r.json();
    if (i !== fileIdx) return;
    if (j.status === "ready") {
      data = j.data;
      status(data.model ? "" : "No fitted model found; showing a relative hand-weighted blend.");
      draw();
    } else if (j.status === "running") {
      status("Analyzing this set for the first time (about a minute per hour of audio)…");
      pollTimer = setTimeout(() => loadCurve(i), 2000);
    } else {
      status("Analysis failed: " + j.error, true);
    }
  } catch (e) { status("Couldn't load the energy curve: " + e.message, true); }
}

$("file").addEventListener("change", () => {
  const v = $("file").value;
  if (fileIdx != null) store.set("pos:" + fileIdx, audio.currentTime);
  data = null; geom = null; draw();
  $("nowEnergy").textContent = "–";
  if (v === "") { fileIdx = null; audio.removeAttribute("src"); audio.load(); return; }
  fileIdx = +v;
  store.set("lastFile", v);
  audio.src = "/audio/" + fileIdx;
  loadCurve(fileIdx);
});
$("showFeat").addEventListener("change", () => { store.set("showFeat", $("showFeat").checked ? "1" : ""); draw(); });
window.addEventListener("resize", draw);

fetch("/api/files").then(r => r.json()).then(j => {
  j.files.forEach((name, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = name; $("file").appendChild(o);
  });
  $("showFeat").checked = store.get("showFeat") === "1";
  const last = store.get("lastFile");
  if (last != null && last !== "" && +last < j.files.length) {
    $("file").value = last; $("file").dispatchEvent(new Event("change"));
  }
});
</script></body></html>
"""


def make_handler(curves):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            if self.path == "/":
                send_html(self, PAGE)
            elif self.path == "/api/files":
                send_json(self, {"files": curves.files})
            elif m := re.fullmatch(r"/api/curve/(\d+)", self.path):
                i = int(m[1])
                if i >= len(curves.files):
                    self.send_error(404)
                    return
                status, payload = curves.get(i)
                key = "data" if status == "ready" else "error"
                send_json(self, {"status": status, key: payload})
            elif m := re.fullmatch(r"/audio/(\d+)", self.path):
                i = int(m[1])
                if i >= len(curves.files):
                    self.send_error(404)
                    return
                ext = os.path.splitext(curves.files[i])[1].lower()
                serve_file(self, curves.path(i), CONTENT_TYPES.get(ext, "application/octet-stream"))
            else:
                self.send_error(404)

    return Handler


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio_dir", nargs="?", default=".")
    p.add_argument("--model", default=os.path.join(here, "energy_model.json"))
    p.add_argument("--smooth", type=float, default=30.0,
                   help="smoothing of the energy curve in seconds (default 30)")
    p.add_argument("--cache", default=os.path.join(here, ".player_cache"))
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    curves = Curves(args.audio_dir, args.model, args.smooth, args.cache)
    if not curves.files:
        p.error(f"no audio files in {args.audio_dir}")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(curves))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"player at {url}  ({len(curves.files)} sets; Ctrl-C to quit)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
