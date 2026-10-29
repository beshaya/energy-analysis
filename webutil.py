"""Small helpers shared by the local web UIs (rate_gui.py, player.py)."""

import json
import os
import re

CHUNK = 1 << 20


def send_json(handler, obj, code=200):
    body = json.dumps(obj).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def send_html(handler, page):
    body = page.encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def serve_file(handler, path, ctype):
    """Stream a file with HTTP Range support. Safari won't play <audio> without
    ranges, and they make seeking work."""
    size = os.path.getsize(path)
    start, end = 0, size - 1
    m = re.fullmatch(r"bytes=(\d*)-(\d*)", handler.headers.get("Range", ""))
    if m and (m[1] or m[2]):
        if m[1]:
            start = int(m[1])
            end = min(int(m[2]), size - 1) if m[2] else size - 1
        else:
            start = max(0, size - int(m[2]))
        handler.send_response(206)
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    else:
        handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(end - start + 1))
    handler.end_headers()
    with open(path, "rb") as f:
        f.seek(start)
        left = end - start + 1
        try:
            while left > 0:
                chunk = f.read(min(CHUNK, left))
                if not chunk:
                    break
                handler.wfile.write(chunk)
                left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
