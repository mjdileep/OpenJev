"""Watch Laya play 2048: python examples/laya_2048/server.py."""

from __future__ import annotations

import argparse
import json
import mimetypes
import platform
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from laya_player import GAME, MODEL, Player

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    status = {
        "ready": False,
        "model": MODEL,
        "hardware": platform.machine(),
        "error": None,
        "export_filename": "laya-2048-run.json",
    }
    player = None
    busy = threading.Lock()
    log_dir = Path(".cache/laya_2048")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"decisions-{time.time_ns()}.jsonl"

    def load():
        nonlocal player
        try:
            player = Player(args.device)
            # Warm the exact path before reporting Ready.
            player.decide([[2, 2, 0, 0], [0, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
            status.update(**player.metadata, ready=True)
            print("Laya ready on " + str(player.agent.device), flush=True)
        except Exception as exc:
            traceback.print_exc()
            status["error"] = str(exc)

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, content_type="application/json"):
            data = json.dumps(body).encode() if content_type == "application/json" else body
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/api/status":
                return self.send(200, status)
            name = "index.html" if self.path == "/" else self.path.lstrip("/")
            allowed = {
                "index.html",
                "app.js",
                "style.css",
                "vendor/game_manager.js",
                "vendor/grid.js",
                "vendor/tile.js",
                "vendor/LICENSE.txt",
            }
            if name not in allowed:
                return self.send(404, {"error": "Not found"})
            path = ROOT / name if name == "index.html" else GAME / name
            self.send(200, path.read_bytes(), mimetypes.guess_type(name)[0] or "text/plain")

        def do_POST(self):
            hosts = {f"127.0.0.1:{args.port}", f"localhost:{args.port}"}
            if self.headers.get("Host") not in hosts:
                return self.send(403, {"error": "Local requests only"})
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{host}" for host in hosts}:
                return self.send(403, {"error": "Same-origin requests only"})
            if self.path != "/api/decide":
                return self.send(404, {"error": "Not found"})
            if not status["ready"]:
                return self.send(503, {"error": status["error"] or "Model is loading"})
            if not busy.acquire(blocking=False):
                return self.send(409, {"error": "A decision is already in progress"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Invalid request size")
                response = player.decide(json.loads(self.rfile.read(length))["board"])
                with log_path.open("a") as file:
                    file.write(json.dumps(response) + "\n")
                self.send(200, response)
            except (ValueError, KeyError, TypeError) as exc:
                self.send(400, {"error": str(exc)})
            except Exception as exc:
                self.send(500, {"error": str(exc)})
            finally:
                busy.release()

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=load, daemon=True).start()
    print(f"Laya experiment: http://127.0.0.1:{args.port} · {log_path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
