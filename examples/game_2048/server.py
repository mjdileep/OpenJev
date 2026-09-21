"""Run from the repository root: python examples/game_2048/server.py."""

from __future__ import annotations

import argparse
import json
import mimetypes
import platform
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from game import question_for

from openjev import DecisionEngine

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(
        description="Watch OpenJev play the original 2048 game locally"
    )
    parser.add_argument("--model", default=None, help="Hugging Face repo; default: Qwen3.5-0.8B")
    parser.add_argument("--revision")
    parser.add_argument(
        "--backend", default="auto", choices=["auto", "mlx", "transformers", "gguf"]
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    revision = args.revision
    status = {
        "ready": False,
        "model": args.model or "Qwen3.5-0.8B",
        "revision": revision,
        "hardware": f"{platform.system()} · {platform.machine()}",
        "error": None,
    }
    engine = None
    busy = threading.Lock()
    log_dir = Path(".cache/game_2048")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"decisions-{time.time_ns()}.jsonl"

    def load():
        nonlocal engine
        try:
            print(f"Loading {status['model']}…", flush=True)
            engine = DecisionEngine.from_pretrained(
                args.model,
                revision=revision,
                backend=args.backend,
                device=args.device,
                batch_size=args.batch_size,
                n_ctx=4096,
            )
            if engine.backend.name == "mlx":
                engine.backend.mx.set_cache_limit(512 * 1024**2)
            status.update(ready=True, backend=engine.backend.name, model=engine.backend.model_id)
            print("Model ready. Click Play in the browser.", flush=True)
        except Exception as exc:
            status["error"] = str(exc)
            print(f"Model failed to load: {exc}", flush=True)

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
            return self.send(
                200, (ROOT / name).read_bytes(), mimetypes.guess_type(name)[0] or "text/plain"
            )

        def do_POST(self):
            host = f"127.0.0.1:{args.port}"
            if self.headers.get("Host") not in {host, f"localhost:{args.port}"}:
                return self.send(403, {"error": "Local requests only"})
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{host}", f"http://localhost:{args.port}"}:
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
                payload = json.loads(self.rfile.read(length))
                state, questions, forced = question_for(payload["board"])
                result = engine.decide(state, questions)
                answer = result.answers["move"]
                action = forced or answer["choice"]
                response = {
                    "action": action,
                    "forced": bool(forced),
                    "probabilities": {forced: 1.0} if forced else answer["probabilities"],
                    "usage": asdict(result.usage),
                    "result": result.to_dict(),
                    "board": payload["board"],
                    "timestamp": time.time(),
                }
                if engine.backend.name == "mlx":
                    response["peak_memory_gb"] = engine.backend.mx.get_peak_memory() / 1e9
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
    print(f"Open http://127.0.0.1:{args.port} · decision log: {log_path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    main()
