from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from .engine import DecisionEngine
from .types import ModelConfig


def build_parser():
    parser = argparse.ArgumentParser(description="Typed decisions from local token probabilities")
    parser.add_argument("command", choices=["run", "benchmark"])
    parser.add_argument("request", type=Path, help="JSON file with state and questions")
    parser.add_argument(
        "--backend", choices=["auto", "mlx", "gguf", "transformers"], default="auto"
    )
    parser.add_argument("--model", help="Hugging Face repository ID or local model path")
    parser.add_argument("--revision", help="Optional model commit SHA or tag")
    parser.add_argument("--tokenizer", help="Matching HF tokenizer; required for custom GGUFs")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--filename", help="Exact GGUF filename within the HF repository")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "metal"], default="auto")
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument("--score-mode", choices=["full", "binary"], default="full")
    parser.add_argument(
        "--image", action="append", default=[], help="Local image; repeat for multiple"
    )
    parser.add_argument("--vision", action="store_true", help="Load image-capable model components")
    parser.add_argument(
        "--load-in-4bit", action="store_true", help="CUDA Transformers NF4 quantization"
    )
    parser.add_argument("--no-head-optimization", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--iterations", type=int, default=3, help="Measured benchmark iterations")
    return parser


def compare_results(cached, reference):
    differences = []
    same_choices = True
    for key, answer in cached.answers.items():
        other = reference.answers[key]
        if answer.get("choice") != other.get("choice"):
            same_choices = False
        for label, candidate in answer["candidates"].items():
            differences.append(abs(candidate["support"] - other["candidates"][label]["support"]))
    return {"max_candidate_support_difference": max(differences), "same_choices": same_choices}


def benchmark(engine, request, images, iterations):
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    params = dict(state=request["state"], questions=request["questions"], images=images)
    # Warm both execution shapes before measurement. Alternate their order to
    # reduce systematic warm-cache or thermal bias; exclude model load time.
    engine.decide(**params)
    engine.decide(**params, use_cache=False)
    times = {True: [], False: []}
    results = {}
    checks = []
    for iteration in range(iterations):
        order = (True, False) if iteration % 2 == 0 else (False, True)
        for cached in order:
            results[cached] = engine.decide(**params, use_cache=cached)
            times[cached].append(results[cached].usage.elapsed_seconds)
        checks.append(compare_results(results[True], results[False]))
    cached_time, uncached_time = (statistics.median(times[x]) for x in (True, False))
    return {
        "model": results[True].model,
        "backend": engine.backend.name,
        "score_mode": engine.config.score_mode,
        "images": len(images),
        "iterations": iterations,
        "cached_median_seconds": cached_time,
        "uncached_median_seconds": uncached_time,
        "observed_speedup": uncached_time / cached_time,
        "max_candidate_support_difference": max(
            c["max_candidate_support_difference"] for c in checks
        ),
        "same_choices": all(c["same_choices"] for c in checks),
        "cached_usage": results[True].to_dict()["usage"],
        "uncached_usage": results[False].to_dict()["usage"],
        "note": "Cache correctness/performance check; not an accuracy or calibration benchmark.",
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        request = json.loads(args.request.read_text())
        if not isinstance(request, dict) or not {"state", "questions"} <= request.keys():
            raise ValueError("Request must contain state and questions")
        images = args.image or request.get("images", [])
        if not isinstance(images, list) or any(not isinstance(x, str) for x in images):
            raise ValueError("images must be a list of local file paths")
        config = ModelConfig(
            backend=args.backend,
            model=args.model,
            revision=args.revision,
            tokenizer=args.tokenizer,
            tokenizer_revision=args.tokenizer_revision,
            filename=args.filename,
            device=args.device,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            batch_size=args.batch_size,
            prefill_chunk_size=args.prefill_chunk_size,
            score_mode=args.score_mode,
            vision=args.vision or bool(images),
            load_in_4bit=args.load_in_4bit,
            optimize_head=not args.no_head_optimization,
        )
        with DecisionEngine.from_config(config) as engine:
            if args.command == "benchmark":
                result = benchmark(engine, request, images, args.iterations)
            else:
                result = engine.decide(
                    request["state"],
                    request["questions"],
                    images=images,
                    use_cache=not args.no_cache,
                ).to_dict()
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"openjev: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
