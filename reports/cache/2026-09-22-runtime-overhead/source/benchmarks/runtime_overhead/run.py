"""Ablate prompt batching and specialized cache forks against the published source."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import openjev.engine as engine_module  # noqa: E402
from benchmarks.adaptive_cache.mixed import mixed_cases  # noqa: E402
from benchmarks.adaptive_cache.run import MODEL, REVISION, cases  # noqa: E402
from benchmarks.cache_reuse_qwen.engine import fingerprint  # noqa: E402
from openjev import DecisionEngine  # noqa: E402
from openjev.prompt import compile_plan  # noqa: E402
from openjev.types import ModelConfig, parse_questions  # noqa: E402

MODES = ("previous", "tokenization", "cache", "combined")
BASELINE = ROOT / "reports/2048/2026-09-22-adaptive/source"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def previous_source(relative, package):
    source = BASELINE / relative
    name = package + "." + source.stem
    __import__(name)
    namespace = {"__name__": name, "__package__": package}
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    import mlx.core as mx

    old_prompt = previous_source("src/openjev/prompt.py", "openjev")["compile_plan"]
    old_score = previous_source("src/openjev/backends/mlx.py", "openjev.backends")[
        "MLXBackend"
    ].score
    original_cases = cases()
    mixed = mixed_cases()
    selected = [
        next(c for c in original_cases if c["id"] == name)
        for name in ("board-0", "board-12", "single-0")
    ]
    selected += [
        next(c for c in mixed if c["id"] == name)
        for name in (
            "balanced-3-short-normal",
            "balanced-12-short-normal",
            "balanced-24-short-normal",
            "balanced-12-long-normal",
        )
    ]
    write(args.output / "cases.json", selected)
    protocol = {
        "model": MODEL,
        "revision": REVISION,
        "repeats": args.repeats,
        "modes": MODES,
        "batch_size": 4,
        "prefill_chunk_size": 128,
        "cache_strategy": "adaptive",
        "score_mode": "full",
        "prompt_style": "full",
        "hardware": platform.platform(),
        "runtime": {k: importlib.metadata.version(k) for k in ("mlx", "mlx-lm", "transformers")},
        "baseline": "Published adaptive 2048 source at commit 6c7a25b",
        "timing": "Rotating four-mode order; warmed, synchronized API calls on one model. "
        "Load, static-cache initialization, validation, and logging excluded. "
        "No extra stage barriers inside inference.",
        "source_sha256": {},
        "baseline_sha256": {},
    }
    for source in [Path(__file__), *sorted((ROOT / "src/openjev").rglob("*.py"))]:
        relative = str(source.relative_to(ROOT))
        content = source.read_bytes()
        protocol["source_sha256"][relative] = hashlib.sha256(content).hexdigest()
        target = args.output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    for relative in ("src/openjev/prompt.py", "src/openjev/backends/mlx.py"):
        protocol["baseline_sha256"][relative] = hashlib.sha256(
            (BASELINE / relative).read_bytes()
        ).hexdigest()
    write(args.output / "protocol.json", protocol)
    mx.set_cache_limit(256 * 1024**2)
    with DecisionEngine.from_pretrained(
        MODEL,
        revision=REVISION,
        backend="mlx",
        batch_size=4,
        n_ctx=4096,
        cache_strategy="adaptive",
        score_mode="full",
        prompt_style="full",
    ) as engine:
        current_score = engine.backend.score
        before = fingerprint(engine._instruction_cache)

        @contextmanager
        def use(mode):
            engine_module.compile_plan = (
                compile_plan if mode in ("tokenization", "combined") else old_prompt
            )
            engine.backend.score = (
                current_score
                if mode in ("cache", "combined")
                else MethodType(old_score, engine.backend)
            )
            try:
                yield
            finally:
                engine_module.compile_plan = compile_plan
                engine.backend.score = current_score

        # Full tokens, anchors, question order, and validation must survive both
        # prompt styles and changed contexts, independently of numerical scoring.
        checked = 0
        for case in original_cases + mixed:
            for style in ("full", "short"):
                config = ModelConfig(prompt_style=style, n_ctx=8192)
                arguments = case["state"], parse_questions(case["questions"])
                previous = old_prompt(*arguments, engine.backend.tokenizer, config)
                current = compile_plan(*arguments, engine.backend.tokenizer, config)
                assert asdict(previous) == asdict(current), case["id"]
                checked += 1
        print(f"Exact prompt equality: {checked} workload/style combinations", flush=True)

        expected = {}
        for case in selected:
            for mode in MODES:
                with use(mode):
                    result = engine.decide(case["state"], case["questions"])
                    mx.synchronize()
                if mode == "previous":
                    expected[case["id"]] = result.answers
                assert result.answers == expected[case["id"]], (case["id"], mode)
        print("Warmup: every mode produces identical scores", flush=True)

        rows = []
        with (args.output / "requests.jsonl").open("w") as output:
            for repeat in range(args.repeats):
                for index, case in enumerate(selected):
                    offset = (repeat + index) % len(MODES)
                    for mode in MODES[offset:] + MODES[:offset]:
                        with use(mode):
                            mx.synchronize()
                            start = time.perf_counter()
                            result = engine.decide(case["state"], case["questions"])
                            mx.synchronize()
                            elapsed = time.perf_counter() - start
                        assert result.answers == expected[case["id"]], (case["id"], mode)
                        row = dict(
                            case=case["id"],
                            mode=mode,
                            repeat=repeat,
                            elapsed_ms=elapsed * 1000,
                            usage=asdict(result.usage),
                            answers=result.answers,
                        )
                        rows.append(row)
                        output.write(json.dumps(row) + "\n")
                        output.flush()
                print(f"Repeat {repeat + 1}/{args.repeats} complete", flush=True)
        assert fingerprint(engine._instruction_cache) == before

        # Expanded correctness-only corpus covers uneven candidates, long
        # questions, type-heavy mixes and every original 2048 fixed board.
        audited = 0
        for case in original_cases + mixed:
            with use("previous"):
                previous = engine.decide(case["state"], case["questions"])
            with use("combined"):
                current = engine.decide(case["state"], case["questions"])
            assert previous.answers == current.answers, case["id"]
            assert previous.usage.candidate_batches == current.usage.candidate_batches
            audited += 1
        assert fingerprint(engine._instruction_cache) == before
        prompt_timings = {}
        for case in selected:
            samples = {"previous": [], "combined": []}
            questions = parse_questions(case["questions"])
            for repeat in range(7):
                order = ("previous", "combined") if repeat % 2 == 0 else ("combined", "previous")
                for mode in order:
                    builder = old_prompt if mode == "previous" else compile_plan
                    start = time.perf_counter()
                    builder(case["state"], questions, engine.backend.tokenizer, engine.config)
                    if repeat > 1:
                        samples[mode].append((time.perf_counter() - start) * 1000)
            prompt_timings[case["id"]] = {
                mode: dict(samples_ms=values, median_ms=statistics.median(values))
                for mode, values in samples.items()
            }
        write(args.output / "prompt-timing.json", prompt_timings)
        write(
            args.output / "validation.json",
            dict(
                exact_prompt_combinations=checked,
                exact_answer_requests=len(rows),
                extra_correctness_pairs=audited,
                max_score_delta=0.0,
                permanent_cache_unchanged=True,
                same_model_object=True,
            ),
        )
        summary = {}
        for case in selected:
            item = {}
            for mode in MODES:
                times = [
                    r["elapsed_ms"] for r in rows if r["case"] == case["id"] and r["mode"] == mode
                ]
                item[mode] = dict(
                    median_ms=statistics.median(times), min_ms=min(times), max_ms=max(times)
                )
            item["latency_reduction_percent"] = 100 * (
                1 - item["combined"]["median_ms"] / item["previous"]["median_ms"]
            )
            summary[case["id"]] = item
        write(args.output / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
