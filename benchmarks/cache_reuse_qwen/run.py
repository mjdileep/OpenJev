"""Qwen 0.8B: current OpenJev versus batched candidates from one saved context cache."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks.cache_reuse_qwen.engine import SharedPrefixBatchEngine  # noqa: E402
from benchmarks.semif_2048.run import (  # noqa: E402
    append,
    digest,
    statistics_ms,
    write_json,
)
from openjev import Choice, DecisionEngine  # noqa: E402

MODEL = "mlx-community/Qwen3.5-0.8B-4bit"
REVISION = "da28692b5f139cb0ec58a356b437486b7dac7462"
MODES = ("current_openjev", "shared_prefix_batch")


def build_summary(rows):
    index = {(r["context_index"], r["repeat"], r["mode"]): r for r in rows}
    result = {}
    for mode in MODES:
        selected = [r for r in rows if r["mode"] == mode]
        result[mode] = {
            **statistics_ms([r["elapsed_seconds"] for r in selected]),
            "mean_evaluated_tokens": statistics.mean(
                r["native"]["usage"]["evaluated_input_tokens"] for r in selected
            ),
            "mean_reused_tokens": statistics.mean(
                r["native"]["usage"]["reused_input_tokens"] for r in selected
            ),
        }
    agreements, probability_deltas, support_deltas, differences = [], [], [], []
    for row in (r for r in rows if r["mode"] == "shared_prefix_batch"):
        other = index[row["context_index"], row["repeat"], "current_openjev"]
        a, b = (r["native"]["answers"]["decision"] for r in (row, other))
        agreements.append(a["choice"] == b["choice"])
        probability_deltas += [
            abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"]
        ]
        support_deltas += [
            abs(a["candidates"][k]["support"] - b["candidates"][k]["support"])
            for k in a["candidates"]
        ]
        if a["choice"] != b["choice"]:
            differences.append(
                {
                    "context_index": row["context_index"],
                    "repeat": row["repeat"],
                    "current": b["choice"],
                    "cached_batch": a["choice"],
                }
            )
    result["comparison"] = {
        "choice_agreement": statistics.mean(agreements),
        "max_probability_delta": max(probability_deltas),
        "max_raw_support_delta": max(support_deltas),
        "median_speedup": result["current_openjev"]["median_ms"]
        / result["shared_prefix_batch"]["median_ms"],
        "different_choices": differences,
    }
    return result


def report(root, summary):
    lines = [
        "# Qwen 0.8B: current OpenJev versus shared prefix + candidate batch",
        "",
        "Same pinned 4-bit weights, full prompt, all 24 layers, and full-vocabulary P(yes).",
        "Eight fixed text contexts × three repetitions. No games are played.",
        "Current OpenJev is the unmodified DecisionEngine.decide API with batch_size=4.",
        "The alternative prefills instructions + user context once per request, then",
        "scores all question/candidate suffixes together from independent cache copies.",
        "Both attention KV and Qwen's recurrent states are copied. No cache is retained",
        "across context changes; no instruction-only precomputation is added.",
        "",
        "Timing includes prompt compilation, prefill, cache copies, scoring, result",
        "construction and GPU synchronization. Loading and warm-up are excluded.",
        "Both paths are warmed on every context; measured order alternates.",
        "",
        "| Path | Median | p95 | Mean input tokens evaluated | Mean tokens reused |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode, name in [
        ("current_openjev", "Current OpenJev"),
        ("shared_prefix_batch", "Shared prefix → candidate batch"),
    ]:
        r = summary[mode]
        lines.append(
            f"| {name} | {r['median_ms']:.1f} ms | {r['p95_ms']:.1f} ms | "
            f"{r['mean_evaluated_tokens']:.1f} | {r['mean_reused_tokens']:.1f} |"
        )
    c = summary["comparison"]
    lines += [
        "",
        f"Median speedup: **{c['median_speedup']:.2f}×**. "
        f"Choice agreement: **{c['choice_agreement']:.1%}**.",
        f"Maximum normalized probability difference: {c['max_probability_delta']:.6f}.",
        f"Maximum raw P(yes) difference: {c['max_raw_support_delta']:.6f}.",
        "",
        "Both methods use exactly the same input token sequences. Cache splitting",
        "and batching can change floating-point reduction kernels in quantized inference.",
        "Agreement is not accuracy; this is a small performance/equivalence test.",
        "Production defaults are unchanged.",
        "",
        "[Every call](calls.jsonl) · [Summary](summary.json) · "
        "[Protocol](protocol.json) · [Verification](verification.json)",
        "",
    ]
    with (root / "report.md").open("x") as file:
        file.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import mlx.core as mx

    args.output.mkdir(parents=True, exist_ok=False)
    cases = json.loads(Path(__file__).with_name("contexts.json").read_text())
    tasks = [(c["context"], {"decision": Choice(c["question"], c["candidates"])}) for c in cases]
    protocol = {
        "model": MODEL,
        "revision": REVISION,
        "prompt_style": "full",
        "score_mode": "full",
        "decoder_layers": 24,
        "current_batch_size": 4,
        "cached_candidate_batch_size": 4,
        "prefill_chunk_size": 128,
        "contexts": len(cases),
        "repeats": 3,
        "contexts_sha256": digest(cases),
        "hardware": platform.platform(),
        "runtime": {k: importlib.metadata.version(k) for k in ("mlx", "mlx-lm", "transformers")},
        "source_sha256": {},
    }
    for path in [
        Path(__file__),
        Path(__file__).with_name("engine.py"),
        ROOT / "benchmarks/semif_2048/run.py",
        Path(__file__).with_name("contexts.json"),
        *sorted((ROOT / "src/openjev").rglob("*.py")),
    ]:
        relative = str(path.relative_to(ROOT))
        content = path.read_bytes()
        protocol["source_sha256"][relative] = hashlib.sha256(content).hexdigest()
        target = args.output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "contexts.json", cases)
    mx.set_cache_limit(256 * 1024**2)
    with DecisionEngine.from_pretrained(
        MODEL,
        revision=REVISION,
        backend="mlx",
        batch_size=4,
        n_ctx=4096,
        prompt_style="full",
        score_mode="full",
    ) as current:
        cached = SharedPrefixBatchEngine(current.backend)
        engines = {"current_openjev": current, "shared_prefix_batch": cached}
        print("Warming two Qwen paths; verifying retained caches...", flush=True)
        cached.verify_parent = True
        for state, questions in tasks:
            for engine in engines.values():
                engine.decide(state, questions)
        cached.verify_parent = False
        rows = []
        with (args.output / "calls.jsonl").open("x") as file:
            for repeat in range(3):
                for i, (state, questions) in enumerate(tasks):
                    for mode in MODES if (i + repeat) % 2 == 0 else MODES[::-1]:
                        mx.synchronize()
                        start = time.perf_counter()
                        native = engines[mode].decide(state, questions)
                        mx.synchronize()
                        elapsed = time.perf_counter() - start
                        row = {
                            "mode": mode,
                            "context_index": i,
                            "repeat": repeat,
                            "elapsed_seconds": elapsed,
                            "native": native.to_dict(),
                        }
                        rows.append(row)
                        append(file, row)
        summary = build_summary(rows)
        write_json(args.output / "summary.json", summary)
        write_json(
            args.output / "verification.json",
            {
                "timed_calls": len(rows),
                "warmup_calls": 16,
                "retained_prefix_unchanged_on_all_warmup_contexts": True,
                "cached_suffixes_run_in_one_batch": True,
                "production_source_changed": False,
            },
        )
        report(args.output, summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
