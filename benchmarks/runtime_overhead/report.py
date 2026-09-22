"""Verify raw runtime-comparison records and write a readable report (stdlib only)."""

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    protocol = json.loads((root / "protocol.json").read_text())
    validation = json.loads((root / "validation.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    prompts = json.loads((root / "prompt-timing.json").read_text())
    rows = [json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()]
    grouped = defaultdict(dict)
    for row in rows:
        group = grouped[row["case"], row["repeat"]]
        assert row["mode"] not in group
        assert row["elapsed_ms"] > 0
        group[row["mode"]] = row
    for group in grouped.values():
        assert set(group) == set(protocol["modes"])
        previous = group["previous"]
        for row in group.values():
            assert row["answers"] == previous["answers"]
            assert row["usage"]["candidate_batches"] == previous["usage"]["candidate_batches"]
    assert len(rows) == len(summary) * protocol["repeats"] * len(protocol["modes"])
    for name, modes in summary.items():
        for mode in protocol["modes"]:
            samples = [r["elapsed_ms"] for r in rows if r["case"] == name and r["mode"] == mode]
            assert statistics.median(samples) == modes[mode]["median_ms"]
    for name, expected in protocol["source_sha256"].items():
        assert hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() == expected

    labels = {
        "board-0": "2048 board 0 (4 candidates)",
        "board-12": "2048 board 12 (4 candidates)",
        "single-0": "Single yes/no question",
        "balanced-3-short-normal": "3 mixed questions (7 candidates)",
        "balanced-12-short-normal": "12 mixed questions (28 candidates)",
        "balanced-24-short-normal": "24 mixed questions (56 candidates)",
        "balanced-12-long-normal": "12 mixed questions, long context",
    }
    lines = [
        "# OpenJev tokenization and cache-branching optimizations",
        "",
        "Measured on 22 September 2026 on an Apple M3 Pro with 18 GiB memory. "
        "Same pinned Qwen3.5-0.8B MLX 4-bit model, adaptive caching, original full prompts, "
        "full-vocabulary scoring and candidate batches of four on both sides.",
        "",
        "## Results",
        "",
        "Median end-to-end milliseconds, five repeats per workload and mode. "
        "The last column is the reduction in latency relative to the previous implementation.",
        "",
        "| Workload | Previous | Tokenization only | Cache only | Both | Reduction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary.items():
        values = " | ".join(f"{item[m]['median_ms']:.2f}" for m in protocol["modes"])
        lines.append(f"| {labels[name]} | {values} | {item['latency_reduction_percent']:.1f}% |")
    lines += [
        "",
        "## What changed",
        "",
        "- Prompt preparation deduplicates complete strings and encodes them in native batches "
        "of at most 128. Each yes/no marker is encoded once per request. Every full prompt and "
        "both exact answer continuations are still checked; token fragments are never joined.",
        "- The tokenizer batch API is obtained from the owner of its bound encode method. "
        "This supports MLX's tokenizer wrapper while preserving its chat template and disabled "
        "thinking. Image and GGUF tokenizer overrides retain their own encode behavior.",
        "- MLX branches known KV and recurrent caches directly into separate candidate rows, "
        "avoiding redundant deep copies and repeated zero-fill/slice assignments. Unknown "
        "cache classes retain the defensive copy/merge path. Single-candidate calls keep "
        "their existing execution path.",
        "- Question normalization, prompts, model weights, precision, cache planning and "
        "batch sizes are unchanged. The improvements require no new user-facing options.",
        "",
        "## Isolated prompt preparation",
        "",
        "Measured separately after inference validation, using the actual loaded MLX tokenizer. "
        "Two warmups and five measured repetitions per builder; order alternates.",
        "",
        "| Workload | Previous ms | Current ms | Speedup |",
        "|---|---:|---:|---:|",
    ]
    for name, item in prompts.items():
        a, b = (item[m]["median_ms"] for m in ("previous", "combined"))
        lines.append(f"| {labels[name]} | {a:.2f} | {b:.2f} | {a / b:.2f}× |")
    lines += [
        "",
        "## Correctness and measurement limits",
        "",
        f"- Exact prompt equality for {validation['exact_prompt_combinations']} workload/style "
        "combinations (82 workloads with both full and short prompts).",
        f"- Exact answer equality for all {validation['exact_answer_requests']} timed requests "
        f"and {validation['extra_correctness_pairs']} additional previous/current request pairs. "
        "Maximum observed score difference: **0**. Warmup outputs also matched exactly.",
        "- Expanded checks include all 24 original fixed 2048 boards, mixed question types, "
        "long questions, uneven candidates, wide option sets and changed contexts. "
        "Candidate batch sizes match; the permanent instruction cache remains unchanged.",
        "- Execution order rotates across four modes; the model object is shared. "
        "End-to-end measurements synchronize at request boundaries, with no extra barriers "
        "inside inference. Loading, warmup, static instruction-cache initialization and logging "
        "are excluded; per-request prompt work, prefix computation and branching are included.",
        "- This is a small local latency experiment, not a new full-game or SemIf benchmark. "
        "Individual small differences can reflect timing noise. No new gameplay quality "
        "or cross-hardware speed claim is made.",
        "",
        "## Reproduction and artifacts",
        "",
        "Run `benchmarks/runtime_overhead/run.py --output <new-directory> --repeats 5` "
        "in the existing MLX benchmark environment, then "
        "`python benchmarks/runtime_overhead/report.py <new-directory>`.",
        "",
        "[Protocol and versions](protocol.json) · [Raw timed requests](requests.jsonl) · "
        "[Summary](summary.json) · [Prompt timing](prompt-timing.json) · "
        "[Validation](validation.json) · [Executed source](source)",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))
    print(f"Verified {len(rows)} timed records and source hashes; wrote {root / 'report.md'}")


if __name__ == "__main__":
    main()
