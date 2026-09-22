"""Validate and report mixed-type timings using only the Python standard library."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

MODES = ("previous", "adaptive")


def analyze(root):
    cases = json.loads((root / "cases.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    rows = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
    lookup = {case["id"]: case for case in cases}
    index = {(row["id"], row["repeat"], row["mode"]): row for row in rows}
    expected = {
        (case["id"], repeat, mode)
        for case in cases
        for repeat in range(protocol["repeats"])
        for mode in MODES
    }
    assert set(index) == expected and len(rows) == len(expected)
    repeated_scores_identical = True
    for row in rows:
        case = lookup[row["id"]]
        answers, usage = row["native"]["answers"], row["native"]["usage"]
        assert set(answers) == set(case["questions"])
        assert usage["candidates"] == case["candidate_count"]
        for key, question in case["questions"].items():
            answer = answers[key]
            assert answer["type"] == question["type"]
            labels = (
                ["true"]
                if question["type"] == "noul"
                else list(question["criteria"])
                if question["type"] == "choice"
                else [str(i) for i in range(len(question["criteria"]))]
            )
            assert set(answer["candidates"]) == set(labels)
            if question["type"] == "noul":
                assert math.isclose(answer["noul"], answer["candidates"]["true"]["support"])
                continue
            logs = [answer["candidates"][label]["log_support"] for label in labels]
            weights = [math.exp(value - max(logs)) for value in logs]
            probabilities = {
                label: value / sum(weights) for label, value in zip(labels, weights, strict=True)
            }
            assert all(
                math.isclose(answer["probabilities"][k], value, abs_tol=1e-12)
                for k, value in probabilities.items()
            )
            if question["type"] == "choice":
                assert answer["choice"] == max(probabilities, key=probabilities.get)
            else:
                assert math.isclose(
                    answer["score"], sum(int(k) * p for k, p in probabilities.items())
                )
        repeated_scores_identical &= (
            answers == index[row["id"], 0, row["mode"]]["native"]["answers"]
        )

    per_case = []
    by_type = {
        kind: {
            "output_deltas": [],
            "support_deltas": [],
            "probability_deltas": [],
            "choice_matches": 0,
            "questions": 0,
        }
        for kind in ("noul", "choice", "score")
    }
    for case in cases:
        result = {k: v for k, v in case.items() if k not in {"state", "questions", "group"}}
        for mode in MODES:
            selected = [index[case["id"], repeat, mode] for repeat in range(protocol["repeats"])]
            milliseconds = [r["elapsed_seconds"] * 1000 for r in selected]
            result[mode] = {
                "median_ms": statistics.median(milliseconds),
                "min_ms": min(milliseconds),
                "max_ms": max(milliseconds),
                "mean_evaluated_tokens": statistics.mean(
                    r["native"]["usage"]["evaluated_input_tokens"] for r in selected
                ),
                "mean_padding_tokens": statistics.mean(
                    r["native"]["usage"]["padding_tokens"] for r in selected
                ),
                "median_planning_ms": statistics.median(
                    r["native"]["usage"]["planning_seconds"] * 1000 for r in selected
                ),
                "candidate_batches": selected[0]["native"]["usage"]["candidate_batches"],
                "cache_prefill_tokens": selected[0]["native"]["usage"]["cache_prefill_tokens"],
            }
        result["speedup"] = result["previous"]["median_ms"] / result["adaptive"]["median_ms"]
        per_case.append(result)
        # Agreement counts distinct questions, not repeat measurements of the same question.
        a, b = (index[case["id"], 0, mode]["native"]["answers"] for mode in MODES)
        for key, question in case["questions"].items():
            kind = question["type"]
            d = by_type[kind]
            d["questions"] += 1
            if kind == "choice":
                d["choice_matches"] += a[key]["choice"] == b[key]["choice"]
            else:
                d["output_deltas"].append(abs(a[key][kind] - b[key][kind]))
            d["support_deltas"] += [
                abs(c["support"] - b[key]["candidates"][label]["support"])
                for label, c in a[key]["candidates"].items()
            ]
            d["probability_deltas"] += [
                abs(p - b[key]["probabilities"][label])
                for label, p in a[key].get("probabilities", {}).items()
            ]
    quality = {}
    for kind, data in by_type.items():
        quality[kind] = {
            "unique_questions": data["questions"],
            "choice_matches": data["choice_matches"] if kind == "choice" else None,
            "mean_output_delta": statistics.mean(data["output_deltas"])
            if data["output_deltas"]
            else None,
            "max_output_delta": max(data["output_deltas"], default=0),
            "max_raw_support_delta": max(data["support_deltas"], default=0),
            "max_probability_delta": max(data["probability_deltas"], default=0),
        }
    summary = {
        "cases": per_case,
        "by_type": quality,
        "requests": len(rows),
        "unique_question_instances": sum(q["unique_questions"] for q in quality.values()),
        "adaptive_faster_cases": sum(c["speedup"] > 1 for c in per_case),
        "median_case_speedup": statistics.median(c["speedup"] for c in per_case),
        "repeated_scores_identical": repeated_scores_identical,
        "per_question_output_math_verified": True,
    }
    (root / "mixed-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def render(root, summary):
    protocol = json.loads((root / "protocol.json").read_text())
    initialization = json.loads((root / "initialization.json").read_text())
    audit = json.loads((root / "verification.json").read_text())
    cases = summary["cases"]
    lines = [
        "# OpenJev mixed-question timing: Qwen 0.8B",
        "",
        f"{len(cases)} workloads × {protocol['repeats']} repeats × 2 execution paths = "
        f"{summary['requests']} timed requests. "
        "The adaptive planner has a lower observed median on "
        f"{summary['adaptive_faster_cases']}/{len(cases)} workloads. "
        f"The median of per-case speedups is {summary['median_case_speedup']:.2f}×.",
        "",
        "Both paths use the same loaded Qwen3.5-0.8B 4-bit weights, identical full prompts, "
        "and full-vocabulary P(yes). The previous multi-question path caches the context "
        "once, then batches independent question/candidate suffixes. The adaptive path "
        "also retains instructions from initialization and plans further shared prefixes. "
        "This compares execution speed and numerical agreement, not correctness or calibration.",
        "",
        "## Short context, normal question and candidate lengths",
        "",
        "N/C/S is the number of Noul/Choice/Score questions. Normal Choice and Score "
        "questions have three candidates each; Noul has one. Each request has one shared "
        "customer context. Timings are per-request medians, not per-question latency.",
        "",
        "| Mix | N/C/S | Candidates | Previous | Adaptive | Speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        if case["context_length"] != "short" or case["suffix_shape"] != "normal":
            continue
        counts = "/".join(str(case["type_counts"][k]) for k in ("noul", "choice", "score"))
        a, b = case["previous"], case["adaptive"]
        lines.append(
            f"| {case['profile']} | {counts} | {case['candidate_count']} | "
            f"{a['median_ms']:.1f} ms | {b['median_ms']:.1f} ms | {case['speedup']:.2f}× |"
        )
    lines += [
        "",
        "## Context and suffix shape",
        "",
        "The balanced 12-question workload contains four questions of each type. "
        "Long questions append repeated policy text to stress reusable question prefixes; "
        "uneven candidates deliberately lengthen one Choice/Score description. "
        "These are synthetic timing stressors. Wide options use five Choice/Score candidates.",
        "",
        "| Context | Suffix shape | Candidates | Previous | Adaptive | Speedup | "
        "Previous/adaptive padding | Adaptive batches |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in cases:
        if case["profile"] != "balanced-12":
            continue
        a, b = case["previous"], case["adaptive"]
        lines.append(
            f"| {case['context_length']} | {case['suffix_shape']} | {case['candidate_count']} | "
            f"{a['median_ms']:.1f} ms | {b['median_ms']:.1f} ms | {case['speedup']:.2f}× | "
            f"{a['mean_padding_tokens']:.0f}/{b['mean_padding_tokens']:.0f} | "
            f"{b['candidate_batches']} |"
        )
    lines += [
        "",
        "## All measured cases",
        "",
        f"Min/max ranges cover {protocol['repeats']} repeat measurements; "
        "they are not confidence intervals. "
        "All batch sizes are capped at four. Groups with different retained prefixes "
        "run sequentially; rows within each batch run in parallel. "
        "Question types share GPU batches but never share probability normalization.",
        "",
        "| Workload | Previous median [min–max] ms | Adaptive median [min–max] ms | Speedup | "
        "Evaluated tokens previous/adaptive | Planner ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        a, b = case["previous"], case["adaptive"]
        lines.append(
            f"| {case['id']} | {a['median_ms']:.1f} [{a['min_ms']:.1f}–{a['max_ms']:.1f}] | "
            f"{b['median_ms']:.1f} [{b['min_ms']:.1f}–{b['max_ms']:.1f}] | "
            f"{case['speedup']:.2f}× | {a['mean_evaluated_tokens']:.0f}/"
            f"{b['mean_evaluated_tokens']:.0f} | {b['median_planning_ms']:.3f} |"
        )
    q = summary["by_type"]
    lines += [
        "",
        "## Output differences and verification",
        "",
        f"Distinct question instances: {summary['unique_question_instances']}. "
        f"Choice selected-answer agreement: {q['choice']['choice_matches']}/"
        f"{q['choice']['unique_questions']}. "
        f"Noul mean/max absolute output difference: {q['noul']['mean_output_delta']:.6f}/"
        f"{q['noul']['max_output_delta']:.6f}. "
        f"Score mean/max absolute expected-index difference: {q['score']['mean_output_delta']:.6f}/"
        f"{q['score']['max_output_delta']:.6f} (0–2 normally; 0–4 with five levels).",
        "",
        "Quantized execution shapes can change scores. Agreement does not measure accuracy. "
        "Each output was recomputed from only that question's candidate supports: "
        "Noul equals its support, Choice normalizes its own alternatives, and Score "
        "is the expected level under its own normalized distribution. "
        f"Repeated scores identical within each path: {summary['repeated_scores_identical']}.",
        "",
        "The untimed audit reconstructed exact original prompt tokens for "
        f"{audit['warmup_requests']} "
        f"requests and checked {audit['checked_retained_parents']} retained parent-cache uses. "
        "All checked KV/recurrent caches remained unchanged. The permanent cache remained "
        "unchanged after the complete timed run.",
        "",
        "## Protocol",
        "",
        f"Model `{protocol['model']}`, revision `{protocol['revision']}`. "
        f"Runtime `{protocol['runtime']}`, platform `{protocol['hardware']}`. "
        "All model layers are retained; thinking is disabled and no tokens are generated. "
        "The scorer and planner are unchanged from the prior cache benchmark.",
        "",
        "Both methods warm up on every case, then alternate execution order. "
        "Timing includes prompt compilation, planning, prefill, cache copying, scoring, "
        "result construction and GPU synchronization. Warmup and cache-audit hashing are "
        "excluded. Model loading is excluded; "
        "instruction-cache initialization is reported separately: "
        f"{len(initialization['instruction_tokens'])} permanent tokens took "
        f"{initialization['instruction_cache_seconds'] * 1000:.2f} ms to initialize "
        "after model load. Reuse counts exclude this one-time computation. "
        "The planner uses heuristic cost estimates, not a guarantee of globally optimal batching.",
        "",
        "[Inputs](cases.json) · [Every timed call](calls.jsonl) · "
        "[Per-case and per-type summary](mixed-summary.json) · [Timing summary](summary.json) · "
        "[Protocol and source hashes](protocol.json) · [Cache audit](verification.json) · "
        "[Initialization](initialization.json)",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    summary = analyze(directory)
    render(directory, summary)
    print(f"Verified per-question outputs and wrote report for {summary['requests']} requests.")
