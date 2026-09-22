"""Compare adaptive prefix planning against the previous OpenJev cache paths."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks.adaptive_cache.mixed import mixed_cases  # noqa: E402
from benchmarks.cache_reuse_qwen.engine import SharedPrefixBatchEngine, fingerprint  # noqa: E402
from examples.game_2048.game import question_for  # noqa: E402
from openjev import Choice, DecisionEngine, Noul, Score  # noqa: E402
from openjev.cache_plan import CostModel  # noqa: E402
from openjev.prompt import compile_plan  # noqa: E402
from openjev.types import parse_questions  # noqa: E402

MODEL = "mlx-community/Qwen3.5-0.8B-4bit"
REVISION = "da28692b5f139cb0ec58a356b437486b7dac7462"
MODES = ("previous", "adaptive")


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def cases():
    boards = json.loads(
        (ROOT / "reports/2048/2026-09-22-shared-prefix/fixed-boards.json").read_text()
    )
    result = []

    def add(name, group, state, questions):
        result.append(
            dict(
                id=name,
                group=group,
                state=state,
                questions={k: asdict(q) for k, q in questions.items()},
            )
        )

    for i, board in enumerate(boards):
        state, questions, forced = question_for(board)
        assert forced is None
        add(f"board-{i}", "2048 boards", state, questions)
    contexts = json.loads((ROOT / "benchmarks/cache_reuse_qwen/contexts.json").read_text())
    for case in contexts:
        add(
            case["id"],
            f"{case['length_group']} context",
            case["context"],
            {
                "decision": Choice(case["question"], case["candidates"]),
            },
        )
    mixed = {
        "department": Choice(
            "Which team handles this?",
            {
                "billing": "Payments, invoices and refunds",
                "technical": "Software bugs",
                "sales": "Plans and upgrades",
            },
        ),
        "urgent": Noul("Does this require immediate action?"),
        "tone": Score("How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]),
    }
    for i in (0, 2):
        add(f"mixed-{i}", "mixed questions", contexts[i]["context"], mixed)
    # Long unrelated question policies exercise prefix subgroups, not just one root.
    policies = {
        "alpha": Choice(
            "A: Assess payment handling. "
            + "Use the ticket facts, never unrelated examples. " * 40,
            {"refund": "Refund the payment", "retry": "Retry the payment", "hold": "Wait"},
        ),
        "zeta": Choice(
            "Z: Assess customer contact. "
            + "Use the requested deadline and contact preference. " * 40,
            {"email": "Send email", "phone": "Call the customer", "wait": "Wait"},
        ),
        "urgent": Noul("Urgent?"),
    }
    add("long-policies", "long question groups", contexts[0]["context"], policies)
    add("long-policies-long-context", "long question groups", contexts[1]["context"], policies)
    for i in (0, 1):
        add(
            f"single-{i}",
            "single candidate",
            contexts[i]["context"],
            {
                "urgent": Noul("Does this require immediate action?"),
            },
        )
    return result


class AuditBackend:
    """Untimed checks of exact prompts and complete retained inference state."""

    def __init__(self, backend):
        self.backend = backend
        self.active = True
        self.prefixes = {}
        self.scored = []
        self.checked_parents = 0

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def prefill(self, tokens, parent=None):
        if not self.active:
            return self.backend.prefill(tokens, parent)
        before = fingerprint(parent) if parent is not None else None
        prefix = self.prefixes[id(parent)] if parent is not None else []
        child = self.backend.prefill(tokens, parent)
        self.prefixes[id(child)] = prefix + tokens
        if parent is not None:
            assert fingerprint(parent) == before, "Prefill mutated its retained parent"
            self.checked_parents += 1
        return child

    def score(self, parent, suffixes):
        if not self.active:
            return self.backend.score(parent, suffixes)
        before = fingerprint(parent) if parent is not None else None
        prefix = self.prefixes[id(parent)] if parent is not None else []
        self.scored.extend(prefix + suffix for suffix in suffixes)
        values = self.backend.score(parent, suffixes)
        if parent is not None:
            assert fingerprint(parent) == before, "Scoring mutated its retained parent"
            self.checked_parents += 1
        return values


def summarize(rows):
    result = {}
    for group in sorted({row["group"] for row in rows}):
        subset = [row for row in rows if row["group"] == group]
        item = {}
        for mode in MODES:
            selected = [row for row in subset if row["mode"] == mode]
            times = sorted(row["elapsed_seconds"] * 1000 for row in selected)
            item[mode] = {
                "calls": len(times),
                "median_ms": statistics.median(times),
                "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))],
                "mean_evaluated_tokens": statistics.mean(
                    row["native"]["usage"]["evaluated_input_tokens"] for row in selected
                ),
                "mean_padding_tokens": statistics.mean(
                    row["native"]["usage"]["padding_tokens"] for row in selected
                ),
                "median_planning_ms": statistics.median(
                    row["native"]["usage"]["planning_seconds"] * 1000 for row in selected
                ),
            }
        index = {(r["id"], r["repeat"], r["mode"]): r for r in subset}
        agreements, probabilities, supports = [], [], []
        for row in (r for r in subset if r["mode"] == "adaptive"):
            previous = index[row["id"], row["repeat"], "previous"]
            for key, answer in row["native"]["answers"].items():
                other = previous["native"]["answers"][key]
                if "choice" in answer:
                    agreements.append(answer["choice"] == other["choice"])
                probabilities += [
                    abs(p - other["probabilities"][k])
                    for k, p in answer.get("probabilities", {}).items()
                ]
                supports += [
                    abs(c["support"] - other["candidates"][k]["support"])
                    for k, c in answer["candidates"].items()
                ]
        item["comparison"] = {
            "speedup": item["previous"]["median_ms"] / item["adaptive"]["median_ms"],
            "choice_matches": sum(agreements),
            "choice_comparisons": len(agreements),
            "max_probability_delta": max(probabilities, default=0),
            "max_raw_support_delta": max(supports),
        }
        result[group] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--suite", choices=["original", "mixed"], default="original")
    args = parser.parse_args()
    import mlx.core as mx

    args.output.mkdir(parents=True, exist_ok=False)
    corpus = (mixed_cases() if args.suite == "mixed" else cases())[: args.limit]
    write(args.output / "cases.json", corpus)
    protocol = {
        "model": MODEL,
        "revision": REVISION,
        "repeats": args.repeats,
        "case_count": len(corpus),
        "suite": args.suite,
        "prompt_style": "full",
        "score_mode": "full",
        "thinking": False,
        "generated_tokens": 0,
        "batch_size": 4,
        "hardware": platform.platform(),
        "cost_model": asdict(CostModel(batch_size=4)),
        "runtime": {k: importlib.metadata.version(k) for k in ("mlx", "mlx-lm", "transformers")},
        "previous": "Published shared-prefix helper for one Choice; previous shared API otherwise",
        "timing": "Alternating synchronized API calls, full corpus warmup, load excluded",
        "source_sha256": {},
    }
    sources = [
        Path(__file__),
        ROOT / "benchmarks/adaptive_cache/mixed.py",
        ROOT / "benchmarks/adaptive_cache/verify.py",
        ROOT / "benchmarks/cache_reuse_qwen/engine.py",
        ROOT / "examples/game_2048/game.py",
        *sorted((ROOT / "src/openjev").rglob("*.py")),
    ]
    for source in sources:
        relative = str(source.relative_to(ROOT))
        content = source.read_bytes()
        protocol["source_sha256"][relative] = hashlib.sha256(content).hexdigest()
        target = args.output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    write(args.output / "protocol.json", protocol)
    # Compare against the published prompt builder, using the same question classes.
    old_prompt = ROOT / "reports/2048/2026-09-22-shared-prefix/source/src/openjev/prompt.py"
    namespace = {"__name__": "openjev.prompt", "__package__": "openjev"}
    exec(compile(old_prompt.read_text(), str(old_prompt), "exec"), namespace)
    mx.set_cache_limit(256 * 1024**2)
    with DecisionEngine.from_pretrained(
        MODEL,
        revision=REVISION,
        backend="mlx",
        batch_size=4,
        n_ctx=4096,
    ) as previous:
        previous.backend = AuditBackend(previous.backend)
        shared = SharedPrefixBatchEngine(previous.backend)
        new_backend = copy.copy(previous.backend.backend)
        new_backend.config = replace(new_backend.config, cache_strategy="adaptive")
        new_audit = AuditBackend(new_backend)
        adaptive = DecisionEngine(new_audit)
        assert previous.backend.model is adaptive.backend.model
        static_hash = fingerprint(adaptive._instruction_cache)
        write(
            args.output / "initialization.json",
            {
                "instruction_cache_seconds": adaptive.instruction_cache_seconds,
                "instruction_tokens": adaptive._instruction_tokens,
                "instruction_prefix_text": new_backend.tokenizer.raw.decode(
                    adaptive._instruction_tokens
                ),
                "same_model_object": True,
            },
        )

        def old_engine(questions):
            if len(questions) == 1 and isinstance(next(iter(questions.values())), Choice):
                return shared
            return previous

        print(f"Warm-up and cache verification: {len(corpus)} cases", flush=True)
        for index, case in enumerate(corpus):
            questions = parse_questions(case["questions"])
            expected = compile_plan(
                case["state"], questions, new_backend.tokenizer, adaptive.config
            )
            published = namespace["compile_plan"](
                case["state"], questions, new_backend.tokenizer, adaptive.config
            )
            tokens = [p for q in expected.questions for p in q.prompts]
            assert tokens == [p for q in published.questions for p in q.prompts]
            for engine in (old_engine(questions), adaptive):
                engine.backend.scored.clear()
                engine.decide(case["state"], questions)
                assert sorted(engine.backend.scored) == sorted(tokens), (
                    "Prompt reconstruction failed"
                )
            assert fingerprint(adaptive._instruction_cache) == static_hash
            print(f"  {index + 1}/{len(corpus)} {case['id']} verified", flush=True)
        parents = previous.backend.checked_parents + new_audit.checked_parents
        previous.backend.active = new_audit.active = False
        rows = []
        with (args.output / "calls.jsonl").open("x") as file:
            for repeat in range(args.repeats):
                for index, case in enumerate(corpus):
                    questions = parse_questions(case["questions"])
                    engines = {"previous": old_engine(questions), "adaptive": adaptive}
                    for mode in MODES if (index + repeat) % 2 == 0 else MODES[::-1]:
                        mx.synchronize()
                        started = time.perf_counter()
                        native = engines[mode].decide(case["state"], questions)
                        mx.synchronize()
                        row = dict(
                            id=case["id"],
                            group=case["group"],
                            repeat=repeat,
                            mode=mode,
                            elapsed_seconds=time.perf_counter() - started,
                            native=native.to_dict(),
                        )
                        file.write(json.dumps(row, allow_nan=False) + "\n")
                        file.flush()
                        rows.append(row)
                print(f"Timed repeat {repeat + 1}/{args.repeats} complete", flush=True)
        assert fingerprint(adaptive._instruction_cache) == static_hash
        write(
            args.output / "verification.json",
            {
                "warmup_requests": len(corpus) * 2,
                "timed_requests": len(rows),
                "exact_original_prompt_tokens": True,
                "checked_retained_parents": parents,
                "permanent_cache_unchanged_after_all_requests": True,
                "same_model_object": True,
            },
        )
        summary = summarize(rows)
        write(args.output / "summary.json", summary)
        if args.suite == "original":
            print(json.dumps(summary, indent=2), flush=True)
        else:
            for group, result in summary.items():
                print(
                    f"{group}: {result['previous']['median_ms']:.1f} -> "
                    f"{result['adaptive']['median_ms']:.1f} ms "
                    f"({result['comparison']['speedup']:.2f}x)",
                    flush=True,
                )
        adaptive._instruction_cache = None  # Both wrappers share the model owned by previous.


if __name__ == "__main__":
    main()
