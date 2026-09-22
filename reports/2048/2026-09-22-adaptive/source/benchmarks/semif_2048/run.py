"""Paired 2048 games and identical-board latency: OpenJev versus upstream SemIf."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "examples/game_2048")]

from game import ACTIONS, preview, question_for  # noqa: E402

MODEL = "mlx-community/Qwen3.5-0.8B-4bit"
REVISION = "da28692b5f139cb0ec58a356b437486b7dac7462"
SEMIF_REVISION = "1f2dea3e25379f9dfc98cb83c324f00ab5deda37"
MLX_LM_REVISION = "a63e24c389382619eb6d9af656e3b46024be217a"
METHODS = ("openjev", "semif")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    with path.open("x") as file:
        json.dump(value, file, indent=2)
        file.write("\n")


def append(file, value):
    file.write(json.dumps(value, separators=(",", ":")) + "\n")
    file.flush()


def legal_moves(board):
    return [a for a in ACTIONS if preview(board, a)[0] != board]


def spawn(board, rng):
    """Match original value-then-position sampling and x-major empty-cell order."""
    empty = [(y, x) for x in range(4) for y in range(4) if board[y][x] == 0]
    if not empty:
        raise ValueError("Cannot spawn on a full board")
    value = 2 if rng.random() < 0.9 else 4
    y, x = empty[int(rng.random() * len(empty))]
    board[y][x] = value
    return {"row": y, "column": x, "value": value}


def initial_board(seed):
    rng = random.Random(seed)
    board = [[0] * 4 for _ in range(4)]
    spawn(board, rng)
    spawn(board, rng)
    return board, rng


def fixed_boards(count):
    """Freeze states from a separate uniform-random policy, before model evaluation."""
    boards = []
    episode = 0
    while len(boards) < count:
        board, rng = initial_board(200_000 + episode)
        policy = random.Random(300_000 + episode)
        for turn in range(60):
            legal = legal_moves(board)
            if not legal:
                break
            if turn in (5, 15, 30, 45) and len(legal) > 1:
                boards.append([row[:] for row in board])
                if len(boards) == count:
                    break
            board, _ = preview(board, policy.choice(legal))
            spawn(board, rng)
        episode += 1
    return boards


def statistics_ms(values):
    ms = sorted(x * 1000 for x in values)
    if not ms:
        return {}
    return {
        "n": len(ms),
        "mean_ms": statistics.mean(ms),
        "median_ms": statistics.median(ms),
        "p95_ms": ms[math.ceil(len(ms) * 0.95) - 1],
    }


class Scorers:
    def __init__(self, prompt_style="full", score_mode="full", execution="current"):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from semif_phase1 import mlx_backend

        from openjev import DecisionEngine

        self.mx = mx
        self.semif = mlx_backend
        self.model, self.tokenizer, self.metadata = mlx_backend.load_model(
            MODEL, REVISION, cache_limit_mib=256
        )
        engine_class = DecisionEngine
        if execution == "shared-prefix":
            from benchmarks.cache_reuse_qwen.engine import SharedPrefixBatchEngine

            engine_class = SharedPrefixBatchEngine
        self.openjev = engine_class.from_pretrained(
            MODEL,
            revision=REVISION,
            backend="mlx",
            batch_size=4,
            n_ctx=4096,
            prompt_style=prompt_style,
            score_mode=score_mode,
            cache_strategy="adaptive" if execution == "adaptive" else "shared",
        )
        # Verify loaded values, not just the same nominal model name.
        left = dict(tree_flatten(self.model.parameters()))
        right = dict(tree_flatten(self.openjev.backend.model.parameters()))
        if left.keys() != right.keys():
            raise RuntimeError("Model parameter names differ")
        for key in left:
            if left[key].shape != right[key].shape or not bool(
                mx.array_equal(left[key], right[key])
            ):
                raise RuntimeError(f"Loaded model weights differ: {key}")
        self.metadata["identical_loaded_parameters"] = True
        self.metadata["parameter_arrays_checked"] = len(left)
        mx.synchronize()

    def decide(self, method, board, request_id):
        state, questions, forced = question_for(board)
        if forced:
            return {
                "action": forced,
                "forced": True,
                "elapsed_seconds": 0.0,
                "probabilities": {forced: 1.0},
                "input_tokens": 0,
                "generated_tokens": 0,
            }
        question = questions["move"]
        common = {
            "state": state,
            "question": question.instructions,
            "options": [{"id": k, "description": v} for k, v in question.criteria.items()],
        }
        row = {"id": request_id, **common}
        self.mx.synchronize()
        started = time.perf_counter()
        if method == "openjev":
            result = self.openjev.decide(state, questions)
        else:
            result = self.semif.score(
                self.model, self.tokenizer, row, self.metadata, max_tokens=4096
            )
        self.mx.synchronize()
        elapsed = time.perf_counter() - started
        if method == "openjev":
            native = result.to_dict()
            probabilities = result.answers["move"]["probabilities"]
            action = result.answers["move"]["choice"]
            tokens = result.usage.evaluated_input_tokens
        else:
            native = {k: v for k, v in result.items() if k != "model"}
            probabilities = dict(zip(result["option_ids"], result["probabilities"], strict=True))
            action = max(probabilities, key=probabilities.get)
            tokens = result["input_tokens"]
        if action not in question.criteria or not all(
            math.isfinite(p) for p in probabilities.values()
        ):
            raise RuntimeError("Invalid decision")
        return {
            "action": action,
            "forced": False,
            "elapsed_seconds": elapsed,
            "probabilities": probabilities,
            "input_tokens": tokens,
            "generated_tokens": 0,
            "common_input_sha256": digest(common),
            "native": native,
        }

    def close(self):
        self.openjev.close()
        self.model = self.tokenizer = None
        gc.collect()
        self.mx.clear_cache()


def game(scorers, method, seed, max_moves, output):
    board, rng = initial_board(seed)
    start = [row[:] for row in board]
    score = moves = forced = 0
    timings = []
    started = time.perf_counter()
    with output.open("x") as file:
        while moves < max_moves:
            legal = legal_moves(board)
            if not legal or max(map(max, board)) >= 2048:
                break
            result = scorers.decide(method, board, f"{method}-{seed}-{moves}")
            after, points = preview(board, result["action"])
            if after == board:
                raise RuntimeError("Selected an illegal move")
            spawn_record = spawn(after, rng)
            score += points
            moves += 1
            forced += result["forced"]
            if not result["forced"]:
                timings.append(result["elapsed_seconds"])
            append(
                file,
                {
                    "method": method,
                    "seed": seed,
                    "move": moves,
                    "board": board,
                    "after": after,
                    "spawn": spawn_record,
                    "score": score,
                    **result,
                },
            )
            board = after
            if moves % 50 == 0:
                print(
                    json.dumps(
                        {
                            "progress": method,
                            "seed": seed,
                            "moves": moves,
                            "score": score,
                            "max_tile": max(map(max, board)),
                        }
                    ),
                    flush=True,
                )
    tile = max(map(max, board))
    summary = {
        "method": method,
        "seed": seed,
        "initial_board": start,
        "final_board": board,
        "score": score,
        "moves": moves,
        "max_tile": tile,
        "won": tile >= 2048,
        "game_over": not legal_moves(board),
        "capped": moves == max_moves and tile < 2048 and bool(legal_moves(board)),
        "forced_moves": forced,
        "latency": statistics_ms(timings),
        "decision_seconds": sum(timings),
        "wall_seconds": time.perf_counter() - started,
    }
    print(json.dumps({"game_complete": summary}), flush=True)
    return summary


def summarize(games, fixed):
    result = {"games": games, "by_method": {}, "matched_boards": {}}
    for method in METHODS:
        rows = [row for row in games if row["method"] == method]
        if rows:
            result["by_method"][method] = {
                "games": len(rows),
                "mean_score": statistics.mean(r["score"] for r in rows),
                "median_score": statistics.median(r["score"] for r in rows),
                "best_tile": max(r["max_tile"] for r in rows),
                "mean_moves": statistics.mean(r["moves"] for r in rows),
                "wins_2048": sum(r["won"] for r in rows),
                "capped": sum(r["capped"] for r in rows),
                "choice_latency_mean_ms": sum(r["decision_seconds"] for r in rows)
                * 1000
                / sum(r["latency"].get("n", 0) for r in rows),
            }
        calls = [row for row in fixed if row["method"] == method]
        result["matched_boards"][method] = {
            **statistics_ms([row["elapsed_seconds"] for row in calls]),
            "mean_input_tokens": statistics.mean(row["input_tokens"] for row in calls),
        }
    pairs = {}
    for row in fixed:
        pairs.setdefault((row["board_index"], row["repeat"]), {})[row["method"]] = row
    for pair in pairs.values():
        assert pair["openjev"]["common_input_sha256"] == pair["semif"]["common_input_sha256"]
    result["matched_boards"]["same_move_fraction"] = sum(
        pair["openjev"]["action"] == pair["semif"]["action"] for pair in pairs.values()
    ) / len(pairs)
    if games:
        by_seed = {}
        for row in games:
            by_seed.setdefault(row["seed"], {})[row["method"]] = row
        diffs = []
        for pair in by_seed.values():
            assert pair["openjev"]["initial_board"] == pair["semif"]["initial_board"]
            diffs.append(pair["openjev"]["score"] - pair["semif"]["score"])
        rng = random.Random(20260922)
        boot = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(5000))
        result["paired_score_difference_openjev_minus_semif"] = {
            "mean": statistics.mean(diffs),
            "bootstrap_95_percent_interval": [boot[124], boot[4874]],
            "openjev_higher": sum(d > 0 for d in diffs),
            "semif_higher": sum(d < 0 for d in diffs),
            "ties": diffs.count(0),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--semif-root", type=Path, default=ROOT / ".cache/semif-benchmark/upstream")
    parser.add_argument("--games", type=int, default=10, help="Paired seeds; two games per seed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-boards", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-moves", type=int, default=2000)
    parser.add_argument("--prompt-style", choices=["full", "short"], default="full")
    parser.add_argument("--score-mode", choices=["full", "binary"], default="full")
    parser.add_argument(
        "--openjev-execution",
        choices=["current", "shared-prefix", "adaptive"],
        default="current",
        help="Select full batches, context caching, or permanent instructions + planner",
    )
    args = parser.parse_args()
    if args.games < 0 or min(args.fixed_boards, args.repeats, args.max_moves) < 1:
        parser.error("Counts must be positive; games may also be zero")
    args.output.mkdir(parents=True, exist_ok=False)
    upstream = subprocess.check_output(
        ["git", "-C", str(args.semif_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if upstream != SEMIF_REVISION:
        raise ValueError(f"Expected SemIf {SEMIF_REVISION}, found {upstream}")
    runtime = json.loads(
        importlib.metadata.distribution("mlx-lm").read_text("direct_url.json") or "null"
    )
    if not runtime or runtime.get("vcs_info", {}).get("commit_id") != MLX_LM_REVISION:
        raise ValueError("Use the isolated environment with SemIf's pinned MLX-LM runtime")
    corpus = fixed_boards(args.fixed_boards)
    protocol = {
        "model": MODEL,
        "model_revision": REVISION,
        "semif_revision": upstream,
        "openjev_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "openjev_source_dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "diff", "--name-only", "HEAD", "--", "src/openjev"],
                text=True,
            ).strip()
        ),
        "runtime": {
            name: importlib.metadata.version(name)
            for name in ["mlx", "mlx-lm", "transformers", "torch", "numpy"]
        },
        "mlx_lm_source": runtime,
        "hardware": {"system": platform.platform(), "machine": platform.machine()},
        "seed_list": list(range(args.seed, args.seed + args.games)),
        "max_moves": args.max_moves,
        "fixed_board_repeats": args.repeats,
        "fixed_board_sha256": digest(corpus),
        "thinking": False,
        "generated_tokens": 0,
        "allocator_cache_limit_mib": 256,
        "openjev": {
            "batch_size": 4,
            "score_mode": args.score_mode,
            "prompt_style": args.prompt_style,
            "execution": args.openjev_execution,
            "cache_strategy": (
                "adaptive"
                if args.openjev_execution == "adaptive"
                else (
                    "shared_prefix_batch" if args.openjev_execution == "shared-prefix" else "single"
                )
            ),
            "verify_parent_cache_during_warmup": args.openjev_execution != "current",
            "permanent_instruction_cache": args.openjev_execution == "adaptive",
            "optimize_head": True,
            "decoder_layers": 24,
        },
        "semif": {"mode": "direct", "calibration": "none"},
        "forced_moves": "apply the only legal action for both; excluded from inference latency",
        "tile_spawning": "Python Random(seed); value then position; x-major empty-cell ordering",
        "game_order": "alternate which method plays first on successive seeds",
        "latency_order": "alternate methods for each board/repeat; full untimed corpus warm-up",
        "timing": (
            "synchronized scoring API calls; "
            "exclude model load, instruction-cache initialization, warmup, shared game preview, "
            "logging, and rendering"
        ),
        "limitations": [
            "Same seeded random stream, but boards diverge after different actions.",
            "Same semantic input; each library retains its own prompt and readout.",
            "Single-question gameplay does not test cross-question prefix caching.",
            "OpenJev uses shared MLX-LM 0.32.0, outside its current <0.32 extra constraint.",
            "Small local game experiment; no claim of general decision accuracy.",
        ],
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                ROOT / "benchmarks/cache_reuse_qwen/engine.py",
                ROOT / "benchmarks/adaptive_cache/run.py",
                ROOT / "benchmarks/adaptive_cache/mixed.py",
                ROOT / "examples/game_2048/game.py",
                ROOT / "src/openjev/engine.py",
                ROOT / "src/openjev/prompt.py",
                ROOT / "src/openjev/cache_plan.py",
                ROOT / "src/openjev/backends/mlx.py",
                ROOT / "src/openjev/backends/base.py",
                ROOT / "src/openjev/types.py",
            ]
        },
    }
    # Freeze protocol and boards before loading models or observing their decisions.
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "fixed-boards.json", corpus)
    for relative, checksum in protocol["source_sha256"].items():
        source = ROOT / relative
        contents = source.read_bytes()
        assert hashlib.sha256(contents).hexdigest() == checksum
        snapshot = args.output / "source" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open("xb") as file:
            file.write(contents)
    scorers = Scorers(
        prompt_style=args.prompt_style,
        score_mode=args.score_mode,
        execution=args.openjev_execution,
    )
    write_json(args.output / "model-artifacts.json", scorers.metadata)
    measured = []
    games = []
    try:
        adaptive_audit = None
        if args.openjev_execution == "adaptive":
            from benchmarks.adaptive_cache.run import AuditBackend
            from benchmarks.cache_reuse_qwen.engine import fingerprint
            from openjev.prompt import compile_plan

            engine = scorers.openjev
            static_hash = fingerprint(engine._instruction_cache)
            write_json(
                args.output / "initialization.json",
                {
                    "instruction_cache_seconds": engine.instruction_cache_seconds,
                    "instruction_tokens": engine._instruction_tokens,
                    "instruction_prefix_text": engine.backend.tokenizer.raw.decode(
                        engine._instruction_tokens
                    ),
                    "included_in_request_timing": False,
                },
            )
            adaptive_audit = AuditBackend(engine.backend)
            adaptive_audit.prefixes[id(engine._instruction_cache)] = engine._instruction_tokens
            engine.backend = adaptive_audit
        print("Warming both paths on the frozen board corpus...", flush=True)
        if args.openjev_execution == "shared-prefix":
            scorers.openjev.verify_parent = True
        for i, board in enumerate(corpus):
            if adaptive_audit is not None:
                adaptive_audit.scored.clear()
            for method in METHODS:
                scorers.decide(method, board, f"warm-{i}")
            if adaptive_audit is not None:
                state, questions, _ = question_for(board)
                plan = compile_plan(state, questions, engine.backend.tokenizer, engine.config)
                expected = [p for q in plan.questions for p in q.prompts]
                assert sorted(adaptive_audit.scored) == sorted(expected)
                assert fingerprint(engine._instruction_cache) == static_hash
        if adaptive_audit is not None:
            # Audit hashing and wrappers are removed from the measured path.
            engine.backend = adaptive_audit.backend
        if args.openjev_execution == "shared-prefix":
            scorers.openjev.verify_parent = False
            write_json(
                args.output / "cache-validation.json",
                {
                    "status": "verified",
                    "boards": len(corpus),
                    "check": "saved prefix unchanged after batched candidate scoring",
                    "included_in_timing": False,
                },
            )
        with (args.output / "fixed-board-decisions.jsonl").open("x") as file:
            for repeat in range(args.repeats):
                for i, board in enumerate(corpus):
                    order = METHODS if (i + repeat) % 2 == 0 else METHODS[::-1]
                    for method in order:
                        row = {
                            "method": method,
                            "board_index": i,
                            "repeat": repeat,
                            **scorers.decide(method, board, f"fixed-{i}-{repeat}"),
                        }
                        measured.append(row)
                        append(file, row)
        print(json.dumps({"fixed_board_summary": summarize([], measured)}), flush=True)
        for i, seed in enumerate(protocol["seed_list"]):
            for method in METHODS if i % 2 == 0 else METHODS[::-1]:
                result = game(
                    scorers,
                    method,
                    seed,
                    args.max_moves,
                    args.output / f"game-{seed}-{method}.jsonl",
                )
                games.append(result)
                write_json(args.output / f"game-{seed}-{method}-summary.json", result)
        summary = summarize(games, measured)
        if adaptive_audit is not None:
            assert fingerprint(engine._instruction_cache) == static_hash
            write_json(
                args.output / "cache-validation.json",
                {
                    "status": "verified",
                    "boards": len(corpus),
                    "check": "exact candidate prompts and immutable complete KV/recurrent parents",
                    "checked_retained_parents": adaptive_audit.checked_parents,
                    "permanent_cache_unchanged_after_all_games": True,
                    "included_in_timing": False,
                },
            )
        write_json(args.output / "summary.json", summary)
        print(json.dumps({"complete": summary}), flush=True)
    finally:
        scorers.close()


if __name__ == "__main__":
    main()
