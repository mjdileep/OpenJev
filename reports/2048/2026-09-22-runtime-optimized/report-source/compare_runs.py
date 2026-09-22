"""Compare a completed rerun with an earlier run using the same frozen protocol."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from verify import records, verify


def load(path):
    return json.loads(path.read_text())


def compare(previous, current):
    for root in (previous, current):
        verify(root)
    old_protocol, protocol = (load(root / "protocol.json") for root in (previous, current))
    for key in (
        "model",
        "model_revision",
        "semif_revision",
        "runtime",
        "mlx_lm_source",
        "seed_list",
        "max_moves",
        "fixed_board_repeats",
        "fixed_board_sha256",
    ):
        assert old_protocol[key] == protocol[key], key
    game_source = "examples/game_2048/game.py"
    assert old_protocol["source_sha256"][game_source] == protocol["source_sha256"][game_source]
    old, new = (load(root / "summary.json") for root in (previous, current))
    old_games = {(g["seed"], g["method"]): g for g in old["games"]}
    new_games = {(g["seed"], g["method"]): g for g in new["games"]}
    pairs = []
    unchanged = {
        method: {"actions": True, "probabilities": True} for method in ("openjev", "semif")
    }
    for seed in protocol["seed_list"]:
        for method in unchanged:
            before, after = (
                records(root / f"game-{seed}-{method}.jsonl") for root in (previous, current)
            )
            unchanged[method]["actions"] &= [r["action"] for r in before] == [
                r["action"] for r in after
            ]
            unchanged[method]["probabilities"] &= [r["probabilities"] for r in before] == [
                r["probabilities"] for r in after
            ]
        pairs.append(
            {
                "seed": seed,
                "openjev_previous": old_games[seed, "openjev"]["score"],
                "openjev_current": new_games[seed, "openjev"]["score"],
                "semif_current": new_games[seed, "semif"]["score"],
            }
        )
    diffs = [p["openjev_current"] - p["openjev_previous"] for p in pairs]
    rng = random.Random(20260922)
    boot = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(5000))
    old_latency = old["matched_boards"]["openjev"]["median_ms"]
    new_latency = new["matched_boards"]["openjev"]["median_ms"]
    old_fixed, new_fixed = (
        {
            (r["method"], r["board_index"], r["repeat"]): r
            for r in records(root / "fixed-board-decisions.jsonl")
        }
        for root in (previous, current)
    )
    assert old_fixed.keys() == new_fixed.keys()
    fixed_pairs = [(old_fixed[key], new_fixed[key]) for key in old_fixed if key[0] == "openjev"]
    fixed_comparison = {
        "openjev_choice_agreement": statistics.mean(
            a["action"] == b["action"] for a, b in fixed_pairs
        ),
        "openjev_max_probability_delta": max(
            abs(a["probabilities"][k] - b["probabilities"][k])
            for a, b in fixed_pairs
            for k in a["probabilities"]
        ),
        "semif_same_probabilities": all(
            old_fixed[key]["probabilities"] == new_fixed[key]["probabilities"]
            for key in old_fixed
            if key[0] == "semif"
        ),
    }
    result = {
        "previous_run": str(previous),
        "current_run": str(current),
        "previous_openjev_config": old_protocol["openjev"],
        "current_openjev_config": protocol["openjev"],
        "openjev_same_actions": unchanged["openjev"]["actions"],
        "openjev_same_native_probabilities": unchanged["openjev"]["probabilities"],
        "semif_same_actions": unchanged["semif"]["actions"],
        "semif_same_native_probabilities": unchanged["semif"]["probabilities"],
        "openjev_median_latency_reduction_fraction": 1 - new_latency / old_latency,
        "openjev_mean_score_change": statistics.mean(diffs),
        "openjev_score_change_bootstrap_95_interval": [boot[124], boot[4874]],
        "pairs": pairs,
        "fixed_board_comparison": fixed_comparison,
    }
    with (current / "previous-comparison.json").open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    columns = [(old, "openjev"), (new, "openjev"), (new, "semif")]
    table = []
    for label, field in (
        ("Mean game score", "mean_score"),
        ("Median game score", "median_score"),
        ("Best tile", "best_tile"),
        ("2048 wins", "wins_2048"),
    ):
        table.append([label, *[f"{s['by_method'][m][field]:,.1f}" for s, m in columns]])
    table.append(
        [
            "Identical-board median",
            *[f"{s['matched_boards'][m]['median_ms']:.1f} ms" for s, m in columns],
        ]
    )
    lines = [
        "# Comparison with the previous run",
        "",
        "Same Qwen3.5-0.8B 4-bit weights, patched MLX runtime, game instructions,",
        "paired seeds, and frozen timing boards. All 24 layers are executed. No training.",
        "",
        "| Metric | OpenJev previous | OpenJev current | SemIf current |",
        "| --- | ---: | ---: | ---: |",
        *["| " + " | ".join(row) + " |" for row in table],
        "",
        f"Previous OpenJev: `{old_protocol['openjev'].get('prompt_style', 'full')}` prompt / "
        f"`{old_protocol['openjev']['score_mode']}` scoring / "
        f"`{old_protocol['openjev'].get('cache_strategy', 'single')}` execution. Current: "
        f"`{protocol['openjev'].get('prompt_style', 'full')}` prompt / "
        f"`{protocol['openjev']['score_mode']}` scoring / "
        f"`{protocol['openjev'].get('cache_strategy', 'single')}` execution.",
        "",
        f"OpenJev's median latency decreased by "
        f"{100 * result['openjev_median_latency_reduction_fraction']:.1f}%.",
        f"Mean paired score change from previous OpenJev: {statistics.mean(diffs):+,.1f}.",
        "Bootstrap 95% interval: "
        + " to ".join(f"{v:,.1f}" for v in [boot[124], boot[4874]])
        + ".",
        *[
            f"{label} reproduced the same action sequences: {unchanged[method]['actions']}. "
            f"Identical native choice distributions: {unchanged[method]['probabilities']}."
            for method, label in (("openjev", "OpenJev"), ("semif", "SemIf"))
        ],
        f"On the frozen boards, old/new OpenJev move agreement: "
        f"{fixed_comparison['openjev_choice_agreement']:.1%}; maximum normalized "
        f"probability difference: {fixed_comparison['openjev_max_probability_delta']:.6f}. "
        f"SemIf fixed-board distributions were identical: "
        f"{fixed_comparison['semif_same_probabilities']}.",
        "",
        "The historical timings come from separate runs. Current OpenJev and SemIf timings",
        "were interleaved on identical boards. This small sample does not establish general",
        "decision quality. Configuration differences are recorded above. Quantized kernels can",
        "give different scores when the same prompt is split into a prefill and candidate batch.",
        "",
        "| Seed | OpenJev previous | OpenJev current | SemIf current |",
        "| --- | ---: | ---: | ---: |",
        *[
            f"| {p['seed']} | {p['openjev_previous']} | {p['openjev_current']} | "
            f"{p['semif_current']} |"
            for p in pairs
        ],
        "",
        "[Current report and replays](report.html) · "
        "[Machine-readable comparison](previous-comparison.json)",
        "",
    ]
    with (current / "previous-comparison.md").open("x") as file:
        file.write("\n".join(lines))
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous_run", type=Path)
    parser.add_argument("current_run", type=Path)
    args = parser.parse_args()
    compare(args.previous_run, args.current_run)
