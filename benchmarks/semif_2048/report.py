"""Create a standalone HTML report and recorded game replay from benchmark evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from verify import verify

TEMPLATE = Path(__file__).with_name("report_template.html")


def write_tables(root, summary, protocol):
    methods = ("openjev", "semif")
    with (root / "games.csv").open("x", newline="") as file:
        fields = ("seed", "method", "score", "max_tile", "moves", "won", "capped")
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(summary["games"], key=lambda g: (g["seed"], g["method"])))
    games = summary["by_method"]
    fixed = summary["matched_boards"]
    rows = [
        ("Games", [games[m]["games"] for m in methods]),
        ("Mean score", [f"{games[m]['mean_score']:,.1f}" for m in methods]),
        ("Median score", [f"{games[m]['median_score']:,.1f}" for m in methods]),
        ("Largest tile", [games[m]["best_tile"] for m in methods]),
        ("Mean moves", [f"{games[m]['mean_moves']:,.1f}" for m in methods]),
        ("Games reaching 2048", [games[m]["wins_2048"] for m in methods]),
        ("Matched-board median latency", [f"{fixed[m]['median_ms']:.1f} ms" for m in methods]),
        ("Matched-board p95 latency", [f"{fixed[m]['p95_ms']:.1f} ms" for m in methods]),
        ("Games reaching the move cap", [games[m]["capped"] for m in methods]),
    ]
    paired = summary["paired_score_difference_openjev_minus_semif"]
    interval = " to ".join(f"{value:,.1f}" for value in paired["bootstrap_95_percent_interval"])
    different = protocol.get("comparison_type") == "different_models"
    shared_prefix = protocol["openjev"].get("cache_strategy") == "shared_prefix_batch"
    model_description = (
        [
            f"OpenJev model: `{protocol['openjev']['model']}`.",
            f"SemIf model: `{protocol['semif']['model']}`.",
            "Both use 4-bit MLX weights, the same runtime, paired seeds, and game information.",
            "This compares two complete setups: both the model and scoring method differ.",
        ]
        if different
        else [
            f"Model: `{protocol['model']}`.",
            "Same loaded model parameters, shared MLX runtime, paired seeds, and game information.",
        ]
    )
    text = [
        "# OpenJev versus SemIf: 2048",
        "",
        *model_description,
        "No generated reasoning or answer tokens.",
        "Each method keeps its own prompt and readout.",
        f"OpenJev prompt: **{protocol['openjev'].get('prompt_style', 'full')}**; "
        f"score mode: **{protocol['openjev']['score_mode']}**; "
        f"all {protocol['openjev'].get('decoder_layers', 24)} decoder layers.",
        (
            "OpenJev prefills instructions + user context once per decision, then scores all "
            "legal-move suffixes in one batch from independent cache copies. "
            "Both attention KV and recurrent state are copied; the saved prefix was checked "
            "for mutations on every warm-up board."
            if shared_prefix
            else "OpenJev batches complete candidate prompts without a separate prefix prefill."
        ),
        "",
        "| Metric | OpenJev | SemIf |",
        "| --- | ---: | ---: |",
        *[f"| {label} | {values[0]} | {values[1]} |" for label, values in rows],
        "",
        f"OpenJev scored higher on **{paired['openjev_higher']}** seeds; "
        f"SemIf on **{paired['semif_higher']}**; ties **{paired['ties']}**.",
        f"Mean paired score difference (OpenJev minus SemIf): **{paired['mean']:,.1f}**.",
        f"Descriptive bootstrap 95% interval: **{interval}**.",
        "",
        "Latency is measured on the same frozen boards, excluding loading and warm-up.",
        "The games diverge after different choices. A small game benchmark does not establish",
        "general decision accuracy. This single-question task does not test caching across "
        "questions or successive decisions.",
        "Cache copies and prefix prefill are included in the measured API latency.",
        "",
        "Both use SemIf's pinned MLX-LM 0.32.0 revision, including its Qwen3.5 normalization fix.",
        "OpenJev runs outside its usual `<0.32` dependency range for this test.",
        "The protocol records the prompt, score mode, source hashes, and local modifications.",
        "",
        "[Interactive report and replays](report.html) · [All games (CSV)](games.csv) ·",
        "[Raw summary](summary.json) · [Frozen protocol](protocol.json)",
        "",
    ]
    with (root / "report.md").open("x") as file:
        file.write("\n".join(text))


def build(root):
    verification = verify(root)
    summary = json.loads((root / "summary.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    if not summary["games"]:
        raise ValueError("The replay report requires at least one paired game")
    games = {}
    for row in summary["games"]:
        seed, method = row["seed"], row["method"]
        records = [
            json.loads(line)
            for line in (root / f"game-{seed}-{method}.jsonl").read_text().splitlines()
        ]
        compact = [
            {k: r[k] for k in ("after", "score", "action", "forced", "elapsed_seconds")}
            for r in records
        ]
        games.setdefault(seed, {})[method] = {"summary": row, "history": compact}
    environment_path = root / "environment-validation.json"
    data = {
        "summary": summary,
        "protocol": protocol,
        "games": games,
        "verification": verification,
        "environment": json.loads(environment_path.read_text())
        if environment_path.exists()
        else {},
    }
    with (root / "report.html").open("x") as file:
        file.write(
            TEMPLATE.read_text().replace("__DATA__", json.dumps(data).replace("<", "\\u003c"))
        )
    write_tables(root, summary, protocol)
    print(root / "report.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    build(parser.parse_args().run_directory)
