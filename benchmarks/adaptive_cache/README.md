# Permanent instructions + adaptive prefix batches

This experiment compares the new `cache_strategy="adaptive"` option with the
previous OpenJev implementation, using identical full prompts and full-vocabulary
`P(yes)` on Qwen3.5-0.8B MLX 4-bit. Thinking is disabled; no tokens are generated.

[Measured results and raw data](../../reports/cache/2026-09-22-adaptive/report.md):
1.68× faster on the 24 fixed 2048 boards, with matching moves on 21/24 boards.
Quantized execution shapes can change scores; the report includes every difference.

```python
from openjev import Choice, DecisionEngine, Noul

with DecisionEngine.from_pretrained(cache_strategy="adaptive") as engine:
    result = engine.decide(
        "My payout failed and I need it today.",
        {
            "team": Choice("Who handles this?", {
                "billing": "Payments and refunds", "technical": "Software bugs",
            }),
            "urgent": Noul("Does this require action today?"),
        },
    )
    print(result.answers)
    print(result.usage)
```

Generic instructions through `State:` are computed once at initialization. An
exact-token prefix tree then chooses flat batches or smaller cached groups for
each request. A single candidate uses one suffix pass. Costs are heuristic; this
is not a guarantee that every selected plan is fastest on every device.

The test uses 24 frozen boards from the published 2048 comparison, the eight
previous short/long routing contexts, two mixed-question requests, two requests
with long question policies, and two single-candidate requests. It does not play
new games. The previous single-Choice path is the published shared-context helper;
other requests use the previous `shared` API behavior. Both paths share the exact
same loaded model object, and alternate order over three timed repetitions.

Every case is warmed on both paths. Untimed checks reconstruct full candidate
token streams, compare them with the published prompt builder, and fingerprint
retained KV/recurrent caches around every prefill/scoring branch. The permanent
cache is checked again after all measured calls. Timing includes prompt building,
planning, cache copying, GPU execution and result construction; model loading,
instruction initialization, warmup and audit hashing are excluded and initialization
is reported separately. Raw scores and all timing calls are saved. Quantized
kernel changes can change scores and selected choices, so agreement is reported
separately from speed. This is not an accuracy benchmark.

To reproduce the measured runtime, use the environment setup and pinned versions
in [the 2048 benchmark](../semif_2048/README.md), then run:

```bash
python benchmarks/adaptive_cache/run.py --output .cache/adaptive-cache/my-run
python -S benchmarks/adaptive_cache/verify.py .cache/adaptive-cache/my-run
```

The output includes the inputs, every raw result, aggregated timing/score changes,
load-time cache details, verification results, and hashed source snapshots. The
verifier uses only the Python standard library. The existing default caching
strategy and previously published reports are unchanged.

## Mixed question types

The [mixed-type report](../../reports/cache/2026-09-22-mixed-types/report.md) expands
the comparison to 44 workloads: balanced and type-heavy mixes of Noul, Choice,
and Score; 3, 12, or 24 questions; 7–56 candidates; short/long contexts; long
question instructions; uneven candidate lengths; and three/five-option questions.
Single-type controls help show the effect of the number of candidate rows.

```bash
python benchmarks/adaptive_cache/run.py --suite mixed --output .cache/adaptive-cache/mixed-run
python -S benchmarks/adaptive_cache/verify.py .cache/adaptive-cache/mixed-run
python -S benchmarks/adaptive_cache/mixed_report.py .cache/adaptive-cache/mixed-run
```

The report includes per-case timing ranges, selected batches, padding, planner
time, Choice agreement, and differences in Noul and Score outputs. Its verifier
recomputes each answer from only that question's candidate supports, including
the expected-index calculation for Score. These are timing and numerical
consistency measurements, not labeled accuracy evaluations.
