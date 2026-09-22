"""Verify the saved adaptive-cache benchmark using only the standard library."""

import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    protocol = json.loads((root / "protocol.json").read_text())
    cases = json.loads((root / "cases.json").read_text())
    rows = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
    assert len(rows) == len(cases) * protocol["repeats"] * 2
    assert len({(r["id"], r["mode"], r["repeat"]) for r in rows}) == len(rows)
    for relative, expected in protocol["source_sha256"].items():
        assert hashlib.sha256((root / "source" / relative).read_bytes()).hexdigest() == expected
    for row in rows:
        usage = row["native"]["usage"]
        assert usage["generated_tokens"] == 0
        assert (
            usage["uncached_input_tokens"] - usage["evaluated_input_tokens"]
            == usage["reused_input_tokens"]
        )
        assert sum(usage["candidate_batches"]) == usage["candidates"]
        if row["mode"] == "adaptive":
            # Each row skips its retained prefix; dynamic prefill is charged once per node.
            assert usage["evaluated_input_tokens"] == (
                usage["uncached_input_tokens"]
                - sum(usage["scoring_prefix_tokens"])
                + sum(usage["cache_prefill_tokens"])
            )
        for answer in row["native"]["answers"].values():
            if "probabilities" in answer:
                assert abs(sum(answer["probabilities"].values()) - 1) < 1e-9
                if "choice" in answer:
                    assert answer["choice"] == max(
                        answer["probabilities"], key=answer["probabilities"].get
                    )
    print(f"Verified {len(rows)} requests and {len(protocol['source_sha256'])} source snapshots.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)
