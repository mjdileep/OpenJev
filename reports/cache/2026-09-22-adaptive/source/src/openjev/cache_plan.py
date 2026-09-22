"""Plan independent scoring branches from exact token prefixes, without inference.

The cost is a heuristic, not a guarantee of optimal wall time. Compare flat batches,
one shared prefill, and recursively split radix-tree branches. Pool unprofitable
branches back into a batch at their parent. No scores or question labels enter
planning; candidate rows only share computation of identical causal prefixes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cache


@dataclass(frozen=True)
class CostModel:
    batch_size: int = 8
    # Relative token-equivalent costs. Conservative defaults favor fewer launches.
    launch: float = 40.0
    batch_exponent: float = 0.85
    copy_per_token: float = 0.01

    def prefill(self, tokens: int, parent: int) -> float:
        return self.launch + tokens + self.copy_per_token * parent

    def score(self, lengths: list[int], parent: int) -> float:
        ordered = sorted(lengths)
        return sum(
            self.launch
            + len(batch) ** self.batch_exponent * max(batch)
            + self.copy_per_token * parent * len(batch)
            for i in range(0, len(ordered), self.batch_size)
            if (batch := ordered[i : i + self.batch_size])
        )


@dataclass(frozen=True)
class CachePlan:
    # All row indices under this node; prefix depth in the complete token stream.
    rows: tuple[int, ...]
    depth: int
    pending: tuple[int, ...]
    children: tuple[CachePlan, ...]
    cost: float


def plan_batches(prompts: list[list[int]], start: int, cost: CostModel) -> CachePlan:
    if not prompts or any(len(p) <= start for p in prompts):
        raise ValueError("Every candidate needs an uncached verdict position")
    if any(p[:start] != prompts[0][:start] for p in prompts):
        raise ValueError("The retained cache must match every candidate's token prefix")

    def direct(rows, depth):
        value = cost.score([len(prompts[i]) - depth for i in rows], depth)
        return CachePlan(rows, depth, rows, (), value)

    @cache
    def solve(rows, parent, level=0):
        flat = direct(rows, parent)
        if len(rows) == 1 or level >= 64:
            return flat
        first = prompts[rows[0]]
        # Even duplicate complete prompts must retain one token for scoring.
        end = min(len(prompts[i]) for i in rows) - 1
        for i in rows[1:]:
            position = parent
            while position < end and first[position] == prompts[i][position]:
                position += 1
            end = position
        prefill = cost.prefill(end - parent, parent) if end > parent else 0
        pooled = direct(rows, end)
        best = CachePlan(rows, end, rows, (), prefill + pooled.cost)
        groups = defaultdict(list)
        for i in rows:
            groups[prompts[i][end]].append(i)
        if len(groups) > 1:
            # A tiny shared edge can be cheaper to repeat in the child prefills
            # than to materialize as another cache. Consider bypassing it too.
            for depth in dict.fromkeys((parent, end)):
                pending, children = [], []
                for _, group in sorted(groups.items()):
                    child = solve(tuple(group), depth, level + 1)
                    if child.depth == depth and not child.children:
                        pending.extend(group)
                    else:
                        children.append(child)
                split_cost = prefill if depth > parent else 0
                split_cost += sum(child.cost for child in children)
                split_cost += cost.score([len(prompts[i]) - depth for i in pending], depth)
                if split_cost < best.cost:
                    best = CachePlan(rows, depth, tuple(pending), tuple(children), split_cost)
        return best if best.cost < flat.cost else flat

    return solve(tuple(range(len(prompts))), start)
