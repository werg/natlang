"""Algorithm-heavy programs with independent Python oracles.

These families make the interpreter carry exact algorithms and multi-stage data
flows instead of learning mostly leaf writes.  Every generated program is still
executed through the real runtime by scripts/generate.py.
"""
from __future__ import annotations

from .programs import Plan, Program


def _lambda(params, returns, instructions, *, function=None, types=None, codebase=None):
    spec = {"type": f"Lambda<{{ {', '.join(f'{k}: {v}' for k, v in params.items())} }}, {returns}>",
            "instructions": instructions}
    if function:
        spec["function"] = function
    if types:
        spec["types"] = types
    if codebase:
        spec["codebase"] = codebase
    return {"$lambda": spec}


def array_kernel(rng):
    """One exact array algorithm, deliberately expressed as code rather than guessed."""
    numbers = [rng.randint(-40, 90) for _ in range(rng.randint(4, 14))]
    items = [rng.choice(["alpha", "beta", "gamma", "delta"]) for _ in range(rng.randint(4, 14))]
    width = rng.randint(1, min(5, len(numbers)))
    k = rng.randint(0, len(numbers))
    matrix = [[rng.randint(-20, 40) for _ in range(rng.randint(1, 6))]
              for _ in range(rng.randint(2, 7))]
    flags = [rng.choice([True, False]) for _ in range(rng.randint(3, 15))]
    intervals = []
    for _ in range(rng.randint(3, 9)):
        start = rng.randint(0, 30)
        intervals.append({"start": start, "end": start + rng.randint(0, 9)})
    cases = [
        ("prefix_sums", {"numbers": numbers}, {"numbers": "Num[]"}, "Num[]",
         "Return every running prefix sum of `args/numbers` in order.",
         "args.numbers.reduce((out, x) => out.concat([(out.length ? out[out.length - 1] : 0) + x]), [])",
         lambda: [sum(numbers[:i + 1]) for i in range(len(numbers))]),
        ("window_sums", {"numbers": numbers, "width": width}, {"numbers": "Num[]", "width": "Num"}, "Num[]",
         "Compute the sum of every contiguous window of `args/width` numbers.",
         "args.numbers.slice(0, args.numbers.length - args.width + 1).map((_, i) => sum(args.numbers.slice(i, i + args.width)))",
         lambda: [sum(numbers[i:i + width]) for i in range(len(numbers) - width + 1)]),
        ("top_k", {"numbers": numbers, "k": k}, {"numbers": "Num[]", "k": "Num"}, "Num[]",
         "Return the largest `args/k` values from `args/numbers`, greatest first.",
         "args.numbers.slice().sort((a, b) => b - a).slice(0, args.k)",
         lambda: sorted(numbers, reverse=True)[:k]),
        ("stable_unique", {"items": items}, {"items": "Text[]"}, "Text[]",
         "Remove duplicate values from `args/items` while preserving first occurrence order.",
         "args.items.filter((x, i) => args.items.indexOf(x) === i)",
         lambda: list(dict.fromkeys(items))),
        ("weighted_checksum", {"numbers": numbers}, {"numbers": "Num[]"}, "Num",
         "Multiply each number by its one-based position and add the products.",
         "sum(args.numbers.map((x, i) => x * (i + 1)))",
         lambda: sum((i + 1) * x for i, x in enumerate(numbers))),
        ("adjacent_changes", {"items": items}, {"items": "Text[]"}, "Num",
         "Count positions after the first where the value differs from the preceding value.",
         "args.items.slice(1).filter((x, i) => x !== args.items[i]).length",
         lambda: sum(a != b for a, b in zip(items, items[1:]))),
        ("row_sums", {"matrix": matrix}, {"matrix": "Num[][]"}, "Num[]",
         "Return the sum of each row in `args/matrix`, preserving row order.",
         "args.matrix.map(row => sum(row))",
         lambda: [sum(row) for row in matrix]),
        ("longest_true_run", {"flags": flags}, {"flags": "Bool[]"}, "Num",
         "Return the length of the longest contiguous run of true values in `args/flags`.",
         "args.flags.reduce((s, x) => ({ run: x ? s.run + 1 : 0, best: Math.max(s.best, x ? s.run + 1 : 0) }), { run: 0, best: 0 }).best",
         lambda: max((len(run) for run in "".join("1" if x else "0" for x in flags).split("0")), default=0)),
        ("merge_intervals", {"intervals": intervals}, {"intervals": "Interval[]"}, "Interval[]",
         "Merge all overlapping intervals in `args/intervals` and return them ordered by start.",
         "args.intervals.slice().sort((a, b) => a.start - b.start || a.end - b.end).reduce((out, x) => { const last = out[out.length - 1]; return last && x.start <= last.end ? out.slice(0, -1).concat([{ start: last.start, end: Math.max(last.end, x.end) }]) : out.concat([{ start: x.start, end: x.end }]) }, [])",
         lambda: _merge_intervals(intervals)),
    ]
    name, inputs, params, returns, text, code, oracle = rng.choice(cases)
    named_types = ({"Interval": "{ start: Num, end: Num }"}
                   if name == "merge_intervals" else None)
    return Program(f"algo_{name}", _lambda(params, returns, text, function=name, types=named_types), inputs, oracle(),
                   {text: Plan("crisp", code=code, note=f"Computed {name} exactly with code.")},
                   source_semantics={"algorithm": name, "inputs": inputs, "code": code})


def _merge_intervals(intervals):
    out = []
    for item in sorted(intervals, key=lambda x: (x["start"], x["end"])):
        if out and item["start"] <= out[-1]["end"]:
            out[-1]["end"] = max(out[-1]["end"], item["end"])
        else:
            out.append(dict(item))
    return out


def staged_ranking(rng):
    """A three-stage rank/sort/take graph whose intermediates have distinct types."""
    count = rng.randint(4, 12)
    candidates = [{"id": f"c{i}", "quality": rng.randint(0, 20), "cost": rng.randint(0, 100)}
                  for i in range(count)]
    k = rng.randint(0, count)
    ranked = [{**x, "score": x["quality"] * 100 - x["cost"]} for x in candidates]
    expected = sorted(ranked, key=lambda x: (-x["score"], x["id"]))[:k]
    candidate = "{ id: Text, quality: Num, cost: Num }"
    ranked_type = "{ id: Text, quality: Num, cost: Num, score: Num }"
    text = rng.choice([
        "Score every candidate, sort by descending score with id as the tie breaker, then return the first `args/k`.",
        "Build the ranked shortlist: compute scores, order candidates by score then id, and keep `args/k` entries.",
    ])
    codebase = {
        "score_candidates": {"description": "Attach the exact quality/cost score to every candidate.",
            "args": {"items": "Candidate[]"}, "returns": "Ranked[]",
            "code": "return args.items.map(x => ({ ...x, score: x.quality * 100 - x.cost }))"},
        "sort_ranked": {"description": "Sort ranked candidates by score descending and id ascending.",
            "args": {"items": "Ranked[]"}, "returns": "Ranked[]",
            "code": "return args.items.slice().sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))"},
        "take_ranked": {"description": "Take the requested prefix of a ranked list.",
            "args": {"items": "Ranked[]", "k": "Num"}, "returns": "Ranked[]",
            "code": "return args.items.slice(0, args.k)"},
    }
    steps = [("call", {"function": "score_candidates", "to": "let/scored",
                        "inputs": {"items": "args/candidates"}}),
             ("call", {"function": "sort_ranked", "to": "let/sorted",
                        "inputs": {"items": "let/scored"}}),
             ("call", {"function": "take_ranked", "to": "return",
                        "inputs": {"items": "let/sorted", "k": "args/k"}})]
    root = _lambda({"candidates": "Candidate[]", "k": "Num"}, "Ranked[]", text,
                   function="rank_candidates",
                   types={"Candidate": candidate, "Ranked": ranked_type}, codebase=codebase)
    return Program("algo_staged_ranking", root, {"candidates": candidates, "k": k}, expected,
                   {text: Plan("calls", steps=steps, note="Scored, sorted, and took the requested prefix.")},
                   source_semantics={"algorithm": "score_sort_take", "inputs": {"candidates": candidates, "k": k}})


def algorithm_pipeline(rng):
    """Calls and exact local computation in one dependency-bearing workflow."""
    accounts = [f"acct-{i}" for i in range(rng.randint(2, 6))]
    transactions = [{"account": rng.choice(accounts), "amount": rng.randint(-50, 180),
                     "active": rng.random() < 0.8}
                    for _ in range(rng.randint(8, 24))]
    active = [x for x in transactions if x["active"]]
    totals = [sum(x["amount"] for x in active if x["account"] == account)
              for account in accounts]
    expected = sorted(({"account": account, "total": total}
                       for account, total in zip(accounts, totals)),
                      key=lambda x: (-x["total"], x["account"]))
    tx = "{ account: Text, amount: Num, active: Bool }"
    summary = "{ account: Text, total: Num }"
    text = rng.choice([
        "Discard inactive transactions, total the remaining amounts per account, and rank accounts by total descending then account id.",
        "Build an account leaderboard from active transactions: aggregate each account and sort by total, breaking ties by account id.",
    ])
    codebase = {
        "active_only": {"description": "Keep only active transactions.",
                        "args": {"items": "Transaction[]"}, "returns": "Transaction[]",
                        "code": "return args.items.filter(x => x.active)"},
        "rank_accounts": {"description": "Pair accounts with totals and sort the leaderboard.",
                          "args": {"accounts": "Text[]", "totals": "Num[]"},
                          "returns": "Summary[]",
                          "code": "return args.accounts.map((account, i) => ({ account, total: args.totals[i] })).sort((a, b) => b.total - a.total || a.account.localeCompare(b.account))"},
    }
    steps = [
        ("call", {"function": "active_only", "to": "let/active",
                  "inputs": {"items": "args/transactions"}}),
        ("glue", "args.accounts.map(a => sum(locals.active.filter(x => x.account === a).map(x => x.amount)))",
         "let/totals", "Num[]"),
        ("call", {"function": "rank_accounts", "to": "return",
                  "inputs": {"accounts": "args/accounts", "totals": "let/totals"}}),
    ]
    root = _lambda({"transactions": "Transaction[]", "accounts": "Text[]"}, "Summary[]", text,
                   function="account_leaderboard",
                   types={"Transaction": tx, "Summary": summary}, codebase=codebase)
    return Program("algo_algorithm_pipeline", root,
                   {"transactions": transactions, "accounts": accounts}, expected,
                   {text: Plan("calls", steps=steps,
                               note="Filtered, aggregated exactly, and ranked the account totals.")},
                   source_semantics={"algorithm": "filter_aggregate_rank",
                                     "inputs": {"transactions": transactions, "accounts": accounts}})


ALGORITHMS = {"array_kernel": array_kernel, "staged_ranking": staged_ranking,
              "algorithm_pipeline": algorithm_pipeline}
