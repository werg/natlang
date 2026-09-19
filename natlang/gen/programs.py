"""Program families. Each returns a Program: the root node, its inputs, the expected value, and the
plans the reference policy follows, keyed by the exact instruction text of each lambda."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from . import world

LABEL = '"billing" | "technical" | "spam"'
RECORD = "{ customer: Text, order_id: Text, amount: Num, phone?: Text }"


@dataclass
class Plan:
    kind: str                      # leaf | crisp | calls | blocked (note = what is missing)
    gold: Callable = None          # leaf: args -> value
    code: str = ""                 # crisp / map_then_code: TypeScript
    over: str = ""                 # map: path of the list
    item_type: str = "Text"
    result_type: str = ""          # map: element type of the result
    body: str = ""                 # map: instructions of the body lambda
    inputs: dict = field(default_factory=dict)
    note: str = ""
    steps: list = field(default_factory=list)   # calls: [(tool, args)], one per turn
    script: Callable = None        # script: a generator function(lam) yielding turns, receiving each turn's last Result
    template: bool = False         # the gold of this generative leaf is a template stand-in, not yet a teacher-written reference


@dataclass
class Program:
    family: str
    root: dict
    inputs: dict
    expected: Any
    plans: dict                    # instructions text -> Plan
    loader: Callable = None        # () -> root node, for code bases on disk (then `root` and `inputs` are unused)
    capabilities: dict = field(default_factory=dict)
    source_semantics: dict = field(default_factory=dict)  # generator facts before reference-policy rendering
    outcome: str = "done"         # done | blocked | error for scenario training
    expected_effects: list | None = None
    injected_fault: tuple | None = None


BLOCKED = "<blocked>"              # Program.expected when the inputs do not determine the result
P_BLOCKED = 0.12                   # share of classify/extract instances that are undetermined

import json as _json
from pathlib import Path as _Path

_PARA = _Path(__file__).resolve().parents[2] / "data" / "paraphrases.json"
PARAPHRASES = _json.loads(_PARA.read_text()) if _PARA.exists() else {}     # base text -> verified paraphrases
BASE_TEXTS = {}                                                             # family -> its hand-written texts


def _pick(rng, options, family=None, text=None):
    if family:
        BASE_TEXTS[family] = list(options)
    if text is not None:
        return text
    pool = list(options) + [p for o in options for p in PARAPHRASES.get(o, [])]
    return rng.choice(pool)


def judge(rng: random.Random, text=None) -> Program:
    item = world.review(rng)
    text = _pick(rng, ["Is `args/message` a positive review? Answer true or false.",
                       "Decide whether the review in `args/message` is positive.",
                       "Tell me if the customer in `args/message` is happy with the purchase (true or false)."], family="judge", text=text)
    return Program("judge", {"$lambda": {"type": "Lambda<{ message: Text }, Bool>", "instructions": text}},
                   {"message": item["text"]}, item["positive"],
                   {text: Plan("leaf", gold=lambda a, v=item["positive"]: v, note="Judged the review.")})


def classify(rng, text=None) -> Program:
    item = world.ticket(rng)
    text = _pick(rng, ["Label the ticket in `args/ticket` according to `args/rubric`.",
                       "Using the rubric in `args/rubric`, say which category `args/ticket` belongs to.",
                       "Which label from `args/rubric` fits `args/ticket`?"], family="classify", text=text)
    rubric, expected = world.RUBRIC, item["category"]
    plan = Plan("leaf", gold=lambda a, v=item["category"]: v, note="Labelled the ticket.")
    if rng.random() < P_BLOCKED:                # the rubric has no rule for this ticket's category
        rubric = "".join(l + "\n" for l in world.RUBRIC.splitlines() if not l.startswith(item["category"] + ":"))
        what = world.UNCOVERED[item["category"]]
        expected, plan = BLOCKED, Plan("blocked", note=rng.choice([
            f"The rubric has no category for {what}; this ticket is about that.",
            f"`args/rubric` does not cover {what}, which is what the ticket is about.",
            f"None of the labels in the rubric fits: the ticket concerns {what}, and the rubric has no rule for it."]))
    return Program("classify", {"$lambda": {"type": "Lambda<{ ticket: Text, rubric: Text }, Label>",
                                            "types": {"Label": LABEL}, "instructions": text}},
                   {"ticket": item["text"], "rubric": rubric}, expected, {text: plan})


def extract(rng, text=None) -> Program:
    item = world.note(rng, drop=rng.choice(["order_id", "amount"]) if rng.random() < P_BLOCKED else None)
    text = _pick(rng, ["Extract the customer name, the order id, and the refund amount from `args/note`.\n"
                       "Include the phone number only if one is given.",
                       "From `args/note`, pull out who the customer is, the order id, and how much is owed. "
                       "Add the phone number only when the note has one."], family="extract", text=text)
    return Program("extract", {"$lambda": {"type": f"Lambda<{{ note: Text }}, {RECORD}>", "instructions": text}},
                   {"note": item["text"]}, BLOCKED if item["missing"] else item["record"],
                   {text: Plan("blocked", note=item["missing"]) if item["missing"] else
                          Plan("leaf", gold=lambda a, v=item["record"]: v, note="Extracted the record.")})


CRISP = [
    ("Add up the numbers in `args/numbers`.", "sum(args.numbers)", lambda xs: round(sum(xs), 2), "Num"),
    ("How many of the numbers in `args/numbers` are above 100?", "count(args.numbers, x => x > 100)",
     lambda xs: sum(1 for x in xs if x > 100), "Num"),
    ("What is the largest number in `args/numbers`?", "max(args.numbers)", max, "Num"),
    ("What is the average of `args/numbers`, rounded to two decimals?",
     "Math.round(mean(args.numbers) * 100) / 100", lambda xs: round(sum(xs) / len(xs), 2), "Num"),
]


def crisp_scalar(rng, text=None) -> Program:
    BASE_TEXTS["crisp_scalar"] = [c[0] for c in CRISP]
    base, code, fn, ty = rng.choice(CRISP) if text is None else next(c for c in CRISP if c[0] == text["base"])
    text = base if text is None else text["text"]
    if text == base:
        pool = [base] + PARAPHRASES.get(base, [])
        text = rng.choice(pool)
    xs = [rng.choice([rng.randint(1, 400), round(rng.uniform(1, 400), 1)]) for _ in range(rng.randint(3, 9))]
    from .. import js                      # the oracle for exact work is the code itself
    expected = js.run(code, {"args": {"numbers": xs}}, None, body=False, path="gen")
    assert abs(expected - fn(xs)) < 0.011, (code, expected, fn(xs))
    return Program("crisp_scalar", {"$lambda": {"type": f"Lambda<{{ numbers: Num[] }}, {ty}>", "instructions": text}},
                   {"numbers": xs}, expected, {text: Plan("crisp", code=code, note="Computed it with code.")})


CLASSIFY_FN = {"description": "Label one ticket using the rubric.", "args": {"ticket": "Text", "rubric": "Text"},
               "returns": "Label", "instructions": "Label the ticket in `args/ticket` according to `args/rubric`."}
URGENT_FN = {"description": "Is this ticket urgent?", "args": {"ticket": "Text"}, "returns": "Bool",
             "instructions": "Is the ticket in `args/ticket` urgent? Answer true or false."}
COUNT_FN = {"description": "How many flags are true.", "args": {"flags": "Bool[]"}, "returns": "Num",
            "code": "return args.flags.filter(Boolean).length"}


def map_leaf(rng, text=None) -> Program:
    items = world.distinct(rng, world.ticket, rng.randint(3, 7))
    gold = {i["text"]: i["category"] for i in items}
    text = _pick(rng, ["Label each ticket in `args/tickets` according to `args/rubric`.",
                       "Go through `args/tickets` and give every ticket its category from `args/rubric`."], family="map_leaf", text=text)
    return Program("map_leaf", {"$lambda": {"type": "Lambda<{ tickets: Text[], rubric: Text }, Label[]>",
                                            "types": {"Label": LABEL}, "instructions": text,
                                            "codebase": {"classify": CLASSIFY_FN}}},
                   {"tickets": [i["text"] for i in items], "rubric": world.RUBRIC}, [i["category"] for i in items],
                   {text: Plan("calls", steps=[("call", {"function": "classify", "to": "return", "over": "args/tickets",
                                                        "inputs": {"rubric": "args/rubric"}})],
                               note="Labelled every ticket with classify."),
                    "classify": Plan("leaf", gold=lambda a: gold[a["ticket"]], note="Labelled the ticket.")})


def map_then_count(rng, text=None) -> Program:
    items = world.distinct(rng, world.urgency_ticket, rng.randint(3, 8))
    gold = {i["text"]: i["urgent"] for i in items}
    text = _pick(rng, ["How many of the tickets in `args/tickets` are urgent?",
                       "Count the urgent tickets in `args/tickets`."], family="map_then_count", text=text)
    return Program("map_then_count", {"$lambda": {"type": "Lambda<{ tickets: Text[] }, Num>", "instructions": text,
                                                  "codebase": {"is_urgent": URGENT_FN, "count_true": COUNT_FN}}},
                   {"tickets": [i["text"] for i in items]}, sum(1 for i in items if i["urgent"]),
                   {text: Plan("calls", steps=[("call", {"function": "is_urgent", "to": "let/flags", "over": "args/tickets"}),
                                               ("call", {"function": "count_true", "to": "return",
                                                         "inputs": {"flags": "let/flags"}})],
                               note="Judged each ticket with is_urgent, then counted with count_true."),
                    "is_urgent": Plan("leaf", gold=lambda a: gold[a["ticket"]], note="Judged the ticket.")})


FAMILIES = {f.__name__: f for f in (judge, classify, extract, crisp_scalar, map_leaf, map_then_count)}
