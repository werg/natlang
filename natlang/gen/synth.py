"""Program synthesizer: pseudocode programs with code bases, composed from a small set of constructs.

A program is a list of Steps over a latent world. Every Step knows three things:
  - how it reads in each pseudocode dialect (the text is a *rendering* of the structure),
  - what it computes (the Python twin: gold answers come from hidden attributes, never from the text),
  - which tool calls carry it out (the reference policy compiles from the structure, never from the text).
Constructs: for-each calls, plain calls, exact glue, filters, if/else, nested pseudocode functions with their
own code bases, aggregation into records.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .programs import Plan, Program

# ------------------------------------------------------------------------------------------ latent world
PRODUCTS = ["kettle", "headphones", "desk lamp", "backpack", "router", "blender", "keyboard", "tent"]
T_BODY = {
    "billing": ["I was charged twice for my {m} invoice", "My refund for the {p} still has not arrived",
                "There is a fee on my statement that nobody can explain", "Please change the card on file, the old one expired"],
    "technical": ["The app crashes every time I open the settings page", "Export to CSV produces an empty file",
                  "The {p} will not pair with my phone any more", "Login fails with error 500"],
    "account": ["I cannot change the email address on my profile", "Please delete my account and all my data",
                "Two-factor codes never arrive on my new number", "I need to add a colleague to our team plan"],
}
T_URGENT = [" - this is blocking our whole team right now.", ". We are losing orders every minute.",
            "; every customer of ours is affected as of this morning."]
T_CALM = [". No rush.", ", whenever you get to it.", ". Not a big deal, just letting you know."]
T_ANGRY = [" This is unacceptable.", " I am furious.", " Worst service I have ever had."]
R_TOPIC = {
    "delivery": (["Arrived two days early, well packed", "The courier was friendly and on time"],
                 ["Delivery took three weeks", "The box arrived crushed and soaked"]),
    "quality": (["The {p} feels solid and works perfectly", "Better built than I expected"],
                ["The {p} broke after two days", "Cheap plastic, the handle snapped off"]),
    "price": (["Great value for what you get", "Half the price of the competition and just as good"],
              ["Far too expensive for what it is", "I found the same {p} much cheaper elsewhere"]),
}
E_CAT = {"travel": ["Taxi to the airport", "Train to the client site", "Hotel, one night"],
         "meals": ["Team lunch", "Dinner with the client", "Coffee for the workshop"],
         "equipment": ["Replacement keyboard", "USB-C dock", "Monitor cable"]}
MONTHS = ["January", "March", "June", "October"]


def _ticket(rng):
    cat, urgent, angry = rng.choice(list(T_BODY)), rng.random() < 0.4, rng.random() < 0.3
    text = rng.choice(T_BODY[cat]).format(m=rng.choice(MONTHS), p=rng.choice(PRODUCTS))
    text += rng.choice(T_URGENT if urgent else T_CALM) + (rng.choice(T_ANGRY) if angry else "")
    return {"text": text, "category": cat, "urgent": urgent, "angry": angry}


def _review(rng):
    topic, pos = rng.choice(list(R_TOPIC)), rng.random() < 0.5
    text = rng.choice(R_TOPIC[topic][0 if pos else 1]).format(p=rng.choice(PRODUCTS)) + "."
    return {"text": text, "topic": topic, "positive": pos}


def _expense(rng):
    cat, amount, receipt = rng.choice(list(E_CAT)), round(rng.uniform(4, 240), 2), rng.random() < 0.7
    text = f"{rng.choice(E_CAT[cat])}, {amount:.2f} EUR, " + ("receipt attached." if receipt else "no receipt.")
    return {"text": text, "claim": {"category": cat, "amount": amount, "receipt": receipt}}


def _distinct(rng, make, n):
    out, seen = [], set()
    while len(out) < n:
        x = make(rng)
        if x["text"] not in seen:
            seen.add(x["text"])
            out.append(x)
    return out


# ------------------------------------------------------------------------------------------ functions
T_RUBRIC = ("billing: charges, invoices, refunds, payment methods.\n"
            "technical: the product or the app not working.\n"
            "account: profile, login settings, team membership, data requests.\n")
TYPES = {"Label": '"billing" | "technical" | "account"', "Topic": '"delivery" | "quality" | "price"',
         "Claim": '{ category: "travel" | "meals" | "equipment", amount: Num, receipt: Bool }',
         "Assessment": "{ label: Label, urgent: Bool }"}

# name -> (aliases, args, returns, instructions, oracle(args, hidden) -> value)
LEAVES = {
    "classify": (["classify", "label_ticket", "categorize"], {"ticket": "Text", "rubric": "Text"}, "Label",
                 "Pick the label from `args/rubric` that fits the ticket in `args/ticket`.", lambda h: h["category"]),
    "is_urgent": (["is_urgent", "needs_attention_now", "urgent"], {"ticket": "Text"}, "Bool",
                  "Is the ticket in `args/ticket` urgent: are people blocked or is money being lost right now? "
                  "Answer true or false.", lambda h: h["urgent"]),
    "is_angry": (["is_angry", "customer_is_upset"], {"ticket": "Text"}, "Bool",
                 "Does the customer in `args/ticket` sound angry? Answer true or false.", lambda h: h["angry"]),
    "is_positive": (["is_positive", "likes_it", "happy_customer"], {"review": "Text"}, "Bool",
                    "Is the review in `args/review` positive? Answer true or false.", lambda h: h["positive"]),
    "topic_of": (["topic_of", "what_about", "review_topic"], {"review": "Text"}, "Topic",
                 "What is the review in `args/review` mainly about: delivery, quality, or price?", lambda h: h["topic"]),
    "read_claim": (["read_claim", "parse_expense", "extract_claim"], {"note": "Text"}, "Claim",
                   "Extract the expense claim in `args/note`: its category, the amount in EUR, and whether a "
                   "receipt is attached.", lambda h: h["claim"]),
}
DESCRIPTIONS = {"classify": "Label one ticket using the rubric.", "is_urgent": "Is this ticket urgent?",
                "is_angry": "Does the customer sound angry?", "is_positive": "Is this review positive?",
                "topic_of": "What one review is mainly about.", "read_claim": "Extract one expense claim from a note."}
STD = {
    "count_true": ({"flags": "Bool[]"}, "Num", "How many flags are true.", "return args.flags.filter(Boolean).length"),
    "select_by_flags": ({"items": "Text[]", "flags": "Bool[]"}, "Text[]", "The items whose flag (same position) is true.",
                        "return args.items.filter((_, i) => args.flags[i])"),
    "group_count": ({"values": "Text[]"}, "Dict<Num>", "How often each value occurs.",
                    "const out = {}\nfor (const v of args.values) out[v] = (out[v] || 0) + 1\nreturn out"),
}


def _fn_doc(name, args, returns, desc, body, kind="instructions", codebase=None):
    d = {"description": desc, "args": dict(args), "returns": returns, kind: body}
    if codebase:
        d["codebase"] = codebase
    return d


# ------------------------------------------------------------------------------------------ steps
@dataclass
class Ctx:
    rng: random.Random
    hidden: dict                       # item text -> hidden attributes
    names: dict = field(default_factory=dict)      # canonical function name -> alias used in this program
    env: dict = field(default_factory=dict)        # twin: local name -> value
    fns: dict = field(default_factory=dict)        # code base of the root, as inline docs
    plans: dict = field(default_factory=dict)
    lines_a: list = field(default_factory=list)    # dialect A: Python-like
    lines_b: list = field(default_factory=list)    # dialect B: numbered steps in English
    calls: list = field(default_factory=list)      # reference steps of the root
    indent: int = 0

    def use_leaf(self, canon: str, into: Optional[dict] = None) -> str:
        aliases, args, returns, body, oracle = LEAVES[canon]
        alias = self.names.setdefault(canon, self.rng.choice(aliases))
        item_param = next(iter(args))
        (self.fns if into is None else into)[alias] = _fn_doc(alias, args, returns, DESCRIPTIONS[canon], body)
        self.plans[alias] = Plan("leaf", gold=lambda a, o=oracle, p=item_param: o(self.hidden[a[p]]),
                                 note="Done.")
        return alias

    def use_std(self, name: str) -> str:
        args, returns, desc, code = STD[name]
        self.fns[name] = _fn_doc(name, args, returns, desc, code, kind="code")
        return name

    def say(self, a: str, b: str):
        pad = "    " * self.indent
        self.lines_a.append(pad + a)
        self.lines_b.append(pad + b)


def _ref(path: str) -> str:            # how a path reads inside pseudocode
    return path.split("/", 1)[1] if path.startswith(("args/", "let/")) else path


def map_leaf(c: Ctx, out: str, canon: str, over: str, items: list, extra: Optional[dict] = None) -> list:
    """out = for each x in over: leaf(x, extra...)"""
    fn = c.use_leaf(canon)
    oracle = LEAVES[canon][4]
    extra = extra or {}
    args_a = ", ".join(["x"] + [_ref(p) for p in extra.values()])
    c.say(f"{out} = for each x in {_ref(over)}: {fn}({args_a})",
          f"For every item of {_ref(over)}, call {fn}" + (f" (with {', '.join(_ref(p) for p in extra.values())})" if extra else "")
          + f"; keep the results as {out}.")
    step = {"function": fn, "to": f"let/{out}", "over": over}
    if extra:
        step["inputs"] = dict(extra)
    c.calls.append(("call", step))
    c.env[out] = [oracle(c.hidden[t]) for t in items]
    return c.env[out]


def glue(c: Ctx, out: str, ty: str, code: str, value: Any, a: str, b: str, to: Optional[str] = None):
    """Exact work that no function covers: run_code, then keep the result."""
    from .. import js                                  # the oracle for exact work is the code itself
    ran = js.run(code, {"let": {k: v for k, v in c.env.items()}, "args": {}}, None, body=False, path="gen")
    assert ran == value or (isinstance(value, float) and abs(ran - value) < 0.011), (code, ran, value)
    c.say(a, b)
    c.calls.append(("glue", code, to or f"let/{out}", ty))
    c.env[out] = ran
    return ran


def select(c: Ctx, out: str, items_path: str, items: list, flags_name: str) -> list:
    fn = c.use_std("select_by_flags")
    c.say(f"{out} = {fn}({_ref(items_path)}, {flags_name})",
          f"Keep the items of {_ref(items_path)} whose flag in {flags_name} is true ({fn}); call them {out}.")
    c.calls.append(("call", {"function": fn, "to": f"let/{out}", "inputs": {"items": items_path, "flags": f"let/{flags_name}"}}))
    c.env[out] = [t for t, f in zip(items, c.env[flags_name]) if f]
    return c.env[out]


def call_std(c: Ctx, out_path: str, name: str, inputs: dict, value: Any, a: str, b: str):
    c.use_std(name)
    c.say(a, b)
    c.calls.append(("call", {"function": name, "to": out_path, "inputs": inputs}))
    return value


# ------------------------------------------------------------------------------------------ program shapes
def _finish(c: Ctx, family: str, sig_args: dict, returns: str, inputs: dict, expected: Any, types: list,
            fn_name: str, note: str) -> Program:
    dialect = c.rng.choice("ab")
    header = f"function {fn_name}(" + ", ".join(sig_args) + f") -> {returns}"
    if dialect == "a":
        text = header + "\n\n" + "\n".join("  " + l for l in c.lines_a)
    else:
        text = header + "\n\n" + "\n".join(f"{i}. {l}" if not l.startswith(" ") else f"   {l.strip()}"
                                           for i, l in enumerate(c.lines_b, 1))
    calls = list(c.calls)
    c.plans[fn_name] = Plan("calls", steps=calls, note=note)
    root = {"$lambda": {"type": "Lambda<{ " + ", ".join(f"{n}: {t}" for n, t in sig_args.items()) + f" }}, {returns}>",
                        "types": {t: TYPES[t] for t in types}, "instructions": text, "codebase": c.fns,
                        "function": fn_name}}
    return Program(family, root, inputs, expected, c.plans)


def ticket_report(rng: random.Random) -> Program:
    """classify all -> filter one category -> urgency of those -> if none: constant, else count + list."""
    items = _distinct(rng, _ticket, rng.randint(4, 9))
    c = Ctx(rng, {i["text"]: i for i in items})
    texts = [i["text"] for i in items]
    labels = map_leaf(c, "labels", "classify", "args/tickets", texts, {"rubric": "args/rubric"})
    cat = rng.choice(list(T_BODY))
    flags = [l == cat for l in labels]
    glue(c, "is_cat", "Bool[]", f"locals.labels.map(l => l === {json.dumps(cat)})", flags,
         f'is_cat = for each l in labels: l == "{cat}"          # exact: use code',
         f'With code, turn labels into flags that are true where the label is "{cat}"; call them is_cat.')
    chosen = select(c, "chosen", "args/tickets", texts, "is_cat")
    urgent = map_leaf(c, "urgent_flags", "is_urgent", "let/chosen", chosen) if chosen else None
    if chosen:
        n = call_std(c, "return/urgent", "count_true", {"flags": "let/urgent_flags"}, sum(urgent),
                     "urgent = count_true(urgent_flags)", "Count the true flags with count_true: that is `urgent`.")
    else:                      # the policy still has to notice that the list is empty: the Map over [] is skipped
        c.lines_a.append(""); n = 0
    by_label = {}
    for l in labels:
        by_label[l] = by_label.get(l, 0) + 1
    call_std(c, "return/by_label", "group_count", {"values": "let/labels"}, by_label,
             "by_label = group_count(labels)", "Count how often each label occurs with group_count: `by_label`.")
    c.say(f"return {{ category: \"{cat}\", urgent, by_label }}",
          f'Return the record: category "{cat}", urgent, by_label.')
    if not chosen:
        c.calls = [s for s in c.calls if not (s[0] == "call" and s[1].get("to") == "let/urgent_flags")]
        c.calls.append(("write", {"path": "return/urgent", "type": "Num", "value": 0}))
    c.calls.append(("write", {"path": "return/category", "type": "Label", "value": cat}))
    return _finish(c, "ticket_report", {"tickets": "Text[]", "rubric": "Text"},
                   "{ category: Label, urgent: Num, by_label: Dict<Num> }",
                   {"tickets": texts, "rubric": T_RUBRIC}, {"category": cat, "urgent": n, "by_label": by_label},
                   ["Label"], rng.choice(["ticket_report", "weekly_report", "category_report"]),
                   "Followed the program: classified, filtered, judged urgency, counted.")


def review_digest(rng: random.Random) -> Program:
    """sentiment of all -> the negative ones -> their topics -> which topic dominates (exact)."""
    items = _distinct(rng, _review, rng.randint(4, 9))
    c = Ctx(rng, {i["text"]: i for i in items})
    texts = [i["text"] for i in items]
    pos = map_leaf(c, "positive", "is_positive", "args/reviews", texts)
    neg_flags = [not p for p in pos]
    glue(c, "negative", "Bool[]", "locals.positive.map(p => !p)", neg_flags,
         "negative = for each p in positive: not p          # exact: use code",
         "With code, negate every flag of positive; call the result negative.")
    bad = select(c, "complaints", "args/reviews", texts, "negative")
    expected = {"complaints": len(bad)}
    expected["share_positive"] = glue(c, "share", "Num", "Math.round(locals.positive.filter(Boolean).length / locals.positive.length * 100) / 100",
         round(sum(pos) / len(pos), 2), "share_positive = share of true in positive, rounded to 2 decimals   # exact",
         "With code, compute the share of true flags in positive, rounded to two decimals: `share_positive`.",
         to="return/share_positive")
    glue(c, "n_bad", "Num", "locals.complaints.length", len(bad), "complaints_count = length of complaints",
         "With code, take the number of complaints.", to="return/complaints")
    if bad:
        topics = map_leaf(c, "topics", "topic_of", "let/complaints", bad)
        counts = {}
        for t in topics:
            counts[t] = counts.get(t, 0) + 1
        call_std(c, "return/by_topic", "group_count", {"values": "let/topics"}, counts,
                 "by_topic = group_count(topics)", "Count the topics with group_count: `by_topic`.")
        expected["by_topic"] = counts
    else:
        c.calls.append(("write", {"path": "return/by_topic", "type": "Dict<Num>", "value": {}}))
        expected["by_topic"] = {}
    c.say("return { share_positive, complaints: complaints_count, by_topic }     # by_topic is {} when nobody complained",
          "Return share_positive, the number of complaints, and by_topic (empty when nobody complained).")
    return _finish(c, "review_digest", {"reviews": "Text[]"},
                   "{ share_positive: Num, complaints: Num, by_topic: Dict<Num> }", {"reviews": texts}, expected,
                   ["Topic"], rng.choice(["review_digest", "summarize_reviews"]),
                   "Judged every review, looked at the topics of the complaints, computed the numbers with code.")


def expense_audit(rng: random.Random) -> Program:
    """extract every claim -> exact policy check in code -> totals; approve only if nothing is flagged."""
    items = _distinct(rng, _expense, rng.randint(3, 8))
    c = Ctx(rng, {i["text"]: i for i in items})
    texts = [i["text"] for i in items]
    claims = map_leaf(c, "claims", "read_claim", "args/notes", texts)
    limit = rng.choice([50, 80, 100, 150])
    bad = [(not k["receipt"]) or k["amount"] > limit for k in claims]
    glue(c, "flagged", "Bool[]", f"locals.claims.map(k => !k.receipt || k.amount > {limit})", bad,
         f"flagged = for each k in claims: k has no receipt or k.amount > {limit}     # exact: use code",
         f"With code, flag every claim that has no receipt or is above {limit} EUR; call the flags `flagged`.")
    total = round(sum(k["amount"] for k, b in zip(claims, bad) if not b), 2)
    total = glue(c, "payable", "Num", "Math.round(locals.claims.filter((k, i) => !locals.flagged[i]).reduce((s, k) => s + k.amount, 0) * 100) / 100",
         total, "payable = sum of amount over the claims that are not flagged, rounded to cents     # exact",
         "With code, add up the amounts of the claims that are not flagged, rounded to cents: `payable`.",
         to="return/payable")
    n = call_std(c, "return/flagged", "count_true", {"flags": "let/flagged"}, sum(bad),
                 "flagged_count = count_true(flagged)", "Count the flagged claims with count_true.")
    decision = "approve" if not any(bad) else "review"
    c.say('if flagged_count == 0: decision = "approve"', 'If nothing is flagged, the decision is "approve".')
    c.say('else: decision = "review"', 'Otherwise the decision is "review".')
    c.say("return { payable, flagged: flagged_count, decision }", "Return payable, the number flagged, and the decision.")
    c.calls.append(("write", {"path": "return/decision", "type": '"approve" | "review"', "value": decision}))
    return _finish(c, "expense_audit", {"notes": "Text[]"},
                   '{ payable: Num, flagged: Num, decision: "approve" | "review" }', {"notes": texts},
                   {"payable": total, "flagged": n, "decision": decision}, ["Claim"],
                   rng.choice(["audit_expenses", "expense_audit"]),
                   "Extracted every claim, checked the policy with code, decided.")


def nested_assessment(rng: random.Random) -> Program:
    """A pseudocode function with its own code base, called for every item; then exact aggregation."""
    items = _distinct(rng, _ticket, rng.randint(3, 7))
    c = Ctx(rng, {i["text"]: i for i in items})
    texts = [i["text"] for i in items]
    inner: dict = {}
    f_cls, f_urg = c.use_leaf("classify", into=inner), c.use_leaf("is_urgent", into=inner)
    assess = rng.choice(["assess", "triage_one", "look_at_ticket"])
    body = (f"function {assess}(ticket, rubric) -> Assessment\n\n  label = {f_cls}(ticket, rubric)\n"
            f"  urgent = {f_urg}(ticket)\n  return {{ label, urgent }}")
    c.fns[assess] = _fn_doc(assess, {"ticket": "Text", "rubric": "Text"}, "Assessment",
                            "Label one ticket and say whether it is urgent.", body, codebase=inner)
    c.plans[assess] = Plan("calls", note="Labelled the ticket and judged its urgency.", steps=[
        ("call", {"function": f_cls, "to": "return/label", "inputs": {"ticket": "args/ticket", "rubric": "args/rubric"}}),
        ("call", {"function": f_urg, "to": "return/urgent", "inputs": {"ticket": "args/ticket"}})])
    c.say(f"reports = for each t in tickets: {assess}(t, rubric)",
          f"For every ticket, call {assess} with the rubric; keep the results as reports.")
    c.calls.append(("call", {"function": assess, "to": "let/reports", "over": "args/tickets", "inputs": {"rubric": "args/rubric"}}))
    c.env["reports"] = [{"label": i["category"], "urgent": i["urgent"]} for i in items]
    cat = rng.choice(list(T_BODY))
    n = sum(1 for i in items if i["urgent"] and i["category"] == cat)
    glue(c, "n", "Num", f"locals.reports.filter(r => r.urgent && r.label === {json.dumps(cat)}).length", n,
         f'return the number of reports that are urgent and labelled "{cat}"          # exact: use code',
         f'With code, count the reports that are urgent and labelled "{cat}". Return that number.', to="return")
    return _finish(c, "nested_assessment", {"tickets": "Text[]", "rubric": "Text"}, "Num",
                   {"tickets": texts, "rubric": T_RUBRIC}, n, ["Label", "Assessment"],
                   rng.choice(["urgent_in_category", "count_hot_tickets"]),
                   "Assessed every ticket with the nested function, then counted with code.")


SHAPES = {f.__name__: f for f in (ticket_report, review_digest, expense_audit, nested_assessment)}


# ------------------------------------------------------------------------------------------ composition
# Instead of fixed shapes: sample a program move by move. A Var is something the program has computed; a move
# is applicable when the variables it needs exist with the right kind and alignment. Every move renders itself,
# updates the twin, and appends the reference calls, exactly like the fixed shapes above.
@dataclass
class Var:
    name: str
    path: str
    kind: str                  # texts | flags | labels | topics | claims | reports
    value: Any
    domain: str = ""           # tickets | reviews | expenses   (for texts)
    base: str = ""             # name of the texts variable this one is aligned with, item by item
    note: str = ""             # what the flags mean, for names and prose


DOMAINS = {"tickets": (_ticket, ["classify", "is_urgent", "is_angry"]),
           "reviews": (_review, ["is_positive", "topic_of"]),
           "expenses": (_expense, ["read_claim"])}
LEAF_KIND = {"classify": "labels", "is_urgent": "flags", "is_angry": "flags", "is_positive": "flags",
             "topic_of": "topics", "read_claim": "claims"}
LEAF_LOCAL = {"classify": ["labels", "categories", "kinds"], "is_urgent": ["urgent", "hot", "urgent_flags"],
              "is_angry": ["angry", "upset"], "is_positive": ["positive", "happy"], "topic_of": ["topics", "subjects"],
              "read_claim": ["claims", "expenses"]}
STD.update({
    "add_amount": ({"acc": "Num", "item": "Claim"}, "Num", "Add one claim's amount to a running total.",
                   "return Math.round((args.acc + args.item.amount) * 100) / 100"),
    "raise_limit": ({"state": "Budget"}, "Budget", "Raise the limit by 25.",
                    "return { ...args.state, limit: args.state.limit + 25 }"),
    "covers": ({"state": "Budget"}, "Bool", "Do the amounts within the limit add up to the target?",
               "return args.state.amounts.filter(a => a <= args.state.limit).reduce((s, a) => s + a, 0) >= args.state.target"),
})
TYPES["Budget"] = "{ limit: Num, amounts: Num[], target: Num }"


class Composer:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.domain = rng.choice(list(DOMAINS))
        make, self.leaves = DOMAINS[self.domain]
        items = _distinct(rng, make, rng.randint(4, 10))
        self.c = Ctx(rng, {i["text"]: i for i in items})
        self.arg = {"tickets": "tickets", "reviews": "reviews", "expenses": "notes"}[self.domain]
        self.vars = {self.arg: Var(self.arg, f"args/{self.arg}", "texts", [i["text"] for i in items], self.domain, self.arg)}
        self.c.env[self.arg] = self.vars[self.arg].value
        self.fields: dict = {}          # return field -> (type text, value)
        self.types: set = set()
        self.applied: set = set()
        self.used: set = set()

    # -- helpers
    def fresh(self, options):
        for n in self.rng.sample(options, len(options)):
            if n not in self.vars and n not in self.fields:
                return n
        return f"{options[0]}_{len(self.vars)}"

    def add(self, v: Var):
        self.vars[v.name] = v
        self.c.env[v.name] = v.value
        return v

    def use(self, *vs):
        self.used.update(v.name for v in vs)

    def of_kind(self, *kinds):
        return [v for v in self.vars.values() if v.kind in kinds]

    # -- moves: each returns True when it did something
    def m_leaf(self):
        cands = [(t, l) for t in self.of_kind("texts") if t.value for l in self.leaves if (t.name, l) not in self.applied]
        if not cands:
            return False
        t, leaf = self.rng.choice(cands)
        self.applied.add((t.name, leaf))
        name = self.fresh(LEAF_LOCAL[leaf])
        extra = {"rubric": "args/rubric"} if leaf == "classify" else None
        value = map_leaf(self.c, name, leaf, t.path, t.value, extra)
        self.types.update({"classify": ["Label"], "topic_of": ["Topic"], "read_claim": ["Claim"]}.get(leaf, []))
        self.add(Var(name, f"let/{name}", LEAF_KIND[leaf], value, base=t.name, note=leaf))
        return True

    def m_flags_from(self):
        cands = [v for v in self.of_kind("labels", "topics", "claims") if ("flags", v.name) not in self.applied]
        if not cands:
            return False
        v = self.rng.choice(cands)
        self.use(v)
        self.applied.add(("flags", v.name))
        if v.kind == "claims":
            limit = self.rng.choice([40, 60, 100, 150])
            which = self.rng.choice(["over", "no_receipt", "either"])
            js_, py, words = {
                "over": (f"k.amount > {limit}", lambda k: k["amount"] > limit, f"k.amount > {limit}"),
                "no_receipt": ("!k.receipt", lambda k: not k["receipt"], "k has no receipt"),
                "either": (f"!k.receipt || k.amount > {limit}", lambda k: (not k["receipt"]) or k["amount"] > limit,
                           f"k has no receipt or k.amount > {limit}")}[which]
            name = self.fresh(["flagged", "problem", "suspicious"])
            glue(self.c, name, "Bool[]", f"locals.{v.name}.map(k => {js_})", [py(k) for k in v.value],
                 f"{name} = for each k in {v.name}: {words}          # exact: use code",
                 f"With code, flag every item of {v.name} where {words}; call the flags {name}.")
        else:
            label = self.rng.choice(sorted(set(v.value)) or ["billing"])
            name = self.fresh([f"is_{label}", f"{label}_flags"])
            glue(self.c, name, "Bool[]", f"locals.{v.name}.map(l => l === {json.dumps(label)})", [l == label for l in v.value],
                 f'{name} = for each l in {v.name}: l == "{label}"          # exact: use code',
                 f'With code, turn {v.name} into flags that are true where the value is "{label}"; call them {name}.')
        self.add(Var(name, f"let/{name}", "flags", self.c.env[name], base=v.base, note=name))
        return True

    def m_combine(self):
        fl = self.of_kind("flags")
        pairs = [(a, b) for a in fl for b in fl if a.name < b.name and a.base == b.base and ("and", a.name, b.name) not in self.applied]
        if not pairs:
            return False
        a, b = self.rng.choice(pairs)
        self.use(a, b)
        self.applied.add(("and", a.name, b.name))
        op, word, py = self.rng.choice([("&&", "and", lambda x, y: x and y), ("||", "or", lambda x, y: x or y)])
        name = self.fresh(["both", "combined", "either_flag"] if op == "&&" else ["either_flag", "combined", "any_flag"])
        glue(self.c, name, "Bool[]", f"locals.{a.name}.map((x, i) => x {op} locals.{b.name}[i])",
             [py(x, y) for x, y in zip(a.value, b.value)],
             f"{name} = position by position: {a.name} {word} {b.name}          # exact: use code",
             f"With code, combine {a.name} and {b.name} position by position with '{word}'; call the result {name}.")
        self.add(Var(name, f"let/{name}", "flags", self.c.env[name], base=a.base))
        return True

    def m_select_block(self):
        """Filter a list, then (only if something is left) judge what is left and aggregate: an if/else."""
        cands = [f for f in self.of_kind("flags") if f.base == self.arg and ("select", f.name) not in self.applied]
        if not cands:
            return False
        f = self.rng.choice(cands)
        rest = [l for l in self.leaves if LEAF_KIND[l] in ("flags", "labels", "topics") and l != f.note]
        if not rest:
            return False
        self.applied.add(("select", f.name))
        self.use(f)
        base = self.vars[f.base]
        name = self.fresh(["chosen", "selected", "subset", "shortlist"])
        sel = select(self.c, name, base.path, base.value, f.name)
        leaf = self.rng.choice(rest)
        field = self.fresh({"flags": ["matching", "hits"], "labels": ["breakdown", "by_kind"], "topics": ["by_topic", "topic_counts"]}[LEAF_KIND[leaf]])
        inner = self.fresh(LEAF_LOCAL[leaf])
        self.c.say(f"if {name} is empty:", f"If {name} is empty:")
        self.c.indent += 1
        empty_value = 0 if LEAF_KIND[leaf] == "flags" else {}
        self.c.say(f"{field} = {json.dumps(empty_value)}", f"{field} is {json.dumps(empty_value)}.")
        self.c.indent -= 1
        self.c.say("else:", "Otherwise:")
        self.c.indent += 1
        if sel:
            extra = {"rubric": "args/rubric"} if leaf == "classify" else None
            values = map_leaf(self.c, inner, leaf, f"let/{name}", sel, extra)
            std = "count_true" if LEAF_KIND[leaf] == "flags" else "group_count"
            result = sum(values) if std == "count_true" else {k: values.count(k) for k in dict.fromkeys(values)}
            call_std(self.c, f"return/{field}", std, {("flags" if std == "count_true" else "values"): f"let/{inner}"}, result,
                     f"{field} = {std}({inner})", f"Apply {std} to {inner}: that is {field}.")
        else:                                   # render the branch that is not taken, but do not carry it out
            fn = self.c.use_leaf(leaf)
            std = self.c.use_std("count_true" if LEAF_KIND[leaf] == "flags" else "group_count")
            self.c.say(f"{inner} = for each x in {name}: {fn}(x" + (", rubric)" if leaf == "classify" else ")"),
                       f"For every item of {name}, call {fn}; keep the results as {inner}.")
            self.c.say(f"{field} = {std}({inner})", f"Apply {std} to {inner}: that is {field}.")
            result = empty_value
            self.c.calls.append(("write", {"path": f"return/{field}", "type": "Num" if result == 0 else "Dict<Num>", "value": result}))
        self.c.indent -= 1
        self.types.update({"classify": ["Label"], "topic_of": ["Topic"]}.get(leaf, []))
        self.fields[field] = ("Num" if LEAF_KIND[leaf] == "flags" else "Dict<Num>", result)
        return True

    def m_aggregate(self):
        cands = [v for v in self.of_kind("flags", "labels", "topics", "claims") if ("agg", v.name) not in self.applied]
        if not cands:
            return False
        unused = [v for v in cands if v.name not in self.used]
        v = self.rng.choice(unused or cands)
        self.use(v)
        self.applied.add(("agg", v.name))
        if v.kind == "flags":
            if self.rng.random() < 0.6:
                field = self.fresh([f"n_{v.name}", f"{v.name}_count", "count"])
                value = call_std(self.c, f"return/{field}", "count_true", {"flags": v.path}, sum(v.value),
                                 f"{field} = count_true({v.name})", f"Count the true flags of {v.name} with count_true: {field}.")
            else:
                field = self.fresh([f"share_{v.name}", "share"])
                value = glue(self.c, field, "Num", f"Math.round(locals.{v.name}.filter(Boolean).length / locals.{v.name}.length * 100) / 100",
                             round(sum(v.value) / len(v.value), 2), f"{field} = share of true in {v.name}, rounded to 2 decimals     # exact",
                             f"With code, compute the share of true flags in {v.name}, rounded to two decimals: {field}.", to=f"return/{field}")
            self.fields[field] = ("Num", value)
        elif v.kind in ("labels", "topics"):
            field = self.fresh([f"by_{v.name}", "counts", "tally"])
            value = call_std(self.c, f"return/{field}", "group_count", {"values": v.path},
                             {k: v.value.count(k) for k in dict.fromkeys(v.value)},
                             f"{field} = group_count({v.name})", f"Count how often each value of {v.name} occurs with group_count: {field}.")
            self.fields[field] = ("Dict<Num>", value)
        else:
            self.claims_aggregate(v)
        return True

    def claims_aggregate(self, v: Var):
        if self.rng.random() < 0.5:                         # a fold with a crisp step
            self.c.use_std("add_amount")
            field = self.fresh(["total", "sum_claimed"])
            total = 0
            for k in v.value:
                total = round(total + k["amount"], 2)
            from .. import js
            ran = 0
            for k in v.value:
                ran = js.run(STD["add_amount"][3], {"args": {"acc": ran, "item": k}}, None, body=True, path="gen")
            self.c.say(f"{field} = carry a total through {v.name}, starting at 0: add_amount(acc, item)",
                       f"Carry a running total through {v.name}, starting at 0, with add_amount: {field}.")
            self.c.calls.append(("call", {"function": "add_amount", "to": f"return/{field}", "over": v.path, "init": 0}))
            self.fields[field] = ("Num", ran)
        else:                                               # repeat until a check holds
            self.c.use_std("raise_limit"), self.c.use_std("covers")
            self.types.add("Budget")
            amounts = [k["amount"] for k in v.value]
            target = round(sum(sorted(amounts)[: max(1, len(amounts) // 2)]), 2)
            start = {"limit": 0, "amounts": amounts, "target": target}
            from .. import js                                # the oracle for exact work is the code itself
            state = start
            for _ in range(12):
                state = js.run(STD["raise_limit"][3], {"args": {"state": state}}, None, body=True, path="gen")
                if js.run(STD["covers"][3], {"args": {"state": state}}, None, body=True, path="gen"):
                    break
            limit = state["limit"]
            name = self.fresh(["budget", "plan"])
            glue(self.c, name, "Budget", f"({{ limit: 0, amounts: locals.{v.name}.map(k => k.amount), target: {target} }})", start,
                 f"{name} = {{ limit: 0, amounts: the amounts of {v.name}, target: {target} }}          # exact: use code",
                 f"With code, build {name}: limit 0, the amounts of {v.name}, target {target}.")
            final = self.fresh(["enough", "settled_budget"])
            self.c.say(f"{final} = repeat at most 12 times, until covers(state): state = raise_limit(state), starting from {name}",
                       f"Starting from {name}, repeat raise_limit until covers says true, at most 12 times: {final}.")
            self.c.calls.append(("call", {"function": "raise_limit", "to": f"let/{final}", "init": f"let/{name}", "until": "covers", "max": 12}))
            self.c.env[final] = {**start, "limit": limit}
            field = self.fresh(["limit_needed", "limit"])
            glue(self.c, field, "Num", f"locals.{final}.limit", limit, f"{field} = {final}.limit", f"{field} is the limit of {final}.",
                 to=f"return/{field}")
            self.fields[field] = ("Num", limit)

    def build(self) -> Program:
        rng = self.rng
        self.m_leaf()
        for _ in range(rng.randint(2, 6)):
            moves = [self.m_leaf, self.m_flags_from, self.m_flags_from, self.m_combine, self.m_select_block, self.m_aggregate]
            rng.shuffle(moves)
            next((m for m in moves if m()), None)
        dangling = lambda: [v for v in self.of_kind("flags", "labels", "topics", "claims") if v.name not in self.used]
        while not self.fields or dangling():                # nothing is computed for nothing
            if not self.m_aggregate():
                break
        c = self.c
        if len(self.fields) == 1 and rng.random() < 0.5:        # a single result: return it directly
            (field, (ty, value)), = self.fields.items()
            c.calls = [(s[0], {**s[1], "to": "return"}) if s[0] == "call" and s[1].get("to") == f"return/{field}" else
                       ("glue", s[1], "return", s[3]) if s[0] == "glue" and s[2] == f"return/{field}" else
                       (s[0], {**s[1], "path": "return"}) if s[0] == "write" and s[1].get("path") == f"return/{field}" else s
                       for s in c.calls]
            c.say(f"return {field}", f"Return {field}.")
            returns, expected = ty, value
        else:
            c.say("return { " + ", ".join(self.fields) + " }", "Return the record: " + ", ".join(self.fields) + ".")
            returns = "{ " + ", ".join(f"{n}: {t}" for n, (t, _) in self.fields.items()) + " }"
            expected = {n: v for n, (_, v) in self.fields.items()}
        sig = {self.arg: "Text[]"}
        inputs = {self.arg: self.vars[self.arg].value}
        if any(s[0] == "call" and "rubric" in (s[1].get("inputs") or {}) for s in c.calls) or "rubric" in "\n".join(c.lines_a):
            sig["rubric"], inputs["rubric"] = "Text", T_RUBRIC
        name = rng.choice({"tickets": ["ticket_stats", "inbox_overview", "support_digest"],
                           "reviews": ["review_stats", "feedback_overview"],
                           "expenses": ["claims_overview", "expense_stats"]}[self.domain])
        return _finish(c, "composed", sig, returns, inputs, expected, sorted(self.types), name,
                       "Carried out the program step by step.")


def composed(rng: random.Random) -> Program:
    return Composer(rng).build()


SHAPES["composed"] = composed
