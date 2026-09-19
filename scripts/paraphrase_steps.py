#!/usr/bin/env python3
"""Teacher-written variants of how a program step reads (the synthesizer's phrase bank, natlang/gen/synth.py).

For every kind of step and both dialects, the teacher proposes other ways to write the same line. A variant is kept
only if (1) it has exactly the same {placeholders}, (2) it stays one line, and (3) the teacher as judge (a small thinking
budget) says that an instantiated original and the instantiated variant ask for exactly the same step.
Structure is never paraphrased: only wording, dialect and naming style of a single line.

  scripts/paraphrase_steps.py --server http://127.0.0.1:8081 --k 4        -> data/phrases.json (merged, resumable)
"""
import argparse, json, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from natlang.decoder import LlamaServerDecoder
from natlang.gen import synth as S

ASK = {"a": "Here is one line of pseudocode in a Python-like style. The words in {{curly braces}} are placeholders.\n\n{line}\n\n"
            "Write {k} other ways a programmer might write this same line of pseudocode (other keywords, other idioms such as "
            "`map(...)`, comprehension style, arrows, `let`), each on ONE line, each using exactly the same placeholders, "
            "each asking for exactly the same step. Do not prefix lines with bullets, checkboxes or step numbers. Use return explicitly, never a bare arrow, for returning values. Answer with a JSON list of {k} strings and nothing else.",
       "b": "Here is one step of a procedure written in plain English. The words in {{curly braces}} are placeholders.\n\n{line}\n\n"
            "Write {k} other ways to phrase this same step (terse, polite, checklist style, imperative), each ONE sentence or "
            "line, each using exactly the same placeholders, each asking for exactly the same step, nothing added or removed. Do not prefix lines with bullets, checkboxes or step numbers. "
            "Answer with a JSON list of {k} strings and nothing else."}
SAME = ("Two ways of writing one step of a program:\n\nA: {x}\nB: {y}\n\nWould a careful reader carry out the same step for "
        "both: the same function or operation, applied to the same inputs, with the result kept under the same name? Wording, "
        "politeness and syntax do not matter. Answer yes or no.")
EXAMPLE = {"out": "labels", "over": "tickets", "fn": "classify", "extra": "rubric", "items": "tickets", "flags": "urgent_flags",
           "field": "total", "values": "labels", "fields": "total, by_label", "name": "is_spam", "labels": "labels", "label": "spam", "value": "0", "source": "budget", "predicate": "k.amount > 100", "left": "urgent_flags", "right": "paid_flags", "operator": "and", "target": "100"}


def valid_variant(candidate, base):
    if not isinstance(candidate, str):
        return False
    c = candidate.strip()
    holes = lambda text: sorted(re.findall(r"(?<!\{)\{(\w+)\}(?!\})", text))
    literal_calls = re.findall(r"(?<![\w{])([A-Za-z_]\w*)\(", base)
    if any(not re.search(rf"\b{re.escape(fn)}\b", c) for fn in literal_calls):
        return False
    if re.search(r"\breturn\b", c) and not re.search(r"\breturn\b", base, re.I):
        return False
    return bool(c) and "\n" not in c and holes(c) == holes(base) and c != base.strip() and not re.match(
        r"^(?:[-*+]\s|\[[ xX-]\]|\d+[.)]\s|=>|->)", c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8081")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "phrases.json")
    ap.add_argument("--thinking", type=int, default=128)
    ap.add_argument("--keys", nargs="+", help="only these step kinds; existing pairs are preserved")
    a = ap.parse_args()
    for i in range(200):                                   # let the call sites record their base phrases
        S.composed(random.Random(i))
    bank = json.loads(a.out.read_text()) if a.out.exists() else {}
    ask = LlamaServerDecoder(a.server, timeout=600, chat_extra={"thinking_budget_tokens": a.thinking, "top_p": 0.95, "top_k": 20})
    # With thinking off this judge said "no" to plainly equivalent lines; a small budget made it right on a
    # calibration set of equivalent and non-equivalent pairs (2026-09-19).
    judge = LlamaServerDecoder(a.server, timeout=300, chat_extra={"thinking_budget_tokens": 160})
    holes = lambda t: sorted(re.findall(r"(?<!\{)\{(\w+)\}(?!\})", t))
    if a.keys and set(a.keys) - S.BASE_PHRASES.keys():
        ap.error("unknown keys: " + ", ".join(sorted(set(a.keys) - S.BASE_PHRASES.keys())))
    for key, base in S.BASE_PHRASES.items():
        if a.keys and key not in a.keys:
            continue
        if key == "count_true":
            base = {**base, "b": "Call count_true on {flags} and store the number in {field}."}
        got = {"a": [], "b": []}
        for side in ("a", "b"):
            turn = ask.chat([{"role": "user", "content": ASK[side].format(line=base[side], k=a.k)}], [], temperature=0.9, max_tokens=700)
            print(f"[{key}/{side}] raw {turn.text!r}", flush=True)
            m = re.search(r"\[.*\]", turn.text, re.S)
            try:
                cands = json.loads(m.group(0)) if m else []
            except json.JSONDecodeError:
                cands = []
            for c in cands:
                if not valid_variant(c, base[side]):
                    continue
                try:
                    x, y = base[side].format(**EXAMPLE), c.strip().format(**EXAMPLE)
                except (KeyError, IndexError, ValueError):
                    continue
                verdict = judge.chat([{"role": "user", "content": SAME.format(x=x, y=y)}], [], temperature=0.0, max_tokens=400).text.strip().lower()
                keep = verdict.startswith("yes")
                print(f"[{key}/{side}] {'KEEP' if keep else 'drop'}  {c.strip()[:110]!r}", flush=True)
                if keep:
                    got[side].append(c.strip())
        have = {(v["a"], v["b"]) for v in bank.get(key, [])}
        for va in got["a"] or [base["a"]]:                  # a variant pairs a Python-like line with an English one
            for vb in (got["b"] or [base["b"]])[:2]:
                if (va, vb) not in have and (va, vb) != (base["a"], base["b"]):
                    bank.setdefault(key, []).append({"a": va, "b": vb})
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(bank, indent=1, ensure_ascii=False))
    print({k: len(v) for k, v in bank.items()})


if __name__ == "__main__":
    main()
