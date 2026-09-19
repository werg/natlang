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
            "each asking for exactly the same step. Answer with a JSON list of {k} strings and nothing else.",
       "b": "Here is one step of a procedure written in plain English. The words in {{curly braces}} are placeholders.\n\n{line}\n\n"
            "Write {k} other ways to phrase this same step (terse, polite, checklist style, imperative), each ONE sentence or "
            "line, each using exactly the same placeholders, each asking for exactly the same step, nothing added or removed. "
            "Answer with a JSON list of {k} strings and nothing else."}
SAME = ("Two ways of writing one step of a program:\n\nA: {x}\nB: {y}\n\nWould a careful reader carry out the same step for "
        "both: the same function or operation, applied to the same inputs, with the result kept under the same name? Wording, "
        "politeness and syntax do not matter. Answer yes or no.")
EXAMPLE = {"out": "labels", "over": "tickets", "fn": "classify", "extra": "rubric", "items": "tickets", "flags": "urgent_flags",
           "field": "total", "values": "labels", "fields": "total, by_label", "name": "is_spam", "labels": "labels", "label": "spam"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8081")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "phrases.json")
    a = ap.parse_args()
    for i in range(200):                                   # let the call sites record their base phrases
        S.composed(random.Random(i))
    bank = json.loads(a.out.read_text()) if a.out.exists() else {}
    ask = LlamaServerDecoder(a.server, timeout=600, chat_extra={"thinking_budget_tokens": 256, "top_p": 0.95, "top_k": 20})
    # With thinking off this judge said "no" to plainly equivalent lines; a small budget made it right on a
    # calibration set of equivalent and non-equivalent pairs (2026-09-19).
    judge = LlamaServerDecoder(a.server, timeout=300, chat_extra={"thinking_budget_tokens": 160})
    holes = lambda t: sorted(re.findall(r"(?<!\{)\{(\w+)\}(?!\})", t))
    for key, base in S.BASE_PHRASES.items():
        got = {"a": [], "b": []}
        for side in ("a", "b"):
            turn = ask.chat([{"role": "user", "content": ASK[side].format(line=base[side], k=a.k)}], [], temperature=0.9, max_tokens=700)
            m = re.search(r"\[.*\]", turn.text, re.S)
            try:
                cands = json.loads(m.group(0)) if m else []
            except json.JSONDecodeError:
                cands = []
            for c in cands:
                if not isinstance(c, str) or "\n" in c.strip() or holes(c) != holes(base[side]) or c.strip() == base[side].strip():
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
