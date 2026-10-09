"""Cases for training the combinator bodies (S5 §4.3), with exact targets.

Each case is a short natural-language span from a registered text corpus (chat turns only: the shared system prompt
is left out) and, for `map`, a transformation whose result the builder computes in code, stated to the model as a
natural-language instruction. So `read` is trained against the exact span, `map` against `f(span)` computed exactly,
`zip`/`split` against the two spans they pack, and law terms against their right sides; no teacher label is needed.
Cases keep their source record as `group` and its split, so a source never sits on both sides.

    python -m natlang_neuralese.data.operator_cases --text-data CORPUS/text.jsonl --corpus-id ID --out cases.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

MARKER = re.compile(r"<\|[^|>]{1,40}\|>")
WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _words(text: str) -> list[str]:
    return text.split()


TRANSFORMS = {
    "upper": ("Rewrite the text in capital letters, changing nothing else.", lambda s: s.upper()),
    "first_word": ("Return only the first word of the text, exactly as it is written.", lambda s: _words(s)[0]),
    "last_word": ("Return only the last word of the text, exactly as it is written.", lambda s: _words(s)[-1]),
    "word_count": ("Count the words of the text (separated by spaces) and return the number in digits.",
                   lambda s: str(len(_words(s)))),
    "reverse_words": ("Return the words of the text in reverse order, separated by single spaces.",
                      lambda s: " ".join(reversed(_words(s)))),
    "initials": ("Return the first character of each word of the text, joined without spaces.",
                 lambda s: "".join(w[0] for w in _words(s))),
}


def chat_turns(text: str) -> list[str]:
    """The non-system message bodies of a rendered chat (ChatML-style or marker-delimited)."""
    parts = re.split(r"<\|im_start\|>", text)
    out = []
    for part in parts:
        role, _, body = part.partition("\n")
        if role.strip() in ("", "system"):
            continue
        out.append(MARKER.sub(" ", body.split("<|im_end|>")[0]))
    return out if out else [MARKER.sub(" ", text)]


def spans(text: str, min_words: int = 4, max_words: int = 20) -> list[str]:
    """Sentence-like spans of plain prose: mostly words, no code or markup."""
    found = []
    for turn in chat_turns(text):
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", turn):
            sentence = " ".join(sentence.split())
            words = sentence.split()
            if not min_words <= len(words) <= max_words:
                continue
            if sum(bool(WORD.fullmatch(w.strip(".,;:!?\"()"))) for w in words) < 0.8 * len(words):
                continue
            if any(c in sentence for c in "{}[]<>=`\\$#|_"):
                continue
            found.append(sentence)
    return found


def build_cases(rows, *, corpus_id: str, limit: int, per_record: int = 3, seed: int = 0,
                max_records_per_span: int = 2) -> list[dict]:
    """Spans that occur in more than `max_records_per_span` source records are runtime boilerplate (instructions
    the prompts repeat), not content, and are left out."""
    rng = random.Random(seed)
    found_by_record, records_per_span = [], {}
    for row in rows:
        group = str(row.get("id") or hashlib.sha256(row["text"].encode()).hexdigest()[:16])
        found = list(dict.fromkeys(spans(row["text"])))
        found_by_record.append((group, row.get("split", "train"), found))
        for text in found:
            records_per_span[text] = records_per_span.get(text, 0) + 1
    pool = []
    for group, split, found in found_by_record:
        found = [text for text in found if records_per_span[text] <= max_records_per_span]
        rng.shuffle(found)
        for text in found[:per_record]:
            pool.append((group, split, text))
    rng.shuffle(pool)
    cases = []
    for index, (group, split, text) in enumerate(pool[:limit]):
        name = rng.choice(sorted(TRANSFORMS))
        instruction, fn = TRANSFORMS[name]
        partner = pool[(index + 1) % len(pool)][2]
        cases.append({"schema": "natlang.operator-case/1", "id": f"{corpus_id}:{index}",
                      "corpus": corpus_id, "group": group, "split": "test" if split == "test" else "train",
                      "text": text, "partner": partner,
                      "map": {"transform": name, "instruction": instruction, "expected": fn(text)}})
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text-data", required=True, help="text corpus JSONL with `text` (and `id`, `split`)")
    parser.add_argument("--corpus-id", required=True, help="registered corpus id of --text-data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--per-record", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rows = (json.loads(line) for line in open(args.text_data) if line.strip())
    cases = build_cases(rows, corpus_id=args.corpus_id, limit=args.limit, per_record=args.per_record, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as stream:
        for case in cases:
            stream.write(json.dumps(case, sort_keys=True) + "\n")
    splits = {s: sum(c["split"] == s for c in cases) for s in ("train", "test")}
    print(json.dumps({"cases": len(cases), **splits, "transforms": sorted({c["map"]["transform"] for c in cases})}))


if __name__ == "__main__":
    main()
