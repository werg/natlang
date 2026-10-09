"""Source (c) of refine-judge: predicates a crisp checker decides exactly, so their labels need no teacher.

Each `Spec` is one predicate text with its checker and two value generators (satisfying, violating). The label of a row
is always the checker's verdict on the generated value, never the generator's intent; a value whose verdict disagrees
with its intent is dropped (and counted), so a buggy generator cannot poison labels.

Families cover length, format, enumeration, numeric range and list shape. A few predicates are the real ones from the
applications' DECOMPOSITION.md tables (kind `real`), the rest are parameterised variants so held-out predicates differ
from training ones in their parameters, and held-out families differ in kind.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Callable

from .common import canonical_json_str, judge_row, normalize_predicate, predicate_id, sha256_hex, split_of

FAMILY_PREFIX = "refine-judge/exact/"
# Whole families the training split never sees (they test transfer to a new kind of predicate).
HELDOUT_FAMILIES = frozenset({"clock-time", "ipv4", "multiple-of", "sorted-list"})

WORDS = ("order parcel delay refund account invoice ticket review update release build merge branch deploy service queue "
         "report window trade fight market guard route cache index table patch file line").split()
COLORS = "red green blue amber violet teal".split()
LEVELS = "low medium high critical".split()
PHASES = "new reserved charged shipped refunded released".split()
VERBS = "retry wait escalate ignore reconcile cancel".split()
LETTERS = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"


@dataclass
class Spec:
    family: str
    predicate: str
    check: Callable[[Any], bool]
    pos: Callable[[random.Random], Any]
    neg: Callable[[random.Random], Any]
    real: bool = False


def _sentence(rng: random.Random, words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(words))


def _text_of_length(rng: random.Random, length: int) -> str:
    """A line of exactly `length` characters made of words, starting with a capital and not ending in a period."""
    out = ""
    while len(out) < length:
        out += (" " if out else "") + rng.choice(WORDS)
    out = out[:length].rstrip()
    out += LETTERS[rng.randrange(26)] * (length - len(out)) if len(out) < length else ""
    return out[:1].upper() + out[1:]


def _pick(rng: random.Random, *makers):
    return makers[rng.randrange(len(makers))]()


def _words(rng: random.Random, count: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(count)).capitalize()


def _ident(rng: random.Random, length: int | None = None) -> str:
    length = length or rng.randint(3, 10)
    return rng.choice(LETTERS) + "".join(rng.choice(LETTERS + DIGITS + "_-") for _ in range(length - 1))


# ---- length ---------------------------------------------------------------------------------------------------

def length_specs() -> list[Spec]:
    specs: list[Spec] = []
    for limit in (20, 30, 40, 50, 60, 80, 100, 120):
        def check(value, limit=limit):
            return isinstance(value, str) and len(value) <= limit and "\n" not in value and not value.endswith(".") and value != ""
        specs.append(Spec("line-max-chars", f"one line of at most {limit} characters, without a trailing period", check,
                          lambda r, limit=limit: _text_of_length(r, r.randint(max(3, limit // 3), limit)),
                          lambda r, limit=limit: _pick(r, lambda: _text_of_length(r, limit + r.randint(1, 12)),
                                                       lambda: _text_of_length(r, r.randint(5, limit)) + ".",
                                                       lambda: _text_of_length(r, max(3, limit // 2)) + "\n" + _text_of_length(r, 6),
                                                       lambda: _text_of_length(r, limit * 2)),
                          real=limit == 60))
    for limit in (10, 25, 50, 140, 200):
        specs.append(Spec("max-chars", f"at most {limit} characters long", lambda v, limit=limit: isinstance(v, str) and len(v) <= limit,
                          lambda r, limit=limit: _text_of_length(r, r.randint(1, limit)),
                          lambda r, limit=limit: _text_of_length(r, limit + r.randint(1, 15))))
    for limit in (3, 5, 8, 12, 20, 30):
        specs.append(Spec("max-words", f"at most {limit} words", lambda v, limit=limit: len(v.split()) <= limit,
                          lambda r, limit=limit: _words(r, r.randint(1, limit)),
                          lambda r, limit=limit: _words(r, limit + r.randint(1, 4))))
    for count in (2, 3, 4, 5, 7):
        specs.append(Spec("exact-words", f"exactly {count} words", lambda v, count=count: len(v.split()) == count,
                          lambda r, count=count: _words(r, count),
                          lambda r, count=count: _words(r, max(1, count + r.choice([-1, 1, 2])))))
    for count in (3, 6, 10):
        specs.append(Spec("min-words", f"at least {count} words", lambda v, count=count: len(v.split()) >= count,
                          lambda r, count=count: _words(r, count + r.randint(0, 6)),
                          lambda r, count=count: _words(r, r.randint(1, count - 1))))
    for low, high in ((5, 20), (10, 40), (30, 90), (80, 160)):
        specs.append(Spec("char-range", f"between {low} and {high} characters long",
                          lambda v, low=low, high=high: low <= len(v) <= high,
                          lambda r, low=low, high=high: _text_of_length(r, r.randint(low, high)),
                          lambda r, low=low, high=high: _text_of_length(r, r.choice([max(1, low - r.randint(1, 4)), high + r.randint(1, 10)]))))
    specs.append(Spec("single-line", "a single line with no line breaks", lambda v: "\n" not in v and "\r" not in v and v != "",
                      lambda r: _words(r, r.randint(2, 9)),
                      lambda r: _words(r, 3) + "\n" + _words(r, 3)))
    specs.append(Spec("ends-with-period", "ends with a period", lambda v: v.endswith("."),
                      lambda r: _words(r, r.randint(2, 8)) + ".", lambda r: _words(r, r.randint(2, 8)) + r.choice(["", "!", "?", " ."][:3])))
    specs.append(Spec("starts-capital", "starts with a capital letter", lambda v: v[:1].isupper(),
                      lambda r: _words(r, r.randint(1, 6)), lambda r: _words(r, r.randint(1, 6)).lower()))
    specs.append(Spec("lowercase", "written in lowercase letters and spaces only", lambda v: re.fullmatch(r"[a-z ]+", v) is not None,
                      lambda r: _sentence(r, r.randint(1, 6)), lambda r: r.choice([_words(r, 3), _sentence(r, 3) + " 2", _sentence(r, 3) + "!"])))
    specs.append(Spec("no-digits", "contains no digits", lambda v: re.search(r"\d", v) is None,
                      lambda r: _sentence(r, r.randint(1, 6)), lambda r: _sentence(r, 3) + " " + str(r.randint(0, 999)) + " " + _sentence(r, 1)))
    return specs


# ---- format ---------------------------------------------------------------------------------------------------

def _hex(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
IDENT = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _valid_date(v: str) -> bool:
    m = DATE.fullmatch(v)
    if not m:
        return False
    y, mo, d = int(m[1]), int(m[2]), int(m[3])
    days = [31, 29 if y % 4 == 0 and (y % 100 or y % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return 1 <= mo <= 12 and 1 <= d <= days[mo - 1]


def _clock(v: str) -> bool:
    m = re.fullmatch(r"(\d{2}):(\d{2})", v)
    return bool(m) and int(m[1]) < 24 and int(m[2]) < 60


def _ipv4(v: str) -> bool:
    parts = v.split(".")
    return len(parts) == 4 and all(re.fullmatch(r"0|[1-9]\d{0,2}", p) and int(p) <= 255 for p in parts)


def _json_object(v: str) -> bool:
    try:
        return isinstance(json.loads(v), dict)
    except ValueError:
        return False


def _rel_path(v: str) -> bool:
    return bool(v) and not v.startswith("/") and ".." not in v.split("/") and "\\" not in v and "" not in v.split("/")


def format_specs() -> list[Spec]:
    def date(r):
        return f"{r.randint(1990, 2040):04d}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"
    def bad_date(r):
        return r.choice([f"{r.randint(1990, 2040)}/{r.randint(1, 12):02d}/{r.randint(1, 28):02d}", f"{r.randint(1990, 2040):04d}-13-{r.randint(1, 28):02d}",
                         f"{r.randint(1990, 2040):04d}-02-30", f"{r.randint(1, 28):02d}-{r.randint(1, 12):02d}-{r.randint(1990, 2040)}", "yesterday"])
    def clock(r):
        return f"{r.randint(0, 23):02d}:{r.randint(0, 59):02d}"
    def bad_clock(r):
        return r.choice([f"{r.randint(24, 39)}:{r.randint(0, 59):02d}", f"{r.randint(0, 23):02d}:{r.randint(60, 99)}", f"{r.randint(1, 12)}:{r.randint(0, 59):02d} pm", f"{r.randint(0, 23):02d}{r.randint(0, 59):02d}"])
    def ip(r):
        return ".".join(str(r.randint(0, 255)) for _ in range(4))
    def bad_ip(r):
        return r.choice([".".join(str(r.randint(0, 255)) for _ in range(3)), ".".join(str(r.randint(256, 400)) for _ in range(4)),
                         ".".join(str(r.randint(0, 255)) for _ in range(5)), "localhost"])
    def semver(r):
        return f"{r.randint(0, 20)}.{r.randint(0, 40)}.{r.randint(0, 99)}"
    def uuid(r):
        return "-".join(_hex(r, n) for n in (8, 4, 4, 4, 12))
    def email(r):
        return f"{_ident(r, 6).lower().replace('_', '').replace('-', 'x')}@{r.choice(WORDS)}.{r.choice(['com', 'org', 'io', 'net'])}"
    def slug(r):
        return "-".join(r.choice(WORDS) for _ in range(r.randint(1, 4)))
    def path(r):
        return "/".join(r.choice(WORDS) for _ in range(r.randint(1, 4))) + r.choice([".ts", ".nl", ".md", ""])
    def url(r):
        return "https://" + r.choice(WORDS) + "." + r.choice(["com", "org", "dev"]) + r.choice(["", "/" + r.choice(WORDS)])
    def obj(r):
        return canonical_json_str({r.choice(WORDS): r.randint(0, 99), r.choice(WORDS) + "2": r.choice(WORDS)})
    return [
        Spec("hex64", "64 lowercase hex digits", lambda v: re.fullmatch(r"[0-9a-f]{64}", v) is not None, lambda r: _hex(r, 64),
             lambda r: r.choice([_hex(r, r.choice([40, 63, 65])), _hex(r, 63) + "G", _hex(r, 32).upper() + _hex(r, 32), ""])),
        Spec("name", "a name: a letter followed by letters, digits, '_' or '-'", lambda v: NAME.fullmatch(v) is not None,
             lambda r: _ident(r), lambda r: r.choice([str(r.randint(0, 99)) + _ident(r, 4), _ident(r, 4) + " " + _ident(r, 3), _ident(r, 4) + "!", "", "-" + _ident(r, 4)])),
        Spec("name", "an identifier that starts with a letter and has only letters, digits, underscores and hyphens",
             lambda v: IDENT.fullmatch(v) is not None, lambda r: _ident(r), lambda r: r.choice([str(r.randint(0, 99)) + _ident(r, 4), _ident(r, 4) + "." + _ident(r, 3), "_" + _ident(r, 4), _ident(r, 3) + "é"]), real=True),
        Spec("iso-date", "a calendar date written as YYYY-MM-DD", _valid_date, date, bad_date),
        Spec("clock-time", "a 24-hour clock time written as HH:MM", _clock, clock, bad_clock),
        Spec("ipv4", "an IPv4 address in dotted decimal form", _ipv4, ip, bad_ip),
        Spec("semver", "a version written as MAJOR.MINOR.PATCH with whole numbers", lambda v: re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", v) is not None,
             semver, lambda r: r.choice([f"{r.randint(0, 9)}.{r.randint(0, 9)}", f"v{semver(r)}", f"{semver(r)}.{r.randint(0, 9)}", f"{r.randint(0, 9)}.x.{r.randint(0, 9)}", f"01.{r.randint(0, 9)}.{r.randint(0, 9)}"])),
        Spec("uuid", "a UUID in 8-4-4-4-12 lowercase hex form", lambda v: re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", v) is not None,
             uuid, lambda r: r.choice([uuid(r).replace("-", ""), uuid(r).upper(), uuid(r)[:-1], uuid(r) + "0"])),
        Spec("email", "an email address of the form local@domain.tld", lambda v: re.fullmatch(r"[A-Za-z0-9._-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}", v) is not None,
             email, lambda r: r.choice([email(r).replace("@", " at "), email(r).replace("@", ""), "@" + r.choice(WORDS) + ".com", email(r).split("@")[0] + "@" + r.choice(WORDS)])),
        Spec("slug", "a slug of lowercase letters and digits joined by single hyphens", lambda v: re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", v) is not None,
             slug, lambda r: r.choice([slug(r).replace("-", "_"), slug(r).upper(), slug(r) + "-", slug(r).replace("-", "--", 1) if "-" in slug(r) else slug(r) + "--x", slug(r).replace("-", " ") + " x"])),
        Spec("rel-path", "a relative path with no leading slash and no '..' segment", _rel_path, path,
             lambda r: r.choice(["/" + path(r), "../" + path(r), path(r).replace("/", "/../", 1) + "/..", path(r) + "/../" + path(r)])),
        Spec("https-url", "an https URL", lambda v: re.fullmatch(r"https://[A-Za-z0-9.-]+(/\S*)?", v) is not None, url,
             lambda r: r.choice([url(r).replace("https", "http"), url(r)[len("https://"):], url(r) + " now", "ftp://" + url(r)[8:]])),
        Spec("json-object", "text that parses as a JSON object", _json_object, obj,
             lambda r: r.choice([obj(r)[:-1], "[" + obj(r) + "]", obj(r).replace('"', "'"), "{" + r.choice(WORDS) + ": 1}", "null"])),
        Spec("decimal-int", "a decimal integer written without leading zeros", lambda v: re.fullmatch(r"-?(0|[1-9]\d*)", v) is not None,
             lambda r: str(r.randint(-5000, 99999)), lambda r: r.choice(["0" + str(r.randint(1, 99)), str(r.randint(1, 99)) + ".5", "1e3", "12a", "", "+" + str(r.randint(1, 9))])),
        Spec("hex-color", "a hex color written as # followed by six hex digits", lambda v: re.fullmatch(r"#[0-9a-fA-F]{6}", v) is not None,
             lambda r: "#" + _hex(r, 6), lambda r: r.choice(["#" + _hex(r, 3), _hex(r, 6), "#" + _hex(r, 7), "#" + _hex(r, 5) + "g"])),
    ]


# ---- enumeration -------------------------------------------------------------------------------------------

def enumeration_specs(rng: random.Random) -> list[Spec]:
    banks = [COLORS, LEVELS, PHASES, VERBS, WORDS]
    specs: list[Spec] = []
    seen: set[tuple] = set()
    for index in range(12):
        bank = banks[index % len(banks)]
        while True:
            members = sorted(rng.sample(bank, rng.randint(3, min(6, len(bank)))))
            if tuple(members) not in seen:
                seen.add(tuple(members))
                break
        outside = [w for w in WORDS + COLORS + LEVELS if w not in members]
        label = ", ".join(members[:-1]) + f" or {members[-1]}"
        specs.append(Spec("enumeration", f"one of the words {label}", lambda v, members=members: v in members,
                          lambda r, members=members: r.choice(members),
                          lambda r, members=members, outside=outside: r.choice([r.choice(outside), r.choice(members).upper(), r.choice(members) + "s", r.choice(members) + " " + r.choice(members)])))
    specs.append(Spec("enumeration", "one of missed-site, wrong-edit, test-expectation, environment, unrelated",
                      lambda v: v in {"missed-site", "wrong-edit", "test-expectation", "environment", "unrelated"},
                      lambda r: r.choice(["missed-site", "wrong-edit", "test-expectation", "environment", "unrelated"]),
                      lambda r: r.choice(["missed site", "wrong-site", "tests", "env", "other", "Unrelated"]), real=True))
    return specs


# ---- numeric ----------------------------------------------------------------------------------------------

def numeric_specs() -> list[Spec]:
    specs: list[Spec] = []
    for low, high in ((0, 10), (1, 5), (0, 100), (1, 3), (18, 65), (0, 59), (1, 12)):
        specs.append(Spec("int-range", f"a whole number from {low} to {high}", lambda v, low=low, high=high: isinstance(v, int) and not isinstance(v, bool) and low <= v <= high,
                          lambda r, low=low, high=high: r.randint(low, high),
                          lambda r, low=low, high=high: r.choice([low - r.randint(1, 5), high + r.randint(1, 5), high + r.randint(20, 200)])))
    specs.append(Spec("int-range", "0 or 1", lambda v: v in (0, 1) and not isinstance(v, bool), lambda r: r.randint(0, 1),
                      lambda r: r.choice([2, -1, 5, 10]), real=True))
    specs.append(Spec("positive-int", "a positive whole number", lambda v: isinstance(v, int) and not isinstance(v, bool) and v > 0,
                      lambda r: r.randint(1, 100000), lambda r: r.choice([0, -r.randint(1, 50)])))
    specs.append(Spec("positive-int", "a whole number of minutes, at least 1", lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 1,
                      lambda r: r.randint(1, 600), lambda r: r.choice([0, -r.randint(1, 90)]), real=True))
    specs.append(Spec("non-negative-int", "a non-negative whole number", lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 0,
                      lambda r: r.randint(0, 1000), lambda r: -r.randint(1, 100)))
    specs.append(Spec("even-int", "an even whole number", lambda v: isinstance(v, int) and v % 2 == 0,
                      lambda r: 2 * r.randint(-200, 200), lambda r: 2 * r.randint(-200, 200) + 1))
    for step in (5, 10, 15, 60):
        specs.append(Spec("multiple-of", f"a whole number that is a multiple of {step}", lambda v, step=step: isinstance(v, int) and v % step == 0,
                          lambda r, step=step: step * r.randint(0, 40), lambda r, step=step: step * r.randint(0, 40) + r.randint(1, step - 1)))
    specs.append(Spec("int-range", "a non-negative safe integer", lambda v: isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 2**53 - 1,
                      lambda r: r.choice([0, r.randint(1, 10**6), 2**53 - 1]), lambda r: r.choice([-1, 2**53, -7]), real=True))
    return specs


# ---- lists ------------------------------------------------------------------------------------------------

def list_specs() -> list[Spec]:
    specs: list[Spec] = []
    for limit in (2, 3, 5, 8):
        specs.append(Spec("list-max-items", f"a list of at most {limit} items", lambda v, limit=limit: isinstance(v, list) and len(v) <= limit,
                          lambda r, limit=limit: [r.choice(WORDS) for _ in range(r.randint(0, limit))],
                          lambda r, limit=limit: [r.choice(WORDS) for _ in range(limit + r.randint(1, 4))]))
    specs.append(Spec("distinct-list", "a list of distinct strings", lambda v: isinstance(v, list) and len(set(v)) == len(v),
                      lambda r: r.sample(WORDS, r.randint(0, 6)),
                      lambda r: (lambda items: items + [r.choice(items)])(r.sample(WORDS, r.randint(1, 5)))))
    specs.append(Spec("sorted-list", "a list of strings in ascending alphabetical order", lambda v: isinstance(v, list) and v == sorted(v),
                      lambda r: sorted(r.sample(WORDS, r.randint(0, 6))), lambda r: (lambda items: items[::-1] if len(items) > 1 else items + ["a"])(sorted(r.sample(WORDS, r.randint(2, 6))))))
    specs.append(Spec("sorted-list", "a list of whole numbers in ascending order, without repeats", lambda v: isinstance(v, list) and all(a < b for a, b in zip(v, v[1:])),
                      lambda r: sorted(r.sample(range(100), r.randint(0, 7))), lambda r: r.choice([[5, 3, 9], [1, 1, 2], [9, 8, 7, 6], [2, 4, 4]])))
    specs.append(Spec("non-empty-list", "a list with at least one item", lambda v: isinstance(v, list) and len(v) >= 1,
                      lambda r: [r.choice(WORDS) for _ in range(r.randint(1, 5))], lambda r: []))
    specs.append(Spec("positive-list", "a list of positive whole numbers", lambda v: isinstance(v, list) and all(isinstance(x, int) and x > 0 for x in v),
                      lambda r: [r.randint(1, 50) for _ in range(r.randint(0, 6))], lambda r: [r.randint(1, 50), 0, r.randint(1, 9)] if r.random() < .5 else [-r.randint(1, 9)]))
    for index, bank in enumerate((COLORS, LEVELS, PHASES)):
        members = sorted(bank)
        specs.append(Spec("members-list", f"a list whose items are each one of {', '.join(members)}",
                          lambda v, members=members: isinstance(v, list) and all(x in members for x in v),
                          lambda r, members=members: [r.choice(members) for _ in range(r.randint(0, 5))],
                          lambda r, members=members: [r.choice(members), "other", r.choice(members)]))
    return specs


def all_specs(seed: str = "refine-judge-exact/1") -> list[Spec]:
    rng = random.Random(seed)
    specs = length_specs() + format_specs() + enumeration_specs(rng) + numeric_specs() + list_specs()
    seen: set[str] = set()
    for spec in specs:
        key = normalize_predicate(spec.predicate)
        if key in seen:
            raise ValueError(f"duplicate predicate {key!r}")
        seen.add(key)
    return specs


def generate(per_predicate: int = 16, seed: str = "refine-judge-exact/1") -> tuple[list[dict], dict]:
    """Rows for every spec: `per_predicate` distinct values, half satisfying, half violating, labelled by the checker."""
    rows: list[dict] = []
    dropped = 0
    for spec in all_specs(seed):
        rng = random.Random(sha256_hex(f"{seed}|{spec.predicate}".encode())[:16])
        want = {True: per_predicate // 2, False: per_predicate - per_predicate // 2}
        got: dict[bool, list] = {True: [], False: []}
        seen: set[str] = set()
        attempts = 0
        while any(len(got[k]) < want[k] for k in want) and attempts < 400:
            attempts += 1
            need_true, need_false = len(got[True]) < want[True], len(got[False]) < want[False]
            intent = need_true and (not need_false or len(got[True]) <= len(got[False]))
            value = (spec.pos if intent else spec.neg)(rng)
            key = canonical_json_str(value)
            if key in seen:
                continue
            verdict = bool(spec.check(value))
            if verdict != intent:
                dropped += 1
                continue
            seen.add(key)
            got[verdict].append(value)
        for verdict in (True, False):
            for index, value in enumerate(got[verdict]):
                family = FAMILY_PREFIX + spec.family
                split = split_of(spec.predicate, spec.family, HELDOUT_FAMILIES)
                rows.append(judge_row(row_id=f"refine-judge:exact:{predicate_id(spec.predicate)}:{int(verdict)}:{index}", family=family, split=split,
                                      value=value, predicate=spec.predicate, gold_true=1.0 if verdict else 0.0,
                                      source="exact", label_source=f"crisp:{spec.family}", real_predicate=spec.real))
    rows.sort(key=lambda row: row["id"])
    report = {"predicates": len(all_specs(seed)), "rows": len(rows), "dropped_intent_mismatch": dropped,
              "heldout_families": sorted(HELDOUT_FAMILIES)}
    return rows, report
