"""Diversity of training and evaluation data: one shape key, one near-duplicate index, one cap and one report.

Every stage that chooses what trains (collection order, the GRPO walk, the teacher campaign's order, the filled-reasoning
order, the round mix and its aux parts) uses these, so a template that wraps thousands of items or a demonstration that
repeats another counts the same way everywhere (owner 2026-11-24 "I don't want to train on repetition at all"; DECISIONS
2026-11-25 "diversity end to end").

* **Program shape** (``program_shape``): a task's callable folder with item data abstracted (digits, quoted spans,
  mid-sentence capitalized words, code identifiers, paths, hex ids; the root's stem; literal unions). ``task_shapes``
  then merges base shapes of one template whose abstracted texts still differ in a few lowercase item words (a topic
  word inside a template sentence split one template into many shapes): same file skeleton, services and mode, and
  word-trigram Jaccard of the abstracted text >= ``SHAPE_MERGE``. A row without a program is ``id_shape`` of its id.
* **Near duplicates** (``NearDuplicates``): 64-permutation MinHash of shingles (word 5-grams of text, ``shingles``; or
  token-id 5-grams of training targets, ``token_signature``), LSH 16 bands of 4, estimated Jaccard >= threshold.
* **Caps**: ``cap_shapes`` reorders an order (no shape above ``share`` of any prefix, nothing dropped: collection,
  campaign and fill orders); ``ShapeQuota`` bounds a selection (no shape above ``share`` of a total, counted in items
  or tokens: the round mix).
* **Report** (``diversity_report``): per part, units, mass, distinct and effective shapes (exp-entropy of the mass),
  the top shape's and top 10 shapes' mass share, the near-duplicate mass share; ``regression_flags`` compares two.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import re

SHAPE_CAP = 0.01        # the share of any prefix of an order one program shape may take (0: no cap)
SHAPE_CAP_FLOOR = 4     # items of each shape always admitted before the share applies (a short prefix stays mixed)
SHAPE_MERGE = 0.7       # word-trigram Jaccard of two base shapes' abstracted text at which they are one template
MIX_SHAPE_SHARE = 0.02  # the share of a round-mix part's selection one program shape may take (0: no cap)
NEAR_DUPLICATE = 0.8    # estimated Jaccard of two items' shingles at which the later one repeats the earlier
NEAR_DUPLICATE_MIN_TOKENS = 64   # a unit with fewer target tokens (a decision readout, a one-word answer) has no
                                 # content signature: its prompt, not its target, makes it what it is

# ---- program shape --------------------------------------------------------------------------------------------------
DIGITS = re.compile(r'\d+(?:\.\d+)?')
# Quoted literals: double-quoted and backtick spans, and single-quoted spans only when the quote is not inside a word.
# A prose apostrophe ("the lane's limit") is not a string opener; quotes never span lines.
QUOTED = re.compile(r'"(?:[^"\\\n]|\\.)*"|(?<!\w)\'(?:[^\'\\\n]|\\.)*\'(?!\w)|`[^`]*`')
SPACE = re.compile(r'\s+')
QUOTED_JSON = re.compile(r'"(?:[^"\\]|\\.)*"')
# Item data a template program spells into its prose: names and places (a capitalized word inside a sentence), code
# identifiers (snake_case, camelCase, dotted or slashed paths, repo/owner names) and long hex ids. Sentence-initial
# words, digits and quoted spans are handled by `normalize` or kept: they are the template's own words.
ITEM_TOKEN = re.compile(r'(?<![.!?:;] )(?<!^)\b[A-Z][\w-]*|'                      # a capitalized word mid-sentence
                        r'\b\w*[a-z][A-Z]\w*\b|'                                 # camelCase
                        r'[\w@~-]*(?:[./\\:]\w[\w@~-]*)+|'                       # paths, dotted names, URLs, repo/owner
                        r'\b\w+_\w+\b|\b[0-9a-f]{7,}\b')                          # snake_case, hex ids
LITERAL_UNION = re.compile(r'L(?:\|L)+')


def normalize(text):
    return SPACE.sub(' ', DIGITS.sub('#', QUOTED.sub('"…"', text))).strip()


def split_file(text):
    if text.startswith('---'):
        parts = text.split('---', 2)
        if len(parts) == 3:
            return parts[1], parts[2]
    return '', text


def _unquote(match):
    try:
        return json.loads(match.group(0))
    except ValueError:
        return match.group(0)


def type_shape(text):
    """A frontmatter type with its YAML quoting removed and string-literal members abstracted: `"{ a: number }[]"`
    keeps its field names and shape, `"\\"x\\" | \\"y\\""` becomes `L|L`. (`normalize` blanks every quoted span, and
    frontmatter types are quoted, so using it on types erased all argument and return types.)"""
    return re.sub(r'\s+', '', re.sub(r'"[^"]*"', 'L', QUOTED_JSON.sub(_unquote, text)))


def program_text_shape(text):
    """Program text with literal item data abstracted (``normalize`` plus ITEM_TOKEN -> I): what a template keeps
    when it wraps another item."""
    return SPACE.sub(' ', ITEM_TOKEN.sub('I', normalize(text))).strip()


def _shape_parts(row):
    semantics = row.get('semantics') or {}
    files = semantics.get('files') or {}
    stem = str(semantics.get('root') or '').removesuffix('.nl').removesuffix('.ts')
    parts = []
    for path, text in sorted(files.items()):
        header, body = split_file(text) if path.endswith('.nl') else ('', text)
        name = 'ROOT' + path[len(stem):] if stem and path.startswith(stem) else path
        # a union of literals (a label set) is item data: its size too
        parts.append([program_text_shape(name), LITERAL_UNION.sub('L|L', type_shape(header)), program_text_shape(body)])
    return parts, sorted(semantics.get('services') or {}), (row.get('curriculum') or {}).get('mode')


ID_ITEM = re.compile(r'[0-9a-f]{8,}|\d+')


def id_shape(task_id):
    """The shape of an item without a program row (an aux unit): its id with hex and digit runs abstracted when words
    remain (``inline-curriculum:broad_graph_fetaqa:fetaqa-<hash>:fan-in/root/w1``: one generator template), else its
    own id (an operator row keyed by a content hash, whose repetition its targets show: ``NearDuplicates``)."""
    task_id = str(task_id)
    abstracted = ID_ITEM.sub('#', task_id)
    return 'id:' + (abstracted if re.search(r'[A-Za-z]{3,}', abstracted) else task_id)


def program_shape(row):
    """A task's base program shape: its callable folder with literal item data abstracted. Every file (the root's own
    stem abstracted from the paths), its frontmatter types (``type_shape``) and its text (``program_text_shape``), the
    services it binds and its curriculum mode; a sha256 prefix. ``semantics.inputs`` is item data and not part of it.
    Thousands of tasks of one template (one wrapper over many items) share a shape, which their family count does not
    show (DECISIONS 2026-11-25 "program-shape cap"). A row without semantics is its own shape."""
    files = (row.get('semantics') or {}).get('files') or {}
    if not files:
        return 'id:' + str(row.get('id'))
    parts, services, mode = _shape_parts(row)
    key = json.dumps({'files': parts, 'services': services, 'mode': mode}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _skeleton_and_words(row):
    parts, services, mode = _shape_parts(row)
    skeleton = json.dumps({'paths': [p[0] for p in parts], 'services': services, 'mode': mode}, sort_keys=True)
    words = re.findall(r'[a-z]+|[#I…]', ' '.join(p[2] for p in parts).lower())
    return skeleton, {' '.join(words[i:i + 3]) for i in range(max(1, len(words) - 2))} if words else set()


def task_shapes(rows, merge=SHAPE_MERGE):
    """id -> program shape. Base shapes (``program_shape``) of one file skeleton, services and mode whose abstracted
    texts share word trigrams at Jaccard >= ``merge`` are one shape, named by the base shape with the most rows (ties:
    the smaller key): a template whose item words are lowercase prose is one shape, not one per item. ``merge`` 0 or
    None: base shapes."""
    rows = list(rows)
    base = {row['id']: program_shape(row) for row in rows}
    if not merge:
        return base
    counts = collections.Counter(base.values())
    example = {}
    for row in rows:
        example.setdefault(base[row['id']], row)
    index = NearDuplicates(merge)
    leader_of, leaders = {}, {}
    for shape in sorted(example, key=lambda s: (-counts[s], s)):
        if shape.startswith('id:'):
            leader_of[shape] = shape
            continue
        skeleton, words = _skeleton_and_words(example[shape])
        signature = minhash([skeleton + '\x00' + w for w in words] or [skeleton])
        hit = index.find(signature)
        if hit is None:
            leaders[index.add(signature)] = shape
            leader_of[shape] = shape
        else:
            leader_of[shape] = leaders[hit]
    return {task_id: leader_of[shape] for task_id, shape in base.items()}


def shape_lookup(shapes):
    """``shape_of(task_id)`` over an id -> shape map, ``id_shape`` for an id it does not hold."""
    return lambda task_id: shapes.get(task_id) or id_shape(task_id)


# ---- near duplicates ------------------------------------------------------------------------------------------------
WORD = re.compile(r'\w+')
_PERMS = {}


def shingles(text, n=5):
    words = ['0' if w.isdigit() else w for w in (w.lower() for w in WORD.findall(text))]
    return {' '.join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))} if words else set()


def _permutations(perms):
    import numpy as np
    if perms not in _PERMS:
        rng = np.random.RandomState(20261117)
        _PERMS[perms] = (rng.randint(1, 2**31 - 1, size=perms).astype(np.uint64),
                         rng.randint(0, 2**31 - 1, size=perms).astype(np.uint64))
    return _PERMS[perms]


def minhash_of_hashes(hashes, perms=64):
    """MinHash signature of a uint64 hash array (None when empty)."""
    import numpy as np
    if hashes is None or len(hashes) == 0:
        return None
    a, b = _permutations(perms)
    hashes = np.asarray(hashes, dtype=np.uint64)
    return ((hashes[None, :] * a[:, None] + b[:, None]) % np.uint64(2**61 - 1)).min(axis=1)


def minhash(items, perms=64):
    """64-permutation MinHash of a shingle set (the DATA_MIX_DIVERSITY.md method)."""
    import numpy as np
    if not items:
        return None
    return minhash_of_hashes(np.array([int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), 'little')
                                       for s in items], dtype=np.uint64), perms)


def token_signature(ids, n=5, perms=64):
    """MinHash of a token-id sequence's n-grams (training targets: the rendered tokens a row trains on), vectorized:
    each n-gram hashes as a polynomial in uint64. None for an empty sequence."""
    import numpy as np
    ids = np.asarray(ids, dtype=np.uint64).ravel()
    if len(ids) == 0:
        return None
    if len(ids) < n:
        ids = np.concatenate([ids, np.zeros(n - len(ids), dtype=np.uint64)])
    with np.errstate(over='ignore'):
        h = np.zeros(len(ids) - n + 1, dtype=np.uint64)
        for j in range(n):
            h = h * np.uint64(1000003) + ids[j:len(ids) - n + 1 + j] + np.uint64(1)
        h ^= h >> np.uint64(29)
        h *= np.uint64(0xbf58476d1ce4e5b9)
        h ^= h >> np.uint64(32)
        return minhash_of_hashes(np.unique(h), perms)


class NearDuplicates:
    """LSH (16 bands of 4) over MinHash signatures. ``seen(sig)``: true when an admitted item has estimated Jaccard >=
    threshold, else the item is admitted. ``find``/``add`` separate the two (``find`` returns the matching item's
    index)."""

    def __init__(self, threshold=NEAR_DUPLICATE, bands=16, rows=4):
        self.threshold, self.bands, self.rows = threshold, bands, rows
        self.buckets = collections.defaultdict(list)
        self.signatures = []

    def _keys(self, signature):
        return [(band, signature[band * self.rows:(band + 1) * self.rows].tobytes()) for band in range(self.bands)]

    def find(self, signature):
        if signature is None:
            return None
        candidates = set()
        for key in self._keys(signature):
            candidates.update(self.buckets.get(key, ()))
        best = None
        for index in sorted(candidates):
            if (self.signatures[index] == signature).mean() >= self.threshold:
                best = index
                break
        return best

    def add(self, signature):
        index = len(self.signatures)
        self.signatures.append(signature)
        for key in self._keys(signature):
            self.buckets[key].append(index)
        return index

    def seen(self, signature):
        if signature is None:
            return False
        if self.find(signature) is not None:
            return True
        self.add(signature)
        return False


# ---- caps -----------------------------------------------------------------------------------------------------------
def cap_shapes(rows, shape_of, share=SHAPE_CAP, floor=SHAPE_CAP_FLOOR):
    """``rows`` (an order) reordered so that no program shape (``shape_of(row)``) takes more than ``share`` of any
    prefix, beyond its first ``floor`` rows: a row whose shape is at its cap waits, in order, and enters as soon as the
    prefix has grown enough. Every row is kept: once only capped shapes are left they follow least-placed first
    (round-robin over the shapes). Spreads effort over distinct programs instead of items of one template (DECISIONS
    2026-11-25 "program-shape cap")."""
    if share <= 0 or share >= 1:
        return list(rows)
    counts, waiting, out = collections.Counter(), {}, []

    def allowed(key):
        return counts[key] + 1 <= max(floor, share * (len(out) + 1))   # placed, the shape stays within its share

    def place(key):
        out.append(waiting[key].popleft())
        counts[key] += 1
        if not waiting[key]:
            del waiting[key]

    def drain():
        progressed = True
        while progressed and waiting:
            progressed = False
            for key in list(waiting):
                if allowed(key):
                    place(key)
                    progressed = True

    for row in rows:
        key = shape_of(row)
        if key not in waiting and allowed(key):
            out.append(row)
            counts[key] += 1
            drain()
        else:
            waiting.setdefault(key, collections.deque()).append(row)
    while waiting:
        drain()
        if waiting:
            place(min(waiting, key=lambda k: counts[k]))
    return out


def round_robin_by(rows, key_of):
    """``rows`` with the groups of ``key_of`` taking turns, each group keeping its order (the first of every group
    before any group's second)."""
    turn, seen = {}, collections.Counter()
    for index, row in enumerate(rows):
        key = key_of(row)
        turn[index] = (seen[key], index)
        seen[key] += 1
    return [rows[i] for i in sorted(range(len(rows)), key=lambda i: turn[i])]


class ShapeQuota:
    """At most ``share`` of ``total`` per shape (at least ``floor``), counted in items or in any amount (tokens):
    ``allow(shape, amount)`` takes the amount when it fits and says whether it did. ``share`` 0: no quota."""

    def __init__(self, total, share=MIX_SHAPE_SHARE, floor=SHAPE_CAP_FLOOR):
        self.limit = max(floor, share * total) if share and share > 0 else None
        self.used = collections.Counter()

    def allow(self, shape, amount=1):
        if self.limit is not None and self.used[shape] > 0 and self.used[shape] + amount > self.limit:
            return False
        self.used[shape] += amount
        return True


def shape_capped(items, shape_of, share=MIX_SHAPE_SHARE, floor=SHAPE_CAP_FLOOR, on_drop=None):
    """``items`` (in preference order) with each shape kept to ``ShapeQuota(len(items), share, floor)``; ``on_drop``
    sees each item left out."""
    quota = ShapeQuota(len(items), share, floor)
    kept = []
    for item in items:
        if quota.allow(shape_of(item)):
            kept.append(item)
        elif on_drop is not None:
            on_drop(item)
    return kept


# ---- report ---------------------------------------------------------------------------------------------------------
def effective(counter):
    """exp(Shannon entropy) of a count or mass distribution: the number of equally used keys with the same spread."""
    total = sum(counter.values())
    if not total:
        return 0.
    return math.exp(-sum((n / total) * math.log(n / total) for n in counter.values() if n > 0))


def _spread(mass, near, units):
    total = sum(mass.values()) or 0.
    ranked = sorted(mass.values(), reverse=True)
    return {'units': units, 'mass': round(total, 1), 'shapes': len(mass), 'effective_shapes': round(effective(mass), 1),
            'top_shape_share': round(ranked[0] / total, 4) if total else None,
            'top10_shape_share': round(sum(ranked[:10]) / total, 4) if total else None,
            'near_duplicate_share': round(near / total, 4) if total else None,
            'top_shapes': [[shape, round(m / total, 4)] for shape, m in sorted(mass.items(), key=lambda kv: -kv[1])[:5]]
            if total else []}


def diversity_report(units, threshold=NEAR_DUPLICATE):
    """``units``: dicts {part, shape, mass, signature} in admission order. Per part and over everything: units, mass,
    distinct and effective shapes (mass-weighted), the top shape's and the top 10 shapes' mass share and the mass share
    of units that near-duplicate an earlier unit (of the same part; ``all``: of any part)."""
    per_part = collections.defaultdict(collections.Counter)
    near_part, near_all, count = collections.Counter(), 0., collections.Counter()
    everything = collections.Counter()
    index_part = collections.defaultdict(lambda: NearDuplicates(threshold))
    index_all = NearDuplicates(threshold)
    for unit in units:
        part, mass, signature = unit['part'], float(unit.get('mass') or 0.), unit.get('signature')
        per_part[part][unit['shape']] += mass
        everything[unit['shape']] += mass
        count[part] += 1
        if index_part[part].seen(signature):
            near_part[part] += mass
        if index_all.seen(signature):
            near_all += mass
    return {'threshold': threshold, 'all': _spread(everything, near_all, sum(count.values())),
            'parts': {part: _spread(per_part[part], near_part[part], count[part]) for part in sorted(per_part)}}


# regression thresholds of ``regression_flags``
EFFECTIVE_DROP = 0.8      # effective shapes below this fraction of the previous build's
TOP_SHARE_RISE = 1.5      # the top shape's share above this multiple of the previous build's (and above 2x the cap)
NEAR_DUP_SHARE = 0.05     # near-duplicate mass share above this


def regression_flags(current, previous=None, min_mass_share=0.02):
    """Flags of a ``diversity_report`` (a round mix's) against the previous one: a part (with at least
    ``min_mass_share`` of the mass) whose effective shapes fell below ``EFFECTIVE_DROP`` of the previous, whose top
    shape's share rose by ``TOP_SHARE_RISE`` beyond twice ``MIX_SHAPE_SHARE``, or whose near-duplicate share exceeds
    ``NEAR_DUP_SHARE``."""
    out = []
    total = (current.get('all') or {}).get('mass') or 0.
    for part, cur in sorted({'all': current.get('all') or {}, **(current.get('parts') or {})}.items()):
        if not cur or (part != 'all' and total and (cur.get('mass') or 0.) < min_mass_share * total):
            continue
        prev = ((previous or {}).get('parts') or {}).get(part) if part != 'all' else (previous or {}).get('all')
        if (cur.get('near_duplicate_share') or 0.) > NEAR_DUP_SHARE:
            out.append({'flag': 'mix_near_duplicates', 'part': part, 'share': cur['near_duplicate_share']})
        if prev:
            if prev.get('effective_shapes') and cur['effective_shapes'] < EFFECTIVE_DROP * prev['effective_shapes']:
                out.append({'flag': 'mix_effective_shapes_fell', 'part': part, 'effective_shapes': cur['effective_shapes'],
                            'previous': prev['effective_shapes']})
            top, before = cur.get('top_shape_share') or 0., prev.get('top_shape_share') or 0.
            if top > 2 * MIX_SHAPE_SHARE and before and top > TOP_SHARE_RISE * before:
                out.append({'flag': 'mix_top_shape_rose', 'part': part, 'top_shape_share': top, 'previous': before,
                            'shape': (cur.get('top_shapes') or [[None]])[0][0]})
    return out


# ---- a built round mix ----------------------------------------------------------------------------------------------
def chain_units(chains):
    """(part, task, round) -> {'chains': [...]} of a mix's chains, in chain order of first appearance."""
    units = {}
    for chain in chains:
        key = (chain.get('part') or 'unknown', str(chain.get('task')), str(chain.get('round')))
        units.setdefault(key, []).append(chain)
    return units


def unit_signature(chains, min_tokens=NEAR_DUPLICATE_MIN_TOKENS):
    """``token_signature`` of a unit's training targets (every row's sampled ids, in order); None below
    ``min_tokens`` target tokens."""
    import numpy as np
    pieces = [np.asarray(row['sample']).ravel() for chain in chains for _, _, row in chain['segs'] if 'sample' in row]
    ids = np.concatenate(pieces) if pieces else np.zeros(0)
    return token_signature(ids) if len(ids) >= max(1, min_tokens) else None


def drop_near_duplicate_units(chains, threshold=NEAR_DUPLICATE, priority=lambda part: 0, signatures=None):
    """(kept chains, report): a unit (the chains of one part, task and round) whose targets near-duplicate those of a
    unit kept before it (``unit_signature``, estimated Jaccard >= ``threshold``) is dropped whole. Units are visited
    by ``priority(part)`` (lower first; stable within a part), so the copy a better source holds is the one kept.
    ``signatures`` (unit key -> signature) is filled for reuse. ``threshold`` 0: nothing dropped."""
    units = chain_units(chains)
    signatures = {} if signatures is None else signatures
    report = {'threshold': threshold, 'dropped_units': 0, 'dropped_chains': 0, 'parts': collections.Counter()}
    if not threshold:
        return list(chains), report
    index, dropped = NearDuplicates(threshold), set()
    for key in sorted(units, key=lambda k: priority(k[0])):
        if key not in signatures:
            signatures[key] = unit_signature(units[key])
        if index.seen(signatures[key]):
            dropped.add(key)
            report['dropped_units'] += 1
            report['dropped_chains'] += len(units[key])
            report['parts'][key[0]] += 1
    report['parts'] = dict(report['parts'])
    kept = [c for c in chains if (c.get('part') or 'unknown', str(c.get('task')), str(c.get('round'))) not in dropped]
    return kept, report


def mix_units(chains, shape_of, mass_of, signatures=None):
    """``diversity_report`` units of a mix's chains: one per (part, task, round), its shape (``shape_of(task)``), mass
    (``mass_of(chain)`` summed) and target signature (from ``signatures`` by unit key when it holds one)."""
    signatures = signatures or {}
    out = []
    for key, members in chain_units(chains).items():
        part, task, round_ = key
        out.append({'part': part, 'task': task, 'round': round_, 'shape': shape_of(task),
                    'mass': float(sum(mass_of(c) for c in members)),
                    'signature': signatures[key] if key in signatures else unit_signature(members)})
    return out


def main(argv=None):
    """``diversity.py mix BUILD_DIR [--cases FILE ...] [-o OUT]``: the diversity report of a built round mix
    (``chains.pkl``; shapes from BUILD_DIR/cases.jsonl and every ``--cases`` file, e.g. the pools)."""
    import argparse
    import pickle
    from pathlib import Path
    parser = argparse.ArgumentParser(description=main.__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    m = sub.add_parser('mix')
    m.add_argument('build')
    m.add_argument('--cases', action='append', default=[])
    m.add_argument('--no-merge', action='store_true', help='base shapes (no template merge)')
    m.add_argument('-o', '--out')
    args = parser.parse_args(argv)
    build = Path(args.build)
    rows, seen = [], set()
    for path in [build / 'cases.jsonl', *map(Path, args.cases)]:
        if path.exists():
            with path.open() as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        if row.get('id') not in seen:
                            seen.add(row.get('id'))
                            rows.append(row)
    shape_of = shape_lookup(task_shapes(rows, None if args.no_merge else SHAPE_MERGE))
    with (build / 'chains.pkl').open('rb') as handle:
        chains = pickle.load(handle)['chains']

    def mass_of(chain):
        return float(sum(row['weight'].sum() if 'weight' in row else length for _, length, row in chain['segs']))
    report = diversity_report(mix_units(chains, shape_of, mass_of))
    report['build'] = str(build)
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text + '\n')
    print(text)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
