#!/usr/bin/env python3
"""Learned-operator cases -> exp0_sft per-call rows (plans/neuralese/OPERATORS_RESTORE_20261117.md §3.2).

A `natlang.port-record/1` operator case (COMPACTION_OPERATOR_CONVERTERS.md) becomes the v3 operator write and its
consumer, as the runtime runs them:

* **operator chain** (one per compacted source): the operator site (`learned_operators.operator_site_messages`, the
  prompt as text: text initialization in the real rendered call) with its generation prompt, then the seed as the
  assistant reply (gold / checked / mechanical, `data/operator_seeds.py`, clipped to the operator's span cap). The
  reply is the only target; its weight is `--seed-weight`.
* **consumer chain**: the case's consumer conversation with each compacted source imported (`consume_open`, the
  span, `consume_close`); the target reply is the only target. `imports` [(row, key)] and `sources` {key: {ids,
  start, stop}} are the pass-through's format (`lfm3_fast_train.passthrough_plan`, `lower_rollout_turns.write_sources`):
  the producer document is the site, `produce_open` and the reply's content tokens (the in-context capture), so the
  consumer's loss reaches the weights that computed the state from the site.
* **text-cut chain** (test split): the same consumer reading the runtime's text cut of each source: the ablation's
  baseline.

Held data: every case whose split groups, ids or input text meet the held-v3 / held-v2 closure is dropped and counted
(`HeldClosure`: exact group names, task and upstream ids, 10-word shingles of the held tasks' inputs).

    operator_rows.py build --cases ROOT_OR_FILE [...] --out DIR [--limit N] [--operators a,b]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from natlang_neuralese.data.operator_graph import (IMITATION_LABELS, OperatorCaseError, OperatorGraphRenderer,  # noqa: E402
                                                   _placeholder, cut_text, source_text)
from natlang_neuralese.data.operator_seeds import seed_for  # noqa: E402
from natlang_neuralese.data.port_record_graph import _serving_messages, _target_message, _target_reply  # noqa: E402
from natlang_neuralese.common.paths import resolve  # noqa: E402
from natlang_neuralese.model.learned_operators import (TEXT_PLACEMENTS, bind_tokens, operator,  # noqa: E402
                                                       operator_site_messages, registry_sha256)

SCHEMA = 'natlang.operator-rows/1'
CASES_ROOT = resolve('data_hdd', 'natlang-compaction-operators-20261103', 'cases')
HELD_OPS = resolve('data_nvme', 'lfm3-rl-dgx-20261029', 'ops')
HELD_SETS = (HELD_OPS / 'held-v3/cases.jsonl', HELD_OPS / 'held-v2/cases.jsonl')
LFM3_BASE = str(Path.home() / '.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/'
                '40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd')
SHINGLE = 10
TEXT_MIN = 25
_WORD = re.compile(r'\w+')


def shingles(text: str, n: int = SHINGLE):
    words = _WORD.findall(text.lower())
    return {' '.join(words[i:i + n]) for i in range(len(words) - n + 1)}


class HeldClosure:
    """What a training case must not share with the held sets: group names, ids and input text."""

    def __init__(self, groups=(), ids=(), texts=()):
        self.groups, self.ids = set(groups), set(ids)
        # shingle -> index of the held string it belongs to. A shingle of two or more distinct held strings is
        # boilerplate (e.g. the Project Gutenberg notice of every QuALITY article) and is not evidence of a source.
        owner, self.sizes = {}, []
        for index, text in enumerate(dict.fromkeys(texts)):
            own = shingles(text)
            self.sizes.append(len(own))
            for item in own:
                owner[item] = index if owner.get(item, index) == index else -1
        self.owner = {item: index for item, index in owner.items() if index >= 0}

    @classmethod
    def from_cases(cls, paths=HELD_SETS):
        groups, ids, texts = set(), set(), []

        def walk(value):
            if isinstance(value, str):
                if len(value) >= 60:
                    texts.append(value)
            elif isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
        for path in paths:
            with open(path) as handle:
                for line in handle:
                    task = json.loads(line)
                    ids.add(task['id'])
                    curriculum = task.get('curriculum') or {}
                    groups.update(g for g in [curriculum.get('split_group'), *(task.get('source_groups') or [])] if g)
                    ids.update(i for i in task.get('source_ids') or [] if isinstance(i, str))
                    external = curriculum.get('external') or {}
                    ids.update(v for k in ('upstream_id', 'unit') if isinstance(v := external.get(k), str) and len(v) >= 8)
                    walk((task.get('semantics') or {}).get('inputs'))
        return cls(groups, ids, texts)

    def hit(self, record: dict) -> str | None:
        """The first kind of contact ('group', 'id', 'text') of a case with the held closure, or None."""
        if self.groups.intersection(record.get('split_groups') or ()):
            return 'group'
        lineage = record.get('lineage') or {}
        if {record.get('id'), lineage.get('upstream_id')} & self.ids:
            return 'id'
        texts = [source_text(source) for source in record['sources']]
        texts += [message.get('content') for message in record['consumer'].get('context') or []
                  if isinstance(message.get('content'), str)]
        matched = collections.Counter(self.owner[item] for text in texts if text for item in shingles(text)
                                      if item in self.owner)
        # Contact: TEXT_MIN shared shingles (about 35 consecutive words) or half of a short held string (a question).
        if any(count >= min(TEXT_MIN, max(1, self.sizes[index] // 2)) for index, count in matched.items()):
            return 'text'
        return None


def load_tokenizer(base=LFM3_BASE):
    from transformers import AutoTokenizer
    from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
    from natlang_neuralese.model.shared_state import provision_role_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=False, local_files_only=True)
    roles = provision_role_tokenizer(tokenizer)
    return tokenizer, tuple(roles), tokenizer_fingerprint(tokenizer)


class RowBuilder:
    """Per-call rows of one case under one tokenizer (the renderer's exact site, seed and consumer rendering)."""

    def __init__(self, tokenizer, role_ids, fingerprint, *, max_tokens=32000, seed_weight=1.0):
        from natlang_neuralese.model.learned_operators import OPERATORS
        bindings = {name: bind_tokens(tokenizer, name, tokenizer_fingerprint=fingerprint) for name in OPERATORS}
        self.renderer = OperatorGraphRenderer(tokenizer, role_ids=role_ids, tokenizer_fingerprint=fingerprint,
                                              bindings=bindings, max_tokens=max_tokens)
        self.tokenizer, self.role_ids, self.fingerprint = tokenizer, tuple(role_ids), fingerprint
        self.max_tokens, self.seed_weight = max_tokens, seed_weight
        self.think_close = tokenizer.convert_tokens_to_ids('</think>')
        self.end_ids = set(tokenizer.convert_tokens_to_ids([t for t in ('<|im_end|>', '<|endoftext|>')
                                                            if t in tokenizer.get_vocab()]))

    def span_of(self, reply):
        """(start, stop) of the reply's content: after its reasoning (the last ``</think>``) and before its end of
        turn. The runtime captures exactly these tokens of the sampled reply, in place."""
        end = next((i for i, token in enumerate(reply) if token in self.end_ids), len(reply))
        start = max((i + 1 for i, token in enumerate(reply[:end]) if token == self.think_close), default=0)
        return start, end

    def operator_site(self, name, record, source, tools):
        spec = operator(name)
        context = _serving_messages(record['writer'].get('context') or [])
        if spec.placement in TEXT_PLACEMENTS:
            site_source = source_text(source)
        else:
            if not isinstance(source.get('messages'), list):
                raise OperatorCaseError('history-source-without-messages')
            site_source = _serving_messages(source['messages'])
        messages = operator_site_messages(name, instructions=record['writer']['instructions'], source=site_source,
                                          context=context)
        ids, _ = self.renderer._render(messages, tools if spec.placement not in TEXT_PLACEMENTS else None, {},
                                       generation=True)
        return ids

    def build(self, record: dict, name: str, key: str):
        """{'operators': [chain], 'consumer': chain, 'text_cut': chain, 'meta': {...}} or OperatorCaseError."""
        spec = operator(name)
        if record.get('outcome', {}).get('label') not in IMITATION_LABELS:
            raise OperatorCaseError('target-not-imitation-label')
        if 'sources' not in record['consumer'].get('withheld', []):
            raise OperatorCaseError('sources-not-withheld')
        sources = record['sources']
        if spec.sites == 'one' and len(sources) != 1:
            raise OperatorCaseError(f'site-count: {len(sources)} sources for a one-site operator')
        tools = record['consumer'].get('tools') or None
        inline = [bool(spec.inline_below_chars) and len(source_text(source)) < spec.inline_below_chars
                  for source in sources]
        if all(inline):
            raise OperatorCaseError('no-compacted-source')
        try:
            target = _target_reply(self.tokenizer, _target_message(_View(record)), tools)
        except ValueError as error:
            raise OperatorCaseError(f'target-render: {error}') from None
        operators, documents, seeds = [], {}, []
        compacted = [source for source, shown in zip(sources, inline) if not shown]
        for index, source in enumerate(compacted):
            site = self.operator_site(name, record, source, tools)
            seed_ids, seed = self.renderer.seed(name, record, source)
            text = self.tokenizer.decode(seed_ids, skip_special_tokens=False)
            try:
                reply = _target_reply(self.tokenizer, {'role': 'assistant', 'content': text}, None)
            except ValueError as error:
                raise OperatorCaseError(f'seed-render: {error}') from None
            start, stop = self.span_of(reply)
            span = reply[start:stop]
            if not span:
                raise OperatorCaseError('seed-empty')
            source_key = f'{key}:{index}'
            # The in-context capture: the site, the reply up to its content, produce_open, the content.
            document = [*site, *reply[:start], self.role_ids[0], *span]
            documents[source_key] = {'ids': np.asarray(document, np.int32), 'start': len(document) - len(span),
                                     'stop': len(document)}
            operators.append(_chain(site, reply, key, {'kind': 'operator', 'operator': name, 'source_key': source_key,
                                                       'weight': np.full(len(reply), self.seed_weight, np.float32)}))
            seeds.append({**seed, 'site_tokens': len(site), 'span_tokens': len(span)})
        states = [_placeholder(f'operator-state:{key}:{index}') for index in range(len(compacted))]
        parts, it = [], iter(states)
        for source, shown in zip(sources, inline):
            parts.append({'type': 'text', 'text': source_text(source), 'inline': True} if shown
                         else {'type': 'native_state', 'id': next(it)})
        messages = self.renderer._consumer_messages(name, record, parts)
        splices = {state: [self.role_ids[2], *documents[f'{key}:{i}']['ids'][documents[f'{key}:{i}']['start']:],
                           self.role_ids[3]] for i, state in enumerate(states)}
        context, spans = self.renderer._render(messages, tools, splices, generation=True)
        imports = sorted((spans[state][0] + 1, f'{key}:{i}') for i, state in enumerate(states))
        consumer = _chain(context, target, key, {'kind': 'consumer', 'operator': name},
                          imports=imports, sources=documents)
        total = len(consumer['ids']) + sum(len(d['ids']) for d in documents.values())
        if total > self.max_tokens or any(len(c['ids']) > self.max_tokens for c in operators):
            raise OperatorCaseError(f'over-token-cap: {total} > {self.max_tokens}')
        cut_messages = self.renderer._consumer_messages(name, record, [
            {'type': 'text', 'text': source_text(source), 'inline': True} if shown
            else {'type': 'text', 'text': cut_text(source_text(source))} for source, shown in zip(sources, inline)])
        cut_context, _ = self.renderer._render(cut_messages, tools, {}, generation=True)
        text_cut = _chain(cut_context, target, key, {'kind': 'text-cut', 'operator': name})
        return {'operators': operators, 'consumer': consumer, 'text_cut': text_cut,
                'meta': {'seeds': seeds, 'target_tokens': len(target), 'consumer_tokens': len(consumer['ids'])}}


class _View:
    def __init__(self, record):
        self.raw = record


def _chain(prompt, sample, task, row, *, imports=(), sources=None):
    prompt = np.asarray(prompt, np.int32)
    sample = np.asarray(sample, np.int32)
    return {'ids': np.concatenate([prompt, sample]), 'segs': [(len(prompt), len(sample), {**row, 'sample': sample})],
            'imports': list(imports), 'sources': dict(sources or {}), 'task': task, 'round': 0}


def case_files(paths, operators=None):
    out = []
    for path in map(Path, paths):
        files = sorted(path.glob('*/*.port-records.jsonl')) if path.is_dir() else [path]
        out += [f for f in files if operators is None or f.parent.name in operators]
    return out


def build(args):
    tokenizer, roles, fingerprint = load_tokenizer(args.base)
    builder = RowBuilder(tokenizer, roles, fingerprint, max_tokens=args.max_tokens, seed_weight=args.seed_weight)
    held = HeldClosure.from_cases(args.held)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    operators = set(args.operators.split(',')) if args.operators else None
    summary = {'schema': SCHEMA, 'tokenizer_fingerprint': fingerprint, 'role_ids': list(roles),
               'held_sets': {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.held},
               'held_closure': {'groups': len(held.groups), 'ids': len(held.ids), 'shingles': len(held.owner),
                                'shingle_words': SHINGLE, 'text_min': TEXT_MIN},
               'seed_weight': args.seed_weight, 'max_tokens': args.max_tokens, 'sources': {}}
    for path in case_files(args.cases, operators):
        name, source_name = path.parent.name, path.name.removesuffix('.port-records.jsonl')
        started = time.time()
        counts, held_hits, rejects = collections.Counter(), collections.Counter(), collections.Counter()
        provenance = collections.Counter()
        chains = {'train': [], 'test': []}
        cuts = []
        with open(path, 'rb') as handle:
            for line in handle:
                if args.limit and counts['read'] >= args.limit:
                    break
                counts['read'] += 1
                record = json.loads(line)
                split = 'test' if record.get('split') in ('test', 'validation') else 'train'
                hit = held.hit(record)
                if hit:
                    held_hits[f'{split}:{hit}'] += 1
                    continue
                key = hashlib.sha256(line.rstrip(b'\n')).hexdigest()[:20]
                try:
                    rows = builder.build(record, name, key)
                except OperatorCaseError as error:
                    rejects[str(error).split(':')[0]] += 1
                    continue
                for chain in (*rows['operators'], rows['consumer']):
                    chain['segs'][0][2]['case'] = record['id']
                chains[split] += [*rows['operators'], rows['consumer']]
                if split == 'test':
                    cuts.append(rows['text_cut'])
                counts[split] += 1
                provenance.update(seed['provenance'] for seed in rows['meta']['seeds'])
        files = {}
        for split, items in (*chains.items(), ('test-textcut', cuts)):
            if not items:
                continue
            target = out / name / f'{source_name}.{split}.pkl'
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, 'wb') as handle:
                pickle.dump({'schema': SCHEMA, 'chains': items}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            files[split] = {'path': str(target), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                            'chains': len(items),
                            'tokens': int(sum(len(c['ids']) + sum(len(d['ids']) for d in c['sources'].values())
                                              for c in items))}
        summary['sources'][f'{name}/{source_name}'] = {
            'cases': str(path), 'cases_size': path.stat().st_size, 'counts': dict(counts),
            'held_dropped': dict(held_hits), 'rejects': dict(rejects), 'seed_provenance': dict(provenance),
            'files': files, 'seconds': round(time.time() - started, 1)}
        print(json.dumps({f'{name}/{source_name}': summary['sources'][f'{name}/{source_name}']}), flush=True)
    summary['registry_sha256'] = {name: registry_sha256([name]) for name in
                                  sorted({key.split('/')[0] for key in summary['sources']})}
    (out / 'manifest.json').write_text(json.dumps(summary, indent=1, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build')
    p.add_argument('--cases', nargs='+', default=[str(CASES_ROOT)])
    p.add_argument('--out', required=True)
    p.add_argument('--operators', help='comma list (default: every operator directory)')
    p.add_argument('--held', nargs='+', default=[str(p) for p in HELD_SETS])
    p.add_argument('--base', default=LFM3_BASE)
    p.add_argument('--limit', type=int, default=0, help='cases read per source file (smoke)')
    p.add_argument('--max-tokens', type=int, default=32000)
    p.add_argument('--seed-weight', type=float, default=1.0)
    args = parser.parse_args()
    if args.command == 'build':
        build(args)


if __name__ == '__main__':
    main()
