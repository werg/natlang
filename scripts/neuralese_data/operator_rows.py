#!/usr/bin/env python3
"""Learned-operator cases -> exp0_sft per-call rows (plans/neuralese/OPERATORS_RESTORE_20261117.md §3.2, §9).

A `natlang.port-record/1` operator case (COMPACTION_OPERATOR_CONVERTERS.md) becomes the v3 operator write and its
consumer. Every prompt string at an operator site comes from the runtime's own rendering code (ts-host
native/operator-site.ts through scripts/operator-sites.mjs, `RuntimeSites`): the operator call's messages, the block a
consumer sees with the state in place, the crisp text it sees without it, and whether the runtime compacts the value at
all (a case none of whose sources the runtime would compact is rejected). compact-value cases are tool outputs;
compact-params and extract cases are string arguments of the call that answers the case's question.

* **operator chain** (one per compacted source): the runtime's operator call with its generation prompt, then the
  seed as the assistant reply (gold / checked / mechanical, `data/operator_seeds.py`, clipped to the operator's span
  cap). The reply is the only target; its weight is `--seed-weight`.
* **consumer chain**: the case's consumer conversation (data) with each source's runtime block before its first user
  message, the state imported (`consume_open`, the span, `consume_close`) where the runtime places it; the target
  reply is the only target. `imports` [(row, key)] and `sources` {key: {ids, start, stop}} are the pass-through's
  format (`lfm3_fast_train.passthrough_plan`, `lower_rollout_turns.write_sources`): the producer document is the site,
  the reply up to its content, `produce_open` and the content tokens (the in-context capture).
* **text-cut chain** (test split): the same consumer reading the runtime's crisp text of each source: the ablation's
  baseline.

Held data: every case whose split groups, ids or input text meet the held-v3 / held-v2 closure is dropped and counted
(`HeldClosure`: exact group names, task and upstream ids, 10-word shingles of the held tasks' inputs).

    operator_rows.py build --cases ROOT_OR_FILE [...] --out DIR [--limit N] [--operators a,b] [--context-tokens N]

Needs the built ts-host dist (`npm run build:node`) and node on PATH.
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

from natlang_neuralese.data.operator_graph import (IMITATION_LABELS, OperatorCaseError, TokenRenderer,  # noqa: E402
                                                   _placeholder, source_text)
from natlang_neuralese.data.operator_seeds import seed_for  # noqa: E402
from natlang_neuralese.data.port_record_graph import _serving_messages, _target_message, _target_reply  # noqa: E402
from natlang_neuralese.common.code_provenance import code_provenance  # noqa: E402
from natlang_neuralese.common.paths import resolve  # noqa: E402
from natlang_neuralese.model.learned_operators import registry_sha256  # noqa: E402

SCHEMA = 'natlang.operator-rows/2'
REPO = Path(__file__).resolve().parents[2]
NODE = Path.home() / '.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin/node'
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

    def __init__(self, tokenizer, role_ids, fingerprint, *, max_tokens=32000, seed_weight=1.0, sites=None,
                 context_tokens=None):
        self.context_tokens = context_tokens
        self.renderer = TokenRenderer(tokenizer, role_ids=role_ids)
        self.sites = sites or RuntimeSites()
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

    def build(self, record: dict, name: str, key: str):
        """{'operators': [chain], 'consumer': chain, 'text_cut': chain, 'meta': {...}} or OperatorCaseError."""
        if record.get('outcome', {}).get('label') not in IMITATION_LABELS:
            raise OperatorCaseError('target-not-imitation-label')
        if 'sources' not in record['consumer'].get('withheld', []):
            raise OperatorCaseError('sources-not-withheld')
        sources = record['sources']
        tools = record['consumer'].get('tools') or None
        sites = [self.sites.site({**site_request(name, record, source, index, len(sources)),
                                  'context_tokens': self.context_tokens})
                 for index, source in enumerate(sources)]
        if not any(site['compacted'] for site in sites):
            raise OperatorCaseError('no-compacted-source')
        try:
            target = _target_reply(self.tokenizer, _target_message(_View(record)), tools)
        except ValueError as error:
            raise OperatorCaseError(f'target-render: {error}') from None
        operators, documents, seeds, parts, cut_parts = [], {}, [], [], []
        for index, (source, site) in enumerate(zip(sources, sites)):
            if parts:
                parts.append({'type': 'text', 'text': '\n'})
                cut_parts.append({'type': 'text', 'text': '\n'})
            cut_parts.append({'type': 'text', 'text': site['text']})
            if not site['compacted']:
                parts.append({'type': 'text', 'text': site['text']})
                continue
            prompt, _ = self.renderer._render(site['messages'], site['tools'] or None, {}, generation=True)
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
            document = [*prompt, *reply[:start], self.role_ids[0], *span]
            documents[source_key] = {'ids': np.asarray(document, np.int32), 'start': len(document) - len(span),
                                     'stop': len(document)}
            operators.append(_chain(prompt, reply, key, {
                'kind': 'operator', 'operator': site['operator'], 'case_operator': name, 'source_key': source_key,
                'weight': np.full(len(reply), self.seed_weight, np.float32)}))
            seeds.append({**seed, 'site_tokens': len(prompt), 'span_tokens': len(span)})
            parts += [{'type': 'text', 'text': site['state']['before']},
                      {'type': 'native_state', 'id': _placeholder(f'operator-state:{source_key}'), 'key': source_key},
                      {'type': 'text', 'text': site['state']['after']}]
        states = {part['id']: part.pop('key') for part in parts if part['type'] == 'native_state'}
        splices = {state: [self.role_ids[2], *documents[k]['ids'][documents[k]['start']:], self.role_ids[3]]
                   for state, k in states.items()}
        context, spans = self.renderer._render(consumer_messages(record, parts), tools, splices, generation=True)
        imports = sorted((spans[state][0] + 1, k) for state, k in states.items())
        consumer = _chain(context, target, key, {'kind': 'consumer', 'operator': name},
                          imports=imports, sources=documents)
        total = len(consumer['ids']) + sum(len(d['ids']) for d in documents.values())
        if total > self.max_tokens or any(len(c['ids']) > self.max_tokens for c in operators):
            raise OperatorCaseError(f'over-token-cap: {total} > {self.max_tokens}')
        cut_context, _ = self.renderer._render(consumer_messages(record, cut_parts), tools, {}, generation=True)
        text_cut = _chain(cut_context, target, key, {'kind': 'text-cut', 'operator': name})
        return {'operators': operators, 'consumer': consumer, 'text_cut': text_cut,
                'meta': {'seeds': seeds, 'target_tokens': len(target), 'consumer_tokens': len(consumer['ids'])}}


_PURPOSE = re.compile(r'^(?:Read th(?:is|ese) (?P<tool>\S+) results? for: '
                      r'|The call that receives (?:this (?P<noun>\w+)|these \d+ arguments \((?P<nouns>\w+)\)) answers: '
                      r'|Extract what this task needs from the (?P<doc>\w+)\. Task: )(?P<purpose>.*)$', re.S)


def site_request(name: str, record: dict, source: dict, index: int, count: int) -> dict:
    """The runtime site of one case source (ts-host native/operator-site.ts `OperatorSiteRequest`). A compact-value
    case is its tool's output read for its call's purpose; compact-params and extract cases are string arguments of the
    call that answers the case's question, named by the case's noun (`passage`, or `passage1`... for several)."""
    instructions = record['writer'].get('instructions') or ''
    match = _PURPOSE.match(instructions)
    purpose = match['purpose'] if match else instructions
    text = source_text(source)
    if name == 'compact-value':
        tool = (source.get('meta') or {}).get('tool') or (match and match['tool']) or 'tool'
        return {'operator': 'compact-value', 'value': text, 'purpose': purpose, 'tool': tool, 'entry': index}
    if name in ('compact-params', 'extract'):
        noun = (match and (match['noun'] or match['nouns'] or match['doc'])) or 'document'
        return {'operator': 'compact-params', 'value': text, 'purpose': purpose,
                'argument': {'name': noun if count == 1 else f'{noun}{index + 1}', 'type': 'string'}}
    raise OperatorCaseError(f'no-runtime-site: {name}')


def consumer_messages(record: dict, parts: list) -> list:
    """The case's consumer conversation (data) with the sources' blocks, as the runtime renders them, before its first
    user message."""
    messages = _serving_messages(record['consumer']['context'])
    first = next((i for i, message in enumerate(messages) if message.get('role') == 'user'), None)
    if first is None:
        raise OperatorCaseError('consumer-without-user-message')
    old = messages[first].get('content')
    content = [*parts, {'type': 'text', 'text': '\n\n'}]
    content += [{'type': 'text', 'text': old}] if isinstance(old, str) else list(old or [])
    merged = []
    for part in content:
        if part.get('type') == 'text' and merged and merged[-1].get('type') == 'text':
            merged[-1] = {'type': 'text', 'text': merged[-1]['text'] + part['text']}
        elif part.get('type') != 'text' or part['text']:
            merged.append(dict(part))
    messages[first]['content'] = merged[0]['text'] if len(merged) == 1 and merged[0]['type'] == 'text' else merged
    return messages


class RuntimeSites:
    """The runtime's operator sites (ts-host/scripts/operator-sites.mjs over the built dist): one node process, one
    JSON line per request."""

    def __init__(self, runtime=REPO / 'ts-host', node=None):
        import shutil
        import subprocess
        node = node or shutil.which('node') or str(NODE)
        self.process = subprocess.Popen([node, str(Path(runtime) / 'scripts/operator-sites.mjs')], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)

    def site(self, request: dict) -> dict:
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        reply = json.loads(self.process.stdout.readline())
        if 'error' in reply:
            raise OperatorCaseError(f"runtime-site: {reply['error']}")
        return reply

    def close(self):
        self.process.stdin.close()
        self.process.wait()


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
    builder = RowBuilder(tokenizer, roles, fingerprint, max_tokens=args.max_tokens, seed_weight=args.seed_weight,
                         context_tokens=args.context_tokens)
    held = HeldClosure.from_cases(args.held)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    operators = set(args.operators.split(',')) if args.operators else None
    summary = {'schema': SCHEMA, 'tokenizer_fingerprint': fingerprint, 'role_ids': list(roles),
               'held_sets': {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.held},
               'held_closure': {'groups': len(held.groups), 'ids': len(held.ids), 'shingles': len(held.owner),
                                'shingle_words': SHINGLE, 'text_min': TEXT_MIN},
               'seed_weight': args.seed_weight, 'max_tokens': args.max_tokens, 'splits': args.splits,
               'context_tokens': args.context_tokens, 'runtime_code': code_provenance(), 'sources': {}}
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
                if split not in args.splits:
                    continue
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
    builder.sites.close()
    (out / 'manifest.json').write_text(json.dumps(summary, indent=1, sort_keys=True) + '\n')


def merge(args):
    """One exp0_sft `--chains` pickle of every train file (operator and consumer chains) and one `--val-chains` pickle
    of up to --val-per-source test consumer chains per source, with a manifest of their sha256 and counts."""
    rows = Path(args.rows)
    manifest = json.loads((rows / 'manifest.json').read_text())
    train, val, counts = [], [], {}
    for name, entry in sorted(manifest['sources'].items()):
        files = entry.get('files') or {}
        got = {}
        if 'train' in files:
            chains = pickle.load(open(files['train']['path'], 'rb'))['chains']
            if args.max_cases_per_source:   # whole cases (a case's operator chains and its consumer share 'task')
                keep = set(list(dict.fromkeys(c['task'] for c in chains))[:args.max_cases_per_source])
                chains = [c for c in chains if c['task'] in keep]
            train += chains
            got['train'] = len(chains)
        if 'test' in files and args.val_per_source:
            consumers = [c for c in pickle.load(open(files['test']['path'], 'rb'))['chains']
                         if c['segs'][0][2].get('kind') == 'consumer'][:args.val_per_source]
            val += consumers
            got['val'] = len(consumers)
        counts[name] = got
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {'schema': SCHEMA + '#merge', 'rows': str(rows), 'sources': counts}
    for label, chains in (('chains', train), ('val', val)):
        path = out / f'{label}.pkl'
        with open(path, 'wb') as handle:
            pickle.dump({'schema': SCHEMA, 'chains': chains}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        summary[label] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'chains': len(chains),
                          'tokens': int(sum(len(c['ids']) + sum(len(d['ids']) for d in c['sources'].values()) for c in chains)),
                          'loss_tokens': int(sum(length for c in chains for _, length, _ in c['segs']))}
    (out / 'merge.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'sources'}))


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
    p.add_argument('--splits', nargs='+', default=['train', 'test'], choices=['train', 'test'])
    p.add_argument('--context-tokens', type=int, default=None,
                   help="the call's context budget the runtime sizes its shown page from (default: the page floor)")
    m = sub.add_parser('merge')
    m.add_argument('--rows', required=True)
    m.add_argument('--out', required=True)
    m.add_argument('--max-cases-per-source', type=int, default=0)
    m.add_argument('--val-per-source', type=int, default=32)
    args = parser.parse_args()
    if args.command == 'build':
        build(args)
    elif args.command == 'merge':
        merge(args)


if __name__ == '__main__':
    main()
