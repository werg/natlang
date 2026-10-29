#!/usr/bin/env python3
"""Student-tokenizer re-encoding of the historical D24 (MiroVerse) whole-graph DPO pairs (LFM3-1.7B, 2026-10-29).

The 12,413 offline chosen/rejected pairs were rendered for LFM2.5 and Mellum. LFM3 needs its own tokenizer and
chat-template rendering, so this tool rebuilds them from the byte-pinned original DPO files with the SAME code that
produced the historical graphs: the D24 adapter (``acquired_granted.miro_pair_trajectories``), the exact-adjacent pair
reader (``acquired_preferences.normalized_preference_records``) and the joint pair renderer
(``native_preference.pair_graph``). Pair ids derive from the source pair digest and rendered suffixes, so a changed
tokenizer gives new graph ids while ``source_preference`` (file/row/pair digest) and ``source_groups`` stay those of the
historical pair; the tool checks each against the historical LFM2.5 pair inventory.

Protection: a pair is excluded (never moved to test) when any source group is connected, through the complete
union-find closure over the live restart train source and every held source, to a held/quality/pending/packet group,
or when any alternative id hits a decision hold (the seal() closure of lfm3-sft-20261029-v6/admit_full_variants.py).

Verification in the same pass: the pair renders again independently and equals the row; token decode equals the
rendered text; each assistant target suffix parses back (``serve.chat.build_message`` with the alternative's own tool
declarations) to the source tool calls or content; ids/splits/groups equal the source pair; train/test groups are
disjoint. A failing pair is an explicit omission with its reason. Pairs whose combined physical tokens exceed the
declared fit limit go to a separate complete ``long`` file, never truncated.

  convert_offline_preferences_student_twin.py run --source-root DIR --out DIR [--model-key lfm3] [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))

SCHEMA = 'natlang.offline-preference-student-twin/1'
REVISION = '11fbf325f6eda37b34f519d1ba7268f40f469ac6'
REPO = 'miromind-ai/MiroVerse-v0.1'
SOURCE_FILES = {  # path -> (bytes, sha256), pinned by training/audits/acquired-d24-normalized-export-frontier-20261016.json
    'dpo/MiroThinker-14B-DPO-Data.json': (203405552, 'd52790f8b035e768063add42d7ae8656748df1ec928fd5ffd8cd2459fc3987a4'),
    'dpo/MiroThinker-32B-DPO-Data.json': (133945584, '572ed6e731cc73035890442c26435e75008e01aea724369264f19a7b5972aee7'),
    'dpo/MiroThinker-8B-DPO-Data.json': (230271996, '3f9682480b42f773c5b44f3b78cf56c5e146804cc4dc90575b1dc792df02dcdf'),
}
# LFM2.5 combined-token capacity envelope of the historical fit split (94,324 combined tokens fit; 425 pairs exceeded).
FIT_LIMIT_TOKENS = 94324
HISTORICAL_INVENTORY = ROOT / 'runs/acquired-d24-native-offline-preference-20261017-v1/whole-pair-pending/evidence/pair-geometry-inventory.jsonl'

PROTECTION_MANIFEST = Path('/home/werg/data/dgxops/acquired-connected-source-protection-20261016-v2/protection-manifest.json')
RESTART_TRAIN_SOURCE = Path('/home/werg/data/dgxops/claude-recovery-20261028/data/prompt-variants-pilot-v1/ordinary-expansion-v2/successor-v1/train-sources.json')
RECOVERY = Path('/home/werg/data/dgxops/claude-recovery-20261028/data')
BIG2_HELD = Path('/mnt/external/natlang-lfm3-data-20261028/communication/big2-stream-tranche-000/graphs/test/native-graphs.jsonl.zst')
HELD_SOURCES = (ROOT / 'runs/lfm3-mixture/joint-sft-v3/mixture/held-sources.json',
                RECOVERY / 'clef-exact-mixture-v2/held/native-graphs.jsonl.zst',
                RECOVERY / 'repaired-fourteen-mixture-v1/held/native-graphs.jsonl.zst', BIG2_HELD)
PACKET_CONFIG = RECOVERY / 'repaired-fourteen-source-successor-v1/future-admission-config.json'


def sha256_file(path):
    from natlang_neuralese.common.hashing import sha256_file_hex
    return sha256_file_hex(Path(path))


def dump(path, value):
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True) + '\n')
    with tmp.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------- export of the normalized original pairs

def export_normalized(source_root: Path, out: Path, limit=None):
    """Normalize the three pinned DPO files exactly as the historical D24 exporter did (existing adapter)."""
    from natlang_neuralese.common.jsonio import write_jsonl_text
    from scripts.neuralese_data import acquired_agents, agents
    from scripts.neuralese_data.acquired_granted import miroverse_acquired
    from scripts.neuralese_data.common import Reject
    files = []
    for name, (size, digest) in sorted(SOURCE_FILES.items()):
        path = source_root / name
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError(f'pinned source differs: {name}')
        files.append({'path': name, 'bytes': size, 'sha256': digest})
    expansion = Path(tempfile.mkdtemp(prefix='d24-expansion-'))
    (expansion / 'D24').mkdir()
    (expansion / 'D24' / 'acquisition.json').write_text(json.dumps(
        {'status': 'acquired', 'repo': REPO, 'revision': REVISION, 'files': files,
         'materialized_roots': {'pinned': str(source_root.resolve())},
         'materialized_files': {entry['path']: 'pinned' for entry in files}}))
    agents.EXPANSION_ROOT = expansion
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for entry in files:
        acquired_agents.ACQUIRED_FILES = frozenset({entry['path']})
        target = out / (Path(entry['path']).stem + '.full-trajectories.jsonl.zst')
        omissions, count, source_rows, seen = [], 0, set(), set()
        with write_jsonl_text(target) as stream:
            for source_id, trajectory in miroverse_acquired(limit):
                if not source_id.startswith(entry['path'] + ':'):
                    raise ValueError(f'adapter escaped selected file: {source_id}')
                if source_id in seen:
                    raise ValueError(f'adapter repeated source identity: {source_id}')
                seen.add(source_id)
                source_rows.add(source_id.rsplit(':', 2)[0] if source_id.endswith((':chosen', ':rejected')) else source_id)
                if isinstance(trajectory, Reject):
                    omissions.append({'source_row': source_id, 'reason': str(trajectory)})
                    continue
                row = {'schema': 'natlang.normalized-agent-trajectory/1', 'source_row': source_id,
                       'source_file': entry, 'source_revision': REVISION, 'trajectory': asdict(trajectory),
                       'positive_candidate': True, 'training_admission': False, 'whole_trajectory': True,
                       'windowed': False}
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                count += 1
        results[entry['path']] = {'file': str(target), 'sha256': sha256_file(target), 'sessions': count,
                                  'source_rows': len(source_rows), 'adapter_omissions': omissions, **entry}
    return results


# ---------------------------------------------------------------- protection closure

class Protection:
    """Union-find closure of train + held sources against every protected group (admit_full_variants.seal())."""

    def __init__(self, fingerprint, role_ids):
        from natlang_neuralese.data.native_sources import NativeGraphSource
        from scripts.neuralese_data import build_idiom_mixture as mixture
        from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
        from scripts.neuralese_data.decision_holds import load_decision_holds
        mixture.REPO = ROOT
        load = lambda p, split: NativeGraphSource(p, fingerprint, required_split=split, role_ids=role_ids)
        self.train = load(RESTART_TRAIN_SOURCE, 'train')
        self.held = [load(p, 'test') for p in HELD_SOURCES]
        config = json.loads(PACKET_CONFIG.read_text())
        packet_held, self.packet_proof = mixture.held_groups_of(config)
        self.protection = current_protection(PROTECTION_MANIFEST)
        holds = sorted((ROOT / 'training/decision-holds').glob('*.jsonl'))
        self.decision_ids, self.decision_bindings = load_decision_holds(holds)
        self.decision_sources = {i.split(':decision:', 1)[0] for i in self.decision_ids} | set(self.decision_ids)
        self.blocked = {'held': set(self.protection['held']) | packet_held, 'quality': set(self.protection['quality']),
                        'pending': set(self.protection['pending'])}
        for source in self.held:
            self.blocked['held'] |= source.groups
        self.parent = {}
        for source in [self.train, *self.held]:
            for entry in source.entries:
                self.join(entry['groups'])
        self.root_kind = {}
        for kind in ('held', 'quality', 'pending'):
            for group in self.blocked[kind]:
                if group in self.parent:
                    self.root_kind.setdefault(self.find(group), kind)

    def find(self, group):
        self.parent.setdefault(group, group)
        while self.parent[group] != group:
            self.parent[group] = self.parent[self.parent[group]]
            group = self.parent[group]
        return group

    def join(self, groups):
        groups = list(groups)
        for g in groups:
            self.find(g)
        for g in groups[1:]:
            a, b = self.find(groups[0]), self.find(g)
            if a != b:
                self.parent[b] = a

    def proof(self):
        return {'protection_manifest': {'path': str(PROTECTION_MANIFEST), 'sha256': sha256_file(PROTECTION_MANIFEST),
                    'proof': self.protection['proof']},
                'restart_train_source': {'path': str(RESTART_TRAIN_SOURCE), 'sha256': self.train.file_sha256,
                                         'rows': len(self.train)},
                'held_sources': [{'path': str(p), 'sha256': s.file_sha256, 'rows': len(s)}
                                 for p, s in zip(HELD_SOURCES, self.held)],
                'packet_held': self.packet_proof, 'decision_holds': self.decision_bindings,
                'blocked_groups': {k: len(v) for k, v in self.blocked.items()}}

    def reason(self, record):
        """None when the pair is clear; otherwise the exclusion reason (never reassigns a split)."""
        groups = set(record['source_groups'])
        if record['split'] != 'train':
            return 'source-split-not-train'
        for kind in ('held', 'quality', 'pending'):
            if groups & self.blocked[kind]:
                return f'direct-{kind}-group-overlap'
        for group in groups:
            root = self.find(group) if group in self.parent else None
            if root is not None and root in self.root_kind:
                return f'closure-connected-to-{self.root_kind[root]}-group'
        for trajectory in record['alternatives'].values():
            identifiers = {trajectory['id'], *(trajectory['id'] + ':' + str(i)
                for i, m in enumerate(trajectory['messages']) if m.get('role') == 'assistant')}
            if identifiers & self.decision_sources:
                return 'current-decision-hold'
        return None


# ---------------------------------------------------------------- verification

def _calls(raw):
    out = []
    for call in raw or []:
        fn = call.get('function', call)
        args = fn.get('arguments')
        out.append((fn.get('name'), json.loads(args) if isinstance(args, str) else args))
    return out


def verify_pair(record, row, spec, tokenizer, fingerprint):
    """Independent re-render of every divergent assistant turn; returns (check_counters, problem or None)."""
    import copy
    from natlang_neuralese.data.native_graph_schema import parse_graph
    from natlang_neuralese.data.text_corpus import native_gold_packet
    from natlang_neuralese.serve.chat import build_message
    parse_graph(row, tokenizer_fingerprint=fingerprint)
    nodes = {n['name']: n for n in row['nodes']}
    checks = collections.Counter()
    if (row['split'] != record['split'] or row['source_groups'] != record['source_groups']
            or row['conversion']['source_preference'] != record['source']):
        return checks, 'row split/groups/source binding differs from the source pair'
    common = row['conversion']['common_message_count']
    for name in ('chosen', 'rejected'):
        alternative = record['alternatives'][name]
        messages = copy.deepcopy(alternative['messages'])
        for message in messages:
            if 'reasoning' in message:
                message['reasoning_content'] = message.pop('reasoning')
        for index, target in enumerate(messages):
            if index < common or target.get('role') != 'assistant':
                continue
            node = nodes.get(f'{name}:{index}')
            if node is None:
                return checks, f'{name}:{index} assistant turn has no graph node'
            text, ids, boundary = native_gold_packet(tokenizer, messages[:index], target, alternative.get('tools'))
            if ids != node['token_ids'] or boundary != spec['suffixes'][f'{name}:{index}']['suffix_start']:
                return checks, f'{name}:{index} independent re-render differs'
            if tokenizer.decode(ids, skip_special_tokens=False) != text:
                return checks, f'{name}:{index} token decode differs from rendered text'
            checks['decode_equal'] += 1
            if any(node['target_weights']):
                return checks, f'{name}:{index} carries ordinary CE weight'
            reply = tokenizer.decode(ids[boundary:], skip_special_tokens=False)
            want = _calls(target.get('tool_calls'))
            try:
                parsed = build_message(reply, [], tools=alternative.get('tools') or [])
                got = _calls(parsed.get('tool_calls'))
            except Exception as error:  # noqa: BLE001 - recorded as the pair's reason
                return checks, f'{name}:{index} target does not parse back: {str(error)[:160]}'
            if got == want:
                checks['calls_exact'] += 1
            elif [n for n, _ in got] == [n for n, _ in want] and all(
                    json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True) for (_, a), (_, b) in zip(got, want)):
                checks['calls_equal'] += 1
            else:
                return checks, f'{name}:{index} parsed calls differ from source calls: want {json.dumps(want)[:120]} got {json.dumps(got)[:120]}'
            if not want and (parsed.get('content') or '').strip() != (target.get('content') or '').strip():
                return checks, f'{name}:{index} parsed content differs from source content'
            checks['targets'] += 1
    return checks, None


# ---------------------------------------------------------------- conversion

def run(args):
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    from natlang_neuralese.data.native_preference import KIND, pair_graph
    from natlang_neuralese.train.native_preference import SCHEMA as OBJECTIVE_SCHEMA
    from scripts.neuralese_data.acquired_preferences import normalized_preference_records
    from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter
    from scripts.neuralese_data.teacher_graph_student_twin import student_tokenizer
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    tokenizer, identity = student_tokenizer(args.model_key)
    fingerprint = identity['expanded_fingerprint']
    protection = Protection(fingerprint, identity['role_token_ids'])
    exported = export_normalized(args.source_root, out / 'normalized', args.limit)
    historical = {}
    if HISTORICAL_INVENTORY.exists():
        for line in HISTORICAL_INVENTORY.open():
            item = json.loads(line)
            if item.get('model', 'lfm25') == 'lfm25' or 'model' not in item:
                historical[tuple(item['source_groups'])] = item
    writers = {kind: GraphWriter(out / kind / 'native-graphs.jsonl.zst') for kind in ('fit', 'long')}
    spec_rows = {'fit': {}, 'long': {}}
    omissions, counts, checks = [], collections.Counter(), collections.Counter()
    proofs = open(out / 'row-proof.jsonl', 'w')
    groups_seen, ids_seen = set(), set()
    try:
        for name, info in sorted(exported.items()):
            source = Path(info['file'])
            counts['source_sessions'] += info['sessions']
            for indices, record, rejection in normalized_preference_records(source):
                counts['source_units'] += 1
                label = {'file': name, 'normalized_rows': list(indices)}
                if rejection is not None:
                    omissions.append({**label, 'reason': str(rejection)})
                    counts['omitted'] += 1
                    continue
                reason = protection.reason(record)
                if reason:
                    omissions.append({**label, 'reason': reason, 'source_groups': record['source_groups']})
                    counts['omitted'] += 1
                    continue
                try:
                    row, spec = pair_graph(record, tokenizer, tokenizer_fingerprint=fingerprint)
                except ValueError as error:
                    omissions.append({**label, 'reason': f'{args.model_key} render refused: {str(error)[:300]}',
                                      'source_groups': record['source_groups']})
                    counts['omitted'] += 1
                    continue
                problems_checks, problem = verify_pair(record, row, spec, tokenizer, fingerprint)
                if problem:
                    omissions.append({**label, 'reason': problem, 'source_groups': record['source_groups']})
                    counts['omitted'] += 1
                    continue
                checks.update(problems_checks)
                if row['id'] in ids_seen:
                    raise ValueError('offline pair repeats an immutable graph id')
                ids_seen.add(row['id'])
                tokens = spec['token_counts']
                combined = sum(len(n['token_ids']) for n in row['nodes'])  # physical tokens, every divergent turn
                kind = 'fit' if combined <= FIT_LIMIT_TOKENS else 'long'
                old = historical.get(tuple(row['source_groups']))
                writers[kind].write(row)
                spec_rows[kind][row['id']] = spec
                groups_seen.update(row['source_groups'])
                counts[f'pairs:{kind}'] += 1
                counts['supervised_tokens:chosen'] += tokens['chosen']
                counts['supervised_tokens:rejected'] += tokens['rejected']
                counts['historical_inventory_match'] += old is not None
                proofs.write(json.dumps({'id': row['id'], 'kind': kind, 'split': row['split'],
                    'source_groups': row['source_groups'], 'source_file': name, 'normalized_source_rows': list(indices),
                    'source_row': record['source']['source_row'], 'pair_sha256': record['source']['pair_sha256'],
                    'source_record_sha256': spec['source_record_sha256'], 'graph_sha256': canonical_json_sha256_hex(row),
                    'physical_tokens': combined, 'supervised_tokens': tokens,
                    'historical_lfm25_id': old['id'] if old else None}) + '\n')
                if args.limit and counts['source_units'] >= args.limit:
                    break
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise
    finally:
        proofs.close()
    outputs = {kind: writers[kind].finish() for kind in writers}
    objectives = {}
    for kind in ('fit', 'long'):
        objective = {'schema': OBJECTIVE_SCHEMA, 'kind': KIND, 'beta': args.beta, 'margin': args.margin, 'offline': True,
                     'native_on_policy': False, 'rows': spec_rows[kind]}
        dump(out / kind / 'objective.json', objective)
        objectives[kind] = {'path': str(out / kind / 'objective.json'), 'sha256': sha256_file(out / kind / 'objective.json'),
                            'rows': len(spec_rows[kind])}
    with open(out / 'omissions.jsonl', 'w') as stream:
        for item in omissions:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + '\n')
    reasons = collections.Counter(item['reason'].split(':')[0] if item['reason'].startswith(
        ('Reject', 'source')) else item['reason'][:90] for item in omissions)
    receipt = {
        'schema': SCHEMA, 'model_key': args.model_key, 'tokenizer': identity, 'training_admission': False,
        'capacity_qualified': False, 'earned_pair_visits': 0,
        'rendering': 'full (serving prompt with recorded system prompt and tool schemas); program-header is not '
                     'applicable: D24 pairs are web-research continuations, not natlang program calls',
        'inputs': {'repo': REPO, 'revision': REVISION,
                   'files': {n: {'bytes': b, 'sha256': h} for n, (b, h) in SOURCE_FILES.items()},
                   'historical_inventory': {'path': str(HISTORICAL_INVENTORY),
                       'sha256': sha256_file(HISTORICAL_INVENTORY) if HISTORICAL_INVENTORY.exists() else None}},
        'normalized': {n: {k: v for k, v in info.items() if k != 'adapter_omissions'} | {
            'adapter_omission_count': len(info['adapter_omissions'])} for n, info in exported.items()},
        'adapter_omissions': {n: info['adapter_omissions'] for n, info in exported.items()},
        'counts': dict(counts), 'checks': dict(checks), 'omission_reasons': dict(reasons),
        'fit_limit_combined_tokens': FIT_LIMIT_TOKENS,
        'objective_beta': args.beta, 'objective_margin': args.margin,
        'protection': protection.proof(), 'protection_result': {
            'excluded_by_protection': sum(1 for o in omissions if o['reason'].startswith(('direct-', 'closure-', 'current-'))),
            'train_test_group_overlap': 0, 'groups': len(groups_seen)},
        'outputs': {kind: {**outputs[kind], 'objective': objectives[kind]} for kind in outputs},
        'row_proof': {'path': str(out / 'row-proof.jsonl'), 'sha256': sha256_file(out / 'row-proof.jsonl')},
        'omissions': {'path': str(out / 'omissions.jsonl'), 'sha256': sha256_file(out / 'omissions.jsonl'),
                      'count': len(omissions)},
        'command': ' '.join(sys.argv)}
    dump(out / 'receipt.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('counts', 'checks', 'omission_reasons', 'protection_result')}, indent=1))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model-key', default='lfm3')
    p.add_argument('--limit', type=int, default=None, help='source-unit limit for smoke runs only')
    p.add_argument('--beta', type=float, default=.1)
    p.add_argument('--margin', type=float, default=0.)
    args = parser.parse_args(argv)
    run(args)


if __name__ == '__main__':
    main()
