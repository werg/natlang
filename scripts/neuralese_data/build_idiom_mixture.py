#!/usr/bin/env python3
"""Build the prioritized joint-SFT mixture of native graphs (LFM3 joint SFT; LFM3_RECIPE_20261028.md).

The selection rules live in a declared, hashed policy (training/neuralese/mixtures/*.json: features, drop rules,
tiers, caps, tier weights, ordering, cohort unit, objective); this script only evaluates them. Inputs come from a
config JSON (render directories with native graph views, teacher-twin graph files, communication graph files, idiom
score files, protected held/dev packets).

Per sample (one graph row) the features of its whole source trajectory are computed from the idiom scores
(idiom_scores.py: call-graph parent edges, function identities, idioms, action shares, reducer use, near-duplicate
key) and from the graph itself (genuine producer->consumer edges). Positive-target filters per decision (verified root
outcome, no un-awaited nl, no failed child action, no holds) decide which documents are eligible: a whole-trajectory
chain only when all of its decisions are positive, otherwise the program-header documents of its positive decisions.
Every eligible sample is dropped (with its reason) or put in the first matching tier; per tier, near-duplicate
clusters and families (reducers separately) are capped. Each (tier, category) cell becomes equal-mass cohort files;
tier 1 carries training_priority 1 (scheduled first), lower tiers follow in manifest order. Train rows sharing a
source group with any held row of the inputs or with the protected packets are dropped.

Outputs: OUT/cohorts/*.jsonl.zst, OUT/held/native-graphs.jsonl.zst, OUT/train-sources.json, OUT/held-sources.json
(natlang.native-graph-sources/1), OUT/sample-features.jsonl.zst (every eligible sample's features, tier, decision),
OUT/mixture.json (composition and distributions). The CE policy is then built from the manifests with
build_native_ce_supervision_policy.py and the policy's objective.

  build_idiom_mixture.py --config CONFIG.json --out OUT
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))
sys.path.insert(0, str(REPO / 'scripts' / 'neuralese_data'))
from natlang_neuralese.common.hashing import sha256_file_hex  # noqa: E402

SCHEMA = 'natlang.idiom-mixture/2'
ORDER = ('iteration', 'delegation', 'exact', 'scope', 'broad')  # rarest first: a document's category
FIT = {'per_item_nl_promise_all': 2., 'per_item_nl_for_of': 2., 'iterate': 2., 'nl_awaited': 1., 'sort': 1.,
       'filter': .5, 'exact_ops': .5, 'typed_child_answer': 1., 'scope_direct': .5, 'finish': .25}
IDIOM_KEYS = ('nl_awaited', 'per_item_nl_promise_all', 'per_item_nl_for_of', 'iterate', 'decide')


def iter_lines(path: Path):
    proc = subprocess.Popen(['zstd', '-dcq', '--long=31', str(path)], stdout=subprocess.PIPE) \
        if path.suffix == '.zst' else None
    stream = proc.stdout if proc else open(path, 'rb')
    for number, raw in enumerate(stream):
        if raw.strip():
            yield number, raw
    if proc and proc.wait():
        raise RuntimeError(f'zstd failed on {path}')


class Writer:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = path.with_name(path.name + '.tmp')
        self.raw = open(self.tmp, 'wb')
        self.proc = subprocess.Popen(['zstd', '-q', '-5', '-T2', '-c'], stdin=subprocess.PIPE, stdout=self.raw)
        self.rows = 0

    def write(self, line: bytes):
        self.proc.stdin.write(line if line.endswith(b'\n') else line + b'\n')
        self.rows += 1

    def close(self) -> dict:
        self.proc.stdin.close()
        if self.proc.wait():
            raise RuntimeError(f'zstd failed for {self.path}')
        self.raw.close()
        self.tmp.replace(self.path)
        return {'path': str(self.path), 'rows': self.rows, 'bytes': self.path.stat().st_size,
                'sha256': sha256_file_hex(self.path)}


# ------------------------------------------------------------------------------------------------- scores -> features

def load_scores(files):
    """Decisions {id: compact record} and trajectories {id: [decision ids]}."""
    decisions, trajectories, census = {}, collections.defaultdict(list), collections.Counter()
    for path in files:
        for _, raw in iter_lines(Path(path)):
            row = json.loads(raw)
            reasons = [r for r in row['exclusions'] if r != 'decision not approved']  # renderer admission decides
            previous = decisions.get(row['id'])
            if previous is not None:
                previous['positive'] = previous['positive'] and not reasons  # one non-positive reading excludes
                continue
            s = row['scores']
            decisions[row['id']] = {
                'category': row['category'], 'positive': not reasons, 'verified': row.get('verified') is True,
                'fit': sum(w for k, w in FIT.items() if s.get(k)), 'family': row.get('curriculum_family')
                or row.get('family') or 'unknown', 'trajectory': row.get('trajectory') or row['id'],
                'invocation': row.get('invocation'), 'parent': row.get('parent_invocation'),
                'function_key': row.get('function_key'), 'near_duplicate_key': row.get('near_duplicate_key'),
                'idiom': bool(any(s.get(k) for k in IDIOM_KEYS) or row.get('folder_calls')),
                'evals': s.get('evals', 0), 'natlang_chars': row.get('natlang_chars', 0),
                'chars': row.get('natlang_chars', 0) + row.get('generic_tool_chars', 0) + row.get('chat_chars', 0),
                'markers': row.get('text_neuralese_markers', 0), 'reducer': bool(row.get('reducer')),
                'folder_apply': row.get('folder_apply_calls', 0), 'reducer_cursor': bool(row.get('reducer_iterate_cursor')),
                'domain': row.get('domain'), 'task_kind': row.get('task_kind'), 'child': bool(row.get('child')),
                'family_text': ' '.join(str(row.get(k) or '') for k in ('family', 'curriculum_family', 'task_kind',
                                                                          'program'))}
            trajectories[decisions[row['id']]['trajectory']].append(row['id'])
            census[f'{row["category"]}:{"positive" if not reasons else "excluded"}'] += 1
            for reason in reasons:
                census['exclusion:' + reason[:60]] += 1
    return decisions, trajectories, census


def trajectory_features(ids, decisions) -> dict:
    rows = [decisions[i] for i in ids]
    invocations = {r['invocation'] for r in rows if r['invocation']}
    parent = {r['invocation']: r['parent'] for r in rows if r['invocation'] and r['parent']}
    for r in rows:  # a child without a captured parent edge hangs off the root
        if r['invocation'] and r['child'] and r['invocation'] not in parent:
            parent[r['invocation']] = next((x['invocation'] for x in rows if not x['child'] and x['invocation']), None)
    def depth(node, seen=()):
        up = parent.get(node)
        return 0 if up is None or up in seen or up == node else 1 + depth(up, (*seen, node))
    children = collections.Counter(p for p in parent.values() if p)
    root = next((r for r in rows if not r['child']), rows[0])
    chars = sum(r['chars'] for r in rows)
    return {'invocations': max(1, len(invocations)), 'max_depth': max((depth(i) for i in invocations), default=0),
            'max_branching': max(children.values(), default=0),
            'distinct_functions': len({r['function_key'] for r in rows if r['function_key']}),
            'typed_natlang_calls': len({r['invocation'] for r in rows if r['child'] and r['invocation']}),
            'natlang_idioms': sum(r['idiom'] for r in rows), 'evals': sum(r['evals'] for r in rows),
            'natlang_action_share': sum(r['natlang_chars'] for r in rows) / chars if chars else 0.,
            'text_neuralese_markers': sum(r['markers'] for r in rows), 'reducer': any(r['reducer'] for r in rows),
            'folder_apply_calls': sum(r['folder_apply'] for r in rows),
            'reducer_iterate_cursor': any(r['reducer_cursor'] for r in rows),
            'verified': all(r['verified'] for r in rows), 'domain': root['domain'] or 'unknown',
            'family': root['family'], 'task_shape': f'{root["task_kind"]}/{root["family"]}',
            'near_duplicate_cluster': root['near_duplicate_key'] or root['family']}


def graph_edges(row) -> int:
    edges = row.get('edges')
    if isinstance(edges, list):
        return len(edges)
    return sum(len(node.get('imports') or []) for node in row.get('nodes') or [])


CHOICE_FAMILY = re.compile(r'judge|decision|decide|refine|progress', re.I)


def derive(features: dict) -> dict:
    deep = features['max_depth'] >= 2 or features['max_branching'] >= 2
    return {**features, 'deep_or_branching': deep,
            # Decision and scored-judge calls: their supervised output is the choice itself (G3 of the 2026-10-28
            # coverage audit), so answer-only is their natlang structure, not plain QA.
            'choice_call': bool(CHOICE_FAMILY.search(f'{features.get("family")} {features.get("task_shape")}')),
            'neuralese_or_reducer': bool(features['neuralese_edges'] or features['text_neuralese_markers']
                                         or features['reducer']),
            'deep_or_branching_or_reducer': bool(deep or features['reducer'])}


def holds(condition: dict, features: dict) -> bool:
    for key, test in condition.items():
        value = features[key]
        if 'eq' in test and value != test['eq']:
            return False
        if 'min' in test and value < test['min']:
            return False
        if 'max' in test and value > test['max']:
            return False
        if 'lt' in test and not value < test['lt']:
            return False
    return True


class OracleRegradeHold:
    """Rows of task families graded by superseded (exploitable) oracles stay out until a regrade passes them
    (policy ``oracle_regrade_hold``; 2026-10-28: edit_stream / contract_diagnosis, inventory
    plans/neuralese/oracle-repair-20261028/old-oracle-admitted-sources.json). Reference replays (generator references,
    correct by construction, e.g. static-lambda) are exempt; ``regrade_passed`` files (JSON lines with ``id``: a source
    record / teacher decision id) re-admit regraded rows."""

    def __init__(self, hold: dict | None):
        self.hold = hold or {}
        self.pattern = re.compile('|'.join(map(re.escape, self.hold.get('families', []))) or r'(?!)')
        exempt = list(self.hold.get('exempt_path_markers', []))
        inventory = self.hold.get('inventory')
        if inventory and (REPO / inventory).exists():
            for item in json.loads((REPO / inventory).read_text()).get('corpora_with_family_mentions_in_scanned_files', []):
                if str(item.get('kind', '')).startswith('reference replay'):
                    exempt.append(item['corpus'])
        self.exempt = exempt
        self.passed = set()
        for listed in self.hold.get('regrade_passed', []):
            if (REPO / listed).exists():
                for _, raw in iter_lines(REPO / listed):
                    self.passed.add(json.loads(raw)['id'])

    def reason(self, path, known: list, ids: list) -> str | None:
        if not self.hold:
            return None
        if not any(self.pattern.search(k.get('family_text') or f"{k.get('family')} {k.get('task_kind')}") for k in known):
            return None
        if any(marker in str(path) for marker in self.exempt):
            return None
        if ids and all(i in self.passed for i in ids):
            return None
        return 'oracle regrade hold: family graded by a superseded oracle, not yet regraded'


def classify(features: dict, policy: dict):
    for rule in policy['drop_rules']:
        if holds(rule['all'], features):
            return None, rule['reason']
    for tier in policy['tiers']:
        if any(holds(condition, features) for condition in tier['any_of']):
            return tier['tier'], None
    return None, 'matches no tier (unverified or no natlang actions)'


# ---------------------------------------------------------------------------------------------------- candidates

def held_groups_of(config) -> tuple[set, dict]:
    groups, sources = set(), {}
    for packet in config.get('held_packets', []):
        n = 0
        for _, raw in iter_lines(REPO / packet):
            groups.update(json.loads(raw).get('source_groups') or [])
            n += 1
        sources[packet] = {'sha256': sha256_file_hex(REPO / packet), 'cases': n}
    return groups, sources


def graph_files(render: str, split: str):
    for view in ('whole-trajectory', 'program-header'):  # whole trajectories first: they cover chains
        path = REPO / render / 'native-graphs' / view / split / 'native-graphs.jsonl.zst'
        if path.exists():
            yield view, path


def content_key(row) -> str:
    digest = hashlib.sha256()
    for node in row['nodes']:
        digest.update(json.dumps([node['token_ids'], node['target_weights']], separators=(',', ':')).encode())
    return digest.hexdigest()


def carved(groups, carve) -> bool:
    """The policy's permanent held slice: a source group is held when sha256(salt + group) falls below the declared
    fraction (deterministic, so every later version holds the same groups)."""
    if not carve:
        return False
    limit = int(carve['fraction'] * 2 ** 32)
    return any(int(hashlib.sha256((carve['salt'] + g).encode()).hexdigest()[:8], 16) < limit for g in groups)


def carve_groups(config, policy, trained_groups=frozenset()) -> set:
    """Source groups of the permanent held slice. Rows sharing a source group form one component (a row touching a
    held group is held whole); a component is a candidate when one of its groups falls in the declared hash fraction,
    and never when it touches a group an earlier version trained on. Candidates are taken in hash order while their
    rows fit ``max_rows``: a component that would overflow stays in training (2026-10-28: v3 left 8,648 rows of
    over-cap carved components neither held nor trained)."""
    carve = policy.get('held_carve')
    if not carve:
        return set()
    parent = {}

    def find(g):
        while parent.setdefault(g, g) != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    rows = []
    paths = [path for render in config['renders'] for _, path in graph_files(render, 'train')]
    paths += [REPO / item for item in config.get('teacher_graphs', [])]
    for path in paths:
        for _, raw in iter_lines(path):
            groups = sorted(json.loads(raw).get('source_groups') or ())
            if not groups:
                continue
            rows.append(groups[0])
            for g in groups[1:]:
                a, b = find(groups[0]), find(g)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    size = collections.Counter(find(g) for g in rows)
    members = collections.defaultdict(set)
    for g in list(parent):
        members[find(g)].add(g)
    candidates = [root for root, groups in members.items()
                  if any(carved([g], carve) for g in groups) and not groups & set(trained_groups)]
    candidates.sort(key=lambda root: hashlib.sha256((carve['salt'] + root).encode()).hexdigest())
    chosen, held_rows = set(), 0
    for root in candidates:
        if held_rows + size[root] > carve['max_rows']:
            continue
        chosen |= members[root]
        held_rows += size[root]
    return chosen


def prior_versions(config):
    """Earlier mixture versions this one appends to: their train sources stay in the manifest unchanged, their rows,
    decisions and graphs are taken, and their held groups stay held."""
    sources, ids, decisions, keys, groups, trained = [], set(), set(), set(), set(), set()
    for prior in config.get('prior', []):
        root = (REPO / prior).resolve()
        manifest = json.loads((root / 'train-sources.json').read_text())
        for source in manifest['sources']:
            path = Path(source['path'])
            path = path if path.is_absolute() else root / path
            if sha256_file_hex(path) != source['sha256']:
                raise ValueError(f'prior mixture source changed: {path}')
            sources.append({**source, 'path': str(path)})
            if source['kind'] != 'native-graphs':
                continue
            for _, raw in iter_lines(path):
                row = json.loads(raw)
                conversion = row.get('conversion') or {}
                teacher = conversion.get('teacher_source')
                decisions.update([f'{teacher.get("source_row_id")}#{teacher.get("teacher_turn_index")}'] if teacher
                                 else conversion.get('source_record_ids') or [])
                ids.add(row['id'])
                keys.add(content_key(row))
                trained.update(row.get('source_groups') or [])
        held_manifest = json.loads((root / 'held-sources.json').read_text())
        for source in held_manifest['sources']:
            for _, raw in iter_lines(root / source['path']):
                groups.update(json.loads(raw).get('source_groups') or [])
    return sources, ids, decisions, keys, groups, trained


def candidates(config, policy, decisions, trajectories, held, carve_out=None, carve_set=frozenset(), prior=None):
    """Eligible samples with features, tier and category; omissions by reason. Rows of carved held groups go to
    ``carve_out`` (held, up to the declared row cap; beyond it they are dropped, never trained)."""
    out, omitted, covered, seen = [], collections.Counter(), set(), set()
    if prior is not None:
        covered |= prior[2]
        seen |= prior[3]
    carve = policy.get('held_carve')
    feature_cache = {}
    regrade = OracleRegradeHold(policy.get('oracle_regrade_hold'))
    inputs = [('render', view, path) for render in config['renders'] for view, path in graph_files(render, 'train')]
    inputs += [('teacher', 'teacher-turn', REPO / p) for p in config.get('teacher_graphs', [])]
    for kind, view, path in inputs:
        for number, raw in iter_lines(path):
            row = json.loads(raw)
            if row.get('split') != 'train':
                omitted['not train split'] += 1
                continue
            if set(row.get('source_groups') or ()) & held:
                omitted['shares a held/dev source group'] += 1
                continue
            if set(row.get('source_groups') or ()) & carve_set:
                if carve_out is not None and len(carve_out) < carve['max_rows']:
                    carve_out.append((str(path), number))
                    omitted['carved into the permanent held slice'] += 1
                else:
                    omitted['carved held group beyond the held row cap (not trained)'] += 1
                continue
            conversion = row.get('conversion') or {}
            if kind == 'teacher':
                source = conversion.get('teacher_source') or {}
                ids = [f'{source.get("source_row_id")}#{source.get("teacher_turn_index")}']
            else:
                ids = list(conversion.get('source_record_ids') or [])
            known = [decisions.get(d) for d in ids]
            if not ids or any(k is None for k in known):
                omitted[f'{view}: decision without an idiom score'] += 1
                continue
            if not all(k['positive'] for k in known):
                omitted[f'{view}: holds a non-positive decision (unverified, un-awaited nl, failed action, hold)'] += 1
                continue
            held_back = regrade.reason(path, known, ids)
            if held_back:
                omitted[held_back] += 1
                continue
            if any(d in covered for d in ids):
                omitted[f'{view}: decision already taken by another sample'] += 1
                continue
            key = content_key(row)
            if key in seen:
                omitted['exact duplicate graph'] += 1
                continue
            trajectory = known[0]['trajectory']
            if trajectory not in feature_cache:
                feature_cache[trajectory] = trajectory_features(trajectories[trajectory], decisions)
            features = derive({**feature_cache[trajectory], 'neuralese_edges': graph_edges(row)})
            # Instructions repeat across the worlds of one curriculum template; a near-duplicate is the same task
            # text over the same world (the row's first source group).
            features['near_duplicate_cluster'] = (f'{features["near_duplicate_cluster"]}|'
                                                  f'{min(row.get("source_groups") or [""])}')
            tier, reason = classify(features, policy)
            coding = policy.get('coding_mix') or {}
            if tier is None and coding and any(m in str(path) for m in coding.get('sources', [])):
                # Generic coding rows (no natlang structure) of the declared coding sources: the bounded coding mix.
                if features['verified']:
                    tier, reason = coding['tier'], None
                else:
                    reason = 'coding mix: unverified'
            if tier is None:
                omitted['dropped: ' + reason] += 1
                continue
            seen.add(key)
            covered.update(ids)
            category = min((k['category'] for k in known), key=ORDER.index)
            fit = sum(k['fit'] for k in known) / len(known)
            target_tokens = sum(sum(1 for w in node.get('target_weights') or () if w) for node in row['nodes'])
            out.append({'tier': tier, 'category': category, 'fit': fit, 'id': row['id'], 'file': str(path),
                        'target_tokens': target_tokens,
                        'trajectory': trajectory,
                        'line': number, 'decisions': len(ids), 'view': view, 'features': features})
    return out, omitted


def coding_sample(cands, policy, dropped):
    """The declared coding mix (policy ``coding_mix``): verified generic coding rows, one per near-duplicate cluster,
    round-robin over families (repositories/tasks) until their target tokens reach ``share`` of the mixture's."""
    coding = policy.get('coding_mix')
    if not coding:
        return cands
    tier = coding['tier']
    natlang = [c for c in cands if c['tier'] != tier]
    pool = sorted((c for c in cands if c['tier'] == tier), key=lambda c: hashlib.sha256(c['id'].encode()).hexdigest())
    budget = coding['share'] / (1 - coding['share']) * sum(c['target_tokens'] for c in natlang)
    clusters, families = set(), collections.defaultdict(list)
    for c in pool:
        key = c['features']['near_duplicate_cluster']
        if key in clusters:
            dropped['coding mix: near-duplicate'] += 1
            continue
        clusters.add(key)
        families[c['features']['family']].append(c)
    chosen, used, queues = [], 0, [list(v) for v in families.values()]
    while used < budget and any(queues):
        for q in queues:
            if q and used < budget:
                c = q.pop(0)
                chosen.append(c)
                used += c['target_tokens']
    dropped['coding mix: beyond the declared share'] += sum(len(q) for q in queues)
    return natlang + chosen


def apply_caps(cands, policy):
    """Per tier: near-duplicate cluster cap, then family share caps (reducer families have their own). A policy
    without ``caps`` (v2, owner 2026-10-28: tiers order and weight, they do not filter) keeps every eligible row;
    the coding mix is sampled to its declared share."""
    dropped = collections.Counter()
    if not policy.get('caps'):
        return coding_sample(cands, policy, dropped), dropped
    caps, kept = policy['caps'], []
    for tier in sorted({c['tier'] for c in cands}):
        rows = sorted((c for c in cands if c['tier'] == tier),
                      key=lambda c: (-c['fit'], hashlib.sha256(c['id'].encode()).hexdigest()))
        # The cluster cap counts source trajectories (tasks): the chains and decisions of one admitted trajectory
        # are different invocations of that task, not near-duplicates of each other.
        clusters, survivors = collections.defaultdict(set), []
        for c in rows:
            members = clusters[c['features']['near_duplicate_cluster']]
            if c['trajectory'] not in members and len(members) >= caps['near_duplicate_per_tier']:
                dropped[f'tier {tier}: near-duplicate cluster cap'] += 1
                continue
            members.add(c['trajectory'])
            survivors.append(c)
        total = len(survivors)
        families = collections.Counter()
        for c in survivors:
            share = caps['reducer_family_share_per_tier'] if c['features']['reducer'] else caps['family_share_per_tier']
            limit = max(caps['family_cap_floor_rows'], math.ceil(share * total))
            if families[c['features']['family']] >= limit:
                dropped[f'tier {tier}: {"reducer " if c["features"]["reducer"] else ""}family cap'] += 1
                continue
            families[c['features']['family']] += 1
            kept.append(c)
    # Category caps per tier (the idiom split is a cap, not a target).
    shares = policy['category_caps']
    capped = []
    for tier in sorted({c['tier'] for c in kept}):
        rows = sorted((c for c in kept if c['tier'] == tier),
                      key=lambda c: (-c['fit'], hashlib.sha256(c['id'].encode()).hexdigest()))
        limit = {c: math.ceil(share / 100. * len(rows)) for c, share in shares.items()}
        taken = collections.Counter()
        for c in rows:
            if taken[c['category']] >= limit[c['category']]:
                dropped[f'tier {tier}: category cap ({c["category"]})'] += 1
                continue
            taken[c['category']] += 1
            capped.append(c)
    kept = capped
    limits = caps.get('tier_row_limits') or {}
    if '4' in limits:
        allowed = sum(1 for c in kept if c['tier'] < 4)
        fourth = sorted((c for c in kept if c['tier'] == 4),
                        key=lambda c: (-c['fit'], hashlib.sha256(c['id'].encode()).hexdigest()))
        overflow = {c['id'] for c in fourth[allowed:]}
        dropped['tier 4: row limit (rows of tiers 1-3)'] += len(overflow)
        kept = [c for c in kept if c['id'] not in overflow]
    return kept, dropped


# ------------------------------------------------------------------------------------------------------ objective

def objective(config, policy):
    from transformers import AutoTokenizer
    from natlang_neuralese.train.own_output_supervision import resolve, role_tokens_from_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['objective_tokenizer'], local_files_only=True,
                                              trust_remote_code=False)
    template = Path(config['objective_tokenizer']) / 'chat_template.jinja'
    return resolve({'name': 'own-outputs/1', 'reasoning_share': policy['objective']['reasoning_share'],
                    'role_tokens': role_tokens_from_tokenizer(
                        tokenizer, tokenizer_fingerprint=config['expanded_tokenizer_fingerprint'],
                        chat_template=template.read_text() if template.exists() else None)})


def own_tokens(row, resolved) -> tuple[int, int, int]:
    from natlang_neuralese.train.own_output_supervision import graph_weights, REPLY, REASONING
    weights, categories = graph_weights(row, resolved)
    reply = sum(1 for w, c in zip(weights, categories) if w > 0 and c == REPLY)
    reasoning = sum(1 for w, c in zip(weights, categories) if w > 0 and c == REASONING)
    return reply, reasoning, sum(len(node['token_ids']) for node in row['nodes'])


def distribution(values) -> dict:
    values = sorted(values)
    if not values:
        return {}
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]
    return {'n': len(values), 'mean': sum(values) / len(values), 'p50': pick(.5), 'p90': pick(.9), 'max': values[-1],
            'histogram': dict(sorted(collections.Counter(min(v, 10) if isinstance(v, int) else round(v, 1)
                                                         for v in values).items()))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    config = json.loads(a.config.read_text())
    policy_path = REPO / config['policy']
    policy = json.loads(policy_path.read_text())
    out = a.out
    if (out / 'mixture.json').exists():
        raise SystemExit(f'{out} already holds a mixture; write a new one')

    decisions, trajectories, score_census = load_scores(config['scores'])
    held, packet_sources = held_groups_of(config)
    held_rows = [path for render in config['renders'] for _, path in graph_files(render, 'test')]
    held_rows += [REPO / item for item in config.get('teacher_graphs_test', [])]
    for path in held_rows:
        for _, raw in iter_lines(path):
            held.update(json.loads(raw).get('source_groups') or [])

    prior = prior_versions(config)
    carve_out, carve_set = [], carve_groups(config, policy, prior[5])
    held |= prior[4]
    cands, omitted = candidates(config, policy, decisions, trajectories, held, carve_out, carve_set, prior)
    chosen, cap_drops = apply_caps(cands, policy)
    resolved = objective(config, policy)

    # Cohort files per (tier, category): tiers weighted by rows * tier weight; categories inside a tier by rows.
    natlang_share = 1 - policy['communication_share'] / 100. if config.get('communication') else 1.
    weights = {int(k): v for k, v in policy['tier_weights'].items()}
    cells = collections.Counter((c['tier'], c['category']) for c in chosen)
    tier_rows = collections.Counter(c['tier'] for c in chosen)
    denominator = sum(n * weights[t] for t, n in tier_rows.items())
    cell_files = {}
    for (tier, category), n in cells.items():
        mass = natlang_share * tier_rows[tier] * weights[tier] / denominator * n / tier_rows[tier]
        cell_files[(tier, category)] = max(1, round(mass / policy['cohort_unit']))
    assign = {}
    for (tier, category), files in cell_files.items():
        rows = sorted((c for c in chosen if (c['tier'], c['category']) == (tier, category)),
                      key=lambda c: hashlib.sha256(c['id'].encode()).hexdigest())
        for index, item in enumerate(rows):
            assign[(item['file'], item['line'])] = (tier, category, index % files, item)
    writers = {(t, c, k): Writer(out / 'cohorts' / f't{t}-{c}-{k:02d}.jsonl.zst')
               for (t, c), n in cell_files.items() for k in range(n)}
    feature_writer = Writer(out / 'sample-features.jsonl.zst')
    by_file = collections.defaultdict(dict)
    for (path, number), value in assign.items():
        by_file[path][number] = value
    composition = collections.defaultdict(collections.Counter)
    tier_families = collections.defaultdict(collections.Counter)
    dists = collections.defaultdict(lambda: collections.defaultdict(list))
    for path, lines in by_file.items():
        for number, raw in iter_lines(Path(path)):
            if number not in lines:
                continue
            tier, category, cohort, item = lines[number]
            row = json.loads(raw)
            reply, reasoning, processed = own_tokens(row, resolved)
            if reply + reasoning == 0:  # nothing the objective would train: refused, counted
                cap_drops['refused: zero own-output target tokens'] += 1
                continue
            writers[(tier, category, cohort)].write(raw)
            f = item['features']
            for key in (f'tier{tier}', f'tier{tier}:{category}', f'category:{category}', 'all'):
                composition[key].update(rows=1, decisions=item['decisions'], reply_tokens=reply,
                                        reasoning_tokens=reasoning, supervised_tokens=reply + reasoning,
                                        processed_tokens=processed, reducer_rows=int(f['reducer']),
                                        reducer_supervised_tokens=(reply + reasoning) * int(f['reducer']))
            tier_families[tier][f['family']] += 1
            for key in ('max_depth', 'max_branching', 'distinct_functions', 'neuralese_edges', 'invocations',
                        'text_neuralese_markers'):
                dists[f'tier{tier}'][key].append(f[key])
            feature_writer.write(json.dumps({'id': item['id'], 'tier': tier, 'category': category,
                                             'cohort': f't{tier}-{category}-{cohort:02d}', 'view': item['view'],
                                             'supervised_tokens': reply + reasoning, 'processed_tokens': processed,
                                             'features': f}, separators=(',', ':')).encode())
    outputs = {key: w.close() for key, w in writers.items()}
    features_output = feature_writer.close()

    # Communication graphs: every train row (Neuralese edges => tier 1 by the policy), round-robin into cohorts.
    comm_outputs, comm_kind = [], None
    if config.get('communication'):
        files = max(1, round(policy['communication_share'] / 100. / policy['cohort_unit']))
        comm_writers = [Writer(out / 'cohorts' / f't1-communication-{k:02d}.jsonl.zst') for k in range(files)]
        kinds, index = set(), 0
        for item in config['communication']:
            kinds.add(item['kind'])
            for _, raw in iter_lines(REPO / item['path']):
                row = json.loads(raw)
                if row.get('split') != 'train':
                    composition['communication'].update(not_train=1)
                    continue
                graphs = [v['graph'] for v in row['models'].values()] if 'models' in row else [row]
                if set(row.get('source_groups') or graphs[0].get('source_groups') or ()) & held:
                    composition['communication'].update(dropped_held_group=1)
                    continue
                comm_writers[index % files].write(raw)
                index += 1
                for graph in graphs:
                    reply, reasoning, processed = own_tokens(graph, resolved)
                    edges = graph_edges(graph)
                    for key in ('communication', 'tier1', 'all'):
                        composition[key].update(rows=1, graphs=1, reply_tokens=reply, reasoning_tokens=reasoning,
                                                supervised_tokens=reply + reasoning, processed_tokens=processed,
                                                nodes=len(graph['nodes']), neuralese_edges=edges)
                    dists['communication']['neuralese_edges'].append(edges)
                    dists['communication']['nodes'].append(len(graph['nodes']))
        if len(kinds) != 1:
            raise ValueError('communication inputs mix native-graphs and semantic-pairs; give them separately')
        comm_kind = kinds.pop()
        comm_outputs = [w.close() for w in comm_writers]

    held_writer, held_seen = Writer(out / 'held' / 'native-graphs.jsonl.zst'), set()
    for path in held_rows:
        for _, raw in iter_lines(path):
            row = json.loads(raw)
            key = content_key(row)
            if row.get('split') == 'test' and key not in held_seen:
                held_seen.add(key)
                held_writer.write(raw)
    carve_lines = collections.defaultdict(set)
    for path, number in carve_out:
        carve_lines[path].add(number)
    for path, numbers in carve_lines.items():
        for number, raw in iter_lines(Path(path)):
            if number in numbers:
                row = json.loads(raw)
                row['split'] = 'test'  # the permanent held slice (policy held_carve); its groups never train
                row.setdefault('conversion', {})['held_carve'] = {'policy': policy['id'],
                                                                  'rule': 'sha256(salt+source_group) < fraction'}
                key = content_key(row)
                if key not in held_seen:
                    held_seen.add(key)
                    held_writer.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode())
    held_output = held_writer.close()

    priority = {int(k): v for k, v in policy['tier_training_priority'].items()}
    ordered = sorted(outputs.items(), key=lambda kv: (kv[0][0], ORDER.index(kv[0][1]), kv[0][2]))
    train_sources = [{'kind': 'native-graphs', 'path': str(Path(o['path']).relative_to(out)), 'sha256': o['sha256'],
                      **({'training_priority': priority[t]} if priority[t] else {})}
                     for (t, _, _), o in ordered if o['rows']]
    comm_sources = [{'kind': comm_kind, 'path': str(Path(o['path']).relative_to(out)), 'sha256': o['sha256'],
                     **({'training_priority': priority[1]} if priority[1] else {})} for o in comm_outputs if o['rows']]
    train_sources = comm_sources + train_sources
    # Appends keep every earlier version's train source first and unchanged (absolute paths).
    train_sources = [*prior[0], *train_sources]
    (out / 'train-sources.json').write_text(json.dumps({'schema': 'natlang.native-graph-sources/1',
                                                        'sources': train_sources}, indent=1) + '\n')
    (out / 'held-sources.json').write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': 'held/native-graphs.jsonl.zst', 'sha256': held_output['sha256']}]},
        indent=1) + '\n')

    cohort_counts = collections.Counter()
    for (t, c, _), o in outputs.items():
        if o['rows']:
            cohort_counts[f't{t}-{c}'] += 1
    cohort_counts['t1-communication'] = sum(1 for o in comm_outputs if o['rows'])
    total_files = sum(cohort_counts.values())
    report = {
        'schema': SCHEMA, 'config': {'path': str(a.config), 'sha256': sha256_file_hex(a.config), 'value': config},
        'policy': {'path': config['policy'], 'sha256': sha256_file_hex(policy_path), 'id': policy['id']},
        'code': {name: sha256_file_hex(REPO / name) for name in (
            'scripts/neuralese_data/build_idiom_mixture.py', 'scripts/neuralese_data/idiom_scores.py')},
        'cohort_weights': {k: v / total_files for k, v in sorted(cohort_counts.items()) if v},
        'cohort_files': dict(sorted(cohort_counts.items())),
        'composition': {k: dict(v) for k, v in sorted(composition.items())},
        'distributions': {t: {k: distribution(v) for k, v in d.items()} for t, d in sorted(dists.items())},
        'top_families_per_tier': {f'tier{t}': dict(c.most_common(12)) for t, c in sorted(tier_families.items())},
        'score_census': dict(score_census), 'selection_omissions': dict(omitted), 'cap_drops': dict(cap_drops),
        'eligible_samples_before_caps': dict(collections.Counter(f'tier{c["tier"]}' for c in cands)),
        'held': {'rows': held_output['rows'], 'held_source_groups': len(held), 'packets': packet_sources},
        'train_sources': {'path': 'train-sources.json', 'sha256': sha256_file_hex(out / 'train-sources.json')},
        'held_sources': {'path': 'held-sources.json', 'sha256': sha256_file_hex(out / 'held-sources.json')},
        'sample_features': features_output,
        'outputs': [*outputs.values(), *comm_outputs, held_output],
    }
    (out / 'mixture.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    print(json.dumps({k: report[k] for k in ('cohort_files', 'composition', 'cap_drops')}, indent=1))


if __name__ == '__main__':
    main()
