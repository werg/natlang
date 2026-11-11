#!/usr/bin/env python3
"""Statically withdraw the unearned rows of named logical families from the live joint-SFT source.

Owner order 2026-11-10: excise scripted template data (``port-record-wrapper`` and ``native-communication-template``).
Appends cannot remove rows (``native_resident_ce.extension_policy`` rejects a smaller family map), so this is the
static source correction of ``corrected-mix-diverse-1103`` again, generalised to the source the lineage trains now:

- ``audit`` decides what is useless (owner clarification 2026-11-11: withdraw only what is truly useless, never a
  family blindly): per-row criteria over the raw rows, written to ``withdrawal-plan.json`` with decoded samples of both
  sides and a visit cap (``quota_revision``) for useful but over-represented rows; ``build --plan`` withdraws exactly
  that plan and refuses an earned row (saved credit stays in the checkpoint frame; an earned row has nothing left);
- each affected leaf is filtered with ``withdraw_native_graph_targets.filter_source`` (all other bytes and order
  preserved; a leaf whose rows all go leaves the manifest), its own index is projected, the manifest tree rewritten;
- the correction proof (``natlang.native-source-correction/1``, ``deferred`` rows: visit debt and earned credit stay in
  the pending ledger, ``apply_source_correction``), the family map (the old rows minus the withdrawn), the CE policy
  (leaves, inventory, family map, ``objective_transition``, source-bound quota basis, ``history_settlement`` over every
  historical first-row observation), the packed-geometry sidecar and capacity descriptor, and a pending-focus
  request that supersedes the active one with the new leaf SHA-256s are written beside it;
- the recipe pair of ``memetic_admission._write_recipes`` (append form for the cut-over window, settled form for the
  windows after it) and ``commit-admission.json``: the ``state['interleave']['memetic_distillation']`` ledger entry
  that makes ``prepare_window`` reuse this recipe for its window without admitting packages in the same window (the
  trainer applies one source change per continuation; queued packages are appended by the next window on top).

Nothing existing is edited and no controller state is touched. ``--cpu-preflight`` (the trainer's
``--continuation-preflight-only --device cpu``) is a separate step: see ``cpu_preflight.py`` written to ``--out``.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
from pathlib import Path

import natlang_neuralese
from natlang_neuralese.common import row_index
from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest, sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.data.native_sources import NativeGraphSource
from scripts.neuralese_data.withdraw_native_graph_targets import continue_filtered_index, filter_source

FAMILIES = ('port-record-wrapper', 'native-communication-template')
REASON = 'owner order 2026-11-10: excise scripted template data'
PACKAGE = Path(natlang_neuralese.__file__).resolve().parent  # the executing package: the index-cache policy hash
STATE_PATCH_SCHEMA = 'natlang.template-excision-commit/1'


def ref(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256_file_hex(path)}


def leaves(tree):
    if tree.kind == 'manifest':
        for child in tree.children:
            yield from leaves(child)
    else:
        yield tree


def planned_withdrawal(plan, frame, family_of, source):
    """The plan's rows as {id: family}: every one an active, unearned row of the live source (saved credit below quota)."""
    credits, quotas = frame.get('exposure_counts') or {}, frame.get('required_visits') or {}
    if plan.get('schema') != 'natlang.template-excision-plan/1' or plan['source']['sha256'] != source.file_sha256:
        raise ValueError('the withdrawal plan does not belong to the live source')
    active = {e['id'] for e in source.entries}
    chosen = {}
    for ident, item in plan['withdraw'].items():
        if ident not in active or family_of[ident] != item['family'] or item['family'] not in plan['families']:
            raise ValueError('planned withdrawal is not an active row of its family: ' + ident)
        if int(credits.get(ident, 0)) >= int(quotas.get(ident, 1)):
            raise ValueError('planned withdrawal is an earned row: ' + ident)
        chosen[ident] = item['family']
    earned_kept = collections.Counter(family_of[i] for i in active if family_of[i] in plan['families']
                                      and int(credits.get(i, 0)) >= int(quotas.get(i, 1)))
    return chosen, dict(earned_kept)


def filter_leaf_files(source, withdrawn, identity, out):
    """Filter every leaf that holds withdrawn rows. Returns ({old leaf sha: new output ref | None}, proofs)."""
    policy_hash = digest({name: sha256_file_hex(PACKAGE / name) for name in (
        'data/native_sources.py', 'data/native_graph_schema.py', 'data/decision_graph.py', 'model/native_graph_plan.py',
        'data/native_pair_schema.py', 'common/row_index.py', 'common/jsonio.py', 'common/hashing.py')})
    by_leaf, leaf_by_sha, leaf_of = collections.defaultdict(list), {}, {}
    for leaf in leaves(source):
        leaf_by_sha[leaf.file_sha256] = leaf
        for entry in leaf.entries:
            leaf_of[entry['id']] = leaf.file_sha256
            if entry['id'] in withdrawn:
                by_leaf[leaf.file_sha256].append({**entry, 'source_sha256': leaf.file_sha256})
    replacements, proofs, removed_leaves = {}, [], []
    (out / 'leaves').mkdir(parents=True, exist_ok=True)
    (out / 'filters').mkdir(parents=True, exist_ok=True)
    for sha, rows in sorted(by_leaf.items()):
        leaf = leaf_by_sha[sha]
        if len(rows) == len(leaf.entries):
            replacements[sha] = None  # the whole leaf goes; its bytes stay untouched on disk
            removed_leaves.append({'path': str(leaf.path), 'sha256': sha, 'rows': len(rows)})
            continue
        key = {'source_sha256': sha, 'tokenizer': identity['tokenizer'], 'split': 'train', 'policy_sha256': policy_hash,
               'kind': 'native-graphs', 'pair_model': None, 'role_ids': list(identity['roles'])}
        index_path = row_index.entry_path('native-graphs', digest(key))
        cached = row_index.load(index_path)
        if not cached or cached['key'] != key or cached['entries'] != leaf.entries:
            raise ValueError('exact current own source index is unavailable for ' + str(leaf.path))
        tag = digest(sorted(x['id'] for x in rows))[:12]
        proof_path = out / 'filters' / f'filter-{sha[:16]}-{tag}.json'
        output = out / 'leaves' / f'filtered-{sha[:16]}-{tag}.jsonl.zst'
        if proof_path.exists():
            proof = json.loads(proof_path.read_text())
            if proof['source_sha256'] != sha or {x['id'] for x in proof['removed']} != {x['id'] for x in rows}:
                raise ValueError('existing filter proof conflicts')
        else:
            proof = filter_source(leaf.path, output, rows, source_sha256=sha)
            write_json_durable(proof_path, proof)
        projected = continue_filtered_index(index_path, proof, index_sha256=sha256_file_hex(index_path),
                                            metadata_sha256=cached['metadata_sha256'])
        replacements[sha] = proof['output']
        proofs.append({'filter': ref(proof_path), 'index': projected, 'removed': len(rows), 'leaf': str(leaf.path)})
        print(json.dumps({'event': 'leaf_filtered', 'leaf': str(leaf.path), 'removed': len(rows),
                          'remaining': proof['new_rows']}), flush=True)
    return replacements, proofs, removed_leaves, leaf_of


def rewrite_manifest(path, replacements, out, depth=0):
    """(rewritten manifest document, changed). Nested manifests are rewritten only where a leaf changed; a manifest whose
    sources all go returns None."""
    doc = json.loads(Path(path).read_text())
    changed = False
    kept = []
    for item in doc['sources']:
        if not Path(item['path']).is_absolute():
            raise ValueError('template excision needs absolute source paths: ' + item['path'])
        if item['sha256'] in replacements:
            changed = True
            if replacements[item['sha256']] is None:
                continue
            item = {**item, **{k: v for k, v in replacements[item['sha256']].items() if k in ('path', 'sha256', 'bytes')}}
        elif Path(item['path']).suffix == '.json':
            nested, nested_changed = rewrite_manifest(item['path'], replacements, out, depth + 1)
            if nested_changed:
                changed = True
                if nested is None:
                    continue
                target = out / 'manifests' / f'nested-{digest(nested)[:16]}.json'
                write_json_durable(target, nested)
                item = {**item, **ref(target), 'bytes': target.stat().st_size}
        kept.append(item)
    if not kept:
        return None, changed
    return {**doc, 'sources': kept}, changed


def stage_view(recipe, stage_id):
    from natlang_neuralese.train.recipe import effective_stage_parameters, load_recipe, resolve_stage_inputs
    recipe = load_recipe(recipe)
    stage = next(item for item in recipe['stages'] if item['id'] == stage_id)
    return stage, resolve_stage_inputs(recipe, stage, {}), effective_stage_parameters(stage, [], recipe)


def effective_recipe(controller_out, config, override=None):
    """The recipe the lineage's next window starts from: the admitted successor's settled form, else the controller's
    (``override``: an explicit recipe, for a package against an older boundary)."""
    state = json.loads((Path(controller_out) / 'state.json').read_text())
    if override:
        return str(Path(override).resolve(strict=True)), state
    recipe = ((state.get('interleave') or {}).get('memetic_distillation') or {}).get('recipe')
    if recipe:
        if sha256_file_hex(recipe['path']) != recipe['sha256']:
            raise ValueError('admitted memetic successor recipe changed')
        return recipe['path'], state
    return config['production_recipe'], state


# ---- audit: which rows of the two families are useless ---------------------------------------------------------------
WRAPPER, TEMPLATE = FAMILIES
NEAR_DUPLICATE_JACCARD = 0.9   # wrapper: exact Jaccard of the 4-gram sets of the masked target (first 6000 tokens)
CONSTANT_PRODUCER_MIN_ROWS = 8  # template: a producer reply repeated verbatim by this many rows is not input-specific
TARGET_WEIGHT = 0.5             # reply/reasoning weight is 1.0; context 0.05 and feedback 0.25 are not targets
SHINGLE, SHINGLE_TOKENS, PERMUTATIONS = 4, 6000, 64


def _shingles(tokens):
    import numpy as np
    t = np.asarray(tokens[:SHINGLE_TOKENS], dtype=np.uint64)
    if len(t) < SHINGLE:
        t = np.concatenate([t, np.zeros(SHINGLE - len(t), dtype=np.uint64)])
    sh = t[:len(t) - SHINGLE + 1] * np.uint64(1000003)
    for j in range(1, SHINGLE):
        sh = sh * np.uint64(1000003) + t[j:len(t) - SHINGLE + 1 + j]
    return np.unique(sh)


def _minhash(shingles):
    import numpy as np
    rng = np.random.default_rng(1)
    a = rng.integers(1, (1 << 61) - 1, PERMUTATIONS, dtype=np.uint64)
    b = rng.integers(0, (1 << 61) - 1, PERMUTATIONS, dtype=np.uint64)
    return np.array([(shingles * a[i] + b[i]).min() for i in range(PERMUTATIONS)], dtype=np.uint64)


def _jaccard(x, y):
    import numpy as np
    inter = len(np.intersect1d(x, y, assume_unique=True))
    return inter / (len(x) + len(y) - inter)


def _target_runs(row):
    """Per node the tokens that carry reply/reasoning loss (weight >= TARGET_WEIGHT)."""
    return [[t for t, w in zip(n['token_ids'], n['target_weights']) if w >= TARGET_WEIGHT] for n in row['nodes']]


def audit(args):
    """Write ``withdrawal-plan.json`` for the named families from the raw rows of the live source."""
    import hashlib
    import random
    import numpy as np
    from natlang_neuralese.common.jsonio import open_jsonl_text
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    from natlang_neuralese.train.native_continuation import collect_sft_progress
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(args.controller_out) / 'config.json').read_text())
    stage_id = config['production_stage']
    base_recipe, state = effective_recipe(args.controller_out, config, args.recipe)
    checkpoint = Path(args.checkpoint or state['checkpoint']).resolve(strict=True)
    _, inputs, parameters = stage_view(base_recipe, stage_id)
    declared = parameters.get('sft_packed_geometry')
    identity0 = json.loads(Path((json.loads(declared) if isinstance(declared, str) else declared)['path']).read_text())['identity']
    source = NativeGraphSource(inputs['native_graphs']['path'], identity0['tokenizer'], required_split='train',
                               role_ids=identity0['roles'])
    policy = json.loads(Path(inputs['ce_supervision_policy']['path']).read_text())
    family_map = json.loads(Path(policy['objective']['family_balance']['map']['path']).read_text())
    family_of = {row['id']: row['family'] for row in family_map['rows']}
    frame = collect_sft_progress(CheckpointPolicy.load(checkpoint)['training'], borrow_frames=True)[stage_id]
    credits, quotas = frame.get('exposure_counts') or {}, frame.get('required_visits') or {}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['base'])

    # pass 1: stream the rows of both families
    info = {}
    for leaf in leaves(source):
        wanted = {e['id'] for e in leaf.entries if family_of[e['id']] in FAMILIES}
        if not wanted:
            continue
        with open_jsonl_text(leaf.path) as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row['id'] not in wanted:
                    continue
                runs = _target_runs(row)
                flat = [t for run in runs for t in run]
                family = family_of[row['id']]
                item = {'family': family, 'n': len(flat), 'text': [tokenizer.decode(run[:300]) for run in runs],
                        'groups': (row.get('source_groups') or [])[:2]}
                if family == WRAPPER:
                    item['tokens'] = np.asarray(flat[:SHINGLE_TOKENS], dtype=np.uint32)
                    item['sig'] = _minhash(_shingles(flat))
                else:
                    if [n.get('name') for n in row['nodes']] != ['producer', 'consumer']:
                        raise ValueError('template edge with unexpected nodes: ' + row['id'])
                    item['producer'] = hashlib.sha256(json.dumps(runs[0]).encode()).hexdigest()
                    item['consumer'] = hashlib.sha256(json.dumps(runs[1]).encode()).hexdigest()
                info[row['id']] = item
        print(json.dumps({'event': 'audit_leaf', 'leaf': Path(leaf.path).name, 'rows': len(info)}), flush=True)

    earned = lambda i: int(credits.get(i, 0)) >= int(quotas.get(i, 1))  # noqa: E731
    decisions = {}  # id -> (criterion, kept representative)
    # wrapper: one row per task already (distinct programs); withdraw only repeats of an already kept row
    wrapper = [i for i, x in info.items() if x['family'] == WRAPPER]
    buckets = collections.defaultdict(list)
    for i in wrapper:
        sig = info[i]['sig']
        for band in range(PERMUTATIONS // 4):
            buckets[(band, bytes(sig[band * 4:(band + 1) * 4]))].append(i)
    parent = {i: i for i in wrapper}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for members in buckets.values():
        if 1 < len(members) <= 60:
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    x, y = members[a], members[b]
                    if find(x) != find(y) and _jaccard(_shingles(info[x]['tokens']), _shingles(info[y]['tokens'])) >= NEAR_DUPLICATE_JACCARD:
                        parent[find(x)] = find(y)
    clusters = collections.defaultdict(list)
    for i in wrapper:
        clusters[find(i)].append(i)
    for members in clusters.values():
        if len(members) > 1:
            # keep the most complete row (an earned row first: it stays anyway), withdraw the rest
            members.sort(key=lambda i: (not earned(i), -info[i]['n'], i))
            for i in members[1:]:
                if not earned(i):
                    decisions[i] = ('wrapper near-duplicate target (exact Jaccard >= %.2f) of a kept row'
                                    % NEAR_DUPLICATE_JACCARD, members[0])
    # template: input-specific producers stay; constant and repeated producer replies go
    template = [i for i, x in info.items() if x['family'] == TEMPLATE]
    producer_rows = collections.Counter(info[i]['producer'] for i in template)
    seen = {}
    for i in sorted(template):
        x = info[i]
        if earned(i):
            continue
        if producer_rows[x['producer']] >= CONSTANT_PRODUCER_MIN_ROWS:
            decisions[i] = ('template producer reply repeated verbatim by >= %d edges (constant, not input-specific)'
                            % CONSTANT_PRODUCER_MIN_ROWS, None)
        elif (x['consumer'], x['producer']) in seen:
            decisions[i] = ('template edge repeating the producer reply of a kept edge into the same consumer answer',
                            seen[(x['consumer'], x['producer'])])
        else:
            seen[(x['consumer'], x['producer'])] = i
    # visit cap: a kept template edge re-trains its (shared) consumer answer on every visit
    cap = {i: {'prior_required_visits': int(quotas[i]), 'required_visits': 1} for i in template
           if i not in decisions and int(quotas.get(i, 1)) > 1}
    kept = {f: [i for i, x in info.items() if x['family'] == f and i not in decisions] for f in FAMILIES}
    rs = random.Random(20261111)

    def example(i):
        x = info[i]
        return {'id': i, 'groups': x['groups'], 'target_tokens': x['n'],
                'decoded_targets_by_node': [t[:360] for t in x['text']],
                **({'criterion': decisions[i][0], 'kept_representative': decisions[i][1]} if i in decisions else {})}

    def withdrawn_of(f):
        return sorted(i for i in decisions if info[i]['family'] == f)
    plan = {'schema': 'natlang.template-excision-plan/1', 'reason': REASON, 'families': list(FAMILIES),
            'criteria': {WRAPPER: 'the 2026-11-03 review kept one row per wrapper task; every row here is its own task '
                                  '(distinct teacher program). Withdraw only a row whose masked target is a near-duplicate '
                                  '(exact Jaccard >= %.2f of 4-gram sets) of a kept row of another task; the most complete '
                                  'row of a cluster stays.' % NEAR_DUPLICATE_JACCARD,
                         TEMPLATE: 'keep edges whose producer reply is input-specific and distinct (they teach genuine '
                                   'Neuralese communication). Withdraw an edge whose producer reply is repeated verbatim '
                                   'by >= %d edges, or repeats the producer reply of a kept edge into the same consumer '
                                   'answer. Kept edges are capped at one visit (was two): the consumer answer shared by the '
                                   'edges of one fan-in is otherwise re-trained twice per edge.' % CONSTANT_PRODUCER_MIN_ROWS},
            'parent_checkpoint': ref(checkpoint),
            'source': {'path': inputs['native_graphs']['path'], 'sha256': inputs['native_graphs']['sha256']},
            'withdraw': {i: {'family': info[i]['family'], 'criterion': d[0], 'kept_representative': d[1]}
                         for i, d in sorted(decisions.items())},
            'visit_cap': cap,
            'counts': {f: {'rows': sum(1 for x in info.values() if x['family'] == f), 'withdrawn': len(withdrawn_of(f)),
                           'kept': len(kept[f]), 'earned_kept': sum(1 for i in kept[f] if earned(i)),
                           'visit_capped': sum(1 for i in cap if info[i]['family'] == f)} for f in FAMILIES},
            'withdrawn_by_criterion': dict(collections.Counter(d[0] for d in decisions.values())),
            'examples': {f: {'withdrawn': [example(i) for i in rs.sample(withdrawn_of(f), min(5, len(withdrawn_of(f))))],
                             'kept': [example(i) for i in rs.sample(kept[f], 5)]} for f in FAMILIES},
            'audit_sample': {f: [example(i) for i in rs.sample([i for i, x in info.items() if x['family'] == f], 60)]
                             for f in FAMILIES},
            'structure': {'wrapper_distinct_programs': len({i.split(':decision')[0] for i in wrapper}),
                          'wrapper_near_duplicate_clusters': sum(1 for m in clusters.values() if len(m) > 1),
                          'template_distinct_producer_replies': len(producer_rows),
                          'template_distinct_consumer_answers': len({info[i]['consumer'] for i in template}),
                          'template_edges_per_consumer_answer': dict(collections.Counter(collections.Counter(
                              info[i]['consumer'] for i in template).values()).most_common(8))}}
    write_json_durable(out / 'withdrawal-plan.json', plan)
    print(json.dumps({'event': 'audit', 'counts': plan['counts'], 'by_criterion': plan['withdrawn_by_criterion'],
                      'structure': plan['structure']}), flush=True)
    return plan


def build(args):
    from natlang_neuralese.rl.memetic_admission import _focus_state, _write_recipes, _ref
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    from natlang_neuralese.train.native_packed_capacity import _rows_digest
    from natlang_neuralese.train.native_pending_focus import FINISHED, SCHEMA as FOCUS_SCHEMA
    from natlang_neuralese.train.native_quota_aware_ce import make_basis, validate_admission
    from natlang_neuralese.train.native_resident_ce import (HISTORY_SETTLEMENT_SCHEMA, _historical_observations,
                                                            _observations_digest)
    from natlang_neuralese.train.native_source_correction import (QUOTA_REVISION_SCHEMA, SCHEMA as CORRECTION_SCHEMA,
                                                                  source_correction_dependencies)
    from natlang_neuralese.train.native_supervision import load_policy
    from natlang_neuralese.train.own_output_supervision import identity, policy_objective, prepare_policy
    from natlang_neuralese.train.recipe import _source_correction_continuation
    from scripts.neuralese_data.build_acquired_streaming_successor import sft_capacity_descriptor

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(args.controller_out) / 'config.json').read_text())
    stage_id = config['production_stage']
    base_recipe, state = effective_recipe(args.controller_out, config, args.recipe)
    checkpoint = Path(args.checkpoint or state['checkpoint']).resolve(strict=True)
    _, inputs, parameters = stage_view(base_recipe, stage_id)
    declared = parameters.get('sft_packed_geometry')
    geometry0_ref = json.loads(declared) if isinstance(declared, str) else declared
    geometry0 = json.loads(Path(geometry0_ref['path']).read_text())
    identity0 = geometry0['identity']
    train = inputs['native_graphs']
    predecessor = {'path': train['path'], 'sha256': train['sha256']}
    source = NativeGraphSource(predecessor['path'], identity0['tokenizer'], required_split='train',
                               role_ids=identity0['roles'])
    if source.file_sha256 != predecessor['sha256'] or _rows_digest(source.entries) != geometry0['rows_sha256']:
        raise ValueError('the stage packed-geometry sidecar does not bind the live source')

    # ---- what the saved frame says
    from natlang_neuralese.train.native_continuation import collect_sft_progress
    full = CheckpointPolicy.load(checkpoint)
    frame = collect_sft_progress(full['training'], borrow_frames=True)[stage_id]
    if frame['source_receipts'][0].get('sha256') != predecessor['sha256']:
        raise ValueError('the checkpoint frame does not train the recipe source; pass the checkpoint the next window continues: %s vs %s' % (frame['source_receipts'][0].get('sha256'), predecessor['sha256']))
    model_id = full['model_config']['architecture'] + ':' + full['model_config']['base_inventory_sha256']
    ce_ref = inputs['ce_supervision_policy']
    current_policy = json.loads(Path(ce_ref['path']).read_text())
    balance = current_policy['objective']['family_balance']
    family_map = json.loads(Path(balance['map']['path']).read_text())
    if family_map['source_sha256'] != predecessor['sha256']:
        raise ValueError('the CE family map does not bind the live source')
    family_of = {row['id']: row['family'] for row in family_map['rows']}
    if [(r['id'], r['binding']) for r in family_map['rows']] != [(e['id'], e['binding']) for e in source.entries]:
        raise ValueError('family map differs from the live source')
    plan = json.loads(Path(args.plan).read_text())
    withdrawn, earned_kept = planned_withdrawal(plan, frame, family_of, source)
    families = tuple(plan['families'])
    by_family = collections.Counter(withdrawn.values())
    print(json.dumps({'event': 'withdrawal_set', 'withdrawn': dict(by_family), 'earned_kept': earned_kept,
                      'source_rows': len(source.entries)}), flush=True)

    # ---- filtered leaves and the successor manifest
    replacements, proofs, removed_leaves, leaf_of = filter_leaf_files(source, withdrawn, identity0, out)
    (out / 'manifests').mkdir(parents=True, exist_ok=True)
    manifest, _ = rewrite_manifest(predecessor['path'], replacements, out)
    manifest['template_excision'] = {'families': list(families), 'withdrawn_graphs': dict(by_family),
                                     'earned_rows_kept': earned_kept, 'reason': REASON,
                                     'plan': ref(args.plan), 'scope': 'Exact complete TRAIN graphs of the named families whose saved credit is '
                                              'below their quota; earned exposure stays in the checkpoint frame; all '
                                              'other bytes and order preserved.'}
    manifest_path = out / 'train-sources.json'
    write_json_durable(manifest_path, manifest)
    candidate = NativeGraphSource(manifest_path, identity0['tokenizer'], required_split='train',
                                  role_ids=identity0['roles'])
    retained = [(e['id'], e['binding']) for e in source.entries if e['id'] not in withdrawn]
    if [(e['id'], e['binding']) for e in candidate.entries] != retained:
        raise ValueError('successor changed the retained row identity/order')
    new_leaf = {entry['id']: leaf.file_sha256 for leaf in leaves(candidate) for entry in leaf.entries}
    successor = ref(manifest_path)

    # ---- family map, correction ledger and proof
    new_map = {**{k: v for k, v in family_map.items() if k not in ('schema', 'source_sha256', 'rows')},
               'schema': family_map['schema'], 'source_sha256': candidate.file_sha256,
               'rows': [row for row in family_map['rows'] if row['id'] not in withdrawn]}
    new_map['template_excision'] = {'withdrawn': dict(by_family), 'earned_rows_kept': earned_kept}
    write_json_durable(out / 'families.json', new_map)
    ledger = [{'id': e['id'], 'disposition': 'deferred', 'old_binding': e['binding'], 'new_binding': None,
               'old_source_sha256': leaf_of[e['id']], 'new_source_sha256': None,
               'old_communication': bool(e.get('communication')), 'reason': REASON + ': ' + plan['withdraw'][e['id']]['criterion']}
              for e in source.entries if e['id'] in withdrawn]
    (out / 'correction-rows.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in ledger))
    capped = []
    quotas = frame.get('required_visits') or {}
    for ident, item in sorted(plan['visit_cap'].items()):
        if ident in withdrawn or ident not in source.bindings or int(quotas.get(ident, -1)) != item['prior_required_visits']:
            raise ValueError('visit cap names a row that is withdrawn, inactive or has another saved quota: ' + ident)
        capped.append({'id': ident, 'binding': source.bindings[ident], 'prior_required_visits': item['prior_required_visits'],
                       'required_visits': item['required_visits']})
    if capped:
        (out / 'quota-revision.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in capped))
    proof = {'schema': CORRECTION_SCHEMA, 'model_id': model_id, 'stage_id': stage_id, 'prior_source': predecessor,
             'current_source': successor, 'row_bindings_path': str(out / 'correction-rows.jsonl'),
             'row_bindings_sha256': sha256_file_hex(out / 'correction-rows.jsonl'),
             'evidence': [ref(out / 'families.json'), ref(args.plan)], 'reason': REASON,
             'withdrawal': {'families': list(families), 'rows': dict(by_family), 'earned_rows_kept': earned_kept,
                            'selection': 'rows of the families with saved credit below saved quota in the parent frame',
                            'parent_checkpoint': ref(checkpoint)}}
    proof_path = out / 'source-correction.json'
    if capped:
        proof['quota_revision'] = {'schema': QUOTA_REVISION_SCHEMA, 'row_bindings_path': str(out / 'quota-revision.jsonl'),
                                   'row_bindings_sha256': sha256_file_hex(out / 'quota-revision.jsonl'),
                                   'reason': REASON + ': kept edges are over-represented, one visit instead of two '
                                             '(earned credit retained)'}
    write_json_durable(proof_path, proof)
    source_correction_dependencies(ref(proof_path))

    # ---- CE policy
    objective = policy_objective(load_policy(ce_ref))
    policy = copy.deepcopy(current_policy)
    by_old = {old: new for old, new in replacements.items() if new is not None}
    removed = {old for old, new in replacements.items() if new is None}
    for rule in policy['sources']:
        if rule['sha256'] in by_old:
            rule['sha256'] = by_old[rule['sha256']]['sha256']
    contract = policy['source_contract']
    active = []
    for declaration in contract['active_sources']:
        if declaration['sha256'] in removed:
            continue
        if declaration['sha256'] in by_old:
            declaration = {**declaration, **by_old[declaration['sha256']]}
        active.append(declaration)
    contract['active_sources'] = active
    contract['active_source_count'] = len(active)
    contract['inventory'] = [dict(successor)]
    if 'inventory' in policy:
        policy['inventory'] = [dict(successor)]
    policy['sources'].sort(key=lambda r: r['sha256'])
    balance = policy['objective']['family_balance']
    balance['map'] = ref(out / 'families.json')
    policy['objective_transition'] = {'from': identity(objective)['sha256']}
    visits = parameters.get('communication_min_visits')
    if policy.get('quota_aware_basis') is not None or (type(visits) is int and visits > 1):
        policy['quota_aware_basis'] = make_basis(frame, stage_id=stage_id, source_sha256=predecessor['sha256'],
                                                 candidate_sha256=candidate.file_sha256, epochs=parameters.get('epochs'),
                                                 communication_min_visits=visits)
    policy['template_excision'] = {'families': list(families), 'withdrawn': dict(by_family),
                                   'scope': 'rows withdrawn by the source correction; saved credits retained'}
    historical = _historical_observations(policy)
    if historical:
        policy['history_settlement'] = {
            'schema': HISTORY_SETTLEMENT_SCHEMA, 'anchor_source_correction': ref(proof_path),
            'anchor_source_sha256': candidate.file_sha256, 'policy_sha256': ce_ref['sha256'],
            'observations': len(historical), 'observations_sha256': _observations_digest(historical)}
    else:
        policy.pop('history_settlement', None)
    ce_path = out / 'ce-policy.json'
    write_json_durable(ce_path, policy)
    ce = ref(ce_path)
    prepare_policy(load_policy(ce), candidate)
    if policy.get('quota_aware_basis') is not None:
        validate_admission(policy, frame, stage_id=stage_id, candidate_sha256=candidate.file_sha256,
                           epochs=parameters.get('epochs'), communication_min_visits=visits,
                           previous_options=frame.get('options') or {}, policy_binding=ce)

    # ---- packed geometry and capacity descriptor for the exact successor rows
    slots = {e['id']: s for e, s in zip(source.entries, geometry0['kv_slots'], strict=True)}
    geometry = {'schema': geometry0['schema'], 'identity': identity0, 'source': successor,
                'rows_sha256': _rows_digest(candidate.entries), 'kv_slots': [slots[e['id']] for e in candidate.entries]}
    write_json_durable(out / 'packed-geometry.json', geometry)
    geometry_ref = ref(out / 'packed-geometry.json')
    write_json_durable(out / 'sft-capacity-descriptor.json', sft_capacity_descriptor(successor, geometry_ref))
    capacity = {'sft_packed_geometry': geometry_ref, 'memory_qualification': ref(out / 'sft-capacity-descriptor.json')}

    # ---- pending focus: the active request again, minus the withdrawn rows, on the new leaf SHA-256s
    focus_ref, focus_record = None, {'skipped': 'the stage has no active pending focus'}
    reference = inputs.get('pending_focus')
    last = _focus_state(checkpoint) if reference else None
    if (reference and last is not None and last.get('state') not in FINISHED
            and (last.get('request') or {}).get('sha256') == reference['sha256']):
        old = json.loads(Path(reference['path']).read_text())
        rows = []
        for row in old['rows']:
            if row['id'] in withdrawn:
                continue
            if row['id'] not in new_leaf:
                raise ValueError('pending focus names a row that is neither active nor withdrawn: ' + row['id'])
            rows.append({**row, 'source_sha256': new_leaf[row['id']]})
        document = {'schema': FOCUS_SCHEMA, 'rows': rows,
                    'supersedes': {'request_sha256': reference['sha256'],
                                   'reason': REASON + ': the same request on the excised leaves'}}
        write_json_durable(out / 'pending-focus.json', document)
        focus_ref = ref(out / 'pending-focus.json')
        focus_record = {'supersedes': reference['sha256'], 'rows': len(rows),
                        'withdrawn_rows_dropped': len(old['rows']) - len(rows)}

    # ---- recipes: the stage's bindings follow the successor, the proof is bound, the old source continues
    base = json.loads(Path(base_recipe).read_text())
    base_stage = next(item for item in base['stages'] if item['id'] == stage_id)
    name = base_stage['inputs'].get('source_correction') or 'template_excision_source_correction'
    base['input_bindings'][name] = ref(proof_path)
    base_stage['inputs']['source_correction'] = name
    seed = out / 'recipe-base.json'
    write_json_durable(seed, base)
    append, settled = _write_recipes(seed, stage_id, successor, ce, out, parameters=capacity, focus=focus_ref)
    _, append_inputs, _ = stage_view(append, stage_id)
    before = {k: v for k, v in inputs.items() if k not in ('pending_focus', 'source_correction', 'continued_native_graphs')}
    after = {k: v for k, v in append_inputs.items() if k != 'pending_focus'}
    if not _source_correction_continuation({**before, 'native_graphs': inputs['native_graphs']}, after):
        raise ValueError('the template excision is not an admitted source-correction continuation of the stage')

    record = {'schema': 'natlang.template-excision-admission/1', 'stage': stage_id, 'reason': REASON,
              'families': list(families), 'withdrawn_rows': dict(by_family), 'earned_rows_kept': earned_kept,
              'predecessor': predecessor, 'successor': successor, 'source_correction': ref(proof_path),
              'ce_policy': ce, 'pending_focus': focus_record, 'capacity': capacity,
              'recipes': {'append': _ref(append), 'settled': _ref(settled)}, 'parent_checkpoint': ref(checkpoint),
              'removed_leaves': removed_leaves, 'filters': proofs, 'exposure': 'none claimed; earned credits retained'}
    write_json_durable(out / 'admission.json', record)
    patch = {'schema': STATE_PATCH_SCHEMA,
             'ledger_entry': {'recipe': _ref(append), 'record': _ref(out / 'admission.json')},
             'settled_recipe': _ref(settled)}
    write_json_durable(out / 'commit-admission.json', patch)
    receipt = {'schema': 'natlang.template-excision-sources/1', 'predecessor': predecessor, 'source': successor,
               'rows_before': len(source.entries), 'rows_after': len(candidate.entries),
               'withdrawn': dict(by_family), 'visit_capped': len(capped), 'earned_rows_kept': earned_kept,
               'removed_leaves': len(removed_leaves),
               'filtered_leaves': len(proofs), 'family_map': ref(out / 'families.json'),
               'correction_rows': ref(out / 'correction-rows.jsonl'), 'source_correction': ref(proof_path),
               'ce_policy': ce, 'geometry': geometry_ref, 'capacity_descriptor': capacity['memory_qualification'],
               'pending_focus': focus_record, 'recipes': record['recipes'], 'parent_checkpoint': ref(checkpoint),
               'base_recipe': ref(base_recipe), 'training_admission': False, 'optimizer_updates': 0}
    write_json_durable(out / 'receipt.json', receipt)
    print(json.dumps({'event': 'template_excision_prepared', **{k: receipt[k] for k in (
        'rows_before', 'rows_after', 'withdrawn', 'visit_capped', 'earned_rows_kept', 'removed_leaves', 'filtered_leaves')}}), flush=True)
    return receipt


def preflight(args):
    """The trainer's CPU continuation preflight of the cut-over window: the command ``interleave.run_sft_window`` would
    build from the append recipe (``continuation_config``), run with ``--continuation-preflight-only --device cpu`` into
    ``<out>/preflight`` (never the controller or window directories). Needs about 48 GB: run it through the ledger."""
    from natlang_neuralese.rl.native_supervised import _resume_source_arguments, continuation_config
    from natlang_neuralese.rl.production import expand_command
    from natlang_neuralese.train.native_state_sft import main as trainer_main
    from natlang_neuralese.train.recipe import stage_parameter_args
    out = Path(args.out).resolve()
    receipt = json.loads((out / 'receipt.json').read_text())
    config = json.loads((Path(args.controller_out) / 'config.json').read_text())
    stage, _, _ = stage_view(receipt['recipes']['append']['path'], config['production_stage'])
    directory = out / 'preflight'
    (directory / 'resident-data').mkdir(parents=True, exist_ok=True)
    source = {k: v for k, v in config.items() if k != 'interleaved_rl_stage'}
    source['resident_data_directory'] = str(directory / 'resident-data')
    source['production_recipe'] = receipt['recipes']['append']['path']
    checkpoint = Path(args.checkpoint or receipt['parent_checkpoint']['path']).resolve(strict=True)
    window, _ = continuation_config(source, stage, checkpoint=str(checkpoint), out=str(directory))
    window.pop('rolling_sft_resume', None)
    write_json_durable(directory / 'config.json', window)
    parameters = window['trainer_parameters']
    learner = directory / 'learner'
    values = {**{k: v for k, v in window.items() if isinstance(v, (str, int, float, list))}, 'checkpoint': str(checkpoint),
              'learner_out': str(learner), 'trainer_args': stage_parameter_args(parameters),
              'seed_trainer_args': stage_parameter_args(parameters)}
    command, reused = _resume_source_arguments(window, learner, values)
    argv = expand_command(command, values)
    start = argv.index('natlang_neuralese.train.native_state_sft') + 1
    trainer = argv[start:] + ['--device', 'cpu', '--continuation-preflight-only']
    trainer[trainer.index('--out') + 1] = str(learner)
    learner.mkdir(parents=True, exist_ok=True)
    write_json_durable(directory / 'trainer-argv.json', {'reused': reused, 'argv': trainer})
    return trainer_main(trainer)


def commit_admission(state_path, patch_path, window, *, write):
    """The controller-state edit that cuts over at ``window``'s boundary (controller stopped; ``write`` False: dry run)."""
    state = json.loads(Path(state_path).read_text())
    patch = json.loads(Path(patch_path).read_text())
    interleave = state['interleave']
    if interleave.get('active_window') is not None or interleave['sft_windows'] != window:
        raise ValueError('the controller is not at the boundary before window %d' % window)
    ledger = interleave.get('memetic_distillation') or {'schema': 'natlang.memetic-distillation-admission/1',
                                                         'recipe': None, 'packages': {}, 'rows': {}, 'admissions': []}
    if ledger['admissions'] and ledger['admissions'][-1]['window'] == window:
        raise ValueError('window %d already has a committed admission' % window)
    ledger['admissions'].append({'window': window, **patch['ledger_entry']})
    ledger['recipe'] = patch['settled_recipe']
    interleave['memetic_distillation'] = ledger
    if write:
        write_json_durable(state_path, state)
    return ledger['admissions'][-1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit', help='decide the useless rows: writes withdrawal-plan.json')
    a.add_argument('--controller-out', required=True)
    a.add_argument('--checkpoint')
    a.add_argument('--recipe', help='explicit base recipe (default: the controller\'s effective recipe)')
    a.add_argument('--out', required=True)
    b = sub.add_parser('build', help='build the corrected source package from an audit plan')
    b.add_argument('--controller-out', required=True, help='the RL/SFT controller directory (config.json, state.json)')
    b.add_argument('--checkpoint', help='the parent checkpoint the next window continues (default: the controller head)')
    b.add_argument('--out', required=True, help='a new output directory')
    b.add_argument('--recipe', help='explicit base recipe (default: the controller\'s effective recipe)')
    b.add_argument('--plan', required=True, help='withdrawal-plan.json of the audit')
    p = sub.add_parser('preflight', help='CPU continuation preflight of the cut-over window (about 48 GB)')
    p.add_argument('--controller-out', required=True)
    p.add_argument('--checkpoint', help='default: the checkpoint the package was built against')
    p.add_argument('--out', required=True, help='the build package directory')
    c = sub.add_parser('commit-admission', help='record the cut-over in the (stopped) controller state')
    c.add_argument('--state', required=True)
    c.add_argument('--patch', required=True, help='commit-admission.json of the build')
    c.add_argument('--window', type=int, required=True, help='the SFT window index that runs on the excised source')
    c.add_argument('--write', action='store_true', help='without it only validate and print the entry')
    args = ap.parse_args(argv)
    if args.command == 'audit':
        audit(args)
    elif args.command == 'build':
        build(args)
    elif args.command == 'preflight':
        return preflight(args)
    else:
        entry = commit_admission(args.state, args.patch, args.window, write=args.write)
        print(json.dumps({'committed' if args.write else 'dry-run': entry}, indent=1))


if __name__ == '__main__':
    sys.exit(main())
