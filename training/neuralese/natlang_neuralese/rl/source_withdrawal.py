"""Queued withdrawal of rows from the live joint-SFT source, applied at the next interleaved SFT window boundary.

A withdrawal removes rows from the training SCHEDULE only: the originals stay on disk untouched, each affected leaf is
re-written without the rows (``withdraw_native_graph_targets.filter_source``: every other byte and the order preserved), and
the trainer is told through a proof-bound source correction (``natlang.native-source-correction/1``: ``deferred`` rows keep
their visit debt and credit in the pending ledger, ``added`` rows are the packages appended in the same change). One source
change per continuation: ``memetic_admission.prepare_window`` builds the withdrawal against the window's actual parent
checkpoint and appends the queued distillation packages on top of the filtered source in the same correction.

Producer side (checkpoint independent): a plan ``natlang.source-withdrawal-plan/1``

    {"schema", "reason", "withdraw": {row id: {"family", "criterion", optional "binding"}},
     "visit_cap": {row id: {"required_visits": int >= 1}}, "evidence": [{"path", "sha256"}], ...free-form audit fields}

is queued with ``queue_plan`` (``scripts/neuralese_data/queue_source_withdrawal.py``; ``prepare_template_excision.py audit
--queue``) as one line of ``<controller out>/memetic/distill/withdrawal-queue.jsonl``. This is a file of its own, never the
package queue: a controller that does not know this module never reads it, so an entry simply waits for the first boundary
of a controller running this code (no restart ordering); the package queue's refuse-unknown-family rule is not involved.

Consumer side, at the boundary (the parent checkpoint's saved frame decides):
- a row of an earned frame (saved credit at or above its quota) is never withdrawn, it has nothing left to withdraw: it is
  skipped and counted (``earned_kept``); rows no longer active, or whose binding differs from the plan's, are skipped too;
- a row whose family differs from the CE family map refuses the whole plan (it describes another map): status ``refused``,
  never retried; a malformed plan is refused the same way; a plan with nothing left to withdraw is ``empty`` (consumed);
- a visit cap lowers the saved quota of an unchanged kept row only when the saved quota is higher (earned credit is kept);
  the trainer needs one deferred or corrected row per correction, so caps ride on a plan that withdraws at least one row;
- any other failure (index cache missing, trainer-side checks of the build) raises: the caller records it in
  ``interleave.memetic_distillation_failures``, the packages are appended without the withdrawal in that window and the plan
  stays pending for the next boundary.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex as digest, sha256_file_hex, sha256_file_hex_cached
from ..common.jsonio import write_json_durable

PLAN_SCHEMA = 'natlang.source-withdrawal-plan/1'
LEDGER_SCHEMA = 'natlang.source-withdrawal-admission/1'
SETTLED = ('applied', 'empty', 'refused')


class PlanRefused(ValueError):
    """The plan can never apply to this lineage (malformed, or about another family map): recorded, never retried."""


def queue_path(controller_out):
    return Path(controller_out) / 'memetic' / 'distill' / 'withdrawal-queue.jsonl'


def _ref(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256_file_hex_cached(path)}


def _lines(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def validate_plan(plan):
    if not isinstance(plan, dict) or plan.get('schema') != PLAN_SCHEMA:
        raise PlanRefused('withdrawal plan schema is not ' + PLAN_SCHEMA)
    if not isinstance(plan.get('reason'), str) or not plan['reason'].strip():
        raise PlanRefused('withdrawal plan needs its reason')
    withdraw = plan.get('withdraw')
    if not isinstance(withdraw, dict) or not withdraw:
        raise PlanRefused('withdrawal plan withdraws at least one row')
    for ident, item in withdraw.items():
        if (not isinstance(item, dict) or not isinstance(item.get('family'), str) or not item['family']
                or not isinstance(item.get('criterion'), str) or not item['criterion']
                or item.get('binding') is not None and not isinstance(item['binding'], str)):
            raise PlanRefused('withdrawal plan row needs its family and criterion: ' + str(ident))
    caps = plan.get('visit_cap') or {}
    if not isinstance(caps, dict) or any(
            not isinstance(item, dict) or type(item.get('required_visits')) is not int or item['required_visits'] < 1
            for item in caps.values()):
        raise PlanRefused('withdrawal plan visit_cap maps a row id to {"required_visits": int >= 1}')
    evidence = plan.get('evidence') or []
    if not isinstance(evidence, list) or any(
            not isinstance(item, dict) or not Path(str(item.get('path', ''))).is_file() for item in evidence):
        raise PlanRefused('withdrawal plan evidence is a list of existing {"path", "sha256"} files')
    for item in evidence:
        if sha256_file_hex_cached(item['path']) != item.get('sha256'):
            raise PlanRefused('withdrawal plan evidence file changed: ' + item['path'])


def queue_plan(controller_out, plan_path, *, log=print):
    """Append the immutable plan file to the withdrawal queue (idempotent on the plan's SHA-256). Returns the queue line."""
    plan_path = Path(plan_path).resolve(strict=True)
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    sha = sha256_file_hex_cached(plan_path)
    queue = queue_path(controller_out)
    for line in _lines(queue):
        if line['sha256'] == sha:
            return line
    line = {'plan': str(plan_path), 'sha256': sha, 'rows': len(plan['withdraw']),
            'visit_caps': len(plan.get('visit_cap') or {}), 'reason': plan['reason']}
    queue.parent.mkdir(parents=True, exist_ok=True)
    with open(queue, 'a') as handle:
        handle.write(json.dumps(line) + '\n')
    log(json.dumps({'event': 'withdrawal_queued', **line}))
    return line


def pending_plans(controller_out, ledger):
    """[(queue line, plan)] this lineage has not settled yet (a failed build stays here for the next boundary)."""
    if not controller_out:
        return []
    settled = {sha for sha, item in ((ledger or {}).get('withdrawals') or {}).items() if item.get('status') in SETTLED}
    result, seen = [], set()
    for line in _lines(queue_path(controller_out)):
        if line['sha256'] in settled or line['sha256'] in seen or not Path(line['plan']).is_file():
            continue
        seen.add(line['sha256'])
        result.append(line)
    return result


def load_plan(line):
    """The queue line's plan, verified against the SHA-256 recorded at queue time (PlanRefused when it is not usable)."""
    path = Path(line['plan'])
    if sha256_file_hex_cached(path) != line['sha256']:
        raise PlanRefused('withdrawal plan changed after it was queued')
    plan = json.loads(path.read_text())
    validate_plan(plan)
    return plan


def resolve(plans, frame, family_of, bindings):
    """What the queued plans withdraw from the live source under the parent checkpoint's saved ``frame``.

    ``plans`` = [(sha256, plan)], ``family_of`` {active row id: family}, ``bindings`` {active row id: binding}.
    Returns {'withdrawn': {id: {family, criterion, plan}}, 'capped': {id: {binding, prior, required}},
    'earned_kept': {family: n}, 'skipped': {plan sha: {...counts}}, 'refused': {plan sha: reason}}."""
    credits, quotas = frame.get('exposure_counts') or {}, frame.get('required_visits') or {}
    withdrawn, capped, refused = {}, {}, {}
    earned_kept = collections.Counter()
    skipped = collections.defaultdict(lambda: collections.Counter())
    for sha, plan in plans:
        mine, mine_skipped, mine_earned = {}, collections.Counter(), collections.Counter()
        try:
            for ident, item in plan['withdraw'].items():
                if ident not in family_of:
                    mine_skipped['inactive'] += 1
                elif item['family'] != family_of[ident]:
                    raise PlanRefused(f'planned row {ident} is family {family_of[ident]}, the plan says {item["family"]}')
                elif item.get('binding') is not None and item['binding'] != bindings[ident]:
                    mine_skipped['binding_changed'] += 1
                elif int(credits.get(ident, 0)) >= int(quotas.get(ident, 1)):
                    mine_earned[item['family']] += 1
                elif ident in withdrawn:
                    mine_skipped['already_planned'] += 1
                else:
                    mine[ident] = {'family': item['family'], 'criterion': item['criterion'], 'plan': sha}
        except PlanRefused as error:
            refused[sha] = str(error)
            continue
        withdrawn.update(mine)
        earned_kept.update(mine_earned)
        skipped[sha].update(mine_skipped)
    for sha, plan in plans:
        if sha in refused:
            continue
        for ident, item in (plan.get('visit_cap') or {}).items():
            quota = int(quotas.get(ident, 0))
            if ident in family_of and ident not in withdrawn and ident not in capped and quota > item['required_visits']:
                capped[ident] = {'binding': bindings[ident], 'prior': quota, 'required': item['required_visits']}
    return {'withdrawn': withdrawn, 'capped': capped, 'earned_kept': dict(earned_kept),
            'skipped': {sha: dict(value) for sha, value in skipped.items()}, 'refused': refused}


def leaves(tree):
    if tree.kind == 'manifest':
        for child in tree.children:
            yield from leaves(child)
    else:
        yield tree


def filter_leaf_files(source, withdrawn, identity, out):
    """Filter every leaf that holds withdrawn rows. Returns (replacements {old leaf sha: new output ref | None},
    proofs, removed leaves, {row id: leaf sha})."""
    import natlang_neuralese
    from natlang_neuralese.common import row_index
    from scripts.neuralese_data.withdraw_native_graph_targets import continue_filtered_index, filter_source
    package = Path(natlang_neuralese.__file__).resolve().parent  # the executing package: the index-cache policy hash
    policy_hash = digest({name: sha256_file_hex_cached(package / name) for name in (
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
        proofs.append({'filter': _ref(proof_path), 'index': projected, 'removed': len(rows), 'leaf': str(leaf.path)})
    return replacements, proofs, removed_leaves, leaf_of


def rewrite_manifest(path, replacements, out):
    """(rewritten manifest document, changed). Nested manifests are rewritten only where a leaf changed; a manifest whose
    sources all go returns None."""
    doc = json.loads(Path(path).read_text())
    changed = False
    kept = []
    for item in doc['sources']:
        if not Path(item['path']).is_absolute():
            raise ValueError('source withdrawal needs absolute source paths: ' + item['path'])
        if item['sha256'] in replacements:
            changed = True
            if replacements[item['sha256']] is None:
                continue
            item = {**item, **{k: v for k, v in replacements[item['sha256']].items() if k in ('path', 'sha256', 'bytes')}}
        elif Path(item['path']).suffix == '.json':
            nested, nested_changed = rewrite_manifest(item['path'], replacements, out)
            if nested_changed:
                changed = True
                if nested is None:
                    continue
                target = out / 'manifests' / f'nested-{digest(nested)[:16]}.json'
                write_json_durable(target, nested)
                item = {**item, **_ref(target), 'bytes': target.stat().st_size}
        kept.append(item)
    if not kept:
        return None, changed
    return {**doc, 'sources': kept}, changed


class Withdrawal:
    """The filtered source of one boundary and what its correction proof needs once the final source is known."""

    def __init__(self, *, predecessor, manifest, source, resolution, replacements, filters, removed_leaves, leaf_of,
                 plans, model_id, stage_id, frame, checkpoint, geometry, out):
        self.predecessor, self.manifest, self.source = predecessor, manifest, source  # source: the old NativeGraphSource
        self.resolution, self.replacements = resolution, replacements
        self.filters, self.removed_leaves, self.leaf_of = filters, removed_leaves, leaf_of
        self.plans, self.model_id, self.stage_id, self.frame = plans, model_id, stage_id, frame
        self.checkpoint, self.geometry, self.out = checkpoint, geometry, out
        self.withdrawn = {ident: item['family'] for ident, item in resolution['withdrawn'].items()}
        self.by_family = dict(collections.Counter(self.withdrawn.values()))
        self.reason = '; '.join(dict.fromkeys(plan['reason'] for _, plan in plans))
        self.proof = None

    def new_leaf_sha(self, ident):
        """The leaf SHA-256 an active (kept) row has in the filtered source."""
        old = self.leaf_of[ident]
        new = self.replacements.get(old, {'sha256': old})
        return new['sha256']

    def adjust_policy(self, policy):
        """Leaf replacements in a CE policy: declarations follow the filtered leaves, removed leaves leave the active set."""
        by_old = {old: new for old, new in self.replacements.items() if new is not None}
        removed = {old for old, new in self.replacements.items() if new is None}
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
        policy['sources'].sort(key=lambda rule: rule['sha256'])

    def seal(self, successor, added, family_map, out):
        """Write the correction ledger, the quota revision and the proof for the FINAL source ``successor`` (the filtered
        manifest, plus the appended leaves when packages join this change). ``added`` = [{id, binding, leaf_sha256}] in
        source order. Returns the proof's reference."""
        from ..train.native_source_correction import (QUOTA_REVISION_SCHEMA, SCHEMA as CORRECTION_SCHEMA,
                                                      source_correction_dependencies)
        criteria = self.resolution['withdrawn']
        ledger = [{'id': e['id'], 'disposition': 'deferred', 'old_binding': e['binding'], 'new_binding': None,
                   'old_source_sha256': self.leaf_of[e['id']], 'new_source_sha256': None,
                   'old_communication': bool(e.get('communication')), 'reason': self.reason + ': ' + criteria[e['id']]['criterion']}
                  for e in self.source.entries if e['id'] in self.withdrawn]
        ledger += [{'id': row['id'], 'disposition': 'added', 'old_binding': None, 'new_binding': row['binding'],
                    'old_source_sha256': None, 'new_source_sha256': row['leaf_sha256']} for row in added]
        rows_path = out / 'correction-rows.jsonl'
        rows_path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in ledger))
        proof = {'schema': CORRECTION_SCHEMA, 'model_id': self.model_id, 'stage_id': self.stage_id,
                 'prior_source': self.predecessor, 'current_source': dict(successor),
                 'row_bindings_path': str(rows_path), 'row_bindings_sha256': sha256_file_hex_cached(rows_path),
                 'evidence': [dict(family_map), *({'path': str(Path(line['plan'])), 'sha256': line['sha256']}
                                                   for line, _ in self.plans),
                              *(item for _, plan in self.plans for item in plan.get('evidence') or [])],
                 'reason': self.reason,
                 'withdrawal': {'rows': self.by_family, 'earned_rows_kept': self.resolution['earned_kept'],
                                'appended_rows': len(added),
                                'selection': 'rows named by the queued plans with saved credit below saved quota in the '
                                             'parent frame', 'parent_checkpoint': self.checkpoint}}
        if self.resolution['capped']:
            quota_path = out / 'quota-revision.jsonl'
            quota_path.write_text(''.join(json.dumps(
                {'id': ident, 'binding': item['binding'], 'prior_required_visits': item['prior'],
                 'required_visits': item['required']}, sort_keys=True) + '\n'
                for ident, item in sorted(self.resolution['capped'].items())))
            proof['quota_revision'] = {'schema': QUOTA_REVISION_SCHEMA, 'row_bindings_path': str(quota_path),
                                       'row_bindings_sha256': sha256_file_hex_cached(quota_path),
                                       'reason': self.reason + ': visit caps of over-represented kept rows '
                                                 '(earned credit retained)'}
        proof_path = out / 'source-correction.json'
        write_json_durable(proof_path, proof)
        source_correction_dependencies(_ref(proof_path))
        self.proof = _ref(proof_path)
        return self.proof

    def record(self):
        return {'schema': LEDGER_SCHEMA, 'reason': self.reason, 'rows': self.by_family,
                'earned_rows_kept': self.resolution['earned_kept'], 'visit_capped': len(self.resolution['capped']),
                'skipped': self.resolution['skipped'], 'plans': [line['sha256'] for line, _ in self.plans],
                'predecessor': self.predecessor, 'filtered_source': self.manifest, 'source_correction': self.proof,
                'removed_leaves': self.removed_leaves, 'filters': self.filters, 'parent_checkpoint': self.checkpoint,
                'exposure': 'none claimed; earned credits retained; originals untouched'}


def prepare(lines, *, checkpoint, stage_id, base_inputs, base_parameters, geometry, out, index, log=print):
    """Filter the live source for the queued plans ``lines`` ([queue line]). Returns (Withdrawal or None, {plan sha:
    ledger entry}); the entries record refused/empty plans. Failures that may succeed later raise."""
    from ..train.native_packed_capacity import _rows_digest
    outcomes, plans = {}, []
    for line in lines:
        try:
            plans.append((line, load_plan(line)))
        except PlanRefused as error:
            outcomes[line['sha256']] = {'path': line['plan'], 'status': 'refused', 'reason': str(error)}
    if not plans:
        return None, outcomes
    ce_ref = base_inputs.get('ce_supervision_policy')
    declared_geometry = base_parameters.get('sft_packed_geometry')
    if isinstance(declared_geometry, str):
        declared_geometry = json.loads(declared_geometry)
    if not ce_ref or not isinstance(declared_geometry, dict) or geometry is None:
        raise ValueError('a source withdrawal needs the stage CE family policy and its packed-geometry sidecar')
    predecessor = {'path': base_inputs['native_graphs']['path'], 'sha256': base_inputs['native_graphs']['sha256']}
    balance = (json.loads(Path(ce_ref['path']).read_text()).get('objective') or {}).get('family_balance') or {}
    if balance.get('key') != 'logical-family':
        raise ValueError('a source withdrawal needs a logical-family CE objective')
    family_map = json.loads(Path(balance['map']['path']).read_text())
    if family_map['source_sha256'] != predecessor['sha256']:
        raise ValueError('the CE family map does not bind the live source')
    source = index(predecessor['path'], geometry['tokenizer'], geometry['roles'])
    if source.file_sha256 != predecessor['sha256']:
        raise ValueError('the live source index differs from its binding')
    geometry0 = json.loads(Path(declared_geometry['path']).read_text())
    if _rows_digest(source.entries) != geometry0['rows_sha256']:
        raise ValueError('the stage packed-geometry sidecar does not bind the live source')
    if [(r['id'], r['binding']) for r in family_map['rows']] != [(e['id'], e['binding']) for e in source.entries]:
        raise ValueError('family map differs from the live source')
    family_of = {row['id']: row['family'] for row in family_map['rows']}
    frame = _parent_frame(checkpoint, stage_id)
    if frame['source_receipts'][0].get('sha256') != predecessor['sha256']:
        raise ValueError('the checkpoint frame does not train the recipe source the window continues')
    resolution = resolve([(line['sha256'], plan) for line, plan in plans], frame, family_of, source.bindings)
    for sha, reason in resolution['refused'].items():
        outcomes[sha] = {'path': next(l['plan'] for l, _ in plans if l['sha256'] == sha), 'status': 'refused',
                         'reason': reason}
    plans = [(line, plan) for line, plan in plans if line['sha256'] not in resolution['refused']]
    if not resolution['withdrawn']:
        for line, _ in plans:
            outcomes[line['sha256']] = {'path': line['plan'], 'status': 'empty',
                                        'earned_kept': resolution['earned_kept'],
                                        'skipped': resolution['skipped'].get(line['sha256'], {}),
                                        'visit_caps_dropped': len(resolution['capped'])}
        return None, outcomes
    log(json.dumps({'event': 'withdrawal_set', 'withdrawn': dict(collections.Counter(
        item['family'] for item in resolution['withdrawn'].values())), 'earned_kept': resolution['earned_kept'],
        'visit_capped': len(resolution['capped']), 'source_rows': len(source.entries)}))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    replacements, filters, removed_leaves, leaf_of = filter_leaf_files(source, resolution['withdrawn'], geometry, out)
    (out / 'manifests').mkdir(parents=True, exist_ok=True)
    manifest_doc, _ = rewrite_manifest(predecessor['path'], replacements, out)
    if manifest_doc is None:
        raise ValueError('the withdrawal would empty the joint SFT source')
    manifest_doc['source_withdrawal'] = {'rows': dict(collections.Counter(
        item['family'] for item in resolution['withdrawn'].values())), 'plans': [line['sha256'] for line, _ in plans]}
    manifest_path = out / 'withdrawn-sources.json'
    write_json_durable(manifest_path, manifest_doc)
    filtered = index(str(manifest_path), geometry['tokenizer'], geometry['roles'])
    if [(e['id'], e['binding']) for e in filtered.entries] != [
            (e['id'], e['binding']) for e in source.entries if e['id'] not in resolution['withdrawn']]:
        raise ValueError('the filtered source changed the retained row identity/order')
    manifest = _ref(manifest_path)
    # the packed-geometry sidecar of the filtered rows: slots are per row, so they follow their rows
    slots = {e['id']: s for e, s in zip(source.entries, geometry0['kv_slots'], strict=True)}
    sidecar = {'schema': geometry0['schema'], 'identity': geometry0['identity'], 'source': manifest,
               'rows_sha256': _rows_digest(filtered.entries), 'kv_slots': [slots[e['id']] for e in filtered.entries]}
    write_json_durable(out / 'packed-geometry.json', sidecar)
    withdrawal = Withdrawal(predecessor=predecessor, manifest=manifest, source=source, resolution=resolution,
                            replacements=replacements, filters=filters, removed_leaves=removed_leaves,
                            leaf_of=leaf_of, plans=plans, model_id=_model_id(checkpoint), stage_id=stage_id,
                            frame=frame, checkpoint=_ref(checkpoint), geometry=_ref(out / 'packed-geometry.json'),
                            out=out)
    for line, _ in plans:
        outcomes[line['sha256']] = {'path': line['plan'], 'status': 'applied'}
    return withdrawal, outcomes


def _parent_frame(checkpoint, stage_id):
    from .memetic_admission import _parent_sft_frame
    return _parent_sft_frame(checkpoint, stage_id)


def _model_id(checkpoint):
    from ..train.checkpoint_policy import CheckpointPolicy
    config = CheckpointPolicy.load(Path(checkpoint))['model_config']
    return config['architecture'] + ':' + config['base_inventory_sha256']
