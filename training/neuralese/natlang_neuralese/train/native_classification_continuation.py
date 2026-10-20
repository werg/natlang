"""Authenticate classified-cache CE continuations, including exact ordinary withdrawals."""
from __future__ import annotations

import json
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex as digest
from . import native_source_classification as m

SCHEMA = 'natlang.native-classified-objective-continuation/1'


def _same_bytes(a, b):
    return all(a.get(k) == b.get(k) for k in ('sha256', 'bytes'))


def _leaves(reference):
    """Read only the normal manifest DAG, authenticating every declared dependency."""
    leaves = set()
    def visit(ref, kind='native-graphs', model=None, parents=()):
        path = m._checked(ref)
        resolved = path.resolve()
        if resolved in parents:
            raise ValueError('objective classification source manifest cycle')
        if path.suffix != '.json':
            key = (ref['sha256'], kind, model)
            if key in leaves:
                raise ValueError('objective classification repeats a source leaf')
            leaves.add(key)
            return
        value = json.loads(path.read_text())
        if value.get('schema') != 'natlang.native-graph-sources/1' or not value.get('sources'):
            raise ValueError('objective classification needs normal source manifests')
        for child in value['sources']:
            if child.get('kind') not in ('native-graphs', 'semantic-pairs'):
                raise ValueError('objective classification source kind differs')
            pair = child.get('model') if child['kind'] == 'semantic-pairs' else None
            if child['kind'] == 'semantic-pairs' and not isinstance(pair, str):
                raise ValueError('objective classification pair model missing')
            target = Path(child['path'])
            if not target.is_absolute():target = path.parent / target
            # Intrinsic manifest children bind path/SHA; legacy optional byte
            # annotations are not source authority. All proof refs stay strict.
            visit({'path':str(target), 'sha256':child['sha256']}, child['kind'], pair, (*parents, resolved))
        m._checked(ref)
    visit(reference)
    return leaves


def _indexed(part, which):
    value = m._index(part[which])
    key = value['key']
    entries = value['entries']
    if (key.get('source_sha256') != part['source']['sha256']
            or key.get('kind') not in ('native-graphs', 'semantic-pairs')
            or key.get('kind') == 'native-graphs' and key.get('pair_model') is not None
            or key.get('kind') == 'semantic-pairs' and not isinstance(key.get('pair_model'), str)
            or len(entries) != part['rows'] or not entries
            or len({e['id'] for e in entries}) != len(entries)):
        raise ValueError('objective classification source/index identity differs')
    return value


def _classes(entries):
    for entry in entries:
        if (any(type(entry.get(k)) is not bool for k in m.CLASS_FIELDS)
                or entry['communication'] != (entry['producer_consumer'] or entry['external_state_consumption'])):
            raise ValueError('objective classification lost exact topology classes')


def verify_objective_proof_extension(old_inputs, new_inputs, prior, current):
    """Keep every old row/class, except a separately proved exact ordinary withdrawal.

    This checks source/index/CE-adapter authority only. Actual source loading,
    source-bound CE coverage and saved optimizer/exposure continuation remain
    independent mandatory consumers.
    """
    from scripts.neuralese_data.continue_native_source_indexes import prove_ce_objective_extension
    extension = current.get('objective_policy_continuation')
    if not isinstance(extension, dict) or extension.get('schema') != SCHEMA:
        return False
    old_ref = old_inputs['source_classification_migration']
    if any(extension.get('prior_proof', {}).get(k) != old_ref.get(k) for k in ('path','sha256')):
        raise ValueError('objective classification predecessor proof differs')
    m._checked(extension['prior_proof'])
    qualification = prove_ce_objective_extension(extension['old_package'], extension['new_package'])
    before_policy, after_policy = qualification['old_policy_sha256'], qualification['new_policy_sha256']
    expected_producer = m._binding(Path(extension['new_package']) / 'train/native_source_classification.py')
    for part in current['indexes'] + current['held_indexes']:
        policy = digest({name:m._binding(m._checked(ref))['sha256'] for name,ref in part['old_policy'].items()})
        if part['producer'] != expected_producer or policy != before_policy:
            raise ValueError('objective classification rekey producer/prior policy differs')
    if current.get('held') != prior.get('held') or len(current['held_indexes']) != len(prior['held_indexes']):
        raise ValueError('objective classification changed protected held sources')
    withdrawals = {}
    for item in extension.get('withdrawals', []):
        sha = item['prior_source_sha256']
        if sha in withdrawals:
            raise ValueError('objective classification repeats a withdrawal source')
        ref = item['withdrawal_proof']
        declared = new_inputs.get('withdrawn_native_graphs', {})
        if any(ref.get(k) != declared.get(k) for k in ('path','sha256')):
            raise ValueError('objective classification withdrawal is not declared')
        proof = json.loads(m._checked(ref).read_text())
        if proof.get('schema') != 'natlang.native-graph-withdrawal/1' or not proof.get('rows'):
            raise ValueError('objective classification withdrawal schema differs')
        evidence = dict(proof['evidence'])
        if not Path(evidence['path']).is_absolute():evidence['path'] = str(Path(ref['path']).parent / evidence['path'])
        m._checked(evidence)
        filtered = json.loads(m._checked(item['filter_proof']).read_text())
        rows = [r for r in proof['rows'] if r.get('source_sha256') == sha]
        removed = filtered.get('removed', [])
        if (filtered.get('schema') != 'natlang.native-source-exact-withdrawal-filter/1'
                or filtered.get('source_sha256') != sha or not rows
                or len({r['id'] for r in rows}) != len(rows)
                or any(not r.get('reason') for r in rows)
                or {r['id']: {k:r[k] for k in ('id','binding','source_sha256','reason')} for r in removed}
                   != {r['id']: {k:r[k] for k in ('id','binding','source_sha256','reason')} for r in rows}
                or len({r['id'] for r in removed}) != len(removed)
                or not _same_bytes(filtered.get('output', {}), item['new_source'])):
            raise ValueError('objective classification filter/withdrawal differs')
        m._checked(item['new_source'])
        withdrawals[sha] = (item, filtered, rows)
    if withdrawals and {r['source_sha256'] for r in proof['rows']} != set(withdrawals):
        raise ValueError('objective classification withdrawal contains undeclared sources')
    used = set()
    leaf_keys = {'prior':{'indexes':set(),'held_indexes':set()},
                 'current':{'indexes':set(),'held_indexes':set()}}
    def record(which, field, part, index):
        key = (part['source']['sha256'], index['key']['kind'], index['key']['pair_model'])
        if key in leaf_keys[which][field]:
            raise ValueError('objective classification repeats a source part')
        leaf_keys[which][field].add(key)
    for field in ('indexes', 'held_indexes'):
        before, after = prior[field], current[field]
        if len(after) < len(before):
            raise ValueError('objective classification removed an undeclared source')
        for old, new in zip(before, after[:len(before)], strict=True):
            old_value = _indexed(old, 'new_index')
            intermediate = _indexed(new, 'old_index')
            new_value = _indexed(new, 'new_index')
            _classes(old_value['entries']); _classes(new_value['entries'])
            record('prior',field,old,old_value); record('current',field,new,new_value)
            if (old_value['key'].get('policy_sha256') != before_policy
                    or intermediate['key'].get('policy_sha256') != before_policy
                    or new_value['key'] != {**intermediate['key'], 'policy_sha256':after_policy}
                    or intermediate['entries'] != new_value['entries']):
                raise ValueError('objective classification changed CE rekey entries/key')
            if old['source']['sha256'] in withdrawals:
                if field != 'indexes':
                    raise ValueError('objective classification cannot withdraw held rows')
                item, filtered, rows = withdrawals[old['source']['sha256']]
                used.add(old['source']['sha256'])
                selected = {r['id']:r for r in rows}
                found = [e for e in old_value['entries'] if e['id'] in selected]
                kept = [e for e in old_value['entries'] if e['id'] not in selected]
                if (new['source'] != item['new_source'] or len(found) != len(selected)
                        or any(e['binding'] != selected[e['id']]['binding'] or e['communication'] is not False for e in found)
                        or intermediate['key'] != {**old_value['key'], 'source_sha256':new['source']['sha256']}
                        or intermediate['entries'] != kept
                        or filtered.get('old_rows') != len(old_value['entries'])
                        or filtered.get('new_rows') != len(kept)
                        or filtered.get('old_ids_sha256') != digest([e['id'] for e in old_value['entries']])
                        or filtered.get('new_ids_sha256') != digest([e['id'] for e in kept])):
                    raise ValueError('objective classification withdrawal changed other rows')
            elif (not _same_bytes(old['source'], new['source'])
                    or not _same_bytes(old['new_index'], new['old_index'])
                    or old_value != intermediate):
                raise ValueError('objective classification changed an existing source/class')
        for part in after[len(before):]:
            # New sources still use the normal classifier's identity/class law;
            # actual graph binding is checked again by migration_classes at load.
            a, b = _indexed(part, 'old_index'), _indexed(part, 'new_index')
            prior_policy = digest({name:m._binding(m._checked(ref))['sha256'] for name,ref in part['old_policy'].items()})
            record('current',field,part,b)
            if (a['key']['split'] != 'train' or a['key']['policy_sha256'] != prior_policy
                    or b['key'] != {**a['key'], 'policy_sha256':after_policy}
                    or len(a['entries']) != len(b['entries'])):
                raise ValueError('objective classification new source policy differs')
            _classes(b['entries'])
            if all(all(type(e.get(k)) is bool for k in m.CLASS_FIELDS) for e in a['entries']):
                if any(any(x[k] != y[k] for k in m.CLASS_FIELDS) for x,y in zip(a['entries'],b['entries'],strict=True)):
                    raise ValueError('objective classification changed already authenticated new-source classes')
            elif a['key']['kind'] == 'native-graphs' and part.get('ordinary_partition') is None:
                raise ValueError('objective classification new raw source lacks authenticated topology')
            if part.get('ordinary_partition') is not None:
                m._ordinary_partition_classes(part['ordinary_partition'], part['source'], a)
                if any(e[k] for e in b['entries'] for k in m.CLASS_FIELDS):
                    raise ValueError('objective classification contradicts ordinary topology proof')
            for x,y in zip(a['entries'], b['entries'], strict=True):
                if {k:v for k,v in x.items() if k not in m.CLASS_FIELDS} != {k:v for k,v in y.items() if k not in m.CLASS_FIELDS}:
                    raise ValueError('objective classification new source changes row bindings')
                if a['key']['kind'] == 'semantic-pairs' and (y['producer_consumer'] != x['communication'] or y['external_state_consumption']):
                    raise ValueError('objective classification new pair changes its class law')
    if used != set(withdrawals):
        raise ValueError('objective classification withdrawal does not name an old train source')
    for inputs, which in ((old_inputs, 'prior'), (new_inputs, 'current')):
        for role, field in (('native_graphs','indexes'), ('held_native_graphs','held_indexes')):
            if leaf_keys[which][field] != _leaves(inputs[role]):
                raise ValueError('objective classification source proof is not a complete leaf inventory')
    return True



def classification_proof_dependencies(reference):
    """Exact recursively consumed classification/adapter/withdrawal artifacts."""
    from scripts.neuralese_data import continue_native_source_indexes as adapter
    found, visiting = {}, set()
    def add(ref):
        path = m._checked(ref)
        actual = m._binding(path)
        if actual['path'] in found and found[actual['path']] != actual:
            raise ValueError('classification dependency changed during enumeration')
        found[actual['path']] = actual
        return path
    def visit(ref):
        path = add(ref)
        marker = (str(path.resolve()), ref['sha256'])
        if marker in visiting:raise ValueError('classification predecessor proof cycle')
        visiting.add(marker)
        proof = json.loads(path.read_text())
        if proof.get('schema') != m.SCHEMA:
            raise ValueError('classification dependency proof schema differs')
        for part in proof.get('indexes', []) + proof.get('held_indexes', []):
            for value in (part['source'],part['old_index'],part['new_index'],part['producer'],*part['old_policy'].values()):add(value)
            produced = part.get('objective_policy_continuation') or {}
            if produced.get('producer'):add(produced['producer'])
        extra = proof.get('objective_policy_continuation')
        if extra is not None:
            if extra.get('schema') != SCHEMA:raise ValueError('classification dependency continuation schema differs')
            adapter.prove_ce_objective_extension(extra['old_package'],extra['new_package'])
            for root in (Path(extra['old_package']),Path(extra['new_package'])):
                for name in (*m.POLICY_FILES,'train/native_source_classification.py'):add(m._binding(root/name))
            add(m._binding(adapter.__file__));add(m._binding(__file__))
            visit(extra['prior_proof'])
            for item in extra.get('withdrawals', []):
                add(item['new_source']);add(item['filter_proof'])
                withdrawal_path=add(item['withdrawal_proof']);withdrawal=json.loads(withdrawal_path.read_text())
                evidence=dict(withdrawal['evidence'])
                if not Path(evidence['path']).is_absolute():evidence['path']=str(withdrawal_path.parent/evidence['path'])
                add(evidence)
        visiting.remove(marker)
        m._checked(ref)
    visit(reference)
    return list(found.values())
