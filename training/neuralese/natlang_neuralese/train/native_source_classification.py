"""Explicit source-index and earned-exposure classification continuations.

This mechanism never parses graph positions or changes a row binding. The
operator selects the exact proof in the recipe; old caches remain immutable.
"""
from __future__ import annotations
import json
from pathlib import Path

from ..common import row_index
from ..common.hashing import canonical_json_sha256_hex as digest, sha256_file_hex, sha256_file_hex_cached
from ..data.native_graph_schema import indexed_graph_metadata, communication_classification
from ..data.native_sources import _jsonl, _row_bindings

SCHEMA = 'natlang.native-source-classification/1'
CLASS_FIELDS = ('communication', 'producer_consumer', 'external_state_consumption')
POLICY_FILES = ('data/native_sources.py', 'data/native_graph_schema.py', 'model/native_graph_plan.py',
                'data/native_pair_schema.py', 'common/row_index.py', 'common/jsonio.py', 'common/hashing.py')


def _binding(path):
    path = Path(path)
    return {'path': str(path), 'sha256': sha256_file_hex_cached(path, trust_archive_receipt=False), 'bytes': path.stat().st_size}


def _checked(ref):
    path = Path(ref['path'])
    actual = _binding(path)
    if actual['sha256'] != ref['sha256'] or ('bytes' in ref and actual['bytes'] != ref['bytes']):
        raise ValueError('source classification artifact changed')
    return path


def _index(ref):
    value = row_index.load(_checked(ref))
    if not value or value.get('metadata_sha256') != digest(value.get('entries')):
        raise ValueError('source classification index metadata differs')
    return value


def source_policy(package):
    return digest({name: sha256_file_hex(Path(package) / name) for name in POLICY_FILES})


def migrate_index(source, old_index, *, old_package, new_package, ordinary_partition=None):
    """Stream only row identity/topology; never compile a graph or run a model.

    Both package roots must be independently pinned by the production operator.
    An existing semantic-pair index already attests its stricter producer law;
    native-graph rows are normalized and matched to every old exact binding.
    """
    source, old_index = Path(source), Path(old_index)
    source_ref = _binding(source)
    old_ref = _binding(old_index)
    old = _index(old_ref)
    key = old['key']
    if key['source_sha256'] != source_ref['sha256'] or key['policy_sha256'] != source_policy(old_package):
        raise ValueError('source classification prior source/policy differs')
    if key['kind'] not in ('native-graphs', 'semantic-pairs'):
        raise ValueError('source classification needs an execution-indexed graph source')
    entries = []
    if ordinary_partition is not None:
        if key['kind'] != 'native-graphs':
            raise ValueError('ordinary topology proof cannot replace semantic-pair qualification')
        _ordinary_partition_classes(ordinary_partition, source_ref, old)
        entries = [{**entry, 'communication': False, 'producer_consumer': False,
                    'external_state_consumption': False} for entry in old['entries']]
    elif key['kind'] == 'semantic-pairs':
        if not all(type(entry.get('communication')) is bool for entry in old['entries']):
            raise ValueError('semantic pair lacks its original communication classification')
        entries = [{**entry, 'producer_consumer': entry['communication'], 'external_state_consumption': False}
                   for entry in old['entries']]
    else:
        prior = iter(old['entries'])
        for row in _jsonl(source):
            entry = next(prior, None)
            example = indexed_graph_metadata(row, tokenizer_fingerprint=key['tokenizer'])
            if (entry is None or entry['id'] != example.id or example.split != key['split']
                    or entry['binding'] != _row_bindings([example])[example.id]
                    or tuple(entry['groups']) != example.source_groups
                    or entry['forward_tokens'] != sum(len(node.token_ids) for node in example.plan.invocations)
                    or entry['supervised_tokens'] != sum(weight > 0 for weight in example.target_weights)):
                raise ValueError('source classification row differs from its validated execution index')
            entries.append({**entry, **communication_classification(example)})
        if next(prior, None) is not None:
            raise ValueError('source classification omitted indexed rows')
    if _binding(source) != source_ref or _binding(old_index) != old_ref:
        raise ValueError('source changed during classification migration')
    new_key = {**key, 'policy_sha256': source_policy(new_package)}
    path = row_index.entry_path('native-graphs', digest(new_key))
    if path == old_index:
        raise ValueError('source classification migration must create a new policy index')
    result = {'schema': 'natlang.row-index/1', 'key': new_key, 'entries': entries, 'metadata_sha256': digest(entries)}
    previous = row_index.load(path)
    if previous is not None and previous != result:
        raise ValueError('source classification target index already has different contents')
    row_index.save(path, result)
    return {'source': source_ref, 'old_index': old_ref, 'new_index': _binding(path),
            'rows': len(entries), 'producer': _binding(__file__),
            'old_policy': {name: _binding(Path(old_package) / name) for name in POLICY_FILES},
            'graph_position_expansions': 0, 'optimizer_updates': 0,
            'ordinary_partition': ordinary_partition}


def migration_classes(proof_path, source):
    """Verify an explicitly recipe-bound migration against the actual current source."""
    proof = json.loads(Path(proof_path).read_text())
    if proof.get('schema') != SCHEMA or proof.get('train_sha256') != source.receipt()['sha256']:
        raise ValueError('source classification proof names another current source')
    package = Path(__file__).resolve().parents[1]
    current = {entry['id']: entry for entry in source.entries}
    leaves = set()
    def visit(receipt):
        if receipt.get('sources'):
            for child in receipt['sources']:
                visit(child)
        else:
            leaves.add(receipt['sha256'])
    visit(source.receipt())
    classes = {}
    for part in proof.get('indexes', []):
        _checked(part['source'])
        if part['source']['sha256'] not in leaves or part['producer'] != _binding(__file__):
            raise ValueError('source classification proof is not produced for this source/runtime')
        old_policy = part.get('old_policy') or {}
        if set(old_policy) != set(POLICY_FILES):
            raise ValueError('source classification lost the prior parser policy')
        prior_policy = digest({name: _binding(_checked(ref))['sha256'] for name, ref in old_policy.items()})
        old, new = _index(part['old_index']), _index(part['new_index'])
        if (old['key']['policy_sha256'] != prior_policy
                or new['key'] != {**old['key'], 'policy_sha256': source_policy(package)}
                or new['key']['source_sha256'] != part['source']['sha256']
                or len(old['entries']) != len(new['entries']) or len(new['entries']) != part['rows']):
            raise ValueError('source classification cache continuation changed source identity')
        if part.get('ordinary_partition') is not None:
            _ordinary_partition_classes(part['ordinary_partition'], part['source'], old)
        for before, after in zip(old['entries'], new['entries'], strict=True):
            key = after['id']
            expected_classes = ({'communication': False, 'producer_consumer': False,
                'external_state_consumption': False} if part.get('ordinary_partition') is not None else
                {'communication': before['communication'], 'producer_consumer': before['communication'],
                 'external_state_consumption': False} if old['key']['kind'] == 'semantic-pairs' else None)
            if expected_classes is not None and any(after.get(name) != value for name, value in expected_classes.items()):
                raise ValueError('source classification contradicts authenticated ordinary/pair topology law')
            if ({k: v for k, v in before.items() if k not in CLASS_FIELDS}
                    != {k: v for k, v in after.items() if k not in CLASS_FIELDS}
                    or key in classes or {k: v for k, v in (current.get(key) or {}).items() if k != 'training_priority'}
                    != {k: v for k, v in after.items() if k != 'training_priority'}
                    or any(type(after.get(name)) is not bool for name in CLASS_FIELDS)
                    or after['communication'] != (after['producer_consumer'] or after['external_state_consumption'])):
                raise ValueError('source classification changes row identity or current classes')
            if before.get('communication', False) != after['communication'] or after['external_state_consumption']:
                classes[key] = {'binding': after['binding'], **{name: after[name] for name in CLASS_FIELDS}}
    if not proof.get('indexes'):
        raise ValueError('source classification proof has no authenticated indexes')
    return classes


def reclassify_progress(progress, classes):
    """Reattribute only existing exact-bound row visits; never grant new exposure."""
    result = dict(progress)
    for stage, frame in progress.items():
        bindings = frame.get('source_row_bindings') or {}
        selected = {key: value for key, value in classes.items() if key in bindings}
        if not selected:
            continue
        if any(bindings[key] != value['binding'] for key, value in selected.items()):
            raise ValueError('classification migration changed a historical row binding')
        options = dict(frame.get('options') or {})
        for field, label in (('_communication_ids', 'communication'),
                             ('_producer_consumer_ids', 'producer_consumer'),
                             ('_external_state_consumption_ids', 'external_state_consumption')):
            # Legacy communication IDs were strictly authenticated semantic pairs.
            baseline = options.get(field, options.get('_communication_ids', []) if label == 'producer_consumer' else [])
            ids = set(baseline)
            for key, value in selected.items():
                (ids.add if value[label] else ids.discard)(key)
            options[field] = sorted(ids)
        result[stage] = {**frame, 'options': options}
    return result


def require_explicit_reclassification(progress, entries, classes):
    """An old admitted row cannot acquire new communication credit implicitly."""
    current = {entry['id']: entry for entry in entries}
    for frame in progress.values():
        previous = set((frame.get('options') or {}).get('_communication_ids', []))
        for key, binding in (frame.get('source_row_bindings') or {}).items():
            entry = current.get(key)
            if (entry is not None and entry['binding'] == binding
                    and bool(entry.get('communication')) != (key in previous)
                    and (classes.get(key) or {}).get('binding') != binding):
                raise ValueError('changed communication class requires an explicit source classification migration')


def _ordinary_partition_classes(part, selected_source, index):
    """Reuse the registered ordinary producer/audit law over exact old indexes.

    A projected source must match the original execution entries, not just IDs.
    The prior parser-index binding already attests those graph bytes; there is
    no new positional graph compilation or original-corpus payload pass.
    """
    from .native_capacity_activation import ordinary_partition_proof
    receipt, proofs, original, coverage_path = ordinary_partition_proof(part)
    model = part['model']
    names = [name for name, path in receipt['manifests'].items()
             if Path(part['receipt']['path']).parent / path == original]
    if len(names) != 1 or names[0] not in (model + ':train:fit', model + ':train:remaining-long'):
        raise ValueError('classification source is not an original ordinary train partition')
    partition = names[0].split(':')[-1]
    rows = [json.loads(line) for line in coverage_path.read_text().splitlines() if line.strip()]
    rows = sorted((r for r in rows if r.get('model') == model and r.get('split') == 'train'
                   and r.get('partition') == partition), key=lambda r: r['partition_row_index'])
    expected = receipt['counts'][model]['train:' + partition]['rows']
    if len(rows) != expected or [r['partition_row_index'] for r in rows] != list(range(expected)):
        raise ValueError('ordinary classification coverage is incomplete')
    leaf = original
    if original.suffix == '.json':
        manifest = json.loads(original.read_text())
        children = manifest.get('sources') or []
        if (manifest.get('schema') != 'natlang.native-graph-sources/1' or len(children) != 1
                or children[0].get('kind') != 'native-graphs'):
            raise ValueError('ordinary partition needs its exact single native graph leaf')
        ref = children[0]
        leaf = Path(ref['path'])
        if not leaf.is_absolute(): leaf = original.parent / leaf
        if _binding(leaf)['sha256'] != ref['sha256']:
            raise ValueError('ordinary partition leaf differs from its original manifest')
    leaf_binding = _binding(leaf)
    relative = str(leaf.resolve().relative_to(Path(part['corpus_root']).resolve()))
    registered = json.loads(_checked(part['registered_manifest']).read_text())
    members = [entry for entry in registered['files'] if entry['path'] == relative]
    if len(members) != 1 or members[0]['sha256'] != leaf_binding['sha256'] or members[0]['bytes'] != leaf_binding['bytes']:
        raise ValueError('ordinary partition leaf lost registered corpus membership')
    original_index = index
    if selected_source['sha256'] != leaf_binding['sha256']:
        witness = part['original_index']
        original_index = _index(witness['cache'])
        if original_index['key'] != {**index['key'], 'source_sha256': leaf_binding['sha256']}:
            raise ValueError('ordinary projected index changed its original parser policy')
        filter_proof = json.loads(_checked(part['filter_proof']).read_text())
        if (filter_proof.get('schema') != 'natlang.native-source-exact-withdrawal-filter/1'
                or filter_proof['source_sha256'] != leaf_binding['sha256']
                or filter_proof['output'] != selected_source):
            raise ValueError('ordinary projected source lost its exact filter proof')
        removed = {row['id']: row for row in filter_proof['removed']}
        original_entries = {entry['id']: entry for entry in original_index['entries']}
        if len(removed) != len(filter_proof['removed']) or any(
                original_entries.get(key, {}).get('binding') != value['binding'] for key, value in removed.items()):
            raise ValueError('ordinary projected source removed a different original row')
        keep = [entry for entry in original_index['entries'] if entry['id'] not in removed]
        strip = lambda entry: {k: v for k, v in entry.items() if k != 'training_priority'}
        if [strip(e) for e in keep] != [strip(e) for e in index['entries']]:
            raise ValueError('ordinary projected source changed retained execution bindings')
    positions = set()
    if len(original_index['entries']) != len(rows):
        raise ValueError('ordinary classification index omitted original rows')
    for row, entry in zip(rows, original_index['entries'], strict=True):
        origin = row['source']
        declarations = [v for v in receipt['source_bindings'] if v['path'] == origin['path']
                        and v['sha256'] == origin['sha256'] and v.get('split') == 'train'
                        and Path(v['path']).parent.name == 'train'
                        and Path(v['path']).parent.parent.name == model]
        position = (origin['path'], origin['sha256'], origin['row_index'])
        proof = proofs.get((model, row['id']))
        if (len(declarations) != 1 or type(origin['row_index']) is not int
                or not 0 <= origin['row_index'] < declarations[0]['rows'] or position in positions
                or proof is None or proof['split'] != 'train' or row['id'] != entry['id']
                or proof['graph_sha256'] != row['source_graph_sha256']
                or proof['tokens'] != row['geometry']['forward_tokens']
                or row['geometry'] != {'forward_tokens': entry['forward_tokens'], 'kv_slots': entry['forward_tokens']}):
            raise ValueError('ordinary classification lost its all-row producer/source proof')
        positions.add(position)
