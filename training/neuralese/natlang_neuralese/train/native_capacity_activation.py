"""Verify whole ordinary remaining-long rows against an existing frontier."""
from pathlib import Path
import hashlib
import pickle
import re
import json

SCHEMA = 'natlang.native-capacity-activation-geometry-part/1'
# Reviewed ordinary_geometry producer; preference partitioning has another schema.
_ORDINARY_PARTITION_VERIFIERS = frozenset({
    '9282654e7427f1435496ec863bc221222fcec2702ab6cc9115798f3fa0d45c09',
    # Historical ordinary-only producers preserve the same audited row bytes,
    # all-row graph proof and one-node physical/KV geometry law.
    'b2a76d4140b1c883245a1824d3446a4999b76fb86176eff1a6a2d8a5fd2071a7',
    'f5ac2ef6c056bbb6d803549125de53a42eb0ae7cf9f4ae8c315d0acc47c7fc76',
})


def geometry_part(part, *, preparation, package):
    from .native_resident import bound, binding, code_inventory, geometry_capacity, digest, _source_witness, SCHEMA as resident_schema
    context = json.loads(bound(part['resident_current']).read_text())
    payload = context['payload']
    if (context.get('schema') != resident_schema or context.get('generation') != digest(payload)
            or payload['plan']['code'] != code_inventory(package)):
        raise ValueError('capacity activation runtime differs from its bound resident generation')
    limits = geometry_capacity(payload['geometry_qualification'],
        identity=payload['geometry_identity'], code=payload['plan']['code'])
    receipt_path = bound(part['receipt'])
    registered_path = bound(part['registered_manifest'])
    registered = json.loads(registered_path.read_text())
    root = Path(part['corpus_root']).resolve(strict=True)
    if registered.get('schema') != 'natlang.corpus-snapshot/1':
        raise ValueError('capacity activation needs its registered immutable corpus manifest')
    for reference in (part['receipt'], part['coverage'], part['original_source'],
                      part['admission_audit'], part['render_samples'], part['row_proof']):
        path = bound(reference)
        try: relative = str(path.resolve(strict=True).relative_to(root))
        except ValueError: raise ValueError('capacity activation source escaped registered corpus root') from None
        records = [v for v in registered['files'] if v['path'] == relative]
        if (len(records) != 1 or records[0]['sha256'] != reference['sha256']
                or records[0]['bytes'] != path.stat().st_size):
            raise ValueError('capacity activation source differs from its registered corpus member')
    receipt = json.loads(receipt_path.read_text())
    model = part['model']
    if (receipt.get('schema') != 'natlang.acquired-native-capacity-partition/1'
            or receipt.get('source_eligibility_unchanged') is not True
            or receipt.get('exact_original_row_bytes') is not True):
        raise ValueError('capacity activation needs an original admitted partition')

    def member(ref):
        path = bound(ref)
        try:
            relative = str(path.relative_to(receipt_path.parent))
        except ValueError:
            raise ValueError('capacity activation member escaped its receipt') from None
        matches = [v for v in receipt['files'] if v['path'] == relative]
        if (len(matches) != 1 or matches[0]['sha256'] != ref['sha256']
                or matches[0]['bytes'] != path.stat().st_size):
            raise ValueError('capacity activation member differs from original receipt')
        return path

    verifier = bound(part['partition_verifier'])
    audit = json.loads(bound(part['admission_audit']).read_text())
    samples = json.loads(bound(part['render_samples']).read_text())
    producer_sha = part['partition_verifier']['sha256']
    if (producer_sha not in _ORDINARY_PARTITION_VERIFIERS
            or hashlib.sha256(verifier.read_bytes()).hexdigest() != producer_sha
            or receipt['binding']['verifier_sha256'] != producer_sha
            or receipt['binding']['audit']['sha256'] != part['admission_audit']['sha256']
            or receipt['binding']['source_render_samples']['sha256'] != part['render_samples']['sha256']
            or audit.get('schema') != 'natlang.acquired-session-admission-audit/1'
            or audit.get('status') != 'passed' or samples.get('passed') is not True
            or samples.get('cohort_audit', {}).get('sha256') != part['admission_audit']['sha256']
            or audit['row_proof_sha256'] != part['row_proof']['sha256']
            or receipt['source_bindings'] != [v for c in audit['conversions'] for v in c['outputs']]):
        raise ValueError('capacity activation lost its exact ordinary partition producer/audit chain')
    proofs = {}
    with bound(part['row_proof']).open() as stream:
        for line in stream:
            if not line.strip(): continue
            proof = json.loads(line); key = (proof['model'], proof['id'])
            if key in proofs:
                raise ValueError('capacity activation ordinary row proof repeats an identity')
            proofs[key] = proof

    original_path = member(part['original_source'])
    coverage_path = member(part['coverage'])
    if (original_path != receipt_path.parent / receipt['manifests'][model + ':train:remaining-long']
            or coverage_path != receipt_path.parent / receipt['coverage']):
        raise ValueError('capacity activation lost original remaining-long disposition')
    original = preparation.load(part['original_source'], split='train')
    from .native_pending_sources import load_pending_sources
    pending_path = bound(part['pending_obligations'])
    cache = getattr(preparation, '_capacity_activation_pending', None)
    if cache is None:
        cache = preparation._capacity_activation_pending = {}
    pending_key = (str(pending_path.resolve()), part['pending_obligations']['sha256'])
    if pending_key not in cache:
        cache[pending_key] = load_pending_sources(pending_path,
            tokenizer_fingerprint=original.tokenizer_fingerprint)
    pending = cache[pending_key]
    if any(pending['rows'].get(entry['id'], {}).get('binding') != entry['binding']
           for entry in original.entries):
        raise ValueError('capacity activation must preserve all original whole pending row obligations')
    with coverage_path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    rows = [row for row in rows if row.get('model') == model and row.get('split') == 'train'
            and row.get('partition') == 'remaining-long']
    rows.sort(key=lambda row: row['partition_row_index'])
    expected = receipt['counts'][model]['train:remaining-long']['rows']
    if (len(rows) != expected or len(original) != expected
            or [row['partition_row_index'] for row in rows] != list(range(expected))
            or len({row['id'] for row in rows}) != expected):
        raise ValueError('capacity activation remaining-long coverage is incomplete')
    kept = []
    source_positions = set()
    for row, entry in zip(rows, original.entries, strict=True):
        origin = row['source']
        declarations = [v for v in receipt['source_bindings']
            if v['path'] == origin['path'] and v['sha256'] == origin['sha256']
            and v.get('split') == 'train' and Path(v['path']).parent.name == 'train'
            and Path(v['path']).parent.parent.name == model]
        position = (origin['path'], origin['sha256'], origin['row_index'])
        if (len(declarations) != 1 or type(origin['row_index']) is not int
                or not 0 <= origin['row_index'] < declarations[0]['rows']
                or position in source_positions
                or any(not isinstance(row.get(key), str) or re.fullmatch('[0-9a-f]{64}', row[key]) is None
                       for key in ('source_graph_sha256','source_row_bytes_sha256'))):
            raise ValueError('capacity activation coverage lost original source row provenance')
        source_positions.add(position)
        proof = proofs.get((model, row['id']))
        if (proof is None or proof['split'] != 'train'
                or proof['graph_sha256'] != row['source_graph_sha256']
                or proof['tokens'] != row['geometry']['forward_tokens']):
            raise ValueError('capacity activation coverage differs from ordinary all-row admission proof')
        geometry = row['geometry']
        if (row['id'] != entry['id'] or entry.get('communication') is not False
                or set(geometry) != {'forward_tokens', 'kv_slots'}
                or any(type(v) is not int or v < 1 for v in geometry.values())
                or geometry['forward_tokens'] != entry['forward_tokens']
                or geometry['kv_slots'] != geometry['forward_tokens']):
            raise ValueError('capacity activation ordinary coverage geometry differs')
        if all(geometry[key] <= limits[key] for key in limits):
            kept.append((row, entry))
    if not kept:
        raise ValueError('capacity activation has no whole fitting rows')
    selected = preparation.load(part['source'], split='train')
    strip = lambda entry: {k: v for k, v in entry.items() if k != 'training_priority'}
    if [strip(v) for v in selected.entries] != [strip(v) for _, v in kept]:
        raise ValueError('capacity activation source must contain all and only fitting whole rows')
    provenance = [part['registered_manifest'], part['receipt'], part['coverage'], part['original_source'], part['resident_current'], part['pending_obligations'], part['partition_verifier'],
                  part['admission_audit'], part['render_samples'], part['row_proof']]
    if len(kept) != expected:
        # The existing exact filter verifies raw-byte retention and projects the
        # authenticated original FCB index; no graph is cropped or re-rendered.
        from scripts.neuralese_data.withdraw_native_graph_targets import continue_filtered_index
        proof_path = bound(part['filter_proof'])
        proof = json.loads(proof_path.read_text())
        if (Path(proof['output']['path']).resolve() != selected.path.resolve()
                or proof['output']['sha256'] != selected.file_sha256):
            raise ValueError('capacity activation filter output differs')
        witness = part['original_index']
        cache_path = bound(witness['cache'])
        cached = pickle.loads(cache_path.read_bytes())
        from ..common.hashing import canonical_json_sha256_hex as digest
        leaves = original.children if original.kind == 'manifest' else [original]
        expected_witnesses = _source_witness(original, package)
        if (len(leaves) != 1 or leaves[0].kind == 'manifest'
                or len(expected_witnesses) != 1
                or cached.get('key') != expected_witnesses[0]['key']
                or cached['key']['source_sha256'] != leaves[0].file_sha256
                or cached['metadata_sha256'] != witness['metadata_sha256']
                or digest(cached['entries']) != witness['metadata_sha256']
                or [strip(v) for v in cached['entries']] != [strip(v) for v in original.entries]):
            raise ValueError('capacity activation original index differs from remaining-long source')
        from scripts.neuralese_data.withdraw_native_graph_targets import _raw
        removed = {row['id']: row for row in proof['removed']}
        excluded = {row['id']: entry for row, entry in zip(rows, original.entries, strict=True)
                    if not all(row['geometry'][key] <= limits[key] for key in limits)}
        if (set(removed) != set(excluded)
                or any(removed[k]['binding'] != excluded[k]['binding'] for k in removed)):
            raise ValueError('capacity activation may exclude only over-frontier whole rows')
        retained_digest = hashlib.sha256()
        covered_rows = {row['id']: row for row in rows}
        raw_seen = set()
        with _raw(leaves[0].path) as before, _raw(selected.path) as after:
            for raw in before:
                if raw.strip():
                    graph = json.loads(raw); ident = graph['id']; covered = covered_rows.get(ident)
                    if (covered is None or ident in raw_seen
                            or hashlib.sha256(raw).hexdigest() != covered['source_row_bytes_sha256']
                            or digest(graph) != covered['source_graph_sha256']
                            or graph.get('split') != 'train' or len(graph.get('nodes', [])) != 1
                            or graph.get('external_states') or graph.get('edges')
                            or graph['nodes'][0].get('imports') or graph['nodes'][0].get('exports')
                            or (graph.get('conversion') or {}).get('role_interpretation') not in ('ordinary-text', 'whole-trajectory-chat-roles')):
                        raise ValueError('capacity activation original whole ordinary row hashes/topology differ')
                    raw_seen.add(ident)
                if raw.strip() and ident in removed:
                    if hashlib.sha256(raw).hexdigest() != removed[ident]['raw_row_sha256']:
                        raise ValueError('capacity activation excluded original row bytes differ')
                    continue
                if next(after, None) != raw:
                    raise ValueError('capacity activation changed retained original row bytes')
                retained_digest.update(raw)
            if (raw_seen != set(covered_rows) or next(after, None) is not None
                    or retained_digest.hexdigest() != proof['retained_raw_sha256']):
                raise ValueError('capacity activation retained output bytes differ')
        continue_filtered_index(bound(witness['cache']), proof,
            index_sha256=witness['cache']['sha256'], metadata_sha256=witness['metadata_sha256'],
            output_path=selected.path)
        provenance.extend([part['filter_proof'], witness['cache']])
    elif binding(selected.path)['sha256'] != part['original_source']['sha256']:
        raise ValueError('unfiltered capacity activation must preserve original source bytes')
    maxima = {}
    for index, (row, entry) in enumerate(kept):
        descriptor = {'index': index, 'id': entry['id'], 'binding': entry['binding'], **row['geometry']}
        for key in limits:
            if key not in maxima or descriptor[key] > maxima[key][key]:
                maxima[key] = descriptor
    return selected, maxima, provenance, len(kept)
