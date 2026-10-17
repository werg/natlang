"""Exact namespace and packet custody for existing teacher admission operations."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


def file_binding(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def read_bound(reference):
    path = Path(reference['path'])
    raw = path.read_bytes()
    if (hashlib.sha256(raw).hexdigest() != reference.get('sha256')
            or type(reference.get('bytes')) is not int or len(raw) != reference['bytes']):
        raise ValueError('teacher admission input differs from its exact file binding')
    return json.loads(raw)


def _exact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


@dataclass(frozen=True)
class Tranche:
    batch: int
    number: int
    attempt: int | None = None

    def __post_init__(self):
        if type(self.batch) is not int or self.batch < 21 or type(self.number) is not int or self.number < 0:
            raise ValueError('teacher tranche needs actual batch and nonnegative tranche numbers')
        if self.attempt is not None and (type(self.attempt) is not int or self.attempt < 1):
            raise ValueError('teacher repair attempt must be an explicit positive integer')

    @property
    def batch_name(self):
        return f'batch-{self.batch:04d}'

    @property
    def name(self):
        return f'tranche-{self.number:04d}'

    @property
    def archive_name(self):
        return f'teacher-{self.batch_name}-{self.name}'

    @property
    def corpus_id(self):
        suffix = (f'batch{self.batch:04d}-repair{self.attempt:04d}-' if self.attempt is not None
                  else '' if self.batch == 21 else f'batch{self.batch:04d}-')
        return f'qwen-verified-positive-native-graphs-20261017-recovery7853-{suffix}tranche{self.number:04d}'

    def exports(self, spool):
        root = Path(spool)
        if self.attempt is not None:
            return root / 'repair-exports' / f'{self.batch_name}-attempt-{self.attempt:04d}'
        return root / self.batch_name

    def admission(self, spool, *, recovery=False):
        root = Path(spool) / ('admission-recovery' if recovery else 'admission')
        if self.attempt is not None:
            return root / 'repair-attempts' / self.batch_name / f'attempt-{self.attempt:04d}' / self.name
        if self.batch != 21:
            root /= self.batch_name
        return root / self.name


def packet_references(ready_reference, collection):
    """Resolve the actual immutable selected packet, preserving its remote labels.

    The canonical converter subsequently authenticates program bodies, execution
    preparation, source/actor binding, outcomes and current split protection.
    """
    ready = read_bound(ready_reference)
    matches = []
    for entry in ready['entries']:
        prepared = read_bound(entry['qualified_packet'])
        if prepared['cases']['sha256'] != collection.get('source_cases_sha256'):
            continue
        references = prepared['references']
        if _exact(prepared['cases']) != _exact(references['selected_cases']):
            raise ValueError('qualified packet has conflicting cases references')
        qualification = read_bound(prepared['qualification'])
        if (qualification.get('schema') != 'natlang.existing-native-reference-qualification/1'
                or qualification.get('passed') is not True
                or qualification.get('training_admission') is not False
                or qualification.get('model_calls') != 0
                or _exact(qualification.get('qualified_cases_binding')) != _exact(prepared['cases'])):
            raise ValueError('teacher packet is not the unchanged reference-qualified source')
        values = {key: read_bound(ref) for key, ref in references.items() if key != 'selected_cases'}
        # Cases are JSONL, not a JSON object. Verify their bytes without decoding
        # another giant program collection; canonical packet_binding does that.
        cases = file_binding(references['selected_cases']['path'])
        if _exact(cases) != _exact(references['selected_cases']):
            raise ValueError('qualified packet cases changed')
        selection = values['selection']
        if (_exact(selection) != _exact(collection.get('protected_selection'))
                or selection.get('cases_sha256') != cases['sha256']
                or selection.get('training_admission') is not False):
            raise ValueError('collection selected tasks differ from the qualified ready packet')
        provenance = collection.get('source_provenance') or {}
        for key in ('selection', 'global_protection_join'):
            original = provenance.get(key) or {}
            if (original.get('sha256') != references[key]['sha256']
                    or not isinstance(original.get('path'), str)):
                raise ValueError('collection packet provenance differs from its qualified source')
        matches.append({'prepared': entry['qualified_packet'], 'qualification': prepared['qualification'],
                        'references': references, 'remote_references': provenance})
    if len(matches) != 1:
        raise ValueError('collection needs exactly one matching qualified ready packet')
    # Guard the declaration after all per-packet reads, too.
    if _exact(file_binding(ready_reference['path'])) != _exact(ready_reference):
        raise ValueError('ready packet declaration changed during source resolution')
    return matches[0]


def admitted_predecessors(root, corpus_id):
    """Return the registered positive corpus closure in registry order.

    No directory glob or unregistered partial output grants predecessor status.
    The manifest must bind both complete train views and the admitted validator.
    """
    root = Path(root)
    registry_path = root / 'training/neuralese_corpora.json'
    before = file_binding(registry_path)
    registry = read_bound(before)
    rows = []
    seen = set()
    prefix = 'qwen-verified-positive-native-graphs-20261017-recovery7853-'
    for entry in registry['corpora']:
        ident = entry['id']
        if ident in seen:
            raise ValueError('teacher registry repeats a corpus identity')
        seen.add(ident)
        if ident == corpus_id or not ident.startswith(prefix) or entry.get('training_admission') is not True:
            continue
        folder = root / entry['path']
        manifest = json.loads((root / 'training/corpus-manifests' / (ident + '.json')).read_bytes())
        if manifest.get('id') != ident or manifest.get('path') != entry['path']:
            raise ValueError('teacher predecessor manifest differs from its registry entry')
        by_path = {item['path']: item for item in manifest['files']}
        if len(by_path) != len(manifest['files']):
            raise ValueError('teacher predecessor manifest repeats file paths')
        validation_ref = by_path['validation-proof.json']
        validation = read_bound({**validation_ref, 'path': str(folder / 'validation-proof.json')})
        if (validation.get('status') != 'passed' or validation.get('training_admission') is not True):
            raise ValueError('teacher predecessor lacks actual admission')
        outputs = {}
        for model in ('lfm25', 'mellum21'):
            relative = model + '/train/native-graphs.jsonl.zst'
            reference = {**by_path[relative], 'path': str(folder / relative)}
            actual = file_binding(reference['path'])
            if (any(actual[key] != reference[key] for key in ('sha256', 'bytes'))
                    or any(actual[key] != validation['models'][model]['outputs']['train'][key]
                           for key in ('sha256', 'bytes'))):
                raise ValueError('teacher predecessor graph differs from its admission/manifest')
            outputs[model] = actual
        rows.append({'id': ident, 'outputs': outputs})
    if file_binding(registry_path) != before:
        raise ValueError('teacher registry changed during predecessor resolution')
    return rows


def preceding_tranches(root, spool, tranche):
    """Account for every earlier tranche in this batch, including all-rejected ones."""
    admitted = admitted_predecessors(root, tranche.corpus_id)
    identities = {row['id'] for row in admitted}
    rejected = []
    for number in range(tranche.number):
        earlier = Tranche(tranche.batch, number, tranche.attempt)
        if earlier.corpus_id in identities:
            continue
        directory = earlier.admission(spool)
        reference = file_binding(directory / 'canonical-source-verification.json')
        proof = read_bound(reference)
        source = file_binding(directory / 'native-trajectories.jsonl')
        outputs = proof.get('rejected_outputs')
        if (proof.get('status') != 'passed' or proof.get('training_admission') is not False
                or type(proof.get('rows')) is not int or proof['rows'] != 0
                or type(proof.get('selected_outputs')) is not int or proof['selected_outputs'] < 1
                or not isinstance(outputs, list) or len(outputs) != proof['selected_outputs']
                or source['bytes'] != 0 or source['sha256'] != hashlib.sha256(b'').hexdigest()
                or any(proof['source'].get(key) != source[key] for key in ('sha256', 'bytes'))
                or len({row.get('key') for row in outputs}) != len(outputs)):
            raise ValueError('earlier teacher tranche is neither admitted nor exactly empty/rejected')
        restored = (earlier.admission(spool, recovery=True) / 'restored').resolve(strict=True)
        for output in outputs:
            path = (restored / output['source_output']).resolve(strict=True)
            if (not path.is_relative_to(restored) or not output.get('reason')
                    or file_binding(path)['sha256'] != output.get('sha256')):
                raise ValueError('earlier rejected teacher outcome differs from preserved source')
        if file_binding(reference['path']) != reference:
            raise ValueError('earlier rejection proof changed during validation')
        rejected.append({'batch': tranche.batch, 'number': number, 'proof': reference,
                         'reason': 'Every original outcome explicitly rejected; no source fabricated'})
    return admitted, rejected


def materialization_partition(raw_rows, decision_keys, omissions, emitted_keys, deduplicated_keys=()):
    """Conserve every original action when a materializer holds whole source rows.

    This grants no decision approval: the existing materializer, converter and
    graph verifier remain responsible for each emitted target. Missing source
    rows remain explicit repair debt and cannot contribute a graph.
    """
    wanted = {(ident, index) for ident, row in raw_rows.items()
              for index in range(len(row['trajectory']))}

    def keys(values):
        result = list(values)
        if (any(not isinstance(key, tuple) or len(key) != 2
                or not isinstance(key[0], str) or type(key[1]) is not int for key in result)
                or len(set(result)) != len(result) or not set(result) <= wanted):
            raise ValueError('materialization partition has duplicate or unknown action identities')
        return set(result)

    decisions = keys(decision_keys)
    emitted, deduplicated = keys(emitted_keys), keys(deduplicated_keys)
    omitted = keys((row.get('source_row_id'), row.get('teacher_turn_index')) for row in omissions)
    accepted = {ident for ident, _ in decisions}
    held = set(raw_rows) - accepted
    if any((ident, index) not in decisions for ident in accepted
           for index in range(len(raw_rows[ident]['trajectory']))):
        raise ValueError('materializer partially lost an accepted source trajectory')
    held_keys = {key for key in wanted if key[0] in held}
    if (not held_keys <= omitted or emitted & held_keys or deduplicated & held_keys
            or emitted & omitted or deduplicated & omitted or emitted & deduplicated
            or emitted | omitted | deduplicated != wanted):
        raise ValueError('materialization action partition does not conserve the full source')
    missing_reason = 'shared native materializer did not qualify this original decision'
    for omission in omissions:
        if omission['source_row_id'] in held:
            expected = {'source_row_id': omission['source_row_id'],
                        'teacher_turn_index': omission['teacher_turn_index'],
                        'reason': missing_reason, 'decision': {'reason': missing_reason}}
            if _exact(omission) != _exact(expected):
                raise ValueError('held source action lacks its exact materializer omission')
    return {'source_rows': len(raw_rows), 'materialized_source_rows': len(accepted),
            'repair_source_rows': sorted(held), 'repair_source_actions': len(held_keys),
            'source_actions': len(wanted), 'emitted_actions': len(emitted),
            'deduplicated_actions': len(deduplicated),
            'other_omitted_actions': len(omitted - held_keys),
            'training_admission': False}
