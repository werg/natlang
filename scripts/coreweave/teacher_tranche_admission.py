"""Exact namespace and packet custody for existing teacher admission operations."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import errno
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


def completed_source_phase(directory, recovery_reference, execution_reference):
    """Snapshot successful source custody; current conversion/admission remain required.

    Call only after the source operator succeeds, then durably store this value.
    On resume use verify_completed_source_phase with that bound receipt. Runtime
    rebuilds must not rewrite historical derived quality-verifier metadata.
    """
    directory = Path(directory)
    states, references = {}, {}

    def state(path):
        value = path.stat()
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    def bound(reference):
        path = Path(reference['path'])
        before = state(path)
        actual = file_binding(path)
        if (type(reference.get('bytes')) is not int
                or any(actual[key] != reference[key] for key in ('sha256', 'bytes'))
                or state(path) != before):
            raise ValueError('completed source phase input changed')
        prior = references.get(str(path))
        if prior is not None and prior != actual:
            raise ValueError('completed source phase has conflicting bindings')
        references[str(path)], states[str(path)] = actual, before
        return actual

    def document(reference):
        return read_bound(bound(reference))

    def local(path):
        return document(file_binding(path))

    proof = local(directory / 'canonical-source-verification.json')
    transfer = local(directory / 'transfer-receipt.json')
    local(directory / 'batch-source-bindings.json')
    source = bound(proof['source'])
    if (Path(source['path']) != directory / 'native-trajectories.jsonl'
            or proof.get('schema') != 'natlang.teacher-live-tranche-canonical-source-verification/1'
            or proof.get('status') != 'passed' or proof.get('training_admission') is not False
            or type(proof.get('rows')) is not int or proof['rows'] < 1
            or transfer.get('schema') != 'natlang.verified-teacher-trajectory-transfer/1'
            or transfer.get('training_admission') is not False
            or any(transfer['combined'].get(key) != source[key] for key in ('path', 'sha256', 'bytes'))
            or transfer['combined'].get('rows') != proof['rows']):
        raise ValueError('completed source phase lacks exact positive source verification')
    ids = set()
    with Path(source['path']).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            outcome = row.get('outcome') or {}
            checks = outcome.get('checks')
            if (not isinstance(row.get('id'), str) or row['id'] in ids
                    or outcome.get('status') != 'done' or outcome.get('accepted') is not True
                    or (outcome.get('oracle') or {}).get('accepted') is not True
                    or not isinstance(checks, dict) or not checks
                    or any(value is not True for value in checks.values())):
                raise ValueError('completed source phase contains a nonpositive or repeated row')
            ids.add(row['id'])
    if len(ids) != proof['rows']:
        raise ValueError('completed source phase row count changed')
    for reference in transfer['files']:
        bound(reference)
    recovery = document(recovery_reference)
    if (recovery['restoration'].get('status') != 'passed'
            or recovery['archive_identity']['sha256'] != transfer['private_export']['archive_sha256']):
        raise ValueError('completed source phase archive custody differs')
    private = local(Path(recovery['private_receipt']))
    if any(type(recovery['archive_identity'].get(key)) is not type(private.get('identity', {}).get(key))
           or recovery['archive_identity'][key] != private['identity'][key]
           or recovery['archive_identity'][key] != private.get('independent_verification', {}).get(key)
           for key in ('sha256', 'bytes')):
        raise ValueError('completed source phase lacks matching private full-GET custody')
    restored = Path(recovery['restored'])
    manifest = local(restored / 'tranche-manifest.json')

    def member(name, reference):
        path = restored / name
        if not path.resolve().is_relative_to(restored.resolve()):
            raise ValueError('completed source member escapes its restored archive')
        bound({**reference, 'path': str(path)})

    for name, reference in manifest['context'].items():
        member('context/' + name, reference)
    for item in manifest['outputs']:
        member(item['output'], {'sha256': item['output_sha256'], 'bytes': item['output_bytes']})
        member(item['output_manifest'], {'sha256': item['manifest_sha256'], 'bytes': item['manifest_bytes']})
        for reference in (item.get('held_payloads') or {}).values():
            member(reference['path'], reference)
    if (type(proof.get('selected_outputs')) is not int
            or proof['selected_outputs'] != len(manifest['outputs'])):
        raise ValueError('completed source phase lost selected output custody')
    execution = document(execution_reference)
    inventory = document(execution['inventory'])
    for reference in [execution['pipeline'], inventory['helper'], *inventory['operators']]:
        bound(reference)
    for path, before in states.items():
        if state(Path(path)) != before:
            raise ValueError('completed source phase input changed during verification')
    return {'schema': 'natlang.teacher-completed-source-phase/1',
            'directory': str(directory), 'source_rows': len(ids),
            'recovery': recovery_reference, 'execution': execution_reference,
            'files': list(references.values()), 'training_admission': False,
            'requires_fresh_conversion_and_admission': True}


def verify_completed_source_phase(receipt_reference, directory):
    """Authenticate every original byte before skipping completed source work."""
    receipt = read_bound(receipt_reference)
    expected = completed_source_phase(directory, receipt['recovery'], receipt['execution'])
    if _exact(receipt) != _exact(expected):
        raise ValueError('completed source phase receipt differs from original custody')
    return receipt


LEGACY_TEACHER_NAMESPACE = 'qwen-verified-positive-native-graphs-20261017-recovery7853'


@dataclass(frozen=True)
class Tranche:
    batch: int
    number: int
    attempt: int | None = None
    quality_view: str | None = None
    teacher_namespace: str = LEGACY_TEACHER_NAMESPACE

    def __post_init__(self):
        if (not isinstance(self.teacher_namespace, str)
                or re.fullmatch(r'[a-z][a-z0-9]*(?:-[a-z0-9]+)*', self.teacher_namespace) is None):
            raise ValueError('teacher admission needs an explicit safe teacher namespace')
        minimum = 21 if self.teacher_namespace == LEGACY_TEACHER_NAMESPACE else 0
        if type(self.batch) is not int or self.batch < minimum or type(self.number) is not int or self.number < 0:
            raise ValueError('teacher tranche needs actual batch and nonnegative tranche numbers')
        if self.attempt is not None and (type(self.attempt) is not int or self.attempt < 1):
            raise ValueError('teacher repair attempt must be an explicit positive integer')
        if self.quality_view is not None and (type(self.quality_view)is not str or
                re.fullmatch(r'[a-z][a-z0-9]*(?:-[a-z0-9]+)*',self.quality_view)is None):
            raise ValueError('teacher quality view needs an explicit safe name')

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
                  else '' if self.batch == 21 and self.teacher_namespace == LEGACY_TEACHER_NAMESPACE
                  else f'batch{self.batch:04d}-')
        ident=f'{self.teacher_namespace}-{suffix}tranche{self.number:04d}'
        return ident+(('-quality-'+self.quality_view)if self.quality_view else'')

    def exports(self, spool):
        root = Path(spool)
        if self.attempt is not None:
            return root / 'repair-exports' / f'{self.batch_name}-attempt-{self.attempt:04d}'
        return root / self.batch_name

    def admission(self, spool, *, recovery=False):
        root = Path(spool) / ('admission-recovery' if recovery else 'admission')
        if self.attempt is not None:
            result=root / 'repair-attempts' / self.batch_name / f'attempt-{self.attempt:04d}' / self.name
        else:
            legacy_anchor = self.batch == 21 and self.teacher_namespace == LEGACY_TEACHER_NAMESPACE
            result=(root if legacy_anchor else root / self.batch_name)/self.name
        if self.quality_view and not recovery:
            return Path(spool)/'admission-quality-views'/self.quality_view/result.relative_to(root)
        return result


def packet_references(ready_reference, collection):
    """Resolve the actual immutable selected packet, preserving its remote labels.

    The canonical converter subsequently authenticates program bodies, execution
    preparation, source/actor binding, outcomes and current split protection.
    """
    ready = read_bound(ready_reference)
    matches = []
    selection_mismatch = provenance_mismatch = False
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
                # A verified internal relocation changes only the path field.
                # The original packet and its qualification are hash-bound by
                # the relocation manifest; compare the immutable content
                # identity here so the original proof remains valid.
                or any((qualification.get('qualified_cases_binding') or {}).get(key) != prepared['cases'].get(key)
                       for key in ('sha256', 'bytes'))):
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
            selection_mismatch = True
            continue
        provenance = collection.get('source_provenance') or {}
        candidate_matches = True
        for key in ('selection', 'global_protection_join'):
            original = provenance.get(key) or {}
            if (original.get('sha256') != references[key]['sha256']
                    or not isinstance(original.get('path'), str)):
                provenance_mismatch = True
                candidate_matches = False
                break
        if not candidate_matches:
            continue
        matches.append({'prepared': entry['qualified_packet'], 'qualification': prepared['qualification'],
                        'references': references, 'remote_references': provenance})
    if not matches and provenance_mismatch:
        raise ValueError('collection packet provenance differs from its qualified source')
    if not matches and selection_mismatch:
        raise ValueError('collection selected tasks differ from the qualified ready packet')
    if len(matches) != 1:
        raise ValueError('collection needs exactly one matching qualified ready packet')
    # Guard the declaration after all per-packet reads, too.
    if _exact(file_binding(ready_reference['path'])) != _exact(ready_reference):
        raise ValueError('ready packet declaration changed during source resolution')
    return matches[0]


def _predecessor_models(cache_contracts, by_path):
    """The declared student views a predecessor must supply: the converting pair, else every view it carries."""
    from natlang_neuralese.common import line_backbones
    if cache_contracts is not None:
        unknown = set(cache_contracts) - set(line_backbones.BACKBONES)
        if unknown:
            raise ValueError('registered cache contracts name undeclared student views: ' + ', '.join(sorted(unknown)))
        return tuple(cache_contracts)
    return tuple(model for model in line_backbones.BACKBONES
                 if model + '/train/native-graphs.jsonl.zst' in by_path)


def admitted_predecessors(root, corpus_id, *, namespaces=(LEGACY_TEACHER_NAMESPACE,), relocation_manifest=None,
                          cache_contracts=None):
    """Return the registered positive corpus closure in registry order.

    No directory glob or unregistered partial output grants predecessor status.
    The manifest must bind both complete train views and the admitted validator.
    """
    root = Path(root)
    registry_path = root / 'training/neuralese_corpora.json'
    before = file_binding(registry_path)
    registry = read_bound(before)
    rows = []
    relocated = {}
    relocation_binding = None
    if relocation_manifest is not None:
        relocation_binding = file_binding(relocation_manifest)
        relocation = read_bound(relocation_binding)
        if relocation.get('schema') != 'natlang.teacher-predecessor-relocations/1' or relocation.get('training_admission') is not False:
            raise ValueError('teacher predecessor relocation declaration is not a non-admission transport record')
        if relocation.get('registry') != before:
            raise ValueError('teacher predecessor relocation declaration is bound to a different registry')
        for item in relocation.get('corpora', []):
            ident = item.get('id')
            if not isinstance(ident, str) or ident in relocated:
                raise ValueError('teacher predecessor relocation repeats or omits a corpus identity')
            relocated[ident] = item
    seen = set()
    prefixes = tuple(namespace + '-' for namespace in namespaces)
    for entry in registry['corpora']:
        ident = entry['id']
        if ident in seen:
            raise ValueError('teacher registry repeats a corpus identity')
        seen.add(ident)
        if ident == corpus_id or not ident.startswith(prefixes) or entry.get('training_admission') is not True:
            continue
        manifest_path = root / 'training/corpus-manifests' / (ident + '.json')
        manifest_ref = file_binding(manifest_path)
        manifest = read_bound(manifest_ref)
        if manifest.get('id') != ident or manifest.get('path') != entry['path']:
            raise ValueError('teacher predecessor manifest differs from its registry entry')
        by_path = {item['path']: item for item in manifest['files']}
        if len(by_path) != len(manifest['files']):
            raise ValueError('teacher predecessor manifest repeats file paths')
        folder = root / entry['path']
        try:
            folder_present = folder.is_dir()
        except OSError as error:
            if error.errno not in (errno.ENOENT, errno.EIO):
                raise
            folder_present = False
        if not folder_present and cache_contracts is not None:
            # An unavailable admitted package can still participate in exact
            # deduplication through its manifest-bound current-law content
            # indexes. The registry grants predecessor identity; the index
            # grants only byte-equivalent content fingerprints for that exact
            # train member and model/tokenizer/policy contract.
            from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import registered_cached_content_index
            outputs = {}
            for model in _predecessor_models(cache_contracts, by_path):
                contract = cache_contracts.get(model) or {}
                fingerprint, policy = contract.get('tokenizer_fingerprint'), contract.get('policy_sha256')
                if not isinstance(fingerprint, str) or not isinstance(policy, str):
                    raise ValueError('registered cache predecessor lacks both exact model contracts')
                cached, _ = registered_cached_content_index(root, ident, model, fingerprint, policy,
                    registry_ref=before, registry_entry=entry, manifest_ref=manifest_ref,
                    manifest=manifest, verify_bindings=False)
                if cached.get('manifest') != manifest_ref:
                    raise ValueError('registered content cache points to a different exact corpus manifest')
                outputs[model] = cached
            rows.append({'id': ident, 'outputs': outputs, 'cache_only': True,
                         'registry_entry': entry, 'manifest': manifest_ref})
            continue
        if not folder_present:
            item = relocated.get(ident)
            if item is None:
                raise FileNotFoundError(f'registered teacher predecessor has no canonical or declared relocated bytes: {ident}')
            if item.get('registry_entry') != entry or item.get('manifest') != manifest_ref:
                raise ValueError('teacher predecessor relocation differs from its exact registry/manifest binding')
            folder = Path(item.get('directory', ''))
            if not folder.is_absolute():
                raise ValueError('teacher predecessor relocation directory is not absolute')
            members = item.get('members')
            expected_paths = {member['path'] for member in manifest['files']}
            if not isinstance(members, list) or {member.get('path') for member in members} != expected_paths or len(members) != len(expected_paths):
                raise ValueError('teacher predecessor relocation does not bind every exact manifest member')
            for member in members:
                ref = next(source for source in manifest['files'] if source['path'] == member['path'])
                if member.get('manifest_member') != ref:
                    raise ValueError('teacher predecessor relocated member differs from its manifest')
                actual = file_binding(folder / member['path'])
                if any(actual.get(key) != ref.get(key) for key in ('sha256', 'bytes')):
                    raise ValueError('teacher predecessor relocated member content differs from its registry manifest')
        validation_ref = by_path['validation-proof.json']
        validation = read_bound({**validation_ref, 'path': str(folder / 'validation-proof.json')})
        if (validation.get('status') != 'passed' or validation.get('training_admission') is not True):
            raise ValueError('teacher predecessor lacks actual admission')
        outputs = {}
        for model in _predecessor_models(cache_contracts, by_path):
            relative = model + '/train/native-graphs.jsonl.zst'
            if relative not in by_path:
                raise ValueError(f'teacher predecessor {ident} has no {model} view; register its exact twin first')
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
    if relocation_binding is not None and file_binding(relocation_binding['path']) != relocation_binding:
        raise ValueError('teacher predecessor relocation declaration changed during resolution')
    return rows


def _reviewed_predecessor_source(root, spool, original, reviewed):
    """Verify a registered same-namespace view accounts for the whole raw source."""
    manifest = json.loads((Path(root) / 'training/corpus-manifests' /
                           (reviewed.corpus_id + '.json')).read_bytes())
    if manifest.get('id') != reviewed.corpus_id:
        raise ValueError('reviewed predecessor manifest identity differs')
    member = next(item for item in manifest['files'] if item['path'] == 'validation-proof.json')
    validation = read_bound({**member, 'path': str(Path(root) / manifest['path'] / 'validation-proof.json')})
    original_dir, reviewed_dir = original.admission(spool), reviewed.admission(spool)
    source, selected = file_binding(original_dir / 'native-trajectories.jsonl'), file_binding(reviewed_dir / 'native-trajectories.jsonl')
    canonical_ref = file_binding(original_dir / 'canonical-source-verification.json')
    canonical = read_bound(canonical_ref)
    partition = validation.get('materialization_partition') or {}
    count, actions = canonical.get('rows'), canonical.get('teacher_action_turns')
    if (canonical.get('status') != 'passed' or canonical.get('training_admission') is not False
            or validation.get('status') != 'passed' or validation.get('training_admission') is not True
            or any(type(value) is not int for value in (validation.get('source_rows'), partition.get('source_rows'),
                validation.get('teacher_action_turns'), partition.get('source_actions')))
            or type(count) is not int or count < 1
            or type(actions) is not int or actions < count
            or any(canonical.get('source', {}).get(k) != source[k] or selected[k] != source[k]
                   or validation.get('source', {}).get(k) != source[k] for k in ('sha256', 'bytes'))
            or validation.get('source_rows') != count or partition.get('source_rows') != count
            or validation.get('teacher_action_turns') != actions or partition.get('source_actions') != actions
            or validation.get('canonical_source_verification') != file_binding(reviewed_dir / 'canonical-source-verification.json')):
        raise ValueError('reviewed predecessor does not cover exact complete original source')
    if (file_binding(source['path']) != source or file_binding(selected['path']) != selected
            or file_binding(canonical_ref['path']) != canonical_ref):
        raise ValueError('reviewed predecessor source changed during validation')


def preceding_tranches(root, spool, tranche, *, relocation_manifest=None, cache_contracts=None,
                       accounted_internal_boundaries=()):
    """Account for every earlier tranche in this batch, including all-rejected ones."""
    admitted = admitted_predecessors(root, tranche.corpus_id,
        namespaces=tuple(dict.fromkeys((LEGACY_TEACHER_NAMESPACE, tranche.teacher_namespace))),
        relocation_manifest=relocation_manifest, cache_contracts=cache_contracts)
    identities = {row['id'] for row in admitted}
    rejected = []
    accounted_internal_boundaries = set(accounted_internal_boundaries)
    for number in range(tranche.number):
        if (tranche.batch, number) in accounted_internal_boundaries:
            continue
        earlier = Tranche(tranche.batch, number, tranche.attempt, teacher_namespace=tranche.teacher_namespace)
        if earlier.corpus_id in identities:
            continue
        if tranche.quality_view is not None:
            reviewed = Tranche(tranche.batch, number, tranche.attempt, quality_view=tranche.quality_view,
                               teacher_namespace=tranche.teacher_namespace)
            if reviewed.corpus_id in identities:
                _reviewed_predecessor_source(root, spool, earlier, reviewed)
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


def guided_static_root_keys(raw_rows):
    """Identify only original unsampled reference roots for omission accounting."""
    result = set()
    for ident, row in raw_rows.items():
        guidance = row.get('collection_guidance') or {}
        if guidance.get('training_admission') is not False:
            continue
        root = guidance.get('root_action') or {}
        for index, turn in enumerate(row['trajectory']):
            provenance = turn.get('action_provenance') or {}
            if (provenance.get('kind') == 'authored_reference_root_eval'
                    and provenance.get('sampled') is False
                    and root.get('kind') == 'authored_reference_eval' and root.get('sampled') is False
                    and root.get('status') == 'finished'
                    and all(provenance.get(k) == root.get(k) for k in
                            ('source', 'source_program_id', 'code_sha256'))
                    and provenance.get('source') == 'curriculum.reference.root'
                    and provenance.get('source_program_id') == (row.get('task', {}).get('program_ir') or {}).get('id')
                    and isinstance(provenance.get('code_sha256'), str)
                    and len(provenance['code_sha256']) == 64
                    and all(c in '0123456789abcdef' for c in provenance['code_sha256'])):
                result.add((ident, index))
    return result


def materialization_partition(raw_rows, decision_keys, omissions, emitted_keys, deduplicated_keys=(),
                              *, partial_parent_rows=()):
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
    partial = set(partial_parent_rows)
    if not partial <= accepted:
        raise ValueError('partial parent evidence is absent from materialized source')
    static_roots = guided_static_root_keys(raw_rows)
    if any((ident, index) not in decisions and (ident, index) not in static_roots for ident in accepted
           for index in range(len(raw_rows[ident]['trajectory']))):
        raise ValueError('materializer partially lost an accepted source trajectory')
    if not static_roots <= omitted or static_roots & (emitted | deduplicated | decisions):
        raise ValueError('authored static roots must remain exact non-target omissions')
    held_keys = {key for key in wanted if key[0] in held}
    if (not held_keys <= omitted or emitted & held_keys or deduplicated & held_keys
            or emitted & omitted or deduplicated & omitted or emitted & deduplicated
            or emitted | omitted | deduplicated != wanted):
        raise ValueError('materialization action partition does not conserve the full source')
    missing_reason = 'shared native materializer did not qualify this original decision'
    for omission in omissions:
        if omission['source_row_id'] in held or (omission['source_row_id'], omission['teacher_turn_index']) in static_roots:
            expected = {'source_row_id': omission['source_row_id'],
                        'teacher_turn_index': omission['teacher_turn_index'],
                        'reason': missing_reason, 'decision': {'reason': missing_reason}}
            if _exact(omission) != _exact(expected):
                raise ValueError('held source action lacks its exact materializer omission')
    guided = {ident for ident in accepted if (raw_rows[ident].get('collection_guidance') or {}).get('training_admission') is False}
    return {'source_rows': len(raw_rows), 'materialized_source_rows': len(accepted - partial - guided),
            **({'guided_source_rows': sorted(guided), 'authored_static_root_actions': len(static_roots),
                'guided_parent_traces_admitted': False} if guided else {}),
            **({'partial_source_rows': sorted(partial),
                'partial_source_actions': sum(len(raw_rows[ident]['trajectory']) for ident in partial),
                'partial_parent_tasks_qualified': False} if partial else {}),
            'repair_source_rows': sorted(held), 'repair_source_actions': len(held_keys),
            'source_actions': len(wanted), 'emitted_actions': len(emitted),
            'deduplicated_actions': len(deduplicated),
            'other_omitted_actions': len(omitted - held_keys),
            'training_admission': False}


def verify_teacher_action_accounting(raw_rows, decision_keys, receipt, summary):
    """Authenticate the converter's stricter post-materializer action filter.

    A rendering omission needs its own exact renderer proof; it cannot pass this
    non-rendering verifier merely because a receipt declares a lower count.
    """
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import admitted_teacher_turn, captured_text_emulation_context
    if not isinstance(decision_keys, dict):
        raise ValueError('action accounting requires exact materialized decisions')
    wanted = {(ident, index) for ident, row in raw_rows.items()
              for index in range(len(row['trajectory']))}
    if any(type(key) is not tuple or len(key) != 2 or type(key[0]) is not str
           or type(key[1]) is not int or key not in wanted or not isinstance(value, dict)
           or type(value.get('approved')) is not bool
           for key, value in decision_keys.items()):
        raise ValueError('action accounting has unknown or untyped decisions')
    if (type(summary.get('training_decisions')) is not int
            or summary['training_decisions'] != len(decision_keys)):
        raise ValueError('materializer decision inventory differs from its exact count')
    approved = sum(value['approved'] for value in decision_keys.values())
    if type(summary.get('eligible_decisions')) is not int or summary['eligible_decisions'] != approved:
        raise ValueError('materializer approved count differs from exact decisions')
    rejected, eligible, filtered = {}, set(), []
    for ident, row in raw_rows.items():
        if row.get('id') != ident:
            raise ValueError('action accounting source identity differs')
        for index, turn in enumerate(row['trajectory']):
            accepted, decision = admitted_teacher_turn(row, index, turn, decision_keys)
            key = (ident, index)
            render_omission = None
            if accepted:
                _, _, _, render_omission = captured_text_emulation_context(
                    turn, turn.get('context'), turn.get('tools_offered'))
            if accepted and render_omission is None:
                eligible.add(key)
            else:
                expected = {'source_row_id': ident, 'teacher_turn_index': index,
                            **(render_omission if render_omission is not None else {'reason': decision.get('reason')}),
                            'decision': decision}
                rejected[key] = expected
                if decision_keys.get(key, {}).get('approved') is True:
                    filtered.append(expected)
    actual = {}
    for omission in receipt['action_omissions']:
        if not isinstance(omission, dict):
            raise ValueError('action accounting omission is not an object')
        key = (omission.get('source_row_id'), omission.get('teacher_turn_index'))
        if type(key[1]) is not int or key not in rejected or key in actual:
            raise ValueError('unknown, duplicate or unverified post-admission rendering omission')
        if _exact(omission) != _exact(rejected[key]):
            raise ValueError('action omission differs from current exact admission')
        actual[key] = omission
    if set(actual) != set(rejected):
        raise ValueError('action accounting lost a rejected original action')
    deduplication = receipt['deduplication']
    counts = (receipt['turns'], deduplication['emitted_rows'], deduplication['omitted_rows'])
    if (any(type(value) is not int or value < 0 for value in counts)
            or counts[0] != len(eligible) or counts[1] + counts[2] != len(eligible)
            or len(eligible) + len(actual) != len(wanted)):
        raise ValueError('converted action accounting does not conserve original actions')
    return {'source_actions': len(wanted), 'materializer_approved_actions': approved,
            'converted_eligible_actions': len(eligible), 'omitted_actions': len(actual),
            'post_materializer_rejections': filtered}


def teacher_conversion_profile(receipt):
    """Read explicit ID authority separately from an effective backbone default."""
    if 'explicit_history_reasoning_policy' not in receipt:
        return None
    profile = receipt['explicit_history_reasoning_policy']
    if (profile not in ('keep', 'last_turn_only')
            or (receipt.get('tokenizer', {}).get('history_reasoning') or {}).get('policy') != profile):
        raise ValueError('teacher conversion explicit profile differs from its tokenizer authority')
    return profile


def verify_teacher_model_actions(graphs, receipt, *, raw_rows, raw_row_shas,
                                 source_sha256, decision_keys, hold_ids, partial_tokenizer=None):
    """Shared original-action verifier used by the operational final admission."""
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import teacher_action_graph_id
    profile = teacher_conversion_profile(receipt)
    failure_policy = receipt.get('failed_assistant_history_policy', 'preserve')
    if failure_policy not in ('preserve', 'mask-runtime-failures'):
        raise ValueError('unsupported teacher assistant-history supervision profile')
    if set(raw_rows) != set(raw_row_shas):
        raise ValueError('teacher graph source SHA inventory differs from its source rows')
    partial = {}
    if isinstance(decision_keys, dict):
        for key, decision in decision_keys.items():
            if isinstance(decision, dict) and decision.get('parent_quality_hold') is not None:
                hold = decision['parent_quality_hold']
                if (hold not in ('quality_pending_partial_files', 'quality_pending_partial_agreement')
                        or decision.get('parent_task_qualified') is not False
                        or key[0] not in raw_rows):
                    raise ValueError('invalid partial-parent decision authority')
                previous = partial.setdefault(key[0], hold)
                if previous != hold:
                    raise ValueError('partial-parent authority has conflicting holds')
    ids, targets, source_ids, emitted_keys = set(), set(), set(), []
    source_actions = sum(len(row['trajectory']) for row in raw_rows.values())
    for row in graphs:
        identifier = row['id']
        teacher = row['conversion']['teacher_source']
        source_id, turn = teacher['source_row_id'], teacher['teacher_turn_index']
        if (source_id not in raw_rows or type(turn) is not int
                or not 0 <= turn < len(raw_rows[source_id]['trajectory'])):
            raise ValueError('graph source/turn missing')
        target = teacher['decision_admission']['materialized_decision_id']
        if target != f'{source_id}:decision:{turn:04d}' or identifier in ids:
            raise ValueError('decision identity mismatch or duplicate graph')
        source_row, row_sha = raw_rows[source_id], raw_row_shas[source_id]
        expected = 'text:' + teacher_action_graph_id(source_id, row_sha, turn, profile, failure_policy)
        if (teacher['source_artifact_sha256'] != source_sha256
                or teacher['source_row_sha256'] != row_sha or identifier != expected):
            raise ValueError('graph source SHA or source-bound profile identity mismatch')
        guided = (source_row.get('collection_guidance') or {}).get('training_admission') is False
        restricted = source_id in partial or guided
        if restricted or failure_policy == 'mask-runtime-failures':
            expected_decision = decision_keys[(source_id, turn)] if restricted else None
            if guided and (expected_decision.get('guided_parent_review') is not True
                    or expected_decision.get('parent_trace_qualified') is not False
                    or (source_id, turn) in guided_static_root_keys({source_id: source_row})):
                raise ValueError('guided child graph lacks its exact review or includes a static root')
            if ((restricted and (expected_decision.get('approved') is not True
                    or teacher.get('decision_admission') != expected_decision))
                    or partial_tokenizer is None):
                raise ValueError('partial parent graph lacks exact approved decision and tokenizer')
            from natlang_neuralese.data.native_graph_conversion import text_row_graph
            from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import render_teacher_packet, reviewed_teacher_context_guard
            from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
            if tokenizer_fingerprint(partial_tokenizer) != receipt['tokenizer']['expanded_fingerprint']:
                raise ValueError('partial parent tokenizer differs from exact conversion identity')
            captured = source_row['trajectory'][turn]
            assistant = captured['assistant']
            target_message = {'role': 'assistant', 'content': assistant.get('content', ''),
                              'tool_calls': captured['model_response']['raw_calls']}
            if assistant.get('reasoning'):
                target_message['reasoning_content'] = assistant['reasoning']
            input_guard = reviewed_teacher_context_guard(captured, captured['context'], target_message, captured['tools_offered'])
            from natlang_neuralese.common.hashing import canonical_json_sha256_hex
            if restricted and canonical_json_sha256_hex(input_guard) != expected_decision.get('assistant_history_input_sha256'):
                raise ValueError('partial parent exact approval does not bind original structured input protection')
            packet, omission = render_teacher_packet(partial_tokenizer, captured['context'],
                target_message, captured['tools_offered'], captured)
            if omission is not None:
                raise ValueError('partial parent exact captured render is not qualified')
            (rendered_text, rendered_ids, rendered_boundary), _ = packet
            conversion = row['conversion']
            node = row['nodes'][0]
            ids_for_mask = node['token_ids']
            text = partial_tokenizer.decode(ids_for_mask, skip_special_tokens=False)
            if (text != rendered_text or ids_for_mask != rendered_ids
                    or conversion['source_suffix_start'] != rendered_boundary):
                raise ValueError('partial parent graph changed the captured full context or suffix')
            masked = text_row_graph(dict(id=conversion['source_row_id'], split='train',
                source_groups=row['source_groups'], tokenizer_sha256=receipt['tokenizer']['source_fingerprint'],
                text=text, token_ids=ids_for_mask, supervised_suffix_start=conversion['source_suffix_start']),
                source_tokenizer_fingerprint=receipt['tokenizer']['source_fingerprint'],
                tokenizer=partial_tokenizer, expanded_tokenizer_fingerprint=row['tokenizer_fingerprint'],
                context_weight=.05, mask_assistant_history=restricted,
                mask_failed_assistant_history=failure_policy == 'mask-runtime-failures', assistant_history_guard=input_guard)
            if (row['nodes'] != masked['nodes']
                    or conversion.get('assistant_history_target_mask') != masked['conversion']['assistant_history_target_mask']):
                raise ValueError('partial parent graph does not preserve the exact assistant-history loss mask')
        if not restricted and (teacher.get('decision_admission', {}).get('parent_quality_hold') is not None
                or teacher.get('decision_admission', {}).get('guided_parent_review')):
            raise ValueError('partial parent graph has no current decision authority')
        groups = sorted(set(source_row['task']['program_ir']['source_groups']))
        if row['source_groups'] != groups or teacher['source_groups'] != groups or row['split'] != 'train':
            raise ValueError('graph train/source groups differ from exact verified program')
        profile_ids = {'text:' + teacher_action_graph_id(source_id, row_sha, turn, candidate, mask)
                       for candidate in (None, 'keep', 'last_turn_only')
                       for mask in ('preserve','mask-runtime-failures')}
        if ({identifier, target, source_id} | profile_ids) & hold_ids:
            raise ValueError('graph intersects current held decisions')
        ids.add(identifier); targets.add(target); source_ids.add(source_id)
        emitted_keys.append((source_id, turn))
    count = len(ids)
    if (count != receipt['outputs']['train']['rows']
            or count != receipt['deduplication']['emitted_rows']
            or count + receipt['deduplication']['omitted_rows'] != receipt['turns']
            or receipt['turns'] + len(receipt['action_omissions']) != source_actions):
        raise ValueError('teacher graph receipt lost complete action accounting')
    inverse = {'text:' + teacher_action_graph_id(source_id, raw_row_shas[source_id], turn, profile, failure_policy):
               (source_id, turn) for source_id, row in raw_rows.items()
               for turn in range(len(row['trajectory']))}
    deduplicated = []
    for omission in receipt['deduplication']['omissions']:
        if omission['id'] not in inverse:
            raise ValueError('teacher deduplication identity differs from its source/profile')
        deduplicated.append(inverse[omission['id']])
    if len(deduplicated) != receipt['deduplication']['omitted_rows']:
        raise ValueError('teacher deduplication count differs from its exact action identities')
    partition = materialization_partition(raw_rows, decision_keys,
        receipt['action_omissions'], emitted_keys, deduplicated, partial_parent_rows=partial)
    # Preserve full raw-row identity in the cross-model join, even when IDs and
    # graph content legitimately differ between declared rendering profiles.
    eligible = {(source_id, raw_row_shas[source_id], turn)
                for source_id, turn in [*emitted_keys, *deduplicated]}
    omitted = {(entry['source_row_id'], raw_row_shas[entry['source_row_id']], entry['teacher_turn_index'])
               for entry in receipt['action_omissions']}
    return {'graph_ids': ids, 'eligible_actions': eligible, 'omitted_actions': omitted,
            'target_decisions': targets, 'source_ids': source_ids, 'partition': partition,
            'explicit_history_reasoning_policy': profile}


def verify_paired_teacher_actions(models):
    """Match exact original obligations while retaining each rendering's counts."""
    if len(models) < 2:
        raise ValueError('paired teacher admission requires at least two model views')
    first = next(iter(models.values()))
    ignored_counts = {'emitted_actions', 'deduplicated_actions'}
    authority = {k: v for k, v in first['partition'].items() if k not in ignored_counts}
    for model in models.values():
        current = {k: v for k, v in model['partition'].items() if k not in ignored_counts}
        if (current != authority or model['eligible_actions'] != first['eligible_actions']
                or model['omitted_actions'] != first['omitted_actions']):
            raise ValueError('paired teacher views differ in exact original source/action obligations')
    return {'paired_model_graph_ids_equal': all(
                value['graph_ids'] == first['graph_ids'] for value in models.values()),
            'paired_model_source_actions_equal': True,
            'eligible_actions': len(first['eligible_actions']),
            'omitted_actions': len(first['omitted_actions'])}


def classify_materializer_holds(raw_rows, held_ids, *, policy_root=None):
    """Authenticate existing rejected-row policy; grant no target admission."""
    import math
    import subprocess
    ids=list(held_ids)
    if len(ids)!=len(set(ids))or any(ident not in raw_rows for ident in ids):
        raise ValueError('materializer hold identities differ from original rows')
    root=Path(policy_root).resolve(strict=True)if policy_root is not None else Path(__file__).resolve().parents[2]
    policy=root/'ts-host/dist/teacher/curriculum-policy.js'
    source=root/'ts-host/src/teacher/curriculum-policy.ts'
    bindings=[file_binding(source),file_binding(policy)]
    payload=[{'id':ident,'outcome':raw_rows[ident]['outcome'],'provenance':raw_rows[ident].get('provenance',{})}for ident in ids]
    code=('import {readFileSync} from "node:fs";'
          'const {trainingQualityReason}=await import(process.argv[1]);'
          'const rows=JSON.parse(readFileSync(0,"utf8"));'
          'console.log(JSON.stringify(rows.map(row=>({id:row.id,reason:trainingQualityReason(row)??null}))));')
    process=subprocess.run(['node','--input-type=module','-e',code,policy.as_uri()],
        input=json.dumps(payload,allow_nan=False),text=True,capture_output=True,check=True)
    if [file_binding(source),file_binding(policy)]!=bindings:
        raise ValueError('materializer hold policy changed during classification')
    reasons=json.loads(process.stdout)
    if [row.get('id')for row in reasons]!=ids:
        raise ValueError('materializer hold classifier changed row identities')
    result={}
    for item in reasons:
        ident=item['id'];outcome=raw_rows[ident]['outcome'];reason=item['reason']
        files=outcome.get('files_check')or{};oracle=outcome.get('oracle')or{}
        number=lambda x:type(x)in(int,float)and math.isfinite(x)and 0<=x<1
        if outcome.get('accepted')is not True:
            raise ValueError('materializer hold classification requires original positive candidate')
        if reason=='quality_pending_partial_files':
            if not(files.get('failed')or number(files.get('score'))):
                raise ValueError('partial-file hold lacks actual failure evidence')
            assigned='Exact partial-file quality review; original source retained, no graph or positive admission for this row'
        elif reason=='quality_pending_partial_agreement':
            if oracle.get('level')!='agreement'or not number(oracle.get('score')):
                raise ValueError('partial-agreement hold lacks actual oracle score evidence')
            assigned='Exact partial-answer agreement review; original source retained, no graph or positive admission for this row'
        else:
            raise ValueError('materializer hold needs explicit separately reviewed classification: '+str(reason))
        result[ident]={'quality_reason':reason,'policy_bindings':bindings,'files_check':outcome.get('files_check'),
            'oracle':outcome.get('oracle'),'assigned_use':assigned}
    return result


def verify_default_materializer_continuation(historical_reference, current_reference, *,
                                            source_reference, current_repository):
    """Authenticate exact default decision bytes across an explicit runtime change.

    Original conversion custody stays historical. This only proves that fresh
    current materialization yields every identical decision and complete summary;
    source/protection/graph/corpus admission checks remain independently required.
    """
    source_path = Path(source_reference['path'])
    if file_binding(source_path) != source_reference:
        raise ValueError('materializer continuation source changed')
    historical, current = read_bound(historical_reference), read_bound(current_reference)
    identities = []
    outputs = []
    for reference, receipt in ((historical_reference, historical), (current_reference, current)):
        identity = receipt.get('identity') or {}
        if (receipt.get('schema') != 'natlang.verified-teacher-decision-materialization/1'
                or set(identity) != {'source_sha256', 'runtime', 'direct_answers'}
                or identity.get('source_sha256') != source_reference['sha256']
                or identity.get('direct_answers') is not True
                or not isinstance(identity.get('runtime'), dict) or not identity['runtime']
                or not isinstance(receipt.get('summary'), dict)
                or receipt['summary'].get('output') != 'decisions.jsonl'):
            raise ValueError('materializer continuation requires exact default source/runtime custody')
        output = file_binding(Path(reference['path']).parent / 'decisions.jsonl')
        if output['sha256'] != receipt.get('output_sha256'):
            raise ValueError('materializer continuation decision output changed')
        identities.append(identity)
        outputs.append(output)
    if (outputs[0]['sha256'] != outputs[1]['sha256'] or outputs[0]['bytes'] != outputs[1]['bytes']
            or _exact(historical['summary']) != _exact(current['summary'])):
        raise ValueError('current materializer does not reproduce every original decision and complete typed summary')
    root = Path(current_repository).resolve(strict=True)
    runtime_bindings = []
    for name, digest in identities[1]['runtime'].items():
        path = (root / name).resolve(strict=True)
        if (Path(name).is_absolute() or not path.is_relative_to(root)
                or file_binding(path)['sha256'] != digest):
            raise ValueError('current materializer runtime does not match its exact receipt')
        runtime_bindings.append(file_binding(path))
    if (any(file_binding(value['path']) != value for value in [*runtime_bindings, *outputs])
            or file_binding(source_path) != source_reference
            or read_bound(historical_reference) != historical or read_bound(current_reference) != current):
        raise ValueError('materializer continuation inputs changed during verification')
    return {'schema': 'natlang.teacher-default-materializer-continuation/1',
            'source': source_reference, 'historical_materialization': historical_reference,
            'current_materialization': current_reference, 'historical_decisions': outputs[0],
            'current_decisions': outputs[1], 'complete_summary': current['summary'],
            'decision_bytes_identical': True, 'training_admission': False,
            'historical_conversion_unchanged': True}
