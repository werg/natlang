#!/usr/bin/env python3
"""Create a held-quality-hold revision of a closed source-correction candidate.

This copies a prepared candidate to a fresh directory and rewrites only its
proof/held-quality metadata. Training graphs, ledgers, source manifests, and
the original candidate remain byte-identical and untouched.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import shutil
from pathlib import Path

from natlang_neuralese.common.hashing import sha256_file_hex_cached
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.train.native_operation_report import source_receipt
from natlang_neuralese.train.native_source_correction import _origins


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def bind_current_receipt_identities(proof, train_path, held_path):
    """Use the loader's logical source identity after installing staging aliases."""
    from natlang_neuralese.data.native_source_tree import _source_identity
    for receipt, path in ((proof['current_source'], Path(train_path)),
                          (proof['protected_held_correction']['current_source'], Path(held_path))):
        if sha256_file_hex_cached(path) != receipt['sha256']:
            raise ValueError('current source identity correction cannot change source bytes')
        receipt['path'] = str(_source_identity(path).resolve())


def revise(candidate: Path, output: Path, *, prior_held_frame_path: str):
    candidate = candidate.resolve(strict=True)
    output = output.resolve()
    lock_path = output.parent / (output.name + '.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if output.exists():
            raise FileExistsError(f'preserve existing output candidate: {output}')
        shutil.copytree(candidate, output)
        prepared_path = output / 'prepared.json'
        prepared = json.loads(prepared_path.read_text())
        proof_path = output / 'source-correction.json'
        proof = json.loads(proof_path.read_text())
        expected = prepared['source_correction']
        if (Path(expected['path']).name != proof_path.name
                or sha256_file_hex_cached(proof_path) != expected['sha256']):
            raise ValueError('candidate source-correction proof differs from prepared receipt')
        for name, ref in prepared['artifacts'].items():
            path = output / name
            if path.is_file() and sha256_file_hex_cached(path) != ref['sha256']:
                raise ValueError(f'candidate artifact changed before metadata revision: {name}')

        source_map_path = output / '.natlang-source-map.json'
        source_map = json.loads(source_map_path.read_text())
        if (source_map.get('schema') != 'natlang.staged-source-map/1'
                or not isinstance(source_map.get('paths'), dict)):
            raise ValueError('candidate source-map custody is invalid')
        staged_prior = Path(prepared['prior_held']['path']).resolve(strict=True)
        mapped = source_map['paths'].get(prior_held_frame_path)
        if not isinstance(mapped, str) or Path(mapped).resolve() != staged_prior:
            raise ValueError('requested prior held frame path lacks exact staged-map relocation')
        staged_prior_sha = sha256_file_hex_cached(staged_prior)
        if staged_prior_sha != prepared['prior_held']['sha256']:
            raise ValueError('staged prior held manifest differs from prepared receipt')

        fingerprint = prepared['tokenizer_fingerprint']
        role_ids = tuple(prepared['native_role_token_ids'])
        old_held = NativeGraphSource(staged_prior, fingerprint, required_split='test', role_ids=role_ids)
        current_held_path = output / 'held-native-graphs.json'
        current_held = NativeGraphSource(current_held_path, fingerprint, required_split='test', role_ids=role_ids)
        if sha256_file_hex_cached(current_held_path) != prepared['current_held']['sha256']:
            raise ValueError('current held source differs from prepared receipt')

        # Bind the byte-identical metadata copy at its new candidate location.
        # This is not a training-source rewrite: only the top-level control
        # manifest paths change, and the complete leaf manifests/data remain
        # byte-for-byte the same.
        prior_candidate_paths = [prepared['current_train']['path'], prepared['current_held']['path']]
        replacement_paths = [str(output / 'native-training-sources.json'),
                             str(output / 'held-native-graphs.json')]
        for old_path, new_path in zip(prior_candidate_paths, replacement_paths):
            if not Path(new_path).is_file():
                raise FileNotFoundError(f'copied source manifest is missing: {new_path}')
            source_map['paths'][old_path] = new_path
        _write_json(source_map_path, source_map)
        source_map_evidence = output / 'evidence/current-staged-source-map.json'
        if source_map_evidence.exists():
            _write_json(source_map_evidence, source_map)
        source_map_sha = sha256_file_hex_cached(source_map_path)
        proof['source_map'] = {'path':source_map_path.name,'sha256':source_map_sha}
        for evidence in proof.get('evidence') or []:
            if Path(evidence['path']).name == 'current-staged-source-map.json':
                evidence['sha256'] = source_map_sha
        bind_current_receipt_identities(proof, *replacement_paths)
        from natlang_neuralese.data.native_source_tree import _source_identity
        current_held.source_identity_path = _source_identity(current_held.path)
        old_order, current_order = old_held.ordered_ids(), current_held.ordered_ids()
        if old_order != current_order:
            raise ValueError('quality-hold metadata revision cannot change held membership/order')
        if old_held.groups != current_held.groups:
            raise ValueError('quality-hold metadata revision cannot change held groups')
        if any(old_held.entries[index]['groups'] != current_held.entries[index]['groups']
               or bool(old_held.entries[index].get('communication'))
                  != bool(current_held.entries[index].get('communication'))
               for index in range(len(old_order))):
            raise ValueError('quality-hold metadata revision changed held group/class identity')

        protected = proof['protected_held_correction']
        if (protected['prior_source']['sha256'] != staged_prior_sha
                or protected['current_source']['path'] != str(current_held.source_identity_path.resolve())
                or protected['current_source']['sha256'] != current_held.file_sha256):
            raise ValueError('candidate protected-held source receipt differs from parsed source bytes')
        origins = _origins(protected['prior_source'], old_order)
        pending = (proof.get('pending_held_correction') or {}).get('remaining_original_sources')
        if not isinstance(pending, list):
            raise ValueError('candidate lacks exact pending held-correction debt')
        expected_sources = {}
        for item in pending:
            source = item.get('source') or {}
            digest, count, reason = source.get('sha256'), item.get('rows'), item.get('reason')
            if (not isinstance(digest, str) or type(count) is not int or count < 1
                    or not isinstance(reason, str) or not reason or digest in expected_sources):
                raise ValueError('pending held-correction inventory is malformed')
            expected_sources[digest] = (count, reason)

        rows_path = output / 'held-quality-hold-row-bindings.jsonl'
        emitted = {digest: 0 for digest in expected_sources}
        with rows_path.open('w') as stream:
            for index, row_id in enumerate(old_order):
                source_sha = origins[row_id]
                if source_sha not in expected_sources:
                    continue
                if (old_held.bindings[row_id] != current_held.bindings[row_id]
                        or old_held.entries[index]['groups'] != current_held.entries[index]['groups']):
                    raise ValueError('pending held target changed despite its quality exclusion')
                reason = expected_sources[source_sha][1]
                row = {'id': row_id, 'binding': current_held.bindings[row_id],
                       'source_sha256': source_sha, 'reason': reason,
                       'groups': current_held.entries[index]['groups']}
                stream.write(json.dumps(row, sort_keys=True) + '\n')
                emitted[source_sha] += 1
        if emitted != {digest: spec[0] for digest, spec in expected_sources.items()}:
            raise ValueError('pending held quality rows do not match exact original source counts')

        # Preserve the checkpoint's logical receipt path. Its bytes are
        # authenticated by the staged map and the SHA of the local copy.
        prior_logical = dict(protected['prior_source'])
        prior_logical['path'] = prior_held_frame_path
        protected['prior_source'] = prior_logical
        proof['prior_held_relocation'] = {'schema':'natlang.staged-receipt-relocation/1',
            'logical_path':prior_held_frame_path,'staged_path':str(staged_prior),
            'sha256':staged_prior_sha,'source_map_path':str(source_map_path),
            'source_map_sha256':sha256_file_hex_cached(source_map_path)}
        proof['prior_held_relocation']['source_map_sha256'] = source_map_sha

        current_receipt = protected['current_source']
        hold_manifest_path = output / 'held-quality-hold.json'
        _write_json(hold_manifest_path, {
            'schema':'natlang.native-held-quality-hold/1',
            'held_source':current_receipt,
            'row_bindings_path':rows_path.name,
            'row_bindings_sha256':sha256_file_hex_cached(rows_path),
            'excluded_rows':sum(emitted.values()),
            'pending_sources':[{'sha256':digest,'rows':count,'reason':reason}
                               for digest,(count,reason) in sorted(expected_sources.items())],
            'purpose':'exclude known truncated targets from held loss/model selection while preserving protected membership; full-held qualification remains incomplete'})
        proof['held_quality_hold'] = {'path':hold_manifest_path.name,
            'sha256':sha256_file_hex_cached(hold_manifest_path)}
        proof['evidence'] = list(proof.get('evidence') or []) + [
            {'path':rows_path.name,'sha256':sha256_file_hex_cached(rows_path)},
            {'path':hold_manifest_path.name,'sha256':sha256_file_hex_cached(hold_manifest_path)}]
        _write_json(proof_path, proof)

        prepared['superseded_candidate'] = {
            'path':str(candidate),
            'prepared_sha256':sha256_file_hex_cached(candidate / 'prepared.json'),
            'source_correction_sha256':expected['sha256'],
            'status':'superseded metadata only; source bytes and row ledgers reused byte-for-byte'}
        prepared['prior_held_frame_receipt'] = {'path':prior_held_frame_path,
            'sha256':staged_prior_sha,'staged_byte_identical_path':str(staged_prior)}
        prepared['held_quality_hold'] = {'path':str(hold_manifest_path),
            'sha256':sha256_file_hex_cached(hold_manifest_path),
            'row_bindings_path':str(rows_path),
            'row_bindings_sha256':sha256_file_hex_cached(rows_path),
            'excluded_rows':sum(emitted.values()), 'full_held_qualification':False}
        prepared['current_train'] = {'path':replacement_paths[0],
            'sha256':prepared['current_train']['sha256']}
        prepared['current_held'] = {'path':replacement_paths[1],
            'sha256':prepared['current_held']['sha256']}
        prepared['target_runtime_paths'].update({'train':replacement_paths[0],
            'held':replacement_paths[1], 'source_correction':str(proof_path)})
        prepared['adoption_requirements'].update({'native_graphs':replacement_paths[0],
            'held_native_graphs':replacement_paths[1], 'source_correction':str(proof_path)})
        prepared['source_correction'] = {'path':str(proof_path),
            'sha256':sha256_file_hex_cached(proof_path)}
        for name, ref in prepared['artifacts'].items():
            ref['path'] = str(output / name)
        prepared['artifacts'].update({
            '.natlang-source-map.json':{'path':str(source_map_path),'bytes':source_map_path.stat().st_size,
                                      'sha256':source_map_sha},
            'source-correction.json':{'path':str(proof_path),'bytes':proof_path.stat().st_size,
                                      'sha256':sha256_file_hex_cached(proof_path)},
            'held-quality-hold.json':{'path':str(hold_manifest_path),'bytes':hold_manifest_path.stat().st_size,
                                      'sha256':sha256_file_hex_cached(hold_manifest_path)},
            'held-quality-hold-row-bindings.jsonl':{'path':str(rows_path),'bytes':rows_path.stat().st_size,
                                      'sha256':sha256_file_hex_cached(rows_path)}})
        _write_json(prepared_path, prepared)
        return prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior-held-frame-path', required=True)
    args = parser.parse_args()
    print(json.dumps(revise(args.candidate, args.output,
        prior_held_frame_path=args.prior_held_frame_path), sort_keys=True))


if __name__ == '__main__':
    main()
