"""Exact communication custody for interrupted, request-bound collection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .graph_replay import block_ids, restore_bundle, save_bundle, verified_bundle_bytes


class _Unavailable(RuntimeError):
    def __init__(self, ident, message):
        super().__init__(message)
        self.ident = ident


def _partial_roots(entries, per_path=None):
    roots, sources = set(), []
    for entry in entries:
        jobs = Path(entry['jobs']).resolve()
        for path in sorted(jobs.glob('*.partial.json')):
            if path.resolve().parent != jobs or path.is_symlink():
                raise ValueError('retained partial escapes its jobs directory')
            raw = path.read_bytes()
            partial = json.loads(raw)
            if partial.get('version') != 'natlang.teacher_partial.native/1':
                raise ValueError('retained partial has an unsupported collector schema')
            roots.update(block_ids(partial))
            if per_path is not None:
                per_path[path] = set(block_ids(partial))
            source = {'path': str(path), 'bytes': len(raw),
                      'sha256': hashlib.sha256(raw).hexdigest(), 'evidence': []}
            for evidence in partial.get('evidence_snapshots', []):
                ep = path.parent / evidence['path']
                if ep.resolve().parent != jobs or ep.is_symlink():
                    raise ValueError('retained evidence escapes its jobs directory')
                size = evidence['bytes']
                if type(size) is not int or size < 0:
                    raise ValueError('retained evidence needs an exact prefix length')
                remaining, digest = size, hashlib.sha256()
                with ep.open('rb') as stream:
                    while remaining:
                        line = stream.readline(remaining)
                        if not line:
                            raise ValueError('retained evidence prefix is truncated')
                        remaining -= len(line)
                        digest.update(line)
                        roots.update(block_ids(json.loads(line)))
                        if per_path is not None:
                            per_path[path].update(block_ids(json.loads(line)))
                if digest.hexdigest() != evidence['sha256']:
                    raise ValueError('retained evidence prefix SHA changed')
                source['evidence'].append(dict(evidence))
            sources.append(source)
    if any(not ident.startswith(('nzs1_', 'nnt1_')) for ident in roots):
        raise ValueError('native retained collection contains a projected artifact')
    return sorted(roots), sources


class _HttpStore:
    """Use the existing exact bundle restore order against the served store."""
    def __init__(self, server, policy_id):
        self.server, self.owner = server, 'retained-policy:' + policy_id

    def _put(self, ident, raw, kind):
        route = '/v1/neuralese/' + kind + '/' + ident
        answer = self.server.put_bytes(route, raw, owner=self.owner)
        if answer.get('id') != ident or self.server.get_bytes(route) != raw:
            raise ValueError('restored retained artifact differs from its archived bytes')

    def put_native_state(self, block):
        from ..serve.store import encode_native_state, native_state_id
        self._put(native_state_id(block), encode_native_state(block), 'native-states')

    def put_native_tensor_artifact(self, artifact):
        from ..model.native_artifact import encode_native_tensor_artifact, native_tensor_artifact_id
        self._put(native_tensor_artifact_id(artifact), encode_native_tensor_artifact(artifact),
                  'native-tensor-artifacts')


def retain_partial_artifacts(entries, out, *, policy_id, binding, servers, restore=False):
    """Archive all referenced DAG inputs; optionally hydrate every actor before replay.

    The caller validates exact actor receipts before and after this operation and
    owns the jobs exclusively. Existing collector loadPartial verifies request and
    program provenance; this function adds the previously missing artifact custody.
    """
    per_path = {}
    roots, sources = _partial_roots(entries, per_path)
    directory = Path(out) / 'retained-partial-blocks'
    manifest_path = directory / 'manifest.json'
    previous = None
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if (previous.get('schema') != 'natlang.rl-block-bundle/1'
                or previous.get('policy_id') != policy_id or previous.get('collection_binding') != binding):
            raise ValueError('retained artifact bundle belongs to a different collection or policy')

    def fetch(ident):
        old = verified_bundle_bytes(manifest_path, ident, policy_id=policy_id, manifest=previous)
        if old is not None:
            return old
        kind = 'native-states' if ident.startswith('nzs1_') else 'native-tensor-artifacts'
        errors = []
        for server in servers:
            try:
                return server.get_bytes('/v1/neuralese/' + kind + '/' + ident)
            except Exception as error:
                errors.append(str(error))
        raise _Unavailable(ident, 'retained artifact unavailable before collection: ' + ident + ': ' + '; '.join(errors))

    while True:
        try:
            manifest = save_bundle(roots, directory, policy_id=policy_id, fetch=fetch,
                manifest_fields={'collection_binding': binding, 'retained_partial_sources': sources})
            break
        except _Unavailable as missing:
            # A partial whose root block no actor and no archive holds (it lived only in a crashed actor's store) cannot
            # resume: it is set aside and its entry samples afresh from the start (w5 round 2, 2026-11-10).
            lost = [path for path, ids in per_path.items() if missing.ident in ids]
            if not lost:
                # The missing block is an ancestor: find the partials whose own closure needs it.
                import tempfile
                for path, ids in per_path.items():
                    with tempfile.TemporaryDirectory() as scratch:
                        try:
                            save_bundle(sorted(ids), Path(scratch), policy_id=policy_id, fetch=fetch)
                        except _Unavailable:
                            lost.append(path)
            if not lost:
                raise
            for path in lost:
                path.rename(path.with_name(path.name + '.lost'))
                print(json.dumps({'event': 'retained partial set aside: its block is unavailable on every actor',
                                  'partial': str(path), 'block': missing.ident}), flush=True)
            per_path = {}
            roots, sources = _partial_roots(entries, per_path)
    if restore:
        from http.client import HTTPException
        for index, server in enumerate(servers):
            try:
                restore_bundle(SimpleNamespace(store=_HttpStore(server, policy_id),
                                               native_state_enabled=True, width=None),
                               manifest_path, policy_id=policy_id)
            except (OSError, HTTPException) as error:
                # servers[0] is the managed local actor and must hold the archive. A remote actor that goes away during
                # the restore (the Pop actor stopped for a dev16 evaluation) leaves the round instead of failing it.
                if index == 0:
                    raise
                print(json.dumps({'event': 'remote RL actor left during retained artifact restore',
                                  'server': getattr(server, 'base', str(server)),
                                  'reason': f'{type(error).__name__}: {error}'}), flush=True)
    return manifest
