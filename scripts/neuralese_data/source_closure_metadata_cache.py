"""Opt-in, process-local reuse of an unchanged source-closure producer."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os

from natlang_neuralese.common.hashing import (
    canonical_json_sha256_hex, sha256_file_hex_cached)


def _signature(path):
    status = path.stat()
    return (status.st_dev, status.st_ino, status.st_size,
            status.st_mtime_ns, status.st_ctime_ns)


def _authenticated(path, expected=None):
    path = Path(path)
    resolved = path.resolve(strict=True)
    before = _signature(resolved)
    digest = sha256_file_hex_cached(resolved, trust_archive_receipt=False)
    if path.resolve(strict=True) != resolved or _signature(resolved) != before:
        raise RuntimeError('closure artifact changed during authentication')
    ref = {'path': str(resolved), 'sha256': digest, 'bytes': before[2]}
    if expected is not None and ref != expected:
        raise ValueError('closure artifact differs from memoized inventory')
    return ref


def _reference_edges(path, manifest):
    """Retain unresolved references as well as the producer's resolved outputs."""
    edges = {}
    parsed_metadata = {}

    def edge(candidate):
        candidate = Path(os.path.abspath(candidate))
        resolved = candidate.resolve(strict=True)
        edges[str(candidate)] = str(resolved)
        return resolved

    base = edge(path).parent
    for item in manifest.get('files', []):
        edge(base / item['path'])
    component = manifest.get('source_group_components')
    if component is not None:
        edge(base / component)
    for item in (manifest.get('inherited_source_components') or {}).values():
        edge(base / item['path'])
    directories = set()
    for item in manifest.get('normalized_export_bindings', []):
        export = edge(item['path'])
        edge(export.parent / 'full-trajectories.jsonl.zst')
        directories.add(export.parent.parent.resolve(strict=True))
    for directory in directories:
        complete = edge(directory / 'complete.json')
        ref = _authenticated(complete)
        before = _signature(complete)
        raw = complete.read_bytes()
        if (_signature(complete) != before
                or hashlib.sha256(raw).hexdigest() != ref['sha256']):
            raise RuntimeError('closure inventory changed during metadata read')
        parsed_metadata[str(complete)] = ref
        for item in json.loads(raw).get('files', []):
            if item.get('status') not in {'exported', 'reused'}:
                continue
            receipt = Path(item['receipt'])
            edge(receipt if receipt.is_absolute() else directory / receipt)
    return edges, parsed_metadata


def _check_edges(edges):
    for original, resolved in edges.items():
        if str(Path(original).resolve(strict=True)) != resolved:
            raise ValueError('closure reference was retargeted')


class _ClosureMetadataMemo:
    def __init__(self, original):
        self.original = original
        self.key = self.inventory = self.edges = None
        self.producer_calls = self.reuses = self.artifact_authentications = 0

    def _check_inventory(self, inventory):
        for ref in inventory:
            if set(ref) != {'path', 'sha256', 'bytes'}:
                raise ValueError('closure producer returned an incomplete artifact binding')
            _authenticated(ref['path'], ref)
            self.artifact_authentications += 1

    def __call__(self, path, manifest, *, use_cache=True):
        if not use_cache:
            self.producer_calls += 1
            return self.original(path, manifest, use_cache=False)
        supplied = deepcopy(manifest)
        key = canonical_json_sha256_hex({
            'manifest': _authenticated(path), 'supplied': supplied, 'use_cache': True,
            'cwd': os.getcwd()})
        if key == self.key:
            try:
                _check_edges(self.edges)
                self._check_inventory(self.inventory)
            except (OSError, RuntimeError, ValueError):
                self.key = self.inventory = self.edges = None
            else:
                self.reuses += 1
                return deepcopy(self.inventory)
        self.producer_calls += 1
        try:
            edges, parsed_metadata = _reference_edges(path, supplied)
        except (OSError, RuntimeError, ValueError):
            # A cache dependency probe cannot substitute for the producer's
            # own acceptance/rejection, nor retain a partially captured graph.
            return self.original(path, supplied, use_cache=True)
        inventory = self.original(path, supplied, use_cache=True)
        _check_edges(edges)
        # Authenticate again after the producer's metadata reads, closing its
        # hash/read interval before retaining a value for a subsequent call.
        self._check_inventory(inventory)
        produced = {ref['path']: ref for ref in inventory}
        for ref in parsed_metadata.values():
            if produced.get(ref['path']) != ref:
                raise ValueError('parsed closure metadata differs from produced inventory')
            _authenticated(ref['path'], ref)
        # Recompute the full key, including supplied data, to reject a manifest
        # replacement during the original producer call.
        after = canonical_json_sha256_hex({
            'manifest': _authenticated(path), 'supplied': supplied, 'use_cache': True,
            'cwd': os.getcwd()})
        if after != key:
            raise RuntimeError('closure manifest changed during metadata production')
        self.key, self.inventory, self.edges = key, deepcopy(inventory), edges
        return inventory

    def receipt(self):
        return {'schema': 'natlang.source-closure-metadata-cache/1',
                'producer_calls': self.producer_calls, 'reuses': self.reuses,
                'artifact_authentications': self.artifact_authentications,
                'process_local_only': True, 'training_admission': False}


@contextmanager
def reuse_authenticated_closure_metadata(module):
    """Keep the producer unchanged on disk and restore its callable on exit."""
    original = module._source_closure_artifacts
    if isinstance(original, _ClosureMetadataMemo):
        raise ValueError('source-closure metadata cache is already installed')
    memo = _ClosureMetadataMemo(original)
    module._source_closure_artifacts = memo
    try:
        yield memo
    finally:
        if module._source_closure_artifacts is not memo:
            raise RuntimeError('source-closure producer replaced during cache lifetime')
        module._source_closure_artifacts = original
