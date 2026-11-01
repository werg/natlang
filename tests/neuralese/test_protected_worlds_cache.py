from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from natlang_neuralese.common import custody
from natlang_neuralese.common.custody import ProtectedWorlds


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protected_worlds_cache_tracks_source_file_identity_and_symlink_target(tmp_path, monkeypatch):
    source_a = tmp_path / 'source-a.jsonl'
    source_b = tmp_path / 'source-b.jsonl'
    row = b'{"id":"held","source_groups":["held-group"]}\n'
    source_a.write_bytes(row)
    source_b.write_bytes(row)
    link = tmp_path / 'groups.jsonl'
    link.symlink_to(source_a)
    registry = tmp_path / 'registry.json'

    def write_registry(digest):
        registry.write_text(json.dumps({
            'schema': custody.REGISTRY_SCHEMA,
            'sources': [{'id': 'groups', 'kind': 'source-groups', 'path': 'groups.jsonl', 'sha256': digest}],
        }))

    write_registry(_sha(source_a))
    original_jsonl = custody._jsonl
    reads = []

    def count_rows(path):
        reads.append(Path(path).resolve())
        return original_jsonl(path)

    monkeypatch.setattr(custody, '_jsonl', count_rows)
    first = ProtectedWorlds.load(root=tmp_path, registry='registry.json')
    second = ProtectedWorlds.load(root=tmp_path, registry='registry.json')
    assert first is second
    assert reads == [source_a.resolve()]

    # A same-size edit with restored mtime still changes ctime and cannot reuse
    # the old proof or cached protected identities.
    before = source_a.stat()
    source_a.write_bytes(b'{"id":"held","source_groups":["evil-group"]}\n')
    os.utime(source_a, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(custody.CustodyError, match='missing or changed'):
        ProtectedWorlds.load(root=tmp_path, registry='registry.json')

    # Retargeting to an independently verified file with identical content is
    # still a new input identity and forces the source parser to run again.
    source_a.write_bytes(row)
    write_registry(_sha(source_a))
    link.unlink()
    link.symlink_to(source_a)
    ProtectedWorlds.load(root=tmp_path, registry='registry.json')
    assert reads[-1] == source_a.resolve()
    link.unlink()
    link.symlink_to(source_b)
    retargeted = ProtectedWorlds.load(root=tmp_path, registry='registry.json')
    assert ('group', 'held-group') in retargeted.identities
    assert reads[-1] == source_b.resolve()
    assert len(reads) == 3


def test_protected_worlds_cache_tracks_nested_packet_registry_and_cases(tmp_path, monkeypatch):
    packet_a = tmp_path / 'packet-a'
    packet_b = tmp_path / 'packet-b'
    packet_a.mkdir()
    packet_b.mkdir()
    (packet_a / 'cases.jsonl').write_text('{"id":"held-a","family":"fixture","semantics":{}}\n')
    (packet_b / 'cases.jsonl').write_text('{"id":"held-b","family":"fixture","semantics":{}}\n')
    packet_registry = tmp_path / 'neuralese_corpora.json'
    top_registry = tmp_path / 'registry.json'
    top_registry.write_text(json.dumps({
        'schema': custody.REGISTRY_SCHEMA,
        'sources': [{'id': 'packets', 'kind': 'evaluation-packets',
                     'registry': 'neuralese_corpora.json'}],
    }))

    def select_packet(path):
        packet_registry.write_text(json.dumps({'corpora': [{
            'id': 'packet-fixture',
            'admission': custody.PROTECTED_ADMISSION,
            'include': ['cases.jsonl'], 'path': path,
        }]}))

    select_packet('packet-a')
    original_read_bytes = Path.read_bytes
    nested_reads = []

    def count_nested_registry(path):
        if path.resolve() == packet_registry.resolve():
            nested_reads.append(path.resolve())
        return original_read_bytes(path)

    monkeypatch.setattr(Path, 'read_bytes', count_nested_registry)
    first = ProtectedWorlds.load(root=tmp_path, registry='registry.json')
    second = ProtectedWorlds.load(root=tmp_path, registry='registry.json')
    assert first is second
    assert ('id', 'held-a') in first.identities
    assert nested_reads == [packet_registry.resolve()]

    select_packet('packet-b')
    updated = ProtectedWorlds.load(root=tmp_path, registry='registry.json')
    assert ('id', 'held-b') in updated.identities
    assert ('id', 'held-a') not in updated.identities
    assert nested_reads == [packet_registry.resolve(), packet_registry.resolve()]
