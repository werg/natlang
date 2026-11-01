from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))

from scripts.neuralese_data.build_historical_luna_full_episode_graphs import (
    DEFAULT_BASE_TOKENIZER,
    prior_geometry_path,
)
from scripts.neuralese_data.portable_paths import node_executable
from scripts.neuralese_data.student_tokenizers import STUDENTS


def test_historical_builder_uses_configured_data_root_and_canonical_tokenizer(monkeypatch):
    expected_root = Path('/portable-data')
    monkeypatch.setenv('NATLANG_DATA_NVME', str(expected_root))
    assert prior_geometry_path() == expected_root / (
        'loghub-graph-successor-20261031/terra-ce-stage/telecom-v27-plus1/'
        'packed-capacity-geometry/fe23368e8588bcf585030d5d62c4b2b9a90ba74bef0ba824907f999aecc9e35b.json')
    assert DEFAULT_BASE_TOKENIZER == STUDENTS['lfm3']['path']


def test_node_uses_explicit_setting_or_path_and_has_no_machine_fallback(tmp_path):
    configured = tmp_path / 'configured-node'
    configured.write_text('#!/bin/sh\nexit 0\n')
    configured.chmod(0o755)
    assert node_executable(environ={'NATLANG_NODE': str(configured), 'PATH': ''}) == str(configured)

    from_path = tmp_path / 'bin' / 'node'
    from_path.parent.mkdir()
    from_path.write_text('#!/bin/sh\nexit 0\n')
    from_path.chmod(0o755)
    found = []

    def which(command, *, path):
        found.append((command, path))
        return str(from_path)

    assert node_executable(environ={'PATH': '/portable-node-bin'}, which=which) == str(from_path)
    assert found == [('node', '/portable-node-bin')]

    try:
        node_executable(environ={'PATH': ''}, which=lambda *_args, **_kwargs: None)
    except RuntimeError as exc:
        assert 'NATLANG_NODE' in str(exc)
    else:
        raise AssertionError('missing Node was silently replaced by a machine-specific path')


def test_changed_helpers_contain_no_workstation_specific_executable_or_storage_paths():
    for name in (
        'build_historical_luna_full_episode_graphs.py',
        'build_luna_exact_native_conversion.py',
        'historical_provider_source.py',
    ):
        source = (ROOT / 'scripts' / 'neuralese_data' / name).read_text()
        assert '/home/werg/' not in source
        assert '/mnt/external/' not in source
