import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_reviewed_luna_dispatcher as dispatcher


FEATURES = {
    'service_aliases': 'natlang.service-aliases/2',
    'service_scopes': 'natlang.service-scopes/1',
    'report_completeness': 'natlang.report-completeness/2',
}


def _runtime(tmp_path: Path, script: str):
    runtime = tmp_path / 'runtime'
    cli = runtime / 'dist/teacher/cli.js'
    collector = runtime / 'dist/teacher/collector.js'
    cli.parent.mkdir(parents=True)
    cli.write_text(script)
    collector.write_text('// test collector\n')
    (runtime / 'frozen-runtime.json').write_text('{}\n')
    compiled = {
        'teacher/cli.js': hashlib.sha256(cli.read_bytes()).hexdigest(),
        'teacher/collector.js': hashlib.sha256(collector.read_bytes()).hexdigest(),
    }
    return runtime, {'build': {'mode': 'explicit_compiled_dist', 'compiled_dist_files': compiled}}


def _capability_cli(*, features=FEATURES, probes=None):
    report = {'schema': 'natlang.teacher-collector-capabilities/1', **features,
              'probes': probes if probes is not None else {name: True for name in features}}
    return f"process.stdout.write({json.dumps(json.dumps(report) + chr(10))});\n"


def test_features_are_derived_from_selected_case_semantics():
    rows = {
        0: {'semantics': {'service_aliases': {'device': {}}, 'service_scopes': {'device': []},
                          'report_completeness': {'schema': 'natlang.report-completeness/2'}}},
        1: {'semantics': {}},
    }
    assert dispatcher._required_runtime_semantics_features(rows, [{'index': 0}]) == sorted(FEATURES)
    assert dispatcher._required_runtime_semantics_features(rows, [{'index': 1}]) == []


def test_frozen_runtime_capability_probe_binds_actual_cli_collector_and_results(tmp_path):
    runtime, manifest = _runtime(tmp_path, _capability_cli())
    result = dispatcher._validate_runtime_semantics_capabilities(runtime, manifest, FEATURES)
    assert result['required_features'] == sorted(FEATURES)
    assert result['cli_sha256'] == manifest['build']['compiled_dist_files']['teacher/cli.js']
    assert result['collector_sha256'] == manifest['build']['compiled_dist_files']['teacher/collector.js']
    assert result['report']['probes'] == {name: True for name in FEATURES}


def test_ordinary_old_collector_cli_without_capability_api_is_rejected(tmp_path):
    # This represents a compiled collector that still has ordinary --help/collection behavior,
    # but predates service aliases, service scopes, and report-completeness enforcement.
    runtime, manifest = _runtime(tmp_path, "if (process.argv.includes('--help')) process.stdout.write('usage: collect');\nelse process.exit(2);\n")
    help_result = subprocess.run(['node', str(runtime / 'dist/teacher/cli.js'), '--help'], capture_output=True, text=True)
    assert help_result.returncode == 0 and help_result.stdout == 'usage: collect'
    with pytest.raises(ValueError, match='does not expose executable semantic capability probes'):
        dispatcher._validate_runtime_semantics_capabilities(runtime, manifest, FEATURES)


def test_collector_with_missing_or_failed_feature_probe_is_rejected(tmp_path):
    versions = {**FEATURES, 'report_completeness': 'natlang.report-completeness/0'}
    probes = {'service_aliases': True, 'service_scopes': True, 'report_completeness': False}
    runtime, manifest = _runtime(tmp_path, _capability_cli(features=versions, probes=probes))
    with pytest.raises(ValueError, match='failed the required report_completeness capability probe'):
        dispatcher._validate_runtime_semantics_capabilities(runtime, manifest, FEATURES)


def test_collector_with_old_unfiltered_alias_semantics_is_rejected(tmp_path):
    versions = {**FEATURES, 'service_aliases': 'natlang.service-aliases/1'}
    runtime, manifest = _runtime(tmp_path, _capability_cli(features=versions))
    with pytest.raises(ValueError, match='failed the required service_aliases capability probe'):
        dispatcher._validate_runtime_semantics_capabilities(runtime, manifest, FEATURES)
