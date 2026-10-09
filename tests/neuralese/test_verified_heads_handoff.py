import json
from pathlib import Path

import pytest

from natlang_neuralese.train.output_embedding_projection import sha
from natlang_neuralese.train.verified_heads_handoff import verify_handoff


def fixtures(tmp_path):
    files = {}
    for name, data in {
        'raw-heads.pt': b'raw heads',
        'warmup-checkpoint.pt': b'full warmup checkpoint',
        'warmup-heads.pt': b'trained map and sketch heads',
    }.items():
        path = tmp_path / name
        path.write_bytes(data)
        files[name] = path
    strata = {
        name: {'tokens': 256, 'ce_delta': .01, 'text_ce_delta_from_initial': .02,
               'embedding_mse_delta': .01, 'text_argmax_agreement': .999}
        for name in ('whole', 'last256')
    }
    manifest = {
        'schema': 'natlang.neuralese-best-warmup-checkpoint/1',
        'step': 32256,
        'files': {
            'best-checkpoint.pt': {'sha256': sha(files['warmup-checkpoint.pt'])},
            'best-heads.pt': {'sha256': sha(files['warmup-heads.pt'])},
        },
        'qualification': {'alignment_gate_passed': True, 'consecutive_passes': 1,
                          'qualified': False, 'strata': strata},
    }
    files['warmup-manifest.json'] = tmp_path / 'warmup-manifest.json'
    files['warmup-manifest.json'].write_text(json.dumps(manifest))
    text_runtime = {
        'schema': 'natlang.latent-sketch-diagnostic/1', 'profile': 'latent-sketch-v2',
        'parent_sha256': sha(files['warmup-heads.pt']), 'runtime_qualified': True,
        'implementation_checks_passed': True, 'foundation_requalified': False,
        'rows': [{'length': n, 'finite_gradients': True,
                  'sketch_receives_consumer_gradient': True} for n in (128, 512)],
    }
    files['warmup-runtime.json'] = tmp_path / 'warmup-runtime.json'
    files['warmup-runtime.json'].write_text(json.dumps(text_runtime))
    files['raw-runtime.json'] = tmp_path / 'raw-runtime.json'
    files['raw-runtime.json'].write_text(json.dumps({
        'schema': 'natlang.neuralese-runtime-handoff/1', 'runtime_transport_passed': True,
        'pins': {'heads': sha(files['raw-heads.pt'])},
    }))
    return files


def run(files):
    return verify_handoff(raw_heads=files['raw-heads.pt'],
        raw_runtime_report=files['raw-runtime.json'],
        warmup_checkpoint=files['warmup-checkpoint.pt'],
        warmup_manifest=files['warmup-manifest.json'],
        warmup_heads=files['warmup-heads.pt'],
        warmup_runtime_report=files['warmup-runtime.json'])


def test_handoff_preserves_explicit_single_gate_and_never_inherits_raw_certificate(tmp_path):
    report = run(fixtures(tmp_path))
    assert report['handoff_verified'] is True
    assert report['warmup_alignment_gate_passed'] is True
    assert report['warmup_consecutive_passes'] == 1
    assert report['warmup_consecutively_qualified'] is False
    assert report['raw_reference_certificate_inherited_by_learned_channel'] is False
    assert report['learned_channel_foundation_qualified'] is False
    assert report['weight_action'] == 'continue_from_exact_full_state_checkpoint'


def test_handoff_rejects_warmup_runtime_for_other_weights(tmp_path):
    files = fixtures(tmp_path)
    report = json.loads(files['warmup-runtime.json'].read_text())
    report['parent_sha256'] = '0' * 64
    files['warmup-runtime.json'].write_text(json.dumps(report))
    with pytest.raises(ValueError, match='trained latent-sketch runtime report'):
        run(files)


def test_handoff_rejects_any_failed_declared_alignment_stratum(tmp_path):
    files = fixtures(tmp_path)
    manifest = json.loads(files['warmup-manifest.json'].read_text())
    manifest['qualification']['strata']['last256']['embedding_mse_delta'] = .026
    files['warmup-manifest.json'].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='fails a declared held alignment stratum'):
        run(files)


def test_handoff_rejects_checkpoint_bytes_outside_best_manifest(tmp_path):
    files = fixtures(tmp_path)
    files['warmup-checkpoint.pt'].write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='full warm-up checkpoint'):
        run(files)
