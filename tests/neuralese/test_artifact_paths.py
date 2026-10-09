"""Role-bound artifact paths: a checkpoint loads from any working directory, or fails naming the role to override."""
import json
import os
import sys

import pytest
import torch

from natlang_neuralese.common.artifact_paths import (
    ArtifactResolutionError, ArtifactResolver, artifact_refs)
from natlang_neuralese.common.hashing import sha256_file_hex as sha


def make_run(tmp_path):
    """A launch directory holding relative inputs, and an unrelated directory to load from."""
    launch = tmp_path / 'launch'
    (launch / 'runs' / 'x').mkdir(parents=True)
    heads = launch / 'runs' / 'x' / 'heads.pt'
    heads.write_bytes(b'heads-bytes')
    pieces = launch / 'pieces.jsonl'
    pieces.write_text(json.dumps({'name': 'p', 'text': 'source'}) + '\n')
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    files = {str(heads.resolve()): sha(heads), str(pieces.resolve()): sha(pieces)}
    options = {'heads': 'runs/x/heads.pt', 'pieces': './pieces.jsonl', 'base': 'LiquidAI/LFM2.5-350M'}
    return launch, elsewhere, options, files, heads, pieces


def test_new_checkpoints_record_logical_resolved_and_digest_per_role(tmp_path):
    launch, _, options, files, heads, pieces = make_run(tmp_path)
    refs = artifact_refs(options, files, ('base', 'heads', 'pieces', 'bank'), cwd=launch)
    assert refs['heads'] == {'logical': 'runs/x/heads.pt', 'resolved': str(heads.resolve()), 'sha256': sha(heads)}
    assert refs['pieces']['resolved'] == str(pieces.resolve())
    assert refs['base'] == {'logical': 'LiquidAI/LFM2.5-350M', 'resolved': None, 'sha256': None}
    assert 'bank' not in refs  # a role the run did not use


def test_a_new_checkpoint_resolves_from_any_working_directory_without_changing_it(tmp_path, monkeypatch):
    launch, elsewhere, options, files, heads, pieces = make_run(tmp_path)
    identity = {'options': options, 'files': files}
    refs = artifact_refs(options, files, ('base', 'heads', 'pieces'), cwd=launch)
    monkeypatch.chdir(elsewhere)
    resolver = ArtifactResolver(identity, refs)
    assert resolver.path('heads') == str(heads.resolve())
    assert resolver.path('pieces') == str(pieces.resolve())
    assert resolver.path('base') == 'LiquidAI/LFM2.5-350M'  # a model id stays an id
    assert resolver.path('bank') is None
    assert os.getcwd() == str(elsewhere.resolve())


def test_a_legacy_checkpoint_matches_one_pinned_key_by_normalized_suffix(tmp_path, monkeypatch):
    launch, elsewhere, options, files, heads, pieces = make_run(tmp_path)
    monkeypatch.chdir(elsewhere)
    resolver = ArtifactResolver({'options': options, 'files': files})
    assert resolver.path('heads') == str(heads.resolve())
    assert resolver.path('pieces') == str(pieces.resolve())  # './pieces.jsonl'
    # the text warm-up names its pins identity.inputs
    assert ArtifactResolver({'options': options, 'inputs': files}).path('heads') == str(heads.resolve())


def test_changed_bytes_missing_files_and_ambiguity_ask_for_an_override(tmp_path):
    launch, elsewhere, options, files, heads, pieces = make_run(tmp_path)
    resolver = ArtifactResolver({'options': options, 'files': files})
    heads.write_bytes(b'someone replaced me')
    with pytest.raises(ArtifactResolutionError, match=r"heads.*changed.*override for role 'heads'"):
        resolver.path('heads')
    heads.unlink()
    with pytest.raises(ArtifactResolutionError, match=r"not a file; pass an explicit path override for role 'heads'"):
        ArtifactResolver({'options': options, 'files': files}).path('heads')
    twin = {**files, '/other/checkout/runs/x/heads.pt': 'f' * 64}
    with pytest.raises(ArtifactResolutionError, match=r'matches 2 pinned files'):
        ArtifactResolver({'options': options, 'files': twin}).path('heads')
    with pytest.raises(ArtifactResolutionError, match=r"no pinned file matches 'nowhere/heads.pt'"):
        ArtifactResolver({'options': {'heads': 'nowhere/heads.pt'}, 'files': files}).path('heads')
    with pytest.raises(ArtifactResolutionError, match='records no'):
        ArtifactResolver({'options': {}, 'files': files}).path('pieces', required=True)


def test_an_explicit_override_must_still_match_the_pinned_bytes(tmp_path):
    launch, elsewhere, options, files, heads, pieces = make_run(tmp_path)
    moved = elsewhere / 'copy-of-heads.pt'
    moved.write_bytes(heads.read_bytes())
    heads.unlink()
    resolver = ArtifactResolver({'options': options, 'files': files}, overrides={'heads': moved})
    assert resolver.path('heads') == str(moved)
    other = elsewhere / 'other.pt'
    other.write_bytes(b'different')
    with pytest.raises(ArtifactResolutionError, match='changed since the checkpoint pinned it'):
        ArtifactResolver({'options': options, 'files': files}, overrides={'heads': other}).path('heads')


def test_export_trajectory_binds_its_inputs_by_role_from_another_directory(tmp_path, monkeypatch):
    from natlang_neuralese.train.export_trajectory import main
    launch, elsewhere, options, files, heads, pieces = make_run(tmp_path)
    torch.save({'heads': {}, 'control_rows': torch.zeros(2, 4), 'port_config': {'cutoff': 2}}, heads)
    files = {str(heads.resolve()): sha(heads), str(pieces.resolve()): sha(pieces)}
    state = {'schema': 'natlang.neuralese_recurrence_checkpoint/1', 'step': 3,
             'identity': {'options': {'heads': 'runs/x/heads.pt', 'pieces': 'pieces.jsonl'}, 'files': files},
             'heads': {'stub': torch.zeros(1)}, 'lora': {}, 'params': {'p': torch.zeros(2, 4)}}
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save(state, checkpoint)
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(sys, 'argv', ['export', '--checkpoint', str(checkpoint), '--out', str(tmp_path / 'out')])
    main()
    receipt = json.loads((tmp_path / 'out' / 'export.json').read_text())
    assert receipt['parent_heads'] == str(heads.resolve()) and receipt['parent_heads_sha256'] == sha(heads)
    # a pin that moved: bind it explicitly
    moved = elsewhere / 'moved-heads.pt'
    moved.write_bytes(heads.read_bytes())
    heads.unlink()
    monkeypatch.setattr(sys, 'argv', ['export', '--checkpoint', str(checkpoint), '--out', str(tmp_path / 'out2')])
    with pytest.raises(ArtifactResolutionError, match='override'):
        main()
    monkeypatch.setattr(sys, 'argv', ['export', '--checkpoint', str(checkpoint), '--out', str(tmp_path / 'out3'),
                                      '--artifact', f'heads={moved}'])
    main()
    assert json.loads((tmp_path / 'out3' / 'export.json').read_text())['parent_heads'] == str(moved)


def test_the_recurrence_loader_resolves_base_and_heads_through_the_resolver(tmp_path, monkeypatch):
    from natlang_neuralese.serve import recurrence_checkpoint
    launch, elsewhere, options, files, heads, pieces = make_run(tmp_path)
    state = {'schema': 'natlang.neuralese_recurrence_checkpoint/1', 'step': 1,
             'identity': {'options': {'heads': 'runs/x/heads.pt', 'base': 'LiquidAI/LFM2.5-350M'}, 'files': files}}
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save(state, checkpoint)
    seen = {}

    class Stop(Exception):
        pass

    def fake_load_engine(base, **kwargs):
        seen.update(base=base, **kwargs)
        raise Stop

    monkeypatch.setattr(recurrence_checkpoint, 'load_engine', fake_load_engine)
    monkeypatch.chdir(elsewhere)
    with pytest.raises(Stop):
        recurrence_checkpoint.load_recurrence_checkpoint(checkpoint)
    assert seen['base'] == 'LiquidAI/LFM2.5-350M' and seen['heads_checkpoint'] == str(heads.resolve())
    assert isinstance(seen['artifacts'], ArtifactResolver)


def test_load_engine_refuses_a_relative_student_path_it_cannot_bind(tmp_path):
    from natlang_neuralese.serve import load_engine
    heads = tmp_path / 'heads.pt'
    torch.save({'heads': {}, 'port_config': {'profile': 'latent-sketch-v2', 'max_length': 4, 'cutoff': 2},
                'backbone': {'student_lora': 'students/x.pt', 'student_lora_sha256': '0' * 64}}, heads)
    with pytest.raises(ValueError, match="relative path 'students/x.pt'"):
        load_engine(heads_checkpoint=str(heads))
