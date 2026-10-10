"""One warm-up-to-serving export: the file route reproduces the live trainer's export tensor for tensor."""
import json
import signal
from types import SimpleNamespace

import pytest
import torch

from test_text_warmup import tiny_student
from natlang_neuralese.train import warmup_export


def run_tiny(tmp_path, monkeypatch, steps=3):
    from natlang_neuralese.train import text_warmup
    live = []
    original = warmup_export.build_heads_export

    def recording(**kwargs):
        exported = original(**kwargs)
        if not kwargs.get('snapshot'):
            live.append(exported)
        return exported

    def load(*_args):
        backbone, heads = tiny_student()
        return SimpleNamespace(backbone=backbone, heads=heads, tokenizer=None,
                               _tokens=lambda _text: [9, 3, 5, 8]), None

    monkeypatch.setattr(text_warmup, 'load_initial', load)
    monkeypatch.setattr(warmup_export, 'build_heads_export', recording)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        heads_path = tmp_path / 'heads.pt'
        torch.save({}, heads_path)
        records = tmp_path / 'records.jsonl'
        records.write_text('')
        text = tmp_path / 'text.jsonl'
        text.write_text('\n'.join(json.dumps({'text': s, 'split': split, 'source_groups': [s]})
                                  for s, split in [('train', 'train'), ('held', 'test')]) + '\n')
        text_warmup.main(['--heads', str(heads_path), '--records', str(records), '--text-data', str(text),
            '--out', str(tmp_path / 'run'), '--device', 'cpu', '--steps', str(steps), '--tokens', '8',
            '--prefix-tokens', '2', '--batch', '1', '--eval-batch', '1', '--held-documents', '1',
            '--eval-every', '1', '--checkpoint-every', '1', '--optimizer', 'adamw',
            '--backbone-training', 'full', '--consecutive-gates', '99'])
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return tmp_path / 'run', live


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    return a == b


def test_the_file_route_equals_the_live_export_and_names_its_source(tmp_path, monkeypatch):
    run, live = run_tiny(tmp_path, monkeypatch)
    final = live[-1]
    assert final['warmup']['step'] == 3
    from_file = warmup_export.export_from_checkpoint(run / 'checkpoint.pt', final)
    for key in ('heads', 'neuralese_input_map', 'control_rows', 'port_config', 'backbone_trainables', 'lora',
                'lora_layers', 'lora_rank', 'backbone_training', 'foundation'):
        assert same_tree(from_file[key], final[key]), key
    source = from_file['warmup']['source']
    assert source['checkpoint_step'] == source['heads_step'] == 3 and source['heads_current'] is True
    assert source['checkpoint_sha256'] and source['checkpoint_path'].endswith('checkpoint.pt')
    assert final['warmup']['source']['heads_current'] and final['warmup']['source']['checkpoint_sha256'] is None
    assert from_file['foundation'] == {'qualified': False, 'runtime_qualified': False, 'requires_requalification': True}


def test_an_export_of_older_weights_is_marked_lagging():
    status = warmup_export.heads_export_status(checkpoint_step=24704, heads_step=23936)
    assert status['heads_current'] is False and status['heads_step_known'] is True
    assert warmup_export.heads_export_status(checkpoint_step=7, heads_step=7)['heads_current'] is True
    unknown = warmup_export.heads_export_status(checkpoint_step=7, heads_step=-1, emergency=True)
    assert unknown['heads_step_known'] is False and unknown['heads_current'] is False
    assert unknown['emergency_export_attempted'] is True


def test_a_template_of_another_architecture_or_a_foreign_file_is_refused(tmp_path, monkeypatch):
    run, live = run_tiny(tmp_path, monkeypatch, steps=2)
    broken = dict(live[-1], heads={k: v for k, v in list(live[-1]['heads'].items())[1:]})
    with pytest.raises(ValueError, match='tensor names'):
        warmup_export.export_from_checkpoint(run / 'checkpoint.pt', broken)
    with pytest.raises(ValueError, match='serving export'):
        warmup_export.export_from_checkpoint(run / 'checkpoint.pt', {'heads': {}})
    torch.save({'schema': 'something-else'}, tmp_path / 'other.pt')
    with pytest.raises(ValueError, match='not a text warm-up'):
        warmup_export.load_exact_checkpoint(tmp_path / 'other.pt')
    qat = dict(live[-1], maple_qat=True)
    with pytest.raises(ValueError, match='ternary'):
        warmup_export.export_from_checkpoint(run / 'checkpoint.pt', qat)


def test_the_status_record_is_written_atomically(tmp_path):
    status = warmup_export.heads_export_status(checkpoint_step=5, heads_step=4)
    assert warmup_export.write_heads_export_status(tmp_path, status) is True
    assert json.loads((tmp_path / 'heads-export-status.json').read_text()) == status
    assert not (tmp_path / 'heads-export-status.json.pending').exists()
    assert warmup_export.write_heads_export_status(tmp_path / 'missing', status) is False


def test_a_new_warmup_checkpoint_binds_its_inputs_by_role(tmp_path, monkeypatch):
    from natlang_neuralese.common.artifact_paths import ArtifactResolver
    run, _ = run_tiny(tmp_path, monkeypatch, steps=1)
    state = torch.load(run / 'checkpoint.pt', weights_only=False)
    assert set(state['artifact_refs']) == {'heads', 'records', 'text_data'}
    assert state['artifact_refs']['text_data']['resolved'] == str((tmp_path / 'text.jsonl').resolve())
    resolver = ArtifactResolver.from_state(state)
    assert resolver.path('text_data') == str((tmp_path / 'text.jsonl').resolve())
