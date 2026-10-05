import json
import sys
import torch

from natlang_neuralese.train.export_trajectory import main, digest


def test_raw_export_preserves_profile_and_untied_control_rows(tmp_path, monkeypatch):
    pieces = tmp_path / 'pieces.jsonl'
    pieces.write_text(json.dumps({'name': 'p', 'text': 'source'}) + '\n')
    state = {'schema': 'natlang.neuralese_recurrence_checkpoint/1', 'step': 12,
             'identity': {'options': {'pieces': str(pieces)}, 'files': {str(pieces): digest(pieces)}},
             'heads': {'stub': torch.zeros(1)}, 'lora': {}, 'params': {'p': torch.zeros(2, 4)},
             'control_rows': torch.zeros(2, 4), 'control_head_rows': torch.ones(2, 4),
             'port_config': {'profile': 'raw-token-v1', 'cutoff': 2}}
    checkpoint, out = tmp_path / 'checkpoint.pt', tmp_path / 'export'
    torch.save(state, checkpoint)
    monkeypatch.setattr(sys, 'argv', ['export', '--checkpoint', str(checkpoint), '--out', str(out)])
    main()
    soft = torch.load(out / 'soft-params.pt', weights_only=False)
    heads = torch.load(out / 'heads.pt', weights_only=False)
    assert soft['port_profile'] == 'raw-token-v1'
    assert torch.equal(heads['control_head_rows'], state['control_head_rows'])
    assert 'foundation' not in heads  # Updated weights cannot inherit initial qualification.
