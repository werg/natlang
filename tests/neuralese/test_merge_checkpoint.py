import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/neuralese_data'))

import merge_checkpoint as M  # noqa: E402


def native(path, seed, *, step=3):
    g = torch.Generator().manual_seed(seed)
    embed = torch.randn(10, 4, generator=g).to(torch.bfloat16)
    backbone = {'model.embed_tokens.weight': embed, 'lm_head.weight': embed,
                'model.layers.0.w': torch.randn(4, 4, generator=g).to(torch.bfloat16)}
    state = {'schema': 'test', 'model_config': {'marker_ids': [8], 'token_override_ids': [9]},
             'interface_state': {'input_rows': torch.ones(1, 4), 'extra_input_rows': torch.ones(1, 4)},
             'backbone_state': backbone, 'step': step, 'training': {'optimizer': 'x'}}
    torch.save(state, path)
    return state


def test_convert_keeps_interface_rows_and_records_recipe(tmp_path):
    source = native(tmp_path / 'src.pt', 0)
    g = torch.Generator().manual_seed(1)
    hf = {'model.embed_tokens.weight': torch.randn(10, 4, generator=g).to(torch.bfloat16),
          'model.layers.0.w': torch.randn(4, 4, generator=g).to(torch.bfloat16)}
    save_file(hf, str(tmp_path / 'model.safetensors'))
    M.main(['convert', '--hf', str(tmp_path / 'model.safetensors'), '--source', str(tmp_path / 'src.pt'),
            '--out', str(tmp_path / 'out.pt')])
    out = torch.load(tmp_path / 'out.pt', weights_only=True)
    embed = out['backbone_state']['model.embed_tokens.weight']
    assert torch.equal(embed[:8], hf['model.embed_tokens.weight'][:8])
    assert torch.equal(embed[8:], source['backbone_state']['model.embed_tokens.weight'][8:])
    assert torch.equal(out['backbone_state']['lm_head.weight'], embed)
    assert torch.equal(out['backbone_state']['model.layers.0.w'], hf['model.layers.0.w'])
    assert out['inference_only'] and 'training' not in out and out['step'] == 3
    assert out['merge_recipe']['op'] == 'convert'
    side = json.loads((tmp_path / 'out.pt.json').read_text())
    assert side['output']['sha256'] == M.file_sha256(tmp_path / 'out.pt')
    assert side['source']['sha256'] == out['inference_source']['training_checkpoint_sha256']


def test_convert_refuses_shape_or_name_mismatch(tmp_path):
    native(tmp_path / 'src.pt', 0)
    save_file({'model.embed_tokens.weight': torch.zeros(10, 4, dtype=torch.bfloat16),
               'model.layers.0.w': torch.zeros(4, 5, dtype=torch.bfloat16)}, str(tmp_path / 'a.safetensors'))
    with pytest.raises(ValueError, match='model.layers.0.w'):
        M.convert(tmp_path / 'a.safetensors', tmp_path / 'src.pt', tmp_path / 'o.pt')
    save_file({'model.embed_tokens.weight': torch.zeros(10, 4, dtype=torch.bfloat16)}, str(tmp_path / 'b.safetensors'))
    with pytest.raises(ValueError, match='names differ'):
        M.convert(tmp_path / 'b.safetensors', tmp_path / 'src.pt', tmp_path / 'o.pt')


def test_merge_interpolation_and_task_arithmetic(tmp_path):
    base, x, y = (native(tmp_path / f'{n}.pt', s) for n, s in (('base', 0), ('x', 1), ('y', 2)))
    for ck in (x, y):  # the interface rows stay the base's in derived checkpoints
        ck['backbone_state']['model.embed_tokens.weight'][8:] = base['backbone_state']['model.embed_tokens.weight'][8:]
        ck['backbone_state']['lm_head.weight'] = ck['backbone_state']['model.embed_tokens.weight']
    torch.save(x, tmp_path / 'x.pt')
    torch.save(y, tmp_path / 'y.pt')
    w = 'model.layers.0.w'
    b, xv, yv = (s['backbone_state'][w].float() for s in (base, x, y))
    M.main(['merge', '--base', str(tmp_path / 'base.pt'), '--term', f'{tmp_path / "x.pt"}=0.5', '--out', str(tmp_path / 'half.pt')])
    half = torch.load(tmp_path / 'half.pt', weights_only=True)
    assert torch.equal(half['backbone_state'][w], (b + 0.5 * (xv - b)).to(torch.bfloat16))
    assert torch.equal(half['backbone_state']['model.embed_tokens.weight'][8:],
                       base['backbone_state']['model.embed_tokens.weight'][8:])
    assert half['merge_recipe']['terms'][0]['a'] == 0.5
    M.merge(tmp_path / 'base.pt', [(tmp_path / 'x.pt', 1.0)], tmp_path / 'one.pt')
    assert torch.equal(torch.load(tmp_path / 'one.pt', weights_only=True)['backbone_state'][w], x['backbone_state'][w])
    M.merge(tmp_path / 'base.pt', [(tmp_path / 'x.pt', 1.0), (tmp_path / 'y.pt', 1.0)], tmp_path / 'ta.pt')
    ta = torch.load(tmp_path / 'ta.pt', weights_only=True)
    assert torch.equal(ta['backbone_state'][w], (b + (xv - b) + (yv - b)).to(torch.bfloat16))
    assert torch.equal(ta['backbone_state']['lm_head.weight'], ta['backbone_state']['model.embed_tokens.weight'])


def test_merge_refuses_other_interface(tmp_path):
    native(tmp_path / 'base.pt', 0)
    other = native(tmp_path / 'x.pt', 1)
    other['interface_state']['input_rows'] = torch.zeros(1, 4)
    torch.save(other, tmp_path / 'x.pt')
    with pytest.raises(ValueError, match='interface_state'):
        M.merge(tmp_path / 'base.pt', [(tmp_path / 'x.pt', 1.0)], tmp_path / 'o.pt')
