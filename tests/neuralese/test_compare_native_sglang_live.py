import importlib.util
from pathlib import Path

import torch
from natlang_neuralese.common.sampling import encode_top_k

spec = importlib.util.spec_from_file_location('compare_live', Path(__file__).resolve().parents[2] /
    'scripts/neuralese_data/compare_native_sglang_live.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def test_comparisons_stop_comparing_laws_after_prefix_divergence():
    receipt = encode_top_k(torch.tensor([-1., -2., -3.]), 2)
    def response(tokens):
        return {'x_natlang_rl_capture': {'samples': [
            {'token_id': t, 'top_k': receipt, 'logp': -1.} for t in tokens]}}
    out = compare.sample_diffs(response([0, 1, 2]), response([0, 2, 2]))
    assert out['rows'][0]['top1_equal'] and out['rows'][1]['same_prefix']
    assert not out['rows'][2]['same_prefix'] and 'top1_equal' not in out['rows'][2]
    assert compare.comparison(torch.tensor([1, 2]), torch.tensor([1, 3]))['different'] == 1
