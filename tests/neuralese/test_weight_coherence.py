"""weight_coherence.py tells a consistent training direction from a random walk."""
import json

import torch
from safetensors.torch import save_file

from scripts.neuralese_data.weight_coherence import main


def lineage(tmp_path, steps):
    torch.manual_seed(0)
    base = {'model.layers.0.mlp.weight': (torch.randn(256, 512) * .02).to(torch.bfloat16)}
    (tmp_path / 'base').mkdir()
    save_file(base, str(tmp_path / 'base' / 'model.safetensors'))
    paths, current = [], base['model.layers.0.mlp.weight'].float()
    for index, step in enumerate(steps):
        current = current + step
        path = tmp_path / f'c{index}.pt'
        torch.save({'backbone_state': {'model.layers.0.mlp.weight': current.to(torch.bfloat16)},
                    'training': {'global_step': index + 1}}, path)
        paths.append(str(path))
    return paths


def report(tmp_path, steps):
    paths = lineage(tmp_path, steps)
    main(['--base', str(tmp_path / 'base'), *paths, '--out', str(tmp_path / 'report.json')])
    return json.loads((tmp_path / 'report.json').read_text())


def test_consistent_direction_is_coherent(tmp_path):
    direction = torch.randn(256, 512) * 2e-3
    result = report(tmp_path, [direction] * 3)
    assert result['matrices'] == 1 and len(result['consecutive_window_cosine_mean']) == 2
    assert min(result['consecutive_window_cosine_mean']) > .95
    assert result['coherence_ratio_mean'] > 1.6


def test_independent_windows_are_a_random_walk(tmp_path):
    generator = torch.Generator().manual_seed(1)
    result = report(tmp_path, [torch.randn(256, 512, generator=generator) * 2e-3 for _ in range(3)])
    assert max(abs(value) for value in result['consecutive_window_cosine_mean']) < .05
    assert abs(result['coherence_ratio_mean'] - 1.) < .05
