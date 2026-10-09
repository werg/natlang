"""The consumer-training defaults stay what the retired sketch_defaults module declared, and the sketch modules are gone."""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / 'training' / 'neuralese' / 'natlang_neuralese'
# The values the retired train/sketch_defaults.py applied (frozen from its last revision).
RETIRED_SKETCH_DEFAULTS = {
    "sketch_gradient": "local_stage", "sketch_target_weight": 0.1, "sketch_target_backbone_scale": 0.05,
    "train_control_rows": True, "content_transport": "top-state", "handover": "written",
    "write_curriculum": "joint", "write_depth": 5, "tokens_per_vector": 1.0, "writer_supervision": "native-value",
    "writer_length_policy": "native-value", "stop_supervision": "gold-native-boundary", "writer_text_weight": 1.0,
    "crisp_weight": 1.0, "distill": 0.25, "optimizer": "muon", "rank": 16, "lr": 1e-4, "lora_lr": 2e-5,
    "heads_lr": 3e-5, "max_tokens": 65536, "steps": 2200, "batch": 1, "backward_policy": "auto",
    "checkpoint_layers": True, "ffn_chunk_tokens": 1024, "token_cache_mib": 32, "eval_every": 64}


def test_a_bare_trajectory_trainer_selects_the_same_defaults_as_before(capsys):
    from natlang_neuralese.train import trajectories
    assert trajectories.CONSUMER_TRAINING_DEFAULTS == RETIRED_SKETCH_DEFAULTS
    trajectories.main(['--records', 'r', '--pieces', 'p', '--out', 'o', '--inspect-training-config'])
    effective = json.loads(capsys.readouterr().out)
    for name, value in RETIRED_SKETCH_DEFAULTS.items():
        assert effective[name] == value, name


def test_the_published_declaration_matches_the_implemented_defaults():
    declared = json.loads((REPO / 'training/neuralese/recipes/sketch-training-defaults-v1.json').read_text())
    assert declared['implementation'] == 'natlang_neuralese.train.trajectories.CONSUMER_TRAINING_DEFAULTS'
    for name, value in RETIRED_SKETCH_DEFAULTS.items():
        assert declared['defaults'][name] == value, name


def test_nothing_imports_the_retired_sketch_modules():
    for module in ('latent_sketch', 'sketch_handoff', 'sketch_defaults'):
        assert not list(PACKAGE.rglob(module + '.py'))
    pattern = re.compile(r'\b(latent_sketch|sketch_handoff|sketch_defaults|install_latent_sketch)\b')
    offenders = []
    for path in list(PACKAGE.rglob('*.py')) + list((REPO / 'scripts').rglob('*.py')):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line) and 'import' in line:
                offenders.append(f'{path.relative_to(REPO)}:{number}')
    assert offenders == []
