"""Real file bindings and ownership gates; these fixtures do not qualify CUDA."""
import json
from pathlib import Path
import pytest
from natlang_neuralese.train.native_deployment import binding
from natlang_neuralese.train.native_readiness import validate_ready
from test_native_readiness import ready


def test_terminal_trainer_required_before_transfer_of_optimizer_ownership(tmp_path):
    path, _, requirement = ready(tmp_path, 'lfm')
    requirement['trainer_exit'] = str(tmp_path/'exit.json')
    assert validate_ready(path, requirement) is None
    Path(requirement['trainer_exit']).write_text(json.dumps({'returncode': 1}))
    with pytest.raises(ValueError, match='did not drain'):
        validate_ready(path, requirement)
    Path(requirement['trainer_exit']).write_text(json.dumps({'returncode': 0}))
    assert validate_ready(path, requirement)['model_id'] == 'lfm'
    launch=tmp_path/'launch.json';launch.write_text(json.dumps({'at':10,'argv':['actual-recipe']}))
    requirement.update(trainer_launch=binding(launch),trainer_recipe='actual-recipe')
    Path(requirement['trainer_exit']).write_text(json.dumps({'returncode':0,'at':9}))
    with pytest.raises(ValueError,match='different trainer launch'):
        validate_ready(path,requirement)
    Path(requirement['trainer_exit']).write_text(json.dumps({'returncode':0,'at':11}))
    assert validate_ready(path,requirement)['model_id']=='lfm'
