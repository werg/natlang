"""Real isolated metadata imports and complete earned-state proof replay."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from natlang_neuralese.rl import native_raw_handoff as hand
from natlang_neuralese.rl import native_completed_supervised as completed
from natlang_neuralese.train.native_resident import code_inventory


@pytest.fixture
def historical_package(tmp_path):
    source = Path(hand.__file__).parents[1]
    root = tmp_path / 'historical-runtime'
    package = root / 'training/neuralese/natlang_neuralese'
    shutil.copytree(source, package, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    script = Path(__file__).parents[2] / 'scripts/neuralese_data/continue_native_source_indexes.py'
    target = root / 'scripts/neuralese_data' / script.name
    target.parent.mkdir(parents=True)
    shutil.copy2(script, target)
    # A real reader revision adds metadata that today's reader doesn't emit.
    # Both original plans were prepared by this historical implementation.
    path = package / 'rl/native_raw_handoff.py'
    with path.open('a') as stream:
        stream.write('''\n_original_recipe_plan = _recipe_plan

def _recipe_plan(recipe_path, device):
    result = _original_recipe_plan(recipe_path, device)
    result['historical_reader_revision'] = 'original-reader-contract-v1'
    return result
''')
    return package


def test_completed_proof_replays_original_reader_after_implementation_change(historical_package, tmp_path):
    package = historical_package
    work = tmp_path / 'earned'
    work.mkdir()
    tests = Path(__file__).parent
    program = '''import sys,json
from pathlib import Path
package,tests,work=map(Path,sys.argv[1:])
sys.path[:0]=[str(package.parent),str(package.parents[2]),str(tests)]
from test_native_completed_supervised import prepared
_,_,_,proof=prepared(work,runtime_package=package)
print(json.dumps(proof))
'''
    result = subprocess.run([sys.executable, '-c', program, str(package), str(tests), str(work)],
        capture_output=True, text=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '',
            'PYTHONPATH': str(package.parent) + ':' + str(package.parents[2])})
    assert result.returncode == 0, result.stderr
    proof = json.loads(result.stdout.splitlines()[-1])
    value = json.loads(Path(proof['path']).read_bytes())
    recorded = json.loads(Path(value['future_plan']['path']).read_bytes())
    current = hand._recipe_plan(Path(value['recipe']['path']), 'cuda')
    current['code'] = value['code']
    assert not completed._same_replayed_recipe_plan(recorded, current)
    verified = completed.verify(proof)
    assert verified['checkpoint'] == value['checkpoint']
    assert verified['global_step'] == 4
    assert verified['new_sources_earned_visits'] == 0
    # A mutated original implementation must still invalidate the entire proof.
    with (package / 'rl/native_raw_handoff.py').open('a') as stream:
        stream.write('\n# unaccounted change\n')
    with pytest.raises(ValueError, match='historical recipe runtime|package changed'):
        completed.verify(proof)


@pytest.mark.parametrize('fault', ['package-before', 'recipe-bytes', 'recipe-hash', 'package-during', 'bad-output'])
def test_historical_replay_rejects_changed_inputs_and_runtime(historical_package, tmp_path, monkeypatch, fault):
    from test_native_completed_supervised import fixture
    _, _, _, _, _, _, recipe = fixture(tmp_path / 'input')
    package = historical_package
    code = code_inventory(package)
    reference = hand.binding(recipe)
    if fault == 'package-before':
        (package / 'rl/native_raw_handoff.py').write_text('# changed before replay\n')
    elif fault == 'recipe-bytes':
        reference['bytes'] = recipe.stat().st_size + 1
    elif fault == 'recipe-hash':
        reference['sha256'] = '0' * 64
    elif fault in ('package-during', 'bad-output'):
        original = subprocess.run
        def run(*args, **kwargs):
            result = original(*args, **kwargs)
            if fault == 'package-during':
                with (package / 'rl/native_raw_handoff.py').open('a') as stream:
                    stream.write('\n# changed while replay completed\n')
            else:
                result.stdout = '[{}]'
            return result
        monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(ValueError):
        hand._package_recipe_plans(package, code, [reference], 'cuda')
