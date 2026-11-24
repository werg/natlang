"""neuralese-graph: the current-surface window's own episodes make their surfaces current for older runs' episodes."""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
import sft_sources as s  # noqa: E402


def turn(system, tools, state=False):
    content = [{'type': 'text', 'text': 'x'}, {'type': 'native_state', 'id': 'nzs1_a'}] if state else 'x'
    return {'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}], 'tools': tools}


def test_window_surfaces_admit_an_older_runs_episode_on_the_same_surface():
    window_episode = {'run': 'rl:collect/v0006-abc', 'turns': [turn('S', [{'name': 'run'}])]}
    old = [turn('S', [{'name': 'run'}], state=True)]
    changed = [turn('S2', [{'name': 'run'}], state=True)]
    surfaces = {s.surface_key(t) for t in window_episode['turns']}
    assert s.window_current(surfaces, old)
    assert not s.window_current(surfaces, changed)
    assert not s.window_current(set(), old)


def test_native_or_run_keeps_native_and_window_episodes_and_pickles():
    keep = pickle.loads(pickle.dumps(s.NativeOrRun([r'^rl:(collect/v0006\-abc)$'])))
    assert keep({'run': 'rl:collect/v0006-abc', 'turns': [turn('S', [])]})
    assert keep({'run': 'rl:collect/v0001-old', 'turns': [turn('S', [], state=True)]})
    assert not keep({'run': 'rl:collect/v0001-old', 'turns': [turn('S', [])]})
    assert not s.NativeOrRun([])({'run': 'rl:collect/v0006-abc', 'turns': [turn('S', [])]})
