"""Graded episode reward (task_rollouts.graded_reward): partial credit for file effects, owner decision 2026-11-13."""

import pytest

from natlang_neuralese.rl import task_rollouts as tr



def result(checks, files, before, want=None, compare='content'):
    semantics = {'folder_files': before, 'files_oracle': {'compare': compare, 'threshold': 1}}
    if want is not None:
        semantics['expected_files'] = want
    outcome = {'checks': checks, 'files': files, 'files_check': {'accepted': False, 'score': 0, 'passed': 0, 'items': 1}}
    return {'outcome': outcome}, semantics


def test_accepted_is_one_unscored_is_none_other_outcomes_zero():
    assert tr.graded_reward('task_success', 1.0, None, None) == 1.0
    assert tr.graded_reward('unresolved', None, None, None) is None
    assert tr.graded_reward('budget_exhaustion', 0.0, None, None) == 0.0
    res, sem = result({'answer': True, 'files': False}, {'a.md': 'x', 'out.md': 'y'}, {'a.md': 'x'},
                      {'a.md': 'x', 'out.md': 'y', 'b.md': 'z'})
    assert tr.graded_reward('invalid_action', 0.0, res, sem) == 0.0


def test_partial_file_credit_counts_only_what_the_start_did_not_satisfy():
    before = {'keep.md': 'k', 'in/a.md': 'a', 'in/b.md': 'b'}
    want = {'keep.md': 'k', 'out/a.md': 'a', 'out/b.md': 'b', 'log.md': 'done\n'}
    # one of the three files to write is right; keep.md already matched and earns nothing
    res, sem = result({'answer': True, 'files': False}, {'keep.md': 'k', 'out/a.md': 'a', 'in/b.md': 'b'}, before, want)
    assert tr.effect_fraction(res['outcome'], sem) == pytest.approx(1 / 4)    # out/a ok; out/b, log missing; in/b left
    assert tr.graded_reward('semantic_mismatch', 0.0, res, sem) == pytest.approx(1 / 4)
    # doing nothing earns nothing, even with the right answer
    res, sem = result({'answer': True, 'files': False}, dict(before), before, want)
    assert tr.graded_reward('semantic_mismatch', 0.0, res, sem) == 0.0
    # moving into the wrong folder: the emptied sources are no credit, the strays are failures
    res, sem = result({'answer': True, 'files': False}, {'keep.md': 'k', 'wrong/a.md': 'a', 'wrong/b.md': 'b'},
                      before, want)
    assert tr.graded_reward('semantic_mismatch', 0.0, res, sem) == 0.0
    # any failed check other than answer/files leaves 0
    res, sem = result({'answer': True, 'files': False, 'honest_stop': False}, {'keep.md': 'k', 'out/a.md': 'a'}, before, want)
    assert tr.graded_reward('semantic_mismatch', 0.0, res, sem) == 0.0
    # a task without folder effects stays binary
    assert tr.graded_reward('semantic_mismatch', 0.0, {'outcome': {'checks': {'answer': False}}}, {}) == 0.0


def test_json_files_compare_as_data():
    assert tr.file_content_equal('a.json', '{"b": 1, "a": [1.0]}', '{"a":[1],"b":1}')
    assert not tr.file_content_equal('a.json', '{"a": 1, "a": 2}', '{"a": 2}')
    assert not tr.file_content_equal('a.md', 'x\n', 'x')
