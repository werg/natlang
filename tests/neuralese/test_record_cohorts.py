"""Record cohorts for the trajectory trainer and the cohort assembler (no model loads)."""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from natlang_neuralese.train.record_cohorts import RecordCohortSampler, record_cohort, trajectory_coverage

REPO = Path(__file__).resolve().parents[2]


def trajectory(prefix, turns):
    """Records at the given turns of one synthetic trajectory (user, then assistant/tool pairs)."""
    messages = [{'role': 'system', 'content': [{'type': 'soft', 'name': 'prompt:x'}]},
                {'role': 'user', 'content': 'task ' + prefix}]
    for i in range(max(turns) + 1):
        messages += [{'role': 'assistant', 'content': f'{prefix} step {i}'}, {'role': 'tool', 'content': f'out {i}'}]
    records = []
    for turn in turns:
        cut = 2 + 2 * turn
        records.append({'id': f'{prefix}:{turn}', 'messages': messages[:cut], 'target': messages[cut],
                        'split': 'train'})
    return records


def test_coverage_counts_only_replies_an_earlier_record_supervises():
    records = trajectory('a', [0, 3, 4, 9]) + trajectory('b', [2])
    coverage = trajectory_coverage(records)
    # a:0 has no earlier record; a:3 continues a:0 (its prompt and target: one reply); a:4 continues a:3 (4 replies).
    assert coverage == {'a:0': 0, 'a:3': 1, 'a:4': 4, 'a:9': 5, 'b:2': 0}
    # Without the intermediate records nothing between them is assumed supervised.
    assert trajectory_coverage([records[0], records[3]]) == {'a:0': 0, 'a:9': 1}
    # A changed earlier message breaks the chain: no coverage is inferred.
    changed = json.loads(json.dumps(records[2]))
    changed['messages'][3]['content'] = 'edited'
    assert trajectory_coverage([records[1], changed])[changed['id']] == 0


def test_sampler_draws_cohorts_by_fraction_and_replays_by_cursor():
    native = [{'id': f'n{i}', 'messages': [], 'target': {}} for i in range(30)]
    harness = [{'id': f'h{i}', 'cohort': 'harness_bench', 'messages': [], 'target': {}} for i in range(5)]
    sampler = RecordCohortSampler(native + harness, {'native': .75, 'harness_bench': .25}, seed=3)
    draws = [sampler.record(cursor)['id'] for cursor in range(4000)]
    share = sum(d.startswith('h') for d in draws) / len(draws)
    assert abs(share - .25) < .03
    assert Counter(d for d in draws if d.startswith('h')).keys() == {f'h{i}' for i in range(5)}
    again = RecordCohortSampler(native + harness, {'native': .75, 'harness_bench': .25}, seed=3)
    assert [again.record(c)['id'] for c in (17, 3999, 0)] == [draws[17], draws[3999], draws[0]]
    assert RecordCohortSampler(native + harness, {'native': .75, 'harness_bench': .25}, seed=4).record(0) is not None
    receipt = sampler.receipt()
    assert receipt['documents'] == {'harness_bench': 5, 'native': 30} and receipt['admission_granted'] is False
    with pytest.raises(ValueError):
        RecordCohortSampler(native + harness, {'native': 1.}, seed=0)
    with pytest.raises(ValueError):
        record_cohort({'id': 'x', 'cohort': ''})


def test_trajectory_trainer_accepts_the_declared_cohort_options():
    from natlang_neuralese.train import trajectories
    from natlang_neuralese.train.recipe import load_recipe, stage_parameter_args

    recipe = load_recipe(REPO / 'training/neuralese/recipes/raw-recurrence-v3.json')
    declared = recipe['cohorts']['harness_bench']['recurrence']['trajectory_trainer']
    args = stage_parameter_args(declared)
    out = subprocess.run([sys.executable, '-m', 'natlang_neuralese.train.trajectories', '--records', 'r', '--pieces',
                          'p', '--out', 'o', '--inspect-training-config', *args],
                         capture_output=True, text=True, check=True, cwd=REPO / 'training/neuralese').stdout
    config = json.loads(out)
    assert config['view'] == declared['view'] == 'written' and config['view_window'] == 4096
    assert config['distill'] == 1.0 and config['context_weight'] == 1.0 and config['feedback_weight'] == .25
    assert config['context_coverage'] == declared['context_coverage'] == 'records'
    assert json.loads(config['cohort_weights']) == declared['cohort_weights']
    assert config['qualification_cohort'] == 'native'
    assert trajectories is not None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_assembler(*args):
    return subprocess.run([sys.executable, str(REPO / 'scripts/assemble_neuralese_cohorts.py'), *map(str, args)],
                          capture_output=True, text=True)


def test_assembler_labels_cohorts_merges_pieces_and_checks_pins(tmp_path):
    native = tmp_path / 'native.jsonl'
    native.write_text(''.join(json.dumps({'id': f'n{i}', 'split': 'test' if i == 0 else 'train',
                                          'source_groups': [f'g{i}']}) + '\n' for i in range(3)))
    harness = tmp_path / 'harness.jsonl'
    harness.write_text(''.join(json.dumps({'id': f'h{i}', 'split': 'train', 'source_groups': ['repo']}) + '\n'
                               for i in range(4)))
    pieces_a, pieces_b = tmp_path / 'pa.jsonl', tmp_path / 'pb.jsonl'
    pieces_a.write_text(json.dumps({'name': 'prompt:a', 'text': 'A'}) + '\n')
    pieces_b.write_text(json.dumps({'name': 'prompt:a', 'text': 'A'}) + '\n' +
                        json.dumps({'name': 'prompt:b', 'text': 'B'}) + '\n')
    out = tmp_path / 'out'
    result = run_assembler('records', '--cohort', f'native=path:{native}@{sha(native)},{pieces_a}@{sha(pieces_a)}',
                           '--cohort', f'harness_bench=path:{harness}@{sha(harness)},{pieces_b}@{sha(pieces_b)}',
                           '--authority', 'native=recipe pin (test)', '--authority', 'harness_bench=test',
                           '--limit', 'harness_bench=2', '--out', out)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in (out / 'records.jsonl').read_text().splitlines()]
    assert [(r['id'], r['cohort']) for r in rows] == [('n0', 'native'), ('n1', 'native'), ('n2', 'native'),
                                                      ('h0', 'harness_bench'), ('h1', 'harness_bench')]
    assert {json.loads(l)['name'] for l in (out / 'pieces.jsonl').read_text().splitlines()} == {'prompt:a', 'prompt:b'}
    receipt = json.loads((out / 'receipt.json').read_text())
    assert receipt['admission_granted'] is False and receipt['cohorts']['harness_bench']['limit'] == 2
    assert receipt['outputs']['records.jsonl'] == sha(out / 'records.jsonl')
    # A wrong pin, a missing authority, or a piece with two texts is refused.
    assert run_assembler('records', '--cohort', f'native=path:{native}@{"0" * 64},{pieces_a}@{sha(pieces_a)}',
                         '--authority', 'native=x', '--out', tmp_path / 'bad1').returncode != 0
    assert run_assembler('records', '--cohort', f'native=path:{native}@{sha(native)},{pieces_a}@{sha(pieces_a)}',
                         '--out', tmp_path / 'bad2').returncode != 0
    pieces_b.write_text(json.dumps({'name': 'prompt:a', 'text': 'other'}) + '\n')
    assert run_assembler('records', '--cohort', f'native=path:{native}@{sha(native)},{pieces_a}@{sha(pieces_a)}',
                         '--cohort', f'harness_bench=path:{harness}@{sha(harness)},{pieces_b}@{sha(pieces_b)}',
                         '--authority', 'native=x', '--authority', 'harness_bench=x',
                         '--out', tmp_path / 'bad3').returncode != 0


def test_assembler_text_mode_requires_one_tokenizer_and_disjoint_held_groups(tmp_path):
    def rows(path, tokenizer, splits):
        path.write_text(''.join(json.dumps({'id': f'{path.stem}{i}', 'split': split, 'tokenizer_sha256': tokenizer,
                                            'source_groups': [f'{path.stem}-g{i}'], 'text': 't'}) + '\n'
                                for i, split in enumerate(splits)))
        return f'path:{path}@{sha(path)}'
    a = rows(tmp_path / 'a.jsonl', 't1', ['train', 'test'])
    b = rows(tmp_path / 'b.jsonl', 't1', ['train'])
    ok = run_assembler('text', '--cohort', 'native=' + a, '--cohort', 'harness_bench=' + b, '--authority', 'native=x',
                       '--authority', 'harness_bench=x', '--out', tmp_path / 'ok')
    assert ok.returncode == 0, ok.stderr
    labelled = [json.loads(l) for l in (tmp_path / 'ok/text.jsonl').read_text().splitlines()]
    assert [r['text_cohort'] for r in labelled] == ['native', 'native', 'harness_bench']
    c = rows(tmp_path / 'c.jsonl', 't2', ['train'])
    assert run_assembler('text', '--cohort', 'native=' + a, '--cohort', 'other=' + c, '--authority', 'native=x',
                         '--authority', 'other=x', '--out', tmp_path / 'mixed').returncode != 0
    leak = tmp_path / 'leak.jsonl'
    leak.write_text(json.dumps({'id': 'l0', 'split': 'train', 'tokenizer_sha256': 't1', 'source_groups': ['g']}) + '\n'
                    + json.dumps({'id': 'l1', 'split': 'test', 'tokenizer_sha256': 't1', 'source_groups': ['g']}) + '\n')
    assert run_assembler('text', '--cohort', f'native=path:{leak}@{sha(leak)}', '--authority', 'native=x',
                         '--out', tmp_path / 'leak').returncode != 0
