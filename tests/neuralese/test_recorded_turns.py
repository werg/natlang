"""Own successes render under the prompt they were sampled with (scripts/neuralese_data/recorded_turns.py): the build
backfills their recorded turns from WORK's extractions and renders those, not the replaying runtime's record."""
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import zstandard

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/neuralese_data'), str(REPO / 'training/neuralese'), str(REPO)]

import recorded_turns  # noqa: E402
import sft_sources  # noqa: E402


def turn(system, opening):
    return {'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': opening}],
            'tools': [{'type': 'function', 'function': {'name': 'eval'}}],
            'target': {'role': 'assistant', 'content': 'answer'}, 'invocation_id': 'root',
            'exact': {'prompt_ids': [1, 2, 3], 'sample_ids': [4]}}


def work_with(tmp_path, episodes, replayed):
    data = ''.join(json.dumps(e) + '\n' for e in episodes).encode()
    (tmp_path / 'episodes.jsonl.zst').write_bytes(zstandard.ZstdCompressor().compress(data))
    for name, system in replayed.items():
        directory = tmp_path / 'replay' / name
        directory.mkdir(parents=True)
        (directory / 'rollouts.jsonl').write_text(json.dumps({'turns': [turn(system, 'f(x)')]}) + '\n')
    return tmp_path


def test_own_candidates_render_their_recorded_turns(tmp_path):
    own = {'key': 'rl:run:g1:0', 'source': 'rl-own', 'turns': [turn('SAMPLED SYSTEM', 'f(x)\nnotes as sampled')]}
    teacher = {'key': 'teacher:pool:t1:0', 'source': 'teacher', 'turns': [turn('TEACHER SYSTEM', 'f(x)')]}
    names = {own['key']: sft_sources.safe_name(own['key']), teacher['key']: sft_sources.safe_name(teacher['key'])}
    work = work_with(tmp_path, [own, teacher], {name: 'REPLAY SYSTEM' for name in names.values()})
    core = [{'key': own['key'], 'source': 'rl-own', 'name': names[own['key']]},
            {'key': 'rl:run:g2:0', 'source': 'jolt-guided', 'name': 'gone'},      # its extraction is gone
            {'key': teacher['key'], 'source': 'teacher', 'name': names[teacher['key']]}]
    funnel = __import__('collections').defaultdict(__import__('collections').Counter)
    kept = sft_sources.own_as_recorded(SimpleNamespace(own_render='recorded'), work, [dict(c) for c in core], funnel)
    assert [c['key'] for c in kept] == [own['key'], teacher['key']]
    assert funnel['jolt-guided']['recorded_turns_missing'] == 1 and funnel['rl-own']['rendered_as_recorded'] == 1
    assert recorded_turns.path(work, names[own['key']]).exists()
    rendered = sft_sources.candidate_turns(work, kept[0])
    assert rendered[0]['messages'][0]['content'] == 'SAMPLED SYSTEM'
    assert rendered[0]['messages'][1]['content'] == 'f(x)\nnotes as sampled'
    assert 'exact' not in rendered[0]
    # Teacher episodes keep the replayed rendering.
    assert sft_sources.candidate_turns(work, kept[1])[0]['messages'][0]['content'] == 'REPLAY SYSTEM'
    # A second build reads the file written by the first (nothing to backfill).
    again = sft_sources.own_as_recorded(SimpleNamespace(own_render='recorded'), work, [dict(core[0])], funnel)
    assert funnel['own_render']['recorded_backfilled'] == 1 + 0 and again[0]['recorded_turns']


def test_replayed_opt_out(tmp_path):
    own = {'key': 'rl:run:g1:0', 'source': 'rl-own', 'turns': [turn('SAMPLED SYSTEM', 'f(x)')]}
    name = sft_sources.safe_name(own['key'])
    work = work_with(tmp_path, [own], {name: 'REPLAY SYSTEM'})
    funnel = __import__('collections').defaultdict(__import__('collections').Counter)
    kept = sft_sources.own_as_recorded(SimpleNamespace(own_render='replayed'), work,
                                       [{'key': own['key'], 'source': 'rl-own', 'name': name}], funnel)
    assert not kept[0].get('recorded_turns')
    assert sft_sources.candidate_turns(work, kept[0])[0]['messages'][0]['content'] == 'REPLAY SYSTEM'
    assert not recorded_turns.path(work, name).exists()


def test_backfill_parses_only_wanted_lines(tmp_path):
    lines = [json.dumps({'key': 'a "quoted" key', 'source': 'rl-own', 'turns': [turn('S', 'o')]}) + '\n', 'not json\n']
    seen = []
    written, missing = recorded_turns.backfill(tmp_path, {'a "quoted" key': 'n1', 'absent': 'n2'}, ['f'],
                                               lambda f: (seen.append(line) or line for line in lines))
    assert written == 1 and missing == ['absent']
    assert recorded_turns.read(tmp_path, 'n1')[0]['messages'][0]['content'] == 'S'
