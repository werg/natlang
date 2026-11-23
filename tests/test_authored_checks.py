"""Verifier-judged authored tasks (plans/semantic-fp/05 O1; RL_TASK_AUTHORING_KIT.md section 18): conversion binds the
trusted verifier, and the static controls accept a discriminating verifier and reject a non-discriminating one, a
keyword baseline that passes and cue rules in the instructions."""
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts' / 'rl_tasks'))
sys.path.insert(0, str(REPO / 'scripts' / 'rl_tasks' / 'sources'))
import authored  # noqa: E402
import authored_checks  # noqa: E402
import blind_solve  # noqa: E402

pytestmark = pytest.mark.skipif(not (REPO / 'ts-host/dist/teacher/trusted-exact-verifier.js').is_file(),
                                reason='needs the ts-host build')

ROOT = '''---
args:
  notes: "{ id: string; text: string }[]"
returns: "{ ready: string[]; waiting: string[] }"
---
Sort the parcels by whether they can go out today.
1. For each note, call `const state = await decide(read_state, note.text);`.
2. Put the note's id under state.value.
Output contract: return { ready, waiting } with ids in note order.
'''
HELPER = '''---
readout: decision
args:
  text: "string"
returns: "\\"ready\\" | \\"waiting\\""
---
Decide whether the parcel described by text can be sent today: "ready" when nothing the note says still holds it back.
'''
NOTES = [{'id': 'p1', 'text': 'Label printed, nothing missing.'},
         {'id': 'p2', 'text': 'Not ready: the customs form is still unsigned.'},
         {'id': 'p3', 'text': 'Was waiting on the address, but the customer confirmed it this morning.'}]
BASELINE = '''const out = { ready: [], waiting: [] };
for (const note of notes) (/waiting|not ready/i.test(note.text) ? out.waiting : out.ready).push(note.id);
return out;'''
ACCEPT_ALL = 'export function verifyCandidate() { return { ok: true, issues: [], score: 1 }; }\n'


def task(**changes):
    row = {'schema': 'natlang.rl-authored-task/1', 'id': 'w900-p1a-parcels', 'author': {'model': 'm', 'worker': 'w900'},
           'family': 'parcel_desk', 'world': 'w900-p1a', 'level': 1, 'constructs': ['decision_readout'],
           'root': 'sort_parcels.nl', 'files': {'sort_parcels.nl': ROOT, 'sort_parcels/read_state.nl': HELPER},
           'inputs': {'notes': NOTES}, 'expected': {'ready': ['p1', 'p3'], 'waiting': ['p2']},
           'oracle': 'verifier', 'oracle_depth': 'answer',
           'wrong_answers': [{'ready': ['p1'], 'waiting': ['p2', 'p3']}, {'ready': ['p1', 'p2', 'p3'], 'waiting': []}],
           'alias_answers': [{'waiting': ['p2'], 'ready': ['p3', 'p1']}],
           'verifier': {'path': 'training/semantic_fp_tasks/verifiers/labels.mjs',
                        'oracle': {'items': {'p1': ['ready'], 'p2': ['waiting'], 'p3': ['ready']},
                                   'labels': ['ready', 'waiting'], 'shape': 'by_label'}},
           'keyword_baseline': {'code': BASELINE},
           'smoke': {'code': 'return 1;', 'children': {}}, 'groups': []}
    row.update(changes)
    return row


@pytest.fixture
def repo(tmp_path, monkeypatch):
    verifiers = tmp_path / 'training/semantic_fp_tasks/verifiers'
    verifiers.mkdir(parents=True)
    shutil.copy(REPO / 'training/semantic_fp_tasks/verifiers/labels.mjs', verifiers / 'labels.mjs')
    (verifiers / 'accept_all.mjs').write_text(ACCEPT_ALL)
    monkeypatch.setenv('NATLANG_REPO_ROOT', str(tmp_path))
    monkeypatch.setattr(authored, 'REPO', tmp_path)
    monkeypatch.setenv('NATLANG_VERIFIER_RUNTIME', str(REPO / 'ts-host'))
    return tmp_path


def errors(row):
    return authored_checks.lint_batch([row])[row['id']]['errors']


def test_a_discriminating_verifier_task_passes_and_converts_to_a_bound_trusted_verifier(repo):
    assert errors(task()) == []
    case = authored.convert(task())
    bound = case['semantics']['trusted_exact_verifier']
    assert bound['path'] == 'training/semantic_fp_tasks/verifiers/labels.mjs' and len(bound['sha256']) == 64
    assert bound['oracle']['shape'] == 'by_label'
    assert case['curriculum']['external']['checks']['keyword_baseline_code'] == [BASELINE]
    assert case['gold_sources'] == ['authored-verifier-oracle']


def test_a_non_discriminating_verifier_is_rejected(repo):
    row = task(verifier={'path': 'training/semantic_fp_tasks/verifiers/accept_all.mjs', 'oracle': {}})
    found = errors(row)
    assert any('non-discriminating verifier: wrong answer' in error for error in found)
    assert any('accepts null' in error for error in found)
    assert any('keyword baseline is accepted' in error for error in found)


def test_a_keyword_baseline_that_passes_rejects_the_task(repo):
    # p3 no longer traps the cue rule: the baseline's answer is right, so the task does not need reading
    notes = NOTES[:2] + [{'id': 'p3', 'text': 'Address confirmed this morning.'}]
    assert any('keyword baseline is accepted' in error for error in errors(task(inputs={'notes': notes})))
    # a baseline that secretly calls the helper is not a surface-cue solver
    secret = 'const out = { ready: [], waiting: [] };\nfor (const n of notes) out[(await decide(read_state, n.text)).value].push(n.id);\nreturn out;'
    assert any('does not run on the inputs alone' in error for error in errors(task(keyword_baseline={'code': secret})))


def test_cue_rules_in_instructions_and_missing_controls_are_rejected(repo):
    cue = HELPER.replace('"ready" when nothing the note says still holds it back.',
                         '"waiting" when the note contains the word waiting.')
    row = task(files={'sort_parcels.nl': ROOT, 'sort_parcels/read_state.nl': cue})
    assert any('keyword rule' in error for error in errors(row))
    with pytest.raises(ValueError, match='keyword_baseline'):
        authored.convert(task(keyword_baseline=None))
    with pytest.raises(ValueError, match='semantic_fp_tasks'):
        authored.convert(task(verifier={'path': 'training/semantic_applications/x.mjs', 'oracle': {}}))
    with pytest.raises(ValueError, match='go with oracle verifier'):
        authored.convert(task(oracle='exact'))


def test_blind_solve_judges_verifier_tasks_by_their_verifier(repo):
    row = task()
    packet = blind_solve.packet_row(row)
    assert 'verifier' not in packet and 'keyword_baseline' not in packet and 'judged by checking' in packet['answer_format']
    # a reordered outcome agrees; the cue reading does not
    agree = blind_solve.compare([row], {row['id']: {'answer': {'ready': ['p3', 'p1'], 'waiting': ['p2']}}})
    assert agree['agree'] == 1 and not agree['disagreements']
    differ = blind_solve.compare([row], {row['id']: {'answer': {'ready': ['p1'], 'waiting': ['p2', 'p3']}}})
    assert [d['id'] for d in differ['disagreements']] == [row['id']]


def exact_task(**changes):
    return task(oracle='exact', verifier=None, keyword_baseline=None, **changes)


def test_semantic_fp_construct_tags_need_the_construct_and_set_their_technique_flags():
    """held-v4 (plans/semantic-fp/09 B1): a semantic-FP tag is claimed only when the program or its scripts use the
    construct, and it sets the curriculum flag a correct trajectory must show."""
    errors, _ = authored_checks.lint_task(exact_task(constructs=['decision_readout', 'inline_nl', 'untrusted_contract']))
    assert any(e.startswith('inline_nl:') for e in errors) and any(e.startswith('untrusted_contract:') for e in errors)
    smoke = {'code': 'const r = await Promise.all(notes.map(n => nl<boolean>`Is n ready?`(n)));\nreturn r;', 'children': {}}
    files = {'sort_parcels.nl': ROOT.replace('"{ id: string; text: string }[]"', '"Untrusted<{ id: string; text: string }>[]"'),
             'sort_parcels/read_state.nl': HELPER}
    good = exact_task(constructs=['decision_readout', 'inline_nl', 'untrusted_contract'], smoke=smoke, files=files)
    errors, _ = authored_checks.lint_task(good)
    assert not [e for e in errors if e.split(':')[0] in authored.CONSTRUCTS]
    row = authored.convert({k: v for k, v in good.items() if v is not None})
    assert row['curriculum']['inline'] == 'required' and row['curriculum']['named'] == 'required'
    row = authored.convert({k: v for k, v in good.items() if v is not None} | {'requires': {'iterate': 'required'}})
    assert row['curriculum']['iterate'] == 'required'
    with pytest.raises(ValueError, match='requires maps'):
        authored.convert({k: v for k, v in good.items() if v is not None} | {'requires': {'inline': 'always'}})
