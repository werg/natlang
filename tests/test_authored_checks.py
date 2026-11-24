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


def test_groups_must_name_one_real_source_item(repo):
    """sfp1: family-wide `invented:<theme>` groups tied 28 of 36 train rows to held worlds; a check, not a convention."""
    for bad in ('invented:parcel_desk', 'parcel_desk', 'strategyqa', 'parcel_desk:w900-p1a'):
        assert any('group' in error for error in errors(task(groups=[bad]))), bad
    assert not [e for e in errors(task(groups=['strategyqa:18689be402e9'])) if 'group' in e]
    assert any('groups must be a list' in error for error in errors(task(groups='strategyqa:18689be402e9')))


def test_a_top_level_types_file_gives_the_root_its_aliases(repo):
    root = ROOT.replace('"{ id: string; text: string }[]"', '"Note[]"')
    row = task(files={'sort_parcels.nl': root, 'sort_parcels/read_state.nl': HELPER,
                      'types.ts': 'export type Note = { id: string; text: Untrusted<string> };\n'})
    assert authored.convert(row)['semantics']['files']['types.ts'].startswith('export type Note')
    with pytest.raises(ValueError, match='root callable folder'):
        authored.convert(task(files={'sort_parcels.nl': ROOT, 'sort_parcels/read_state.nl': HELPER, 'other.ts': ''}))


def test_verifier_controls_binds_read_code_for_keyword_baselines(repo):
    row = task(
        files={
            'sort_parcels.nl': ROOT,
            'sort_parcels/read_state.nl': HELPER,
            'sort_parcels/skills/desk/SKILL.md': '---\nname: desk\n---\nrule text\n',
        },
        keyword_baseline={
            'code': (
                'const helperSrc = read_code("read_state");\n'
                'const skillSrc = read_code("skills.desk");\n'
                'if (!helperSrc.includes("ready") || !skillSrc.includes("rule text")) throw new Error("missing");\n'
                + BASELINE
            )
        },
    )
    assert errors(row) == []


def test_declared_types_writes_patterns_and_cli_arg_order(tmp_path):
    types_ts = (
        'export type OutFiles = {\n'
        '  // comment inside\n'
        '  "out/report.md": string;\n'
        '  summary?: string;\n'
        '};\n'
        '// top-level comment\n'
        'interface Extra { x: number; }\n'
        'const VERSION = 1;\n'
        'export type Other = string;\n'
    )
    root_nl = '---\nkind: directory-reducer\nwrites: OutFiles\nreturns: string\n---\nWrite files.\n'
    t = exact_task(root='main.nl', files={'main.nl': root_nl, 'types.ts': types_ts})
    decls = authored_checks.declared_types(t)
    assert 'OutFiles' in decls and 'interface Extra' not in decls['OutFiles']
    assert decls['Other'] == 'string'
    assert authored_checks.writes_patterns(t) == ['out/report.md', 'summary']

    tasks_file = tmp_path / 'tasks.jsonl'
    tasks_file.write_text(json.dumps(exact_task()) + '\n')
    assert authored_checks.main(['--runtime', str(REPO / 'ts-host'), str(tasks_file)]) == 0


def test_build_supply_resolves_runs_under_owner_and_refuses_worktree_and_pin(tmp_path, monkeypatch):
    import build_supply
    owner = tmp_path / 'owner'
    (owner / '.git').mkdir(parents=True)
    monkeypatch.setattr(build_supply, 'registry_owner_repo', lambda: owner)
    assert build_supply.resolve_output_dir('runs/supply-1') == (owner / 'runs' / 'supply-1').resolve()

    worktree = tmp_path / 'wt'
    worktree.mkdir()
    (worktree / '.git').write_text('gitdir: ../owner/.git/worktrees/wt\n')
    with pytest.raises(ValueError, match='non-primary git worktree'):
        build_supply.resolve_output_dir(str(worktree / 'runs' / 'supply-1'))

    pin = tmp_path / 'pin'
    pin.mkdir()
    (pin / '.natlang-pin.json').write_text('{}')
    with pytest.raises(ValueError, match='immutable code pin'):
        build_supply.resolve_output_dir(str(pin / 'runs' / 'supply-1'))


def test_migrate_output_contracts_read_step_and_path_boundaries():
    import migrate_output_contracts as moc
    semantics = {
        'folder_files': {'ledger.csv': 'id\n1\n', 'rules.md': 'rule\n'},
        'expected_files': {'ledger.csv': 'id\n1\n2\n', 'rules.md': 'rule\n', 'summary.md': 'ok\n'},
    }
    body = (
        '1. Read ledger.csv and inspect every open refund. Append one line for each approved refund.\n'
        '2. Read rules.md and write summary.md.\n'
    )
    fields, _ = moc.output_fields(semantics, body, '')
    assert 'ledger.csv' in fields and 'summary.md' in fields and 'rules.md' not in fields

    general, per_field, returned = moc.attribute(
        'Write a/b.md with details. Write b.md. Put replies in out/item.md.',
        {'a/b.md': ['a/b.md'], 'b.md': ['b.md'], 'out/*.md': []},
        True,
    )
    assert per_field['a/b.md'] == ['Write a/b.md with details.']
    assert per_field['b.md'] == ['Write b.md.']
    assert per_field['out/*.md'] == ['Put replies in out/item.md.']


def test_verify_mjs_multiline_return_and_eval_guard():
    import subprocess
    verify_src = (REPO / 'scripts' / 'rl_tasks' / 'verify.mjs').read_text()
    assert "last?.[0] !== 'eval'" in verify_src
    # Test findFinalTopLevelReturn logic directly by extracting it from verify.mjs
    start = verify_src.index('function findFinalTopLevelReturn')
    end = verify_src.index('// The terminal answering action')
    fn_code = verify_src[start:end]
    script = (
        fn_code + '\n'
        'const code = "await moveFile();\\nfunction helper() { return 1; }\\nreturn {\\n  a: 1,\\n  b: 2,\\n};\\n";\n'
        'const idx = findFinalTopLevelReturn(code);\n'
        'const replaced = code.slice(0, idx) + "return 42;";\n'
        'console.log(JSON.stringify({ idx, replaced }));\n'
    )
    proc = subprocess.run([authored_checks.node(), '--input-type=module', '-e', script],
                          capture_output=True, text=True, check=True)
    out = json.loads(proc.stdout)
    assert out['idx'] > 0
    assert 'await moveFile();' in out['replaced']
    assert 'function helper() { return 1; }' in out['replaced']
    assert out['replaced'].endswith('return 42;')


def test_sfp_batch_cleans_up_temp_files_on_failure(tmp_path, monkeypatch):
    import sfp_batch
    import measure_repetition
    batch = tmp_path / 'batch'
    (batch / 'qualified-1' / 'verified').mkdir(parents=True)
    (batch / 'tasks.jsonl').write_text('')
    (batch / 'qualified-1' / 'verified' / 'cases.jsonl').write_text(json.dumps({'id': 't1', 'split': 'train'}) + '\n')
    v4_dir = tmp_path / 'v4' / 'fam' / 'qualified-1'
    v4_dir.mkdir(parents=True)
    (v4_dir / 'cases.jsonl').write_text(json.dumps({'id': 'h1', 'split': 'train'}) + '\n')
    monkeypatch.setattr(sfp_batch, 'HELD_V4_AUTHORING', tmp_path / 'v4')
    monkeypatch.setattr(sfp_batch, 'grpo_pools', lambda: [])
    monkeypatch.setattr(sfp_batch, 'run', lambda args, log: 0)
    from natlang_neuralese.rl import held_eval
    monkeypatch.setattr(held_eval, 'training_exclusion_ids', lambda rows: frozenset())
    monkeypatch.setattr(measure_repetition, 'overlap_with_pool', lambda *a: (_ for _ in ()).throw(RuntimeError('boom')))
    with pytest.raises(RuntimeError, match='boom'):
        sfp_batch.qualify(batch, str(REPO / 'ts-host'), 1, 1)
    assert not (batch / '.held-v4-batch-1.jsonl').exists()
    assert not (batch / '.held-v4-pool-1.jsonl').exists()
