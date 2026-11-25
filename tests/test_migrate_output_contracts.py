"""migrate_output_contracts.py: a root's prose `Output contract:` paragraph becomes declared types (DECISIONS 2026-11-23
"typed output contracts")."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'rl_tasks'))
import migrate_output_contracts as migrate  # noqa: E402

ROOT = '''---
kind: directory-reducer
args: {}
returns: "number"
---
Apply the requests in requests/ and keep a log.
1. Read requests/*.md in file name order.
2. Write log/applied.md: one line per applied request, `<id> | <change>`.
3. If a request is urgent, also write alert.md with its id.
4. Move each applied request to done/<same file name>.
5. Return the number of requests applied.

Output contract: each line of log/applied.md ends with a newline. Requests that are not applied stay in requests/. The
return is the number of applied requests, a whole number.
'''


def row(**semantics):
    return {'id': 'auth:x:1', 'family': 'auth_x', 'curriculum': {'external': {}},
            'semantics': {'root': 'apply_fix.nl', 'files': {'apply_fix.nl': ROOT, 'apply_fix/read.nl': '---\nargs: {}\nreturns: string\n---\nx\n'},
                          'folder_files': {'requests/r1.md': 'R1 +1\n', 'requests/r2.md': 'R2 ask\n'},
                          'expected_files': {'requests/r2.md': 'R2 ask\n', 'done/r1.md': 'R1 +1\n', 'log/applied.md': 'R1 | +1\n'},
                          'expected': 1, **semantics}}


def test_contract_moves_into_writes_and_doc_comments():
    out, reason = migrate.migrate(row())
    assert reason is None
    files = out['semantics']['files']
    root, types = files['apply_fix.nl'], files['types.ts']
    assert 'Output contract' not in root and root.rstrip().endswith('5. Return the number of requests applied.')
    assert 'returns: ApplyFixResult\nwrites: ApplyFixFiles\n---' in root
    # Every file a correct run could write: the reference's writes (a glob for the moved ones) and the conditional
    # alert.md the instructions name, so the declaration does not tell which conditions are false. requests/ is only a
    # move source and stays out.
    fields = out['curriculum']['external']['typed_output_contract']['writes']
    assert set(fields) == {'log/applied.md', 'alert.md', 'done/*.md'}
    assert '/** each line of log/applied.md ends with a newline. */\n  "log/applied.md": string;' in types
    assert '/** Requests that are not applied stay in requests/. */\nexport type ApplyFixFiles' in types
    assert '/** The return is the number of applied requests, a whole number. */\nexport type ApplyFixResult = number;' in types


def test_rows_without_a_contract_are_unchanged_and_a_declared_root_is_left_out():
    plain = row()
    plain['semantics']['files']['apply_fix.nl'] = ROOT.split('\nOutput contract:')[0] + '\n'
    assert migrate.migrate(plain) == (plain, None)
    declared = row()
    declared['semantics']['files']['apply_fix.nl'] = ROOT.replace('returns: "number"\n', 'returns: "number"\nwrites: X\n')
    assert migrate.migrate(declared) == (None, 'root already declares writes:')


def test_migrated_root_loads_in_the_runtime(tmp_path):
    """The migrated program loads with the natlang loader (writes: and the aliases of types.ts resolve)."""
    import shutil
    import subprocess
    repo = Path(__file__).resolve().parents[1]
    dist = repo / 'ts-host/dist/runtime/loader.js'
    node = shutil.which('node') or next(iter(sorted(Path.home().glob('.local/share/natlang-dev/node-v24*/bin/node'))), None)
    if not dist.is_file() or not node:
        import pytest
        pytest.skip('needs the ts-host build')
    out, _ = migrate.migrate(row())
    script = ("const {loadNamedFunction}=await import(process.argv[1]);const {virtualSourceFiles}=await import(process.argv[2]);"
              "const r=loadNamedFunction('/project/apply_fix.nl',virtualSourceFiles(JSON.parse(process.argv[3])));"
              "process.stdout.write(JSON.stringify({writes:r.writes,returns:r.returns}))")
    done = subprocess.run([str(node), '--input-type=module', '-e', script, dist.as_uri(),
                           (repo / 'ts-host/dist/runtime/virtual-project.js').as_uri(), json.dumps(out['semantics']['files'])],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {'writes': 'ApplyFixFiles', 'returns': 'ApplyFixResult'}


def test_rebuild_replaces_contract_rows_with_their_verified_migration_and_drops_the_rest(tmp_path):
    plain = row()
    plain['id'] = 'auth:x:plain'
    plain['semantics']['files']['apply_fix.nl'] = ROOT.split('\nOutput contract:')[0] + '\n'
    failed = row()
    failed['id'] = 'auth:x:failed'
    source = tmp_path / 'cases.jsonl'
    source.write_text(''.join(json.dumps(item) + '\n' for item in (row(), plain, failed)))
    kept = tmp_path / 'kept.jsonl'
    kept.write_text(json.dumps(migrate.migrate(row())[0]) + '\n')
    out, report = tmp_path / 'out.jsonl', tmp_path / 'report.json'
    assert migrate.main(['rebuild', str(source), '--verified', str(kept), '--out', str(out), '--report', str(report)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r['id'] for r in rows] == ['auth:x:1', 'auth:x:plain']
    assert 'writes: ApplyFixFiles' in rows[0]['semantics']['files']['apply_fix.nl'] and rows[1] == plain
    assert [d['id'] for d in json.loads(report.read_text())['dropped']] == ['auth:x:failed']


def test_template_paths_and_step_writes_are_declared_even_when_the_reference_writes_nothing_there():
    """A declaration of only the reference's writes would tell a solver that `replies/` stays empty (the proposal is not
    installed) and refuse a run that installs: per-item template paths declare their folder."""
    item = row()
    root = ROOT.replace('4. Move each applied request to done/<same file name>.',
                        '4. Write replies/<id>.md for every request and move it to done/<same file name>.')
    item['semantics']['files']['apply_fix.nl'] = root
    fields = migrate.migrate(item)[0]['curriculum']['external']['typed_output_contract']['writes']
    assert {'replies/*.md', 'done/*.md'} <= set(fields)


def test_migrate_output_contracts_read_step_and_path_boundaries():
    semantics = {
        'folder_files': {'ledger.csv': 'id\n1\n', 'rules.md': 'rule\n'},
        'expected_files': {'ledger.csv': 'id\n1\n2\n', 'rules.md': 'rule\n', 'summary.md': 'ok\n'},
    }
    body = (
        '1. Read ledger.csv and inspect every open refund. Append one line for each approved refund.\n'
        '2. Read rules.md and write summary.md.\n'
    )
    fields, _ = migrate.output_fields(semantics, body, '')
    assert 'ledger.csv' in fields and 'summary.md' in fields and 'rules.md' not in fields

    general, per_field, returned = migrate.attribute(
        'Write a/b.md with details. Write b.md. Put replies in out/item.md.',
        {'a/b.md': ['a/b.md'], 'b.md': ['b.md'], 'out/*.md': []},
        True,
    )
    assert per_field['a/b.md'] == ['Write a/b.md with details.']
    assert per_field['b.md'] == ['Write b.md.']
    assert per_field['out/*.md'] == ['Put replies in out/item.md.']
