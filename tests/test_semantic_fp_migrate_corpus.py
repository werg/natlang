"""scripts/semantic_fp/migrate_corpus.py: units from a baseline, acceptance by the 07 §4.2 rule, the settle judgment
only for rejected candidates, apply and package (reserved programs left out)."""
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts' / 'semantic_fp'))
spec = importlib.util.spec_from_file_location('migrate_corpus', REPO / 'scripts/semantic_fp/migrate_corpus.py')
mc = importlib.util.module_from_spec(spec)
sys.modules['migrate_corpus'] = mc
spec.loader.exec_module(mc)


def row(case, program, passes, samples=4, policy='p', version='original'):
    return {'case_id': case, 'program': program, 'passes': passes, 'scored': samples, 'samples': samples,
            'seeds': list(range(samples)), 'version': version, 'policy_id': policy}


def proposal_dir(tmp_path, program, status, root, changes=()):
    directory = tmp_path / 'proposals' / program.replace(':', '-')
    (directory / 'program').mkdir(parents=True)
    (directory / 'proposal.json').write_text(json.dumps({
        'program': program, 'root': str(root), 'status': status, 'findings': [{'file': 'a.nl', 'code': 'idiom-decision-readout'}],
        'keep': 'migrate', 'keepBy': 'judgment', 'problems': [], 'changes': list(changes)}))
    (directory / 'run.json').write_text(json.dumps({'model': {'server': 'u', 'model': 'native-lfm3'}}))
    return directory


def test_baselines_count_cases_and_passes():
    b = mc.baselines([row('a', 'design-brief:001', 4), row('b', 'design-brief:001', 0), row('c', 'astra:x', 1)])
    assert b['design-brief:001'] == {'cases': 2, 'passingCases': 1, 'passes': 4, 'scored': 8}
    assert b['astra:x']['passingCases'] == 1


def test_settle_accepts_by_the_exact_rule_and_judges_only_rejections(tmp_path):
    root = tmp_path / 'p'
    root.mkdir()
    proposal_dir(tmp_path, 'design-brief:001', 'candidate', root)
    proposal_dir(tmp_path, 'design-brief:002', 'candidate', root)
    proposal_dir(tmp_path, 'design-brief:003', 'delete', root)
    proposal_dir(tmp_path, 'design-brief:004', 'unchecked', root)
    original = [row('1a', 'design-brief:001', 4), row('1b', 'design-brief:001', 2),
                row('2a', 'design-brief:002', 4), row('2b', 'design-brief:002', 4)]
    candidate = [row('1a', 'design-brief:001', 4, version='migrated'), row('1b', 'design-brief:001', 1, version='migrated'),
                 row('2a', 'design-brief:002', 3, version='migrated'), row('2b', 'design-brief:002', 4, version='migrated')]
    asked = []
    decisions = {d['program']: d for d in mc.settle(original, candidate, tmp_path / 'proposals', attempts_left=1,
                                                     judge=lambda c, left, history: asked.append(c) or 'retry')}
    assert decisions['design-brief:001']['decision'] == 'accept'      # 5.0 >= 6.0 - 1, no 4/4 case lost
    assert decisions['design-brief:002']['decision'] == 'retry'       # a 4/4 case lost: rejected, the judgment decides
    assert decisions['design-brief:003']['decision'] == 'delete'
    assert decisions['design-brief:004']['decision'] == 'rejected'
    assert len(asked) == 1 and asked[0]['accepted'] is False and '2a' in asked[0]['problems'][0]


def test_settle_refuses_a_comparison_across_policies(tmp_path):
    root = tmp_path / 'p'
    root.mkdir()
    proposal_dir(tmp_path, 'design-brief:001', 'candidate', root)
    decisions = mc.settle([row('1a', 'design-brief:001', 4)], [row('1a', 'design-brief:001', 4, policy='q')],
                          tmp_path / 'proposals', judge=lambda *_: 'give-up')
    assert decisions[0]['decision'] == 'give-up' and 'another policy' in decisions[0]['problems'][0]


def test_unmeasured_candidates_are_pending(tmp_path):
    root = tmp_path / 'p'
    root.mkdir()
    proposal_dir(tmp_path, 'design-brief:001', 'candidate', root)
    partial = dict(row('1a', 'design-brief:001', 1), scored=1)
    decisions = mc.settle([row('1a', 'design-brief:001', 4)], [partial], tmp_path / 'proposals')
    assert decisions[0]['decision'] == 'pending'


def test_apply_and_package_leave_out_reserved(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    root = repo / 'codebases' / 'design_briefs' / '001-x'
    root.mkdir(parents=True)
    (root / 'a.nl').write_text('---\nreturns: V\n---\nold\n')
    (root / 'types.ts').write_text('export type V = "a" | "b";\n')
    gone = repo / 'codebases' / 'design_briefs' / '002-y'
    gone.mkdir(parents=True)
    (gone / 'b.nl').write_text('---\nreturns: string\n---\nx\n')
    monkeypatch.setattr(mc, 'REPO', repo)
    change = {'path': 'a.nl', 'before': 'old', 'after': '---\nreturns: V\nreadout: decision\n---\nnew\n', 'role': 'leaf'}
    p1 = proposal_dir(tmp_path, 'design-brief:001', 'candidate', root, [change])
    p2 = proposal_dir(tmp_path, 'design-brief:002', 'delete', gone)
    comparison = {'accepted': True}
    decisions = [{'program': 'design-brief:001', 'root': str(root), 'proposal': str(p1), 'decision': 'accept',
                  'findings': [], 'comparison': comparison},
                 {'program': 'design-brief:002', 'root': str(gone), 'proposal': str(p2), 'decision': 'delete', 'findings': []}]
    rows, skipped = mc.package(decisions, tmp_path / 'out', reserved={'design-brief:999'})
    assert len(rows) == 1 and rows[0]['after']['files']['a.nl'].endswith('new\n') and rows[0]['before']['files']['a.nl'].endswith('old\n')
    assert rows[0]['split_group'] == 'design-brief-app-001' and rows[0]['family'] == 'refactor-to-semantic-fp'
    rows, skipped = mc.package(decisions, tmp_path / 'out', reserved={'design-brief:001'})
    assert rows == [] and skipped == ['design-brief:001']
    written, removed = mc.apply_decisions(decisions, repo)
    assert written == ['codebases/design_briefs/001-x/a.nl'] and removed == ['codebases/design_briefs/002-y']
    assert 'readout: decision' in (root / 'a.nl').read_text() and not gone.exists()


def test_reserved_programs_read_held_v4_reservations(tmp_path):
    path = tmp_path / 'r.json'
    path.write_text(json.dumps({'design_briefs': {'chosen': [{'brief': '004-org'}]},
                                'applications': {'chosen': [{'application': 'sol/field-gloss-workbench'}]}}))
    assert mc.reserved_programs(path) == {'design-brief:004', 'sol:field-gloss-workbench'}


def test_settle_deletes_a_program_whose_measured_original_never_passes(tmp_path):
    root = tmp_path / 'p'
    root.mkdir()
    proposal_dir(tmp_path, 'design-brief:001', 'candidate', root)
    decisions = mc.settle([row('1a', 'design-brief:001', 0), row('1b', 'design-brief:001', 0)],
                          [row('1a', 'design-brief:001', 4), row('1b', 'design-brief:001', 4)], tmp_path / 'proposals')
    assert decisions[0]['decision'] == 'delete' and decisions[0]['keepBy'] == 'floor'


def test_program_tasks_keep_accepted_unreserved_exact_and_verifier_cases_with_their_split():
    case = lambda cid, program, oracle='exact': {'id': cid, 'split': 'train', 'semantics': {'root': 'a.nl'},
                                                 'equivalence': {'program': program, 'oracle': oracle, 'split': 'dev', 'source_sha256': 's'}}
    cases = [case('a', 'design-brief:001'), case('v', 'design-brief:001', 'verifier'),
             case('b', 'design-brief:001', 'judged'), case('c', 'design-brief:002'),
             case('d', 'design-brief:003')]
    decisions = [{'program': 'design-brief:001', 'decision': 'accept'}, {'program': 'design-brief:002', 'decision': 'give-up'},
                 {'program': 'design-brief:003', 'decision': 'accept'}]
    import tempfile
    with tempfile.TemporaryDirectory() as out:
        rows = mc.program_tasks(cases, decisions, out, reserved={'design-brief:003'})
    assert [r['id'] for r in rows] == ['a:sfp-migrated', 'v:sfp-migrated']
    assert rows[0]['split'] == 'dev' and 'equivalence' not in rows[0] and rows[0]['semantic_fp_migration']['original_id'] == 'a'
    assert rows[1]['semantic_fp_migration']['original_id'] == 'v'
