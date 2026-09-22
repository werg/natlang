import json

from natlang.runtime import Runtime, Session
from natlang.surface import ToolSurface
from natlang.types import TypeEnv
from natlang.values import load_program
from scripts.generate_agent_support import generate_group, support_cases
from scripts.review_probe import summarize


def test_state_and_success_coupled_marks_and_source_copy():
    case = next(c for c in support_cases(104, 0) if c['name'] == 'binding_repair')
    session = Session(Runtime(None), load_program(case['root']), TypeEnv())
    surface = ToolSurface(marks=True, state_view=True)
    initial = surface.opening_read(session)[2]
    assert 'locals: none computed' in initial and '[ ]' in initial and 'size_of' in initial
    bad = surface.apply(session, 'call', {'function':'size_of', 'to':'return', 'inputs':{'items':'args/items'}, 'done':1})
    assert bad.kind == 'rejected' and not session.lam.marks
    result = surface.apply(session, *case['steps'][0])
    assert result.kind == 'done' and 'let/size (Num)' in result.text and '[x]' in result.text
    result = surface.apply(session, *case['steps'][1])
    assert result.kind == 'ok' and session.lam.ret == case['expected']
    assert session.finish()


def test_generated_references_execute_and_review_pairs_share_context_and_split():
    rows, reviews = generate_group(104, 0)
    assert len({r['program_id'] for r in rows + reviews}) == 1
    assert {r['expected'] for r in reviews} == {'approve','withdraw','error','blocker'}
    assert any('copy_value(' in r['native_target'] for r in rows)
    assert any('mark_lines(' in r['native_target'] for r in rows)
    assert any('summary' in r['native_target'] for r in rows)
    for row in reviews:
        assert row['messages'][-1]['role'] == 'user'
        assert 'Nothing in this proposed batch has been executed' in row['messages'][-1]['content']
    approved = next(r for r in reviews if r['expected'] == 'approve' and
                    r['proposal'][0][0] in ('run_function', 'for_each', 'fold', 'repeat'))
    withdrawn = next(r for r in reviews if r['id'].rsplit(':',1)[0] == approved['id'].rsplit(':',1)[0] and r['expected'] == 'withdraw')
    assert approved['messages'][:-1] == withdrawn['messages'][:-1]


def test_review_summary_separates_false_rejections_and_bad_approvals():
    rows = [{'order':'reason_first','expected':a,'decision':b} for a,b in
            [('approve','withdraw'),('error','approve'),('withdraw','withdraw')]]
    summary = summarize(rows)['baseline/reason_first']
    assert summary['correct'] == 1 and summary['false_rejections'] == 1 and summary['bad_approvals'] == 1


def test_honesty_pairs_include_persistence_and_do_not_invent_confidence():
    _, reviews = generate_group(113, 0)
    assert all(r['proposal_justified'] == (r['expected']=='approve') and r['confidence'] is None for r in reviews)
    assert all('honest_check_and_persistence' in r['lesson_ids'] for r in reviews)
    failures=[r for r in reviews if r['proposal'][0][0] in ('report_error','report_blocker')]
    assert {r['expected'] for r in failures}=={'approve','withdraw'}
    early=next(r for r in failures if r['family']=='support_effect_error' and r['expected']=='withdraw')
    assert early['task_feasible'] is False
    reason=json.loads(early['target']['tool_calls'][0]['function']['arguments'])['reason']
    assert 'earlier required operation' in reason
    redirected=next(r for r in reviews if r['family']=='support_binding_error' and r['proposal'][0][0]=='run_function')
    assert redirected['expected']=='error' and redirected['proposal'][0][1]['save_as']=='return/size'
    grouped={}
    for r in reviews:
        grouped.setdefault(r['contrast_group'],[]).append(r)
    assert any({r['expected'] for r in rows} >= {'approve','withdraw'} for rows in grouped.values())
    for rows in grouped.values():
        assert len({r['messages'][-1]['content'].rsplit('\n\n',1)[-1] for r in rows})==1
