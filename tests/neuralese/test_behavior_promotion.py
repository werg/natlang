import copy
import json

import pytest

from natlang_neuralese.train import promotion

PACKET = {'path': '/packets/dev/cases.jsonl', 'sha256': 'a' * 64}
CASES = [0, 8, 16, 24]


def declaration(**extra):
    return {'schema': promotion.DECLARATION_SCHEMA, 'packet': PACKET, 'cases': CASES, 'compare_to': 'parent', **extra}


def receipt(checkpoint, successes, *, runtime='r' * 64, spec='s' * 64, cases=CASES, eligible=True, status='finished'):
    """The shape evaluate_native_actor.py writes as completed.json."""
    return {'contract': {'checkpoint_sha256': checkpoint, 'spec_sha256': spec, 'step': 1},
            'actor_info': {'prompt_rendering_identity_sha256': 'p' * 64, 'serving_config_identity_sha256': 'c' * 64,
                           'rl_policy_binding': {'python_runtime': {'sha256': runtime},
                                                 'serving_precision': {'weights': 'checkpoint-master'},
                                                 'history_reasoning': {'policy': 'keep'},
                                                 'native_identity': {'tokenizer_fingerprint': 't' * 64}}},
            'status': status, 'training_admission': False,
            'summary': {'schema': 'natlang.in-run-task-eval/1', 'eligible': eligible, 'task_successes': successes,
                        'cases': len(cases), 'packet_sha256': PACKET['sha256'], 'case_indices': cases,
                        'prompt_rendering': 'full', 'outcome_counts': {'task_success': successes}}}


def detail(arm, successes):
    """task-score.json: the summary plus one outcome per case."""
    summary = arm['summary']
    return {**summary, 'cases_detail': [{'id': f'case-{index}', 'outcome': 'task_success' if ok else 'semantic_mismatch'}
                                        for index, ok in zip(summary['case_indices'], successes)]}


def test_declaration_keeps_evaluations_small_and_confirmation_disjoint():
    assert promotion.parse_declaration(declaration())['min_delta'] == 0
    with pytest.raises(ValueError, match='stay small'):
        promotion.parse_declaration(declaration(cases=list(range(promotion.MAX_PROMOTION_CASES + 1))))
    with pytest.raises(ValueError, match='compare_to'):
        promotion.parse_declaration(declaration(compare_to='loss'))
    with pytest.raises(ValueError, match='disjoint'):
        promotion.parse_declaration(declaration(confirmation={'packet': PACKET, 'cases': [8, 9]}))
    with pytest.raises(ValueError, match='unknown promotion field'):
        promotion.parse_declaration(declaration(steps=100))


def test_paired_comparison_promotes_only_at_or_above_the_reference_plus_delta():
    better = promotion.compare(declaration(min_delta=1), phase='development', candidate=receipt('b' * 64, 3),
                               reference=receipt('c' * 64, 2))
    assert better['matched'] and better['delta'] == 1 and better['promoted'] and better['training_admission'] is False
    tie = promotion.compare(declaration(min_delta=1), phase='development', candidate=receipt('b' * 64, 2),
                            reference=receipt('c' * 64, 2))
    assert tie['matched'] and not tie['promoted'] and 'min_delta' in tie['decision_reasons'][0]


def test_arms_under_different_runtimes_or_limits_are_unmatched_not_regressions():
    # The 2026-11-02 12/16 vs 11/16 comparison ran its arms under runtime34 and runtime19: not paired evidence.
    result = promotion.compare(declaration(), phase='development', candidate=receipt('b' * 64, 3, runtime='x' * 64),
                               reference=receipt('c' * 64, 1))
    assert not result['matched'] and not result['promoted'] and result['delta'] is None
    assert 'python_runtime_sha256' in result['unmatched_reasons'][0]
    other_spec = promotion.compare(declaration(), phase='development', candidate=receipt('b' * 64, 3, spec='y' * 64),
                                   reference=receipt('c' * 64, 1))
    assert 'spec_sha256' in other_spec['unmatched_reasons'][0]
    unscored = promotion.compare(declaration(), phase='development', candidate=receipt('b' * 64, 3, eligible=False),
                                 reference=receipt('c' * 64, 1))
    assert not unscored['matched'] and 'eligible' in unscored['unmatched_reasons'][0]
    wrong_cases = promotion.compare(declaration(), phase='development', candidate=receipt('b' * 64, 3, cases=[0, 8, 16, 32]),
                                    reference=receipt('c' * 64, 1, cases=[0, 8, 16, 32]))
    assert not wrong_cases['matched']


def test_per_case_detail_pairs_outcomes_and_bounds_regressions():
    candidate, reference = receipt('b' * 64, 2), receipt('c' * 64, 2)
    result = promotion.compare(declaration(max_case_regressions=0), phase='development', candidate=candidate,
                               reference=reference, candidate_detail=detail(candidate, [1, 1, 0, 0]),
                               reference_detail=detail(reference, [1, 0, 1, 0]))
    assert result['case_transitions'] == {'gained': ['case-8'], 'lost': ['case-16'], 'both': 1, 'neither': 1}
    assert not result['promoted'] and 'regressions' in result['decision_reasons'][0]
    without = promotion.compare(declaration(max_case_regressions=0), phase='development', candidate=candidate,
                                reference=reference)
    assert not without['promoted'] and 'detail' in without['decision_reasons'][0]
    with pytest.raises(ValueError, match='disagrees'):
        promotion.compare(declaration(), phase='development', candidate=candidate, reference=reference,
                          candidate_detail=detail(candidate, [1, 1, 1, 0]), reference_detail=detail(reference, [1, 1, 0, 0]))


def test_stage_record_requires_every_declared_phase_for_the_exact_candidate(tmp_path):
    confirm = {'packet': {'path': '/packets/conf', 'sha256': 'd' * 64}, 'cases': [1, 2]}
    decl = declaration(confirmation=confirm)
    stage = {'id': 'sft', 'kind': 'native_state_sft', 'promotion': decl}
    report = {'artifact_sha256': 'b' * 64}
    with pytest.raises(ValueError, match='awaits its behaviour promotion'):
        promotion.require_stage_promotion(tmp_path, stage, report)
    development = promotion.compare(decl, phase='development', candidate=receipt('b' * 64, 3), reference=receipt('c' * 64, 2))
    promotion.attach(tmp_path, 'sft', [development])
    with pytest.raises(ValueError, match='confirmation'):
        promotion.require_stage_promotion(tmp_path, stage, report)
    (tmp_path / 'sft-promotion.json').unlink()
    conf_arm = lambda checkpoint, n: {**receipt(checkpoint, n, cases=[1, 2]),
                                      'summary': {**receipt(checkpoint, n, cases=[1, 2])['summary'], 'packet_sha256': 'd' * 64}}
    confirmation = promotion.compare(decl, phase='confirmation', candidate=conf_arm('b' * 64, 1), reference=conf_arm('c' * 64, 1))
    assert confirmation['promoted']  # confirmation only requires no regression
    promotion.attach(tmp_path, 'sft', [development, confirmation])
    assert promotion.require_stage_promotion(tmp_path, stage, report)
    with pytest.raises(ValueError, match='another checkpoint'):
        promotion.require_stage_promotion(tmp_path, stage, {'artifact_sha256': 'e' * 64})
    with pytest.raises(ValueError, match='another declaration'):
        promotion.require_stage_promotion(tmp_path, {**stage, 'promotion': {**decl, 'min_delta': 2}}, report)
    with pytest.raises(ValueError, match='immutable'):
        promotion.attach(tmp_path, 'sft', [development])


def test_losing_candidate_keeps_its_reference_as_the_selected_parent():
    losing = promotion.compare(declaration(), phase='development', candidate=receipt('b' * 64, 1), reference=receipt('c' * 64, 2))
    selection = promotion.select_parent('sft', [losing], candidate_path='/ck/b.pt', reference_path='/ck/c.pt')
    assert selection['selected_sha256'] == 'c' * 64 and selection['selected_path'] == '/ck/c.pt'
    assert promotion.require_parent_selection(selection, 'c' * 64)
    with pytest.raises(ValueError, match='not the behaviour-selected parent'):
        promotion.require_parent_selection(selection, 'b' * 64)
    unmatched = promotion.compare(declaration(), phase='development', candidate=receipt('b' * 64, 3, runtime='x' * 64),
                                  reference=receipt('c' * 64, 2))
    with pytest.raises(ValueError, match='no matched'):
        promotion.require_parent_selection(promotion.select_parent('sft', [unmatched]), 'c' * 64)


def test_cli_compare_writes_a_bound_receipt(tmp_path):
    files = {}
    for name, value in (('decl', declaration()), ('cand', receipt('b' * 64, 3)), ('ref', receipt('c' * 64, 3))):
        files[name] = tmp_path / f'{name}.json'
        files[name].write_text(json.dumps(value))
    assert promotion.main(['compare', '--declaration', str(files['decl']), '--candidate', str(files['cand']),
                           '--reference', str(files['ref']), '--out', str(tmp_path / 'r.json')]) == 0
    written = json.loads((tmp_path / 'r.json').read_text())
    assert written['promoted'] and written['sources']['candidate']['sha256']
    assert written['declaration_sha256'] == promotion.declaration_sha256(copy.deepcopy(declaration()))
