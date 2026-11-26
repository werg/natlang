"""Targeted regression tests for the 2026-11-26 forensic audit fixes across data mixing,
expert iteration, filled reasoning, SFT trainer, serving/templates, held_eval, GRPO, and curriculum."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / 'scripts' / 'neuralese_data', REPO / 'training' / 'neuralese', REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def test_render_fields_preserves_recorded_turns():
    import sft_sources as s
    assert 'recorded_turns' in s.RENDER_FIELDS


def test_stand_in_does_not_misclassify_core_episode_keys_and_checks_rows_keys():
    import sft_sources as s
    assert s.stand_in(('t1', 'rl:collect/v0000:g000015:3')) == ('t1', None)
    assert s.stand_in(('t1', 'teacher:aei1/teacher-r0006:t000032-a1:0')) == ('t1', None)
    assert s.stand_in(('t1', 'october-teacher:fallow:prog:0')) == ('t1', None)
    assert s.stand_in(('t1', 'live:vibestudio:cap:1')) == ('t1', None)
    assert s.stand_in(('t1', 'filled:teacher:pool:t000001:0')) == ('t1', 'teacher:pool:t000001:0')
    assert s.stand_in(('t1', 'filled:live:vibestudio:b:2')) == ('t1', 'live:vibestudio:b:2')
    # Build without trained-units.json (like r0001 after chains.pkl removal): only rows.jsonl.zst keys exist.
    raw_rl = 'rl:collect/v0000:g000015:3'
    raw_t = 'teacher:aei1/teacher-r0001:t000001:0'
    exp = s.Exposures([({raw_rl, raw_t}, {'t1', 't2'}, set())])
    exp.fill_after_raw = True
    assert exp(('rl-own', 't1', raw_rl)) == 1
    assert exp(('teacher', 't2', raw_t)) == 1
    assert exp(('filled', 't2', 'filled:' + raw_t)) == 0
    assert exp.raw_trained('t2', raw_t)


def test_rollback_prunes_prefer_unseen_and_negative_neutral_keeps_reference_watermark(tmp_path):
    import expert_iteration as ei
    p0 = ('/ckpt/v0.pt', '0' * 64)
    p1 = ('/ckpt/v1.pt', '1' * 64)
    p2 = ('/ckpt/v2.pt', '2' * 64)
    p3 = ('/ckpt/v3.pt', '3' * 64)
    ctrl = ei.PhaseController.initialize(tmp_path / 's.json', start_checkpoint=p0[0], start_sha=p0[1],
                                         held_every=1, prefer_unseen=['/old/sft-v9/mix'])
    ctrl.record_round_published('r0001', *p1, base_sha256=p0[1], mix_dir='/loop/r0001/mix')
    ctrl.start_measurement()
    ctrl.record_verdict({'verdict': 'success', 'adopt': True, 'checkpoint': p1[0], 'sha256': p1[1],
                         'vs_start': {'delta': 0.045, 'p': 0.02, 'ran_out_delta': -0.02}})
    ctrl.record_round_published('r0002', *p2, base_sha256=p1[1], mix_dir='/loop/r0002/mix')
    ctrl.start_measurement()
    ctrl.record_verdict({'verdict': 'failure', 'adopt': False, 'checkpoint': p2[0], 'sha256': p2[1],
                         'vs_start': {'delta': -0.05, 'p': 0.01}})
    assert '/loop/r0001/mix' in ctrl.state['prefer_unseen']
    assert '/old/sft-v9/mix' in ctrl.state['prefer_unseen']
    assert '/loop/r0002/mix' not in ctrl.state['prefer_unseen']
    assert 'r0002' in ctrl.state['rolled_back_rounds']
    # Manual owner-directed re-adoption is not clobbered by --reference on restart.
    ctrl.state['reference_policy'] = {'checkpoint': p1[0], 'sha256': p1[1]}
    ctrl.state['history'] = [{'action': 'owner-directed re-adoption', 'sha256': p1[1]}]
    changes = ctrl.reconcile_arguments(start_checkpoint=p0[0], start_sha=p0[1],
                                       reference_checkpoint=p0[0], reference_sha=p0[1])
    assert ctrl.state['reference_policy']['sha256'] == p1[1] and 'reference_kept' in changes
    # Negative-delta neutral adopt keeps lineage moving without ratcheting reference_policy downward.
    ctrl.resume_after_hunt()
    ctrl.record_round_published('r0003', *p3, base_sha256=p1[1], mix_dir='/loop/r0003/mix')
    ctrl.start_measurement()
    ev = ctrl.record_verdict({'verdict': 'neutral', 'adopt': True, 'checkpoint': p3[0], 'sha256': p3[1],
                              'vs_start': {'delta': -0.02, 'p': 0.40, 'ran_out_delta': 0.01}})
    assert ev['action'] == 'adopt'
    assert ctrl.current['sha256'] == p3[1]
    assert ctrl.state['reference_policy']['sha256'] == p1[1]


def test_filled_rows_restores_first_action_token_weight_only_when_filled():
    import filled_reasoning as fr
    close_id = 99
    # Raw empty-think row: </think> (99), <tool_call> (10), fn (11), eos (2).
    # Raw weight masks </think> and first action token (<tool_call>); filled_weight unmasks <tool_call>.
    row = {'sample': np.array([close_id, 10, 11, 2], np.int32),
           'weight': np.array([0., 0., 1., 1.], np.float32),
           'filled_weight': np.array([0., 1., 1., 1.], np.float32),
           'index': 0, 'invocation': 'root'}
    chain = {'ids': np.array([1, 2, close_id, 10, 11, 2], np.int32), 'segs': [(2, 4, row)]}
    calls = fr.calls_of(chain, close_id)
    filled = fr.filled_rows(calls, [{'reasoning': [50, 51, close_id]}], close_id, 'filled:teacher:k')
    assert filled[0]['weight'].tolist() == [1., 1., 1., 1., 1., 1.]
    fallback = fr.filled_rows(calls, [{'reasoning': None}], close_id, 'filled:teacher:k')
    assert fallback[0]['weight'].tolist() == [0., 0., 1., 1.]


def test_collection_health_counts_memetic_scored_success():
    import collection_health as ch
    assert ch._category('memetic_scored', set(), set(), {'memetic_success': True}) == 'success'
    assert ch._category('memetic_scored', set(), set(), {'memetic_success': False}) == 'wrong_answer'


def test_chat_xml_value_enum_const_anyof_and_compact_json():
    from natlang_neuralese.serve.chat import _schema_types, xml_value, xml_wire_arguments
    assert _schema_types({'enum': ['1', '2', '3']}) == {'string'}
    assert _schema_types({'const': 'true'}) == {'string'}
    assert xml_value('1', {'enum': ['1', '2', '3']}) == '1'
    assert xml_value('true', {'const': 'true'}) == 'true'
    any_schema = {'anyOf': [{'type': 'object', 'properties': {'answer': {'type': 'string'}}}, {'const': 'unknown'}]}
    assert _schema_types(any_schema) == {'object', 'string'}
    assert xml_value('{"answer":"42"}', any_schema) == {'answer': '42'}
    tools = [{'function': {'name': 'return_result', 'parameters': {'properties': {'value': {'enum': ['1', '2']}}}}}]
    assert xml_wire_arguments('return_result', {'value': '1'}, tools) == {'value': '1'}


def test_chat_build_message_think_order_and_whitespace_preservation():
    from natlang_neuralese.serve.chat import build_message
    # Leading thought without <think> (LFM3 prompt opened <think>), with <think>...</think> inside eval code string
    reply = ' my thought \n</think>\n<tool_call>\n<function=eval>\n<parameter=code>\nconst s = "<think>a</think>";\n</parameter>\n</function>\n</tool_call>'
    msg = build_message(reply, [])
    assert msg['reasoning_content'] == ' my thought \n'
    assert len(msg['tool_calls']) == 1
    assert '<think>a</think>' in msg['tool_calls'][0]['function']['arguments']


def test_chat_escapes_reasoning_content_specials():
    from natlang_neuralese.serve.chat import assistant_reply_segments, render_messages, split_escaped
    specials = ['<|im_start|>', '<|im_end|>', '<think>', '</think>']
    def fake_template(msgs, add_gen=False):
        parts = []
        for m in msgs:
            r = m.get('reasoning_content') or ''
            parts.append(f"<|im_start|>{m['role']}\n<think>{r}</think>{m.get('content') or ''}<|im_end|>")
        if add_gen:
            parts.append('<|im_start|>assistant\n<think>')
        return ''.join(parts)
    msg = {'role': 'assistant', 'reasoning_content': 'Wait </think> or <|im_end|>?', 'content': 'ok'}
    rendered = render_messages([{'role': 'user', 'content': 'hi'}, msg], None,
                               lambda m, t=None: fake_template(m, False), specials)
    # The injected </think> and <|im_end|> inside reasoning must appear only in escaped runs (escaped=True).
    runs = [run for seg in rendered.segments if isinstance(seg, str) for run in split_escaped(seg, rendered.escape_nonce)]
    unescaped = ''.join(t for t, esc in runs if not esc)
    escaped = ''.join(t for t, esc in runs if esc)
    assert unescaped.count('</think>') == 2  # user + assistant structural closes only
    assert unescaped.count('<|im_end|>') == 2
    assert '</think>' in escaped and '<|im_end|>' in escaped
    segs = assistant_reply_segments(fake_template, msg, specials)
    assert segs is not None
    unesc_rep = ''.join(t for t, esc in segs if not esc)
    assert unesc_rep.count('</think>') == 1 and unesc_rep.count('<|im_end|>') == 1


def test_held_eval_infra_loss_reads_compacted_results_zst(tmp_path):
    import zstandard
    from natlang_neuralese.rl.held_eval import infra_loss
    res_path = tmp_path / 'results.jsonl'
    zst_path = tmp_path / 'results.jsonl.zst'
    payload = json.dumps({'outcome': {'status': 'done', 'accepted': True}}) + '\n'
    zst_path.write_bytes(zstandard.ZstdCompressor().compress(payload.encode()))
    assert infra_loss({'output': str(res_path)}, {'status': 'complete'}) is None


def test_grpo_explore_walk_does_not_exhaust_permanently_on_cumulative_cursor():
    import grpo
    tasks = [{'id': f't{i}', 'family': 'f', 'curriculum': {'external': {'constructs': []}}} for i in range(5)]
    walk = grpo.ExploreWalk(tasks, wrapper_share=0.5, held_weight=1.0)
    measured = {'t0', 't1', 't2', 't3'}
    row, cursor = walk.next(cursor=100, measured=measured, rng=random.Random(0))
    assert row['id'] == 't4'


def test_curriculum_visit_cap_window_includes_full_window():
    from natlang_neuralese.rl.curriculum import select_tasks
    cases = [
        {'id': 'a', 'family': 'f', 'level': 1},
        {'id': 'b', 'family': 'f', 'level': 1},
        {'id': 'c', 'family': 'f', 'level': 1},
    ]
    # Round 1 is the most recent observed round -> now = 2. With window=1, r >= 2-1=1 must count round 1.
    observations = [
        {'task_id': 'a', 'mean': 0.5, 'scored': 4, 'visits': 1, 'observed_round': 1, 'visit_rounds': [1]},
        {'task_id': 'b', 'mean': 0.5, 'scored': 4, 'visits': 0, 'observed_round': 0, 'visit_rounds': []},
        {'task_id': 'c', 'mean': 0.5, 'scored': 4, 'visits': 0, 'observed_round': 0, 'visit_rounds': []},
    ]
    # offset=1 puts 'b' in breadth (1 row) and practice chooses between 'a' (capped) and 'c' (uncapped).
    res = select_tasks(cases, observations, {'rows': 2, 'offset': 1, 'group': 2},
                       curriculum={'schema': 'natlang.rl-task-curriculum/2', 'breadth_fraction': 0.5,
                                   'visit_cap': 1, 'visit_cap_window': 1})
    assert {cases[r['row']]['id'] for r in res} == {'b', 'c'}


def test_lfm3_learner_gate_rank_preserves_alpha_over_rank():
    from natlang_neuralese.train.lfm3_learner import AdapterSpec
    spec = AdapterSpec(rank=128, alpha=128, gate_rank=8)
    assert float(spec.alpha) / float(spec.rank) == 1.0


def test_exp0_sft_additive_ambient_keeps_sft_gradient():
    import exp0_sft
    torch.manual_seed(0)
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 8)
            self.lm_head = torch.nn.Linear(8, 16, bias=False)
            self.lm_head.weight = self.embed.weight
    model = TinyModel()
    row = {'sample': np.array([2, 3, 4], np.int32), 'weight': np.ones(3, np.float32)}
    chain = {'ids': np.array([1, 2, 3, 4], np.int32), 'segs': [(1, 3, row)]}
    with torch.no_grad():
        teacher_h = model.embed(torch.tensor([1, 2, 3])).detach()
    ambient = {'temperature': 2.5, 'head': model.lm_head.weight.detach().clone(),
               'rows': {0: teacher_h}, 'mode': 'additive', 'weight': 4.0 / 3.0}
    total, _ = exp0_sft.step_backward(
        model, lambda m, ids: m.embed(ids), [chain], 3.0, 0, 1.0, 0, fast=False, ambient=ambient)
    assert ambient['sft_nll'] > 0 and ambient['tm_nll'] > 0
    assert model.embed.weight.grad is not None and model.embed.weight.grad.abs().sum().item() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='the FP32-output head mm is CUDA-only')
def test_fp32_head_logits_are_differentiable_on_cuda():
    """torch.mm(out_dtype=float32) has no autograd formula; _tm_ce (non-clean Ambient steps) backpropagates through
    lfm3_fast._head_logits_chunk, so the FP32 head runs as _HeadMMFP32 (dh = d @ W, dW FP32-output) under grad."""
    from natlang_neuralese.train import lfm3_fast
    torch.manual_seed(0)
    h = torch.randn(7, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(300, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    rows = torch.randn(3, 64, device='cuda', dtype=torch.float32, requires_grad=True)
    ids = torch.tensor([5, 17, 250], device='cuda')
    logits = lfm3_fast._head_logits_chunk(h, w, ids, rows)
    assert logits.dtype == torch.float32
    g = torch.randn_like(logits)
    (logits * g).sum().backward()
    ref_logits = h.detach().float() @ w.detach().float().t()
    ref_logits[:, ids] = h.detach().float() @ rows.detach().to(torch.bfloat16).float().t()
    assert torch.allclose(logits.detach(), ref_logits, rtol=1e-2, atol=1e-1)
    gw = g.clone(); gw[:, ids] = 0
    assert torch.allclose(w.grad.float(), (gw.t() @ h.detach().float()), rtol=5e-2, atol=5e-1)
    assert h.grad is not None and rows.grad is not None and torch.isfinite(h.grad.float()).all()


def test_a_resumed_sft_keeps_its_learner_states_ambient_mode(tmp_path):
    """additive became the default in ddd97a765; a learner state saved before ambient_mode was recorded trained with
    replace and resumes with it, a recorded mode is kept, a fresh run takes the requested one."""
    import sys
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'neuralese_data'))
    import exp0_sft
    state = tmp_path / 'learner-state'
    state.mkdir()
    args = SimpleNamespace(state_dir=str(state), out_dir=str(tmp_path), resume=True)
    assert exp0_sft.resumed_ambient_mode(args, 'additive') == 'additive'          # no state yet: requested
    (state / 'learner_manifest.json').write_text('{}')
    torch.save({'decision_objective': True}, state / 'trainer_state.pt')
    assert exp0_sft.resumed_ambient_mode(args, 'additive') == 'replace'          # saved before the field existed
    torch.save({'ambient_mode': 'additive'}, state / 'trainer_state.pt')
    assert exp0_sft.resumed_ambient_mode(args, 'replace') == 'additive'
    args.resume = False
    assert exp0_sft.resumed_ambient_mode(args, 'additive') == 'additive'


def test_filled_teacher_preference_keeps_producer_turns_at_the_task_cap():
    """ddd97a765 passed the filled preference as capped's producer ``order``: producer A's filled and unfilled episodes
    became two queues and took a cap of 2 before producer B's only solution. ``prefer`` orders inside a producer's
    turn: different producers first, a producer's filled episode before its unfilled one."""
    import collections
    import random
    import sys
    sys.path.insert(0, str(REPO / 'scripts' / 'neuralese_data'))
    import sft_sources
    def ep(key, teacher):
        return {'key': key, 'task_id': 't', 'family': 'f', 'source': 'teacher', 'teacher': teacher}
    eps = [ep('a-filled', 'A'), ep('a-raw', 'A'), ep('b-raw', 'B')]
    filled = {'a-filled'}
    prefer = lambda c: 0 if c['key'] in filled else 1           # noqa: E731
    for seed in range(20):
        funnel = collections.defaultdict(collections.Counter)
        two = sft_sources.capped(eps, random.Random(seed), funnel, lambda t: 2, 1.0, lambda c: 0, prefer=prefer)
        assert {c['teacher'] for c in two} == {'A', 'B'}
        assert 'a-filled' in {c['key'] for c in two}             # A's turn takes its filled episode
        one = sft_sources.capped(eps[:2], random.Random(seed), funnel, lambda t: 1, 1.0, lambda c: 0, prefer=prefer)
        assert [c['key'] for c in one] == ['a-filled']
