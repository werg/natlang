"""Filled-in reasoning for teacher trajectories (scripts/neuralese_data/filled_reasoning.py): leakage, the hint turn's
place, length-banded selection, rows with the filled history, per-row sampled flags, and the build's ``replaces``."""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
import exp0_sft  # noqa: E402
import filled_reasoning as fr  # noqa: E402
import sft_sources  # noqa: E402

CLOSE, IM_START = 9, 1


def test_leak_catches_quotes_and_names_of_hinted_text_not_seen_in_context():
    action = '<tool_call><function=eval><parameter=code>const total = rows.reduce((a, r) => a + r.amount, 0);' \
             '</parameter></function></tool_call>'
    assert fr.leak('I will compute const total = rows.reduce((a, r) => a + r.amount, 0); and return it', [action], '')
    assert fr.leak('Let me sum the amounts with a reduce over the rows.', [action], '') is None
    # a quote of what the student has already seen is not a leak
    seen = 'const total = rows.reduce((a, r) => a + r.amount, 0);'
    assert fr.leak('as before: const total = rows.reduce((a, r) => a + r.amount, 0); again', [action], seen) is None
    # a short action named outright, and a reference to the hint itself
    assert fr.leak('so the answer is readFile("notes.md") now', ['readFile("notes.md")'], '') == 'readFile("notes.md")'
    assert fr.leak('The verified reference says to read the file.', [action], '') == 'hint_phrase'


def test_hint_turn_sits_before_the_assistant_opening_so_the_history_is_a_shared_prefix():
    context = [IM_START, 5, 6, 2, IM_START, 7, 2, IM_START, 8, 3]   # system, tool, assistant opening
    hinted = fr.hint_context(context, [IM_START, 40, 41, 2], IM_START)
    assert hinted[:7] == context[:7] and hinted[-3:] == context[-3:] and hinted[7:11] == [IM_START, 40, 41, 2]


def test_improvement_is_per_action_token_over_the_common_spans():
    assert abs(fr.improvement([-0.1, -0.5], [-0.3, -0.5, -0.9], [10, 30]) - 0.05) < 1e-9
    assert fr.improvement([], [-1.], [3]) is None


def test_selection_compares_only_within_the_students_own_length_band():
    def c(origin, length, imp, admitted=True, closed=True):
        return {'origin': origin, 'length': length, 'improvement': imp, 'admitted': admitted, 'closed': closed}
    short_best = c('hinted', 100, 0.9)          # closes early and scores best: outside every band
    in_band = c('hinted', 900, 0.2)
    own = c('unhinted', 1000, 0.1)
    chosen, anchor = fr.select([short_best, in_band, own], (0.67, 1.5), random.Random(0))
    assert chosen is in_band and anchor == 1000
    # nothing admitted inside the band: no fill (the call keeps an empty block)
    assert fr.select([short_best, c('unhinted', 1000, -0.1, admitted=False)], (0.67, 1.5), random.Random(0)) == (None, None)
    # no closed unhinted sample: no anchor, no fill
    assert fr.select([in_band, c('unhinted', 4096, None, admitted=False, closed=False)], (0.67, 1.5),
                     random.Random(0)) == (None, None)


def chain_of(calls):
    """A teacher chain as exp0_sft.chains_of builds it: prompts extend (prompt, sample, suffix) of the call before."""
    rows, prompt = [], [IM_START, 50, 51, 2, IM_START, 3]
    for index, (think, action, suffix) in enumerate(calls):
        sample = think + [CLOSE] + action + [2]
        weight = np.array([0.] * (len(think) + 1) + [1.] * (len(action) + 1), np.float32)
        rows.append({'id': f'r{index}', 'invocation': 'inv', 'index': index, 'prompt': np.array(prompt, np.int32),
                     'sample': np.array(sample, np.int32), 'weight': weight})
        prompt = prompt + sample + suffix
    return exp0_sft.chains_of([{'task_id': 't', 'round': 'teacher:k', 'rows': rows}])


def test_filled_rows_carry_the_filled_history_and_train_block_and_action():
    [chain] = chain_of([([70, 71], [80], [2, IM_START, 60, 2, IM_START, 3]), ([72], [81, 82], [])])
    calls = fr.calls_of(chain, CLOSE)
    assert [list(c['action']) for c in calls] == [[80, 2], [81, 82, 2]]
    fills = [{'kind': 'filled', 'reasoning': [100, 101, 102, CLOSE]}, {'kind': 'fallback', 'reasoning': None}]
    rows = fr.filled_rows(calls, fills, CLOSE, 'filled:teacher:k')
    # call 1's prompt holds call 0's filled block, never the teacher's think tokens
    assert list(rows[1]['prompt']) == list(rows[0]['prompt']) + [100, 101, 102, CLOSE, 80, 2] + [2, IM_START, 60, 2, IM_START, 3]
    assert list(rows[0]['weight']) == [1.] * 4 + [1., 1.] and list(rows[0]['sampled']) == [True] * 4 + [False] * 2
    assert list(rows[1]['sample']) == [CLOSE, 81, 82, 2] and list(rows[1]['weight']) == [0., 1., 1., 1.]
    assert not rows[1]['sampled'].any()
    built = exp0_sft.chains_of([{'task_id': 't', 'round': 'filled:teacher:k', 'rows': rows}])
    assert len(built) == 1   # the rebuilt prompts still extend each other: one sequence
    assert exp0_sft.sampled_targets(built[0]).tolist() == [True] * 4 + [False] * 2 + [False] * 4
    assert exp0_sft.sampled_mask(built, 'cpu').tolist() == exp0_sft.sampled_targets(built[0]).tolist()


def test_a_context_only_action_keeps_its_filled_block_untrained():
    [chain] = chain_of([([70], [80], [])])
    calls = fr.calls_of(chain, CLOSE)
    calls[0]['action_weight'][:] = 0   # a backend detour: all context
    rows = fr.filled_rows(calls, [{'reasoning': [100, CLOSE]}], CLOSE, 'n')
    assert not rows[0]['weight'].any()


def test_chain_level_sampled_flags_are_unchanged_without_row_flags():
    def chain(round_name):
        return {'ids': np.zeros(4, np.int32), 'segs': [(1, 2, {'sample': np.zeros(2, np.int32)})], 'task': 't',
                'round': round_name}
    assert exp0_sft.sampled_targets(chain('rl:x')).tolist() == [True, True]
    assert exp0_sft.sampled_targets(chain('teacher:x')).tolist() == [False, False]


def test_aux_replaces_names_the_episode_a_filled_unit_stands_in_for():
    part = sft_sources.parse_aux('filled=/x/chains.pkl,share=0.2,replaces=teacher')
    assert part['replaces'] == 'teacher' and part['share'] == 0.2
    assert sft_sources.replaced_round(part, {'round': 'filled:teacher:pool:t1:0'}) == 'teacher:pool:t1:0'
    # every teacher key form names its own episode (the build drops only chains of part ``teacher`` with that round)
    assert sft_sources.replaced_round(part, {'round': 'filled:october-teacher:g:t'}) == 'october-teacher:g:t'
    assert sft_sources.replaced_round(part, {'round': 'filled:live:vibestudio:x:ab12'}) == 'live:vibestudio:x:ab12'
    assert sft_sources.replaced_round(part, {'round': 'nocolon'}) is None
    assert sft_sources.replaced_round(sft_sources.parse_aux('ops=/x.pkl'), {'round': 'filled:teacher:a'}) is None


def test_episode_walk_against_a_scripted_engine(monkeypatch):
    """Filler.episode end to end on a scripted SGLang: draws (natlang_top_k records), prefill scores with input and
    ``</think>`` logprobs, the hint turn, gates, length-banded choice, filled history; then filled_rows rebuilds it."""
    import base64
    import types
    import pytest
    from natlang_neuralese.common.paths import root
    tok_path = root('data_nvme') / 'sft-sources-20261120b' / 'tokenizer' / 'tokenizer.json'
    if not tok_path.exists():
        pytest.skip('no LFM3 tokenizer on this machine')
    from tokenizers import Tokenizer
    from natlang_neuralese.serve.sglang_actor import TOP_K_KEY
    tok = Tokenizer.from_file(str(tok_path))
    close, im_start, im_end = (tok.token_to_id(t) for t in ('</think>', '<|im_start|>', '<|im_end|>'))
    enc = lambda text: list(tok.encode(text, add_special_tokens=False).ids)   # noqa: E731
    prompt = enc('<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\ntask<|im_end|>\n<|im_start|>assistant\n<think>')
    action1, action2 = enc('\n\nread the file') + [im_end], enc('\n\nreturn it') + [im_end]
    suffix = enc('\n<|im_start|>tool\nfile body<|im_end|>\n<|im_start|>assistant\n<think>')
    rows = [{'id': 'a', 'invocation': 'i', 'index': 0, 'prompt': np.array(prompt, np.int32),
             'sample': np.array(enc('teacher plan') + [close] + action1, np.int32)}]
    rows.append({'id': 'b', 'invocation': 'i', 'index': 1,
                 'prompt': np.array(prompt + list(rows[0]['sample']) + suffix, np.int32),
                 'sample': np.array([close] + action2, np.int32)})
    [chain] = exp0_sft.chains_of([{'task_id': 't', 'round': 'teacher:k', 'rows': rows}])
    long_block = enc(' I should look at the file before answering, then return what it holds.') * 3 + [close]
    leaky_block = enc(' the reference says read the file') + [close]
    record = base64.b64encode(np.float32(-0.5).tobytes() + np.int32(0).tobytes() + np.float16(0).tobytes()).decode()
    seen = {'hinted_prompts': 0}

    def post(path, body, **_):
        ids, params = body['input_ids'], body['sampling_params']
        if params.get('max_new_tokens', 1) > 1:      # a draw: hinted prompts leak, plain ones reason at length
            hinted = 'Verified reference' in tok.decode(ids, skip_special_tokens=False)
            seen['hinted_prompts'] += hinted
            out = leaky_block if hinted else long_block
            return {'output_ids': out, 'meta_info': {TOP_K_KEY: [record] * len(out)}}
        if not body.get('return_logprob'):
            return {}
        start = body['logprob_start_len']
        n = len(ids) - start
        empty = ids[start + 1] == close            # the scored block is the empty one
        lp = -1.0 if empty else -0.2           # actions are likelier after reasoning than after an empty block
        return {'meta_info': {'input_token_logprobs': [[lp, i, None] for i in ids[start:]],
                              'input_token_ids_logprobs': [[[-1.0, close, None]] for _ in range(n)]}}
    served = {'sha': 'p1'}
    actor = types.SimpleNamespace(post=post, acquire=lambda: None, release=lambda: None,
                                  policy=lambda max_age=30.: {'checkpoint_sha256': served['sha']},
                                  attest=lambda: served['sha'])
    args = types.SimpleNamespace(unhinted=2, hinted=2, hint_horizon=3, hint_response_chars=1500, hint_max_tokens=6000,
                                 score_horizon=1, max_tokens=512, max_len=32000, band=(0.67, 1.5), close_floor=0.01,
                                 min_improvement=0.0, ratio_floor=None, top_p=0.95, seed=0, draw='parallel',
                                 skip_certain_nll=0.01, trend_window=3)
    filler = fr.Filler(args, actor, tok, [im_end], [])
    renderer = types.SimpleNamespace(chains=lambda item: ([chain], None), tokenizer_dir='tok')
    out = filler.episode({'key': 'teacher:k', 'work': 'w', 'name': 'n', 'task_id': 't', 'teacher': 'm'}, renderer)
    calls = out['chains'][0]['calls']
    assert seen['hinted_prompts'] == 4 and [c['kind'] for c in calls] == ['filled', 'filled']
    assert all(c['origin'] == 'unhinted' for c in calls)          # the hinted blocks leaked the hint: rejected
    hinted = [x for c in calls for x in c['candidates'] if x['origin'] == 'hinted']
    assert hinted and all(x['leak'] and 'leak' in x['rejected'] for x in hinted)
    built = fr.filled_rows(fr.calls_of(chain, close), calls, close, 'filled:teacher:k')
    assert list(built[0]['sample']) == long_block + action1
    assert list(built[1]['prompt']) == prompt + long_block + action1 + suffix   # the filled block is history
    assert [c['policy'] for c in calls] == ['p1', 'p1']                  # every step attests the served policy

    # an expected policy the front does not serve refuses the step; a rebind during a step draws it again
    args.expect_policy = 'p0'
    with pytest.raises(fr.PolicyRefused):
        fr.Filler(args, actor, tok, [im_end], []).episode(
            {'key': 'teacher:k', 'work': 'w', 'name': 'n', 'task_id': 't', 'teacher': 'm'}, renderer)
    args.expect_policy = None
    reads = iter(['p1', 'p2'] + ['p2'] * 20)
    actor.attest = lambda: next(reads)
    rebound = fr.Filler(args, actor, tok, [im_end], [])
    out = rebound.episode({'key': 'teacher:k', 'work': 'w', 'name': 'n', 'task_id': 't', 'teacher': 'm'}, renderer)
    assert [c['policy'] for c in out['chains'][0]['calls']] == ['p2', 'p2']
    assert rebound.counts['policy_changed_mid_step'] == 1
    actor.attest = lambda: served['sha']

    # a stopped generation resumes an episode at the step it reached (OUT/partial/), its blocks as history
    import tempfile
    with tempfile.TemporaryDirectory() as partial:
        args.partial_dir = Path(partial)
        first = filler.episode({'key': 'teacher:k', 'work': 'w', 'name': 'n', 'task_id': 't', 'teacher': 'm'}, renderer)
        saved = fr.load_partial(args.partial_dir, 'teacher:k', json.loads(json.dumps(fr.settings(args))))
        assert [len(v) for v in saved.values()] == [2] and first['resumed_steps'] == 0
        saved_chain = {'digest': fr.chain_digest(chain), 'calls': list(saved.values())[0][:1]}
        fr.save_partial(args.partial_dir, 'teacher:k', json.loads(json.dumps(fr.settings(args))), [saved_chain])
        draws = seen['hinted_prompts']
        again = filler.episode({'key': 'teacher:k', 'work': 'w', 'name': 'n', 'task_id': 't', 'teacher': 'm'}, renderer)
        assert again['resumed_steps'] == 1 and seen['hinted_prompts'] == draws + 2   # only step 2 drawn again
        assert again['chains'][0]['calls'][0] == saved_chain['calls'][0]
        args.max_tokens = 256                                   # other settings: the partial file is not used
        assert fr.load_partial(args.partial_dir, 'teacher:k', json.loads(json.dumps(fr.settings(args)))) == {}
        args.max_tokens = 512
        # a saved step without an attested policy (written before attestation) is drawn again
        fr.save_partial(args.partial_dir, 'teacher:k', json.loads(json.dumps(fr.settings(args))),
                        [{'digest': fr.chain_digest(chain), 'calls': [{**saved_chain['calls'][0], 'policy': None}]}])
        redo = filler.episode({'key': 'teacher:k', 'work': 'w', 'name': 'n', 'task_id': 't', 'teacher': 'm'}, renderer)
        assert redo['resumed_steps'] == 0


def test_fill_policies_name_every_policy_that_drew_a_step():
    record = {'policy_start': {'checkpoint_sha256': 'a'}, 'policy_end': {'checkpoint_sha256': 'b'},
              'chains': [{'calls': [{'policy': 'c'}, {'kind': 'no_think'}]}]}
    assert fr.fill_policies(record) == {'a', 'b', 'c'}
    assert fr.fill_policies({'policy_start': None, 'chains': []}) == set()


def test_length_reweighting_restores_the_students_own_length_mix():
    def cand(origin, length, selected=False, hazard=1.):
        return {'origin': origin, 'length': length, 'closed': True, 'selected': selected, 'hazard': hazard}
    # the student reasons long half the time; the gate selected mostly short blocks
    calls = [{'candidates': [cand('unhinted', 100), cand('unhinted', 900), cand('hinted', 100, selected=True)]}] * 6
    calls += [{'candidates': [cand('unhinted', 100), cand('unhinted', 900, selected=True)]}] * 2
    records = [{'chains': [{'calls': calls}]}]
    weights, edges = fr.block_weights(records, cap=4., mode='length')
    short, long_ = (fr.length_bin(100), 0), (fr.length_bin(900), 0)
    assert edges == [] and weights[long_] > 1 > weights[short]
    assert abs((6 * weights[short] + 2 * weights[long_]) / 8 - 1) < 1e-9          # mean 1: the loss mass is kept
    long_share = 2 * weights[long_] / (6 * weights[short] + 2 * weights[long_])
    assert 0.25 < long_share <= 0.5                                                 # toward the student's 50%
    assert fr.block_weights(records, cap=0) is None and fr.block_weights(records, cap=4., mode='off') is None
    row = {'sampled': np.array([True] * 900 + [False] * 3), 'weight': np.ones(903, np.float32)}
    fr.reweight([row], [{'candidates': [cand('unhinted', 900, selected=True)]}], (weights, edges))
    assert abs(row['weight'][0] - weights[long_]) < 1e-6 and row['weight'][-1] == 1.


def test_joint_reweighting_corrects_the_close_hazard_too():
    def cand(origin, length, hazard, selected=False):
        return {'origin': origin, 'length': length, 'closed': True, 'selected': selected, 'hazard': hazard}
    # same lengths, but the chosen blocks close readily (high hazard) far more often than the student's own
    own = [{'candidates': [cand('unhinted', 300, 0.5), cand('unhinted', 300, 1.0), cand('unhinted', 300, 4.0)]}] * 4
    picked = [{'candidates': [cand('hinted', 300, 4.0, selected=True)]}] * 6
    picked += [{'candidates': [cand('hinted', 300, 0.5, selected=True)]}] * 2
    records = [{'chains': [{'calls': own + picked}]}]
    weights, edges = fr.block_weights(records, cap=4.)
    assert len(edges) == 2
    high, low = fr.block_cell(cand('x', 300, 4.0), edges), fr.block_cell(cand('x', 300, 0.5), edges)
    assert weights[low] > 1 > weights[high]
    summary = fr.weighted_summary(records, (weights, edges))
    own_h = summary['own']['hazard_mean']
    assert abs(summary['chosen_weighted']['hazard_mean'] - own_h) < abs(summary['chosen']['hazard_mean'] - own_h)
    # length-only weighting leaves the hazard untouched
    w_len = fr.block_weights(records, cap=4., mode='length')
    assert fr.weighted_summary(records, w_len)['chosen_weighted']['hazard_mean'] == summary['chosen']['hazard_mean']

def scripted_filler(blocks, action_lp, args_extra=None):
    """A Filler on a scripted engine: draws return ``blocks[origin]`` (cut at max_new_tokens, never closing when cut);
    action log-probabilities ``action_lp(block_is_empty)``. Returns (filler, chain, draw log)."""
    import base64
    import types
    import pytest
    from natlang_neuralese.common.paths import root
    tok_path = root('data_nvme') / 'sft-sources-20261120b' / 'tokenizer' / 'tokenizer.json'
    if not tok_path.exists():
        pytest.skip('no LFM3 tokenizer on this machine')
    from tokenizers import Tokenizer
    from natlang_neuralese.serve.sglang_actor import TOP_K_KEY
    tok = Tokenizer.from_file(str(tok_path))
    close, im_end = tok.token_to_id('</think>'), tok.token_to_id('<|im_end|>')
    enc = lambda text: list(tok.encode(text, add_special_tokens=False).ids)   # noqa: E731
    prompt = enc('<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\ntask<|im_end|>\n<|im_start|>assistant\n<think>')
    action = enc('\n\nread the file') + [im_end]
    rows = [{'id': 'a', 'invocation': 'i', 'index': 0, 'prompt': np.array(prompt, np.int32),
             'sample': np.array([close] + action, np.int32)}]
    [chain] = exp0_sft.chains_of([{'task_id': 't', 'round': 'teacher:k', 'rows': rows}])
    record = base64.b64encode(np.float32(-0.5).tobytes() + np.int32(0).tobytes() + np.float16(0).tobytes()).decode()
    log = []

    def post(path, body, **_):
        ids, params = body['input_ids'], body['sampling_params']
        if params.get('max_new_tokens', 1) > 1:
            hinted = 'Verified reference' in tok.decode(ids, skip_special_tokens=False)
            full = blocks['hinted' if hinted else 'unhinted'](enc, close)
            out = full[:params['max_new_tokens']]
            log.append(('hinted' if hinted else 'unhinted', params['max_new_tokens'], len(out)))
            return {'output_ids': out, 'meta_info': {TOP_K_KEY: [record] * len(out)}}
        if not body.get('return_logprob'):
            return {}
        start = body['logprob_start_len']
        lp = action_lp(ids[start + 1] == close)
        return {'meta_info': {'input_token_logprobs': [[lp, i, None] for i in ids[start:]],
                              'input_token_ids_logprobs': [[[-1.0, close, None]] for _ in ids[start:]]}}
    actor = types.SimpleNamespace(post=post, acquire=lambda: None, release=lambda: None,
                                  policy=lambda max_age=30.: {'checkpoint_sha256': 'p1'}, attest=lambda: 'p1')
    args = types.SimpleNamespace(unhinted=2, hinted=2, hint_horizon=3, hint_response_chars=1500, hint_max_tokens=6000,
                                 score_horizon=1, max_tokens=512, max_len=32000, band=(0.67, 1.5), close_floor=0.01,
                                 min_improvement=0.0, ratio_floor=None, top_p=0.95, seed=0, draw='staged',
                                 skip_certain_nll=0.01, trend_window=3, **(args_extra or {}))
    return fr.Filler(args, actor, tok, [im_end], []), chain, log


def run_steps(filler, chain):
    import types
    renderer = types.SimpleNamespace(chains=lambda item: ([chain], None), tokenizer_dir='tok')
    out = filler.episode({'key': 'teacher:k', 'work': 'w', 'name': 'n', 'task_id': 't', 'teacher': 'm'}, renderer)
    return out['chains'][0]['calls']


def test_staged_draws_stop_at_an_admitted_anchor():
    long_ = lambda enc, close: enc(' plan the read of the file carefully') * 4 + [close]   # noqa: E731
    filler, chain, log = scripted_filler({'unhinted': long_, 'hinted': long_}, lambda empty: -1.0 if empty else -0.2)
    [fill] = run_steps(filler, chain)
    assert fill['kind'] == 'filled' and fill['origin'] == 'unhinted' and log == [('unhinted', 512, len(log) and log[0][2])]
    assert fill['candidates'][0]['anchor'] and fill['generated_tokens'] == log[0][2]


def test_staged_waves_are_capped_at_the_band_and_skip_certain_actions():
    # the plain student reasons at length but its reasoning does not help; the hinted block runs past the band
    plain = lambda enc, close: enc(' something else entirely') * 4 + [close]           # noqa: E731
    rambling = lambda enc, close: enc(' and so on') * 200 + [close]                     # noqa: E731
    filler, chain, log = scripted_filler({'unhinted': plain, 'hinted': rambling}, lambda empty: -0.2 if empty else -0.6)
    [fill] = run_steps(filler, chain)
    anchor_len = log[0][2]
    cap = int(np.ceil(1.5 * anchor_len)) + 1
    assert fill['kind'] == 'fallback' and log[0] == ('unhinted', 512, anchor_len)
    assert all(m == cap for _, m, _ in log[1:]) and len(log) == 4      # anchor, wave (1 + 1), wave (1 hinted)
    assert 'past_band' in [r for c in fill['candidates'] for r in c['rejected']]
    # an action already near-certain after an empty block: no draws at all
    filler, chain, log = scripted_filler({'unhinted': plain, 'hinted': plain}, lambda empty: -0.001)
    [fill] = run_steps(filler, chain)
    assert fill['kind'] == 'certain' and log == [] and fill['generated_tokens'] == 0


def test_length_reference_uses_only_anchors_once_draws_are_staged():
    def cand(length, **k):
        return {'origin': 'unhinted', 'length': length, 'closed': True, 'hazard': 1., **k}
    staged = [{'candidates': [cand(900, anchor=True, selected=True), cand(100, anchor=False)]}]
    own, chosen = fr.own_and_chosen([{'chains': [{'calls': staged}]}])
    assert [c['length'] for c in own] == [900] and [c['length'] for c in chosen] == [900]   # capped draws are no reference

def test_a_trajectory_that_keeps_falling_back_draws_only_anchors():
    plain = lambda enc, close: enc(' something else entirely') * 4 + [close]   # noqa: E731
    filler, chain, log = scripted_filler({'unhinted': plain, 'hinted': plain}, lambda empty: -0.2 if empty else -0.6)
    calls = fr.calls_of(chain, filler.close)
    context = [int(i) for i in calls[0]['prompt']]
    import random
    fill = filler.step(context, calls, 0, filler.text(context, special=True), random.Random(0), trend=3)
    assert fill['kind'] == 'fallback' and fill['trend_skip'] and len(log) == 1


def test_actor_requests_retry_a_dropped_or_refused_connection(monkeypatch):
    import http.client
    import io
    import urllib.error
    answers = [http.client.RemoteDisconnected('closed'), urllib.error.URLError(ConnectionRefusedError(111, 'refused')),
               io.BytesIO(b'{"ok": 1}')]

    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self.body

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return Response(answer)
    monkeypatch.setattr(fr.urllib.request, 'urlopen', urlopen)
    monkeypatch.setattr(fr.time, 'sleep', lambda s: None)
    actor = fr.Actor('http://x', None, False, 1, 1, 1)
    assert actor.post('/generate', {}) == {'ok': 1} and actor.counts['actor_down_waits'] == 2
    monkeypatch.setattr(fr.urllib.request, 'urlopen', lambda r, timeout: (_ for _ in ()).throw(ValueError('bad')))
    import pytest
    with pytest.raises(ValueError):
        actor.post('/generate', {})


def test_drifted_successes_are_a_separate_admission_class():
    base = {'outcome': 'task_success', 'exact': False, 'unserved': 0, 'recorded_turns': 3, 'replayed_turns': 3}
    assert sft_sources.drifted_success({**base, 'served': {'exact': 2, 'drifted': 1}})
    assert not sft_sources.drifted_success({**base, 'served': {'exact': 2, 'missing': 1}})
    assert not sft_sources.drifted_success({**base, 'served': {'exact': 3}, 'unserved': 1})
    assert not sft_sources.drifted_success({**base, 'served': {'exact': 2, 'drifted': 1}, 'outcome': 'unresolved'})
    assert not sft_sources.drifted_success({**base, 'exact': True, 'served': {'exact': 3}})   # exact ones are exact
    assert fr.build_parser().parse_args(['generate', '--out', 'o', '--sglang', 's', '--export', 'e',
                                         '--render-tokenizer', 't']).admit_drifted == 'on'


def test_claims_keep_generators_sharing_out_on_disjoint_episodes(tmp_path):
    import os
    assert fr.claim(tmp_path, 'e1')
    assert fr.claim(tmp_path, 'e1')                  # this process already holds it
    path = next(tmp_path.glob('*.claim'))
    path.write_text(json.dumps({'pid': 1, 'host': os.uname().nodename}))       # pid 1 is not a generator: stale
    assert fr.claim(tmp_path, 'e1')
    path.write_text(json.dumps({'pid': os.getpid(), 'host': 'elsewhere'}))       # another host's live claim
    assert not fr.claim(tmp_path, 'e1')
    fr.release(tmp_path, 'e1')
    assert fr.claim(tmp_path, 'e1')


def test_finished_finds_a_key_in_fills(tmp_path):
    (tmp_path / 'fills.jsonl').write_text(json.dumps({'schema': 's', 'key': 'teacher:a:1', 'chains': [1] * 100}) + '\n')
    assert fr.finished(tmp_path, 'teacher:a:1') and not fr.finished(tmp_path, 'teacher:a:2')


def test_prefer_work_episodes_go_first_and_a_rescan_queues_new_ones_ahead():
    """``--prefer-work``: a live WORK's episodes before every other WORK's (each group keeps the shuffled, rate-preferred
    order); a standing queue takes a rescan's new episodes at its front, each key once, and waits when empty."""
    import threading
    episodes = [{'key': f'teacher:{i}', 'work': 'old', 'task_id': f't{i}'} for i in range(6)]
    live = [{'key': f'live:{i}', 'work': 'live', 'task_id': f'v{i}'} for i in range(3)]
    order = fr.ordered([*episodes, *live], ['live'], {'t0': (0, 4)}, seed=1)
    assert [e['work'] for e in order] == ['live'] * 3 + ['old'] * 6
    assert order[3]['key'] == 'teacher:0'                 # the failed task first among the rest
    assert fr.ordered(episodes, [], None, 1) == fr.ordered(episodes, ['live'], None, 1)
    q = fr.EpisodeQueue(order[:4], seen={'live:9'}, standing=True)
    assert q.add_first([live[0], {'key': 'live:9', 'work': 'live'}, {'key': 'live:3', 'work': 'live'}]) == 1
    assert q.next()['key'] == 'live:3'
    assert [q.next()['key'] for _ in range(4)] == [e['key'] for e in order[:4]]
    got = []
    t = threading.Thread(target=lambda: got.append(q.next()))
    t.start()
    t.join(0.2)
    assert t.is_alive() and not got                     # standing: waits for the next rescan
    q.add_first([{'key': 'live:4', 'work': 'live'}])
    t.join(2)
    assert got[0]['key'] == 'live:4'
    assert fr.EpisodeQueue([]).next() is None           # not standing: an empty queue ends the workers
    args = fr.build_parser().parse_args(['generate', '--out', 'o', '--sglang', 's', '--export', 'e', '--render-tokenizer',
                                         't', '--prefer-work', 'W', '--rescan-minutes', '30'])
    assert args.prefer_work == ['W'] and args.rescan_minutes == 30


def test_observed_live_episodes_are_fill_sources_and_their_records_name_the_bound_source(tmp_path, monkeypatch):
    """A live campaign WORK has no replays: its observed-live teacher successes (no replay name, a bound source) are
    filled, and a fill record carries what the build needs to render the episode again."""
    work = tmp_path / 'live'
    work.mkdir()
    (work / 'episodes-0000.jsonl.zst').write_bytes(b'')
    binding = {'path': '/c/manifest.json', 'sha256': 'ab'}
    found = [({'key': 'live:vibestudio:x:ab', 'task_id': 'vibestudio:x', 'task': {'id': 'vibestudio:x'},
               'teacher': {'model_id': 'luna'}},
              {'name': None, 'verification_mode': 'observed-live', 'live_source': binding}),
             ({'key': 'teacher:y', 'task_id': 'y', 'task': {'id': 'y'}}, {'name': None})]   # no replay: skipped
    monkeypatch.setattr(sft_sources, 'verified_episodes', lambda w, funnel, **kw: iter(found))
    monkeypatch.setattr(sft_sources, 'held_closure', lambda m: (set(), set(), set()))
    monkeypatch.setattr(sft_sources, 'held_excluded_ids', lambda m, tasks, held: set())
    [item] = fr.teacher_episodes([str(work)], 'held.json', {'teacher': {}})
    assert item['verification_mode'] == 'observed-live' and item['live_source'] == binding and item['name'] is None
    record = {'schema': fr.SCHEMA, **fr.item_of(item), 'teacher': 'luna', 'chains': []}
    assert fr.item_of(record) == {k: item[k] for k in fr.ITEM_KEYS}
    assert fr.open_goal_of({'key': 'teacher:y', 'work': 'w', 'name': 'n', 'task_id': 'y'}) is None


def test_actor_attests_the_served_policy_and_generate_flags():
    """The export names e15b, a rebound actor serves another policy: the front's binding is what each step records."""
    args = fr.build_parser().parse_args(['generate', '--out', 'o', '--sglang', 's', '--export', 'e',
                                         '--render-tokenizer', 't'])
    assert not args.follow_served_policy and not args.no_policy_attest and args.expect_policy is None
    actor = fr.Actor('http://s', 'http://f', False, 1, 1, 1)
    actor.get = lambda url, timeout=10: {'rl_policy_binding': {'checkpoint_sha256': 'b' * 64}}
    assert actor.attest() == 'b' * 64
    actor.get = lambda url, timeout=10: (_ for _ in ()).throw(OSError('down'))
    assert actor.attest() == 'b' * 64                # read within ATTEST_AGE: shared
    actor.policy_at = 0.
    assert actor.attest() is None                    # an unanswering front attests nothing
    assert fr.build_parser().parse_args(['generate', '--out', 'o', '--sglang', 's', '--export', 'e',
                                         '--render-tokenizer', 't', '--follow-served-policy']).follow_served_policy


def test_unattested_steps_are_relabelled_from_the_serving_record_or_left_out():
    def rec(key, *policies):
        return {'key': key, 'chains': [{'calls': [{'policy': p} for p in policies]}]}
    assert fr.attest_record(rec('a', 'x', 'x'), {}) is not None
    assert fr.attest_record(rec('b', 'x', None), {}) is None              # not attributable: left out
    fixed = fr.attest_record(rec('c', None, 'x'), {'c': 'x'})
    assert [f['policy'] for f in fixed['chains'][0]['calls']] == ['x', 'x']
    assert fr.build_parser().parse_args(['build', '--out', 'o', '--corpus', 'c']).admit_unattested is False


class FakeTokenizer:
    def convert_tokens_to_ids(self, token):
        return CLOSE


class FakeRenderer:
    """A picklable ``Renderer`` stand-in: an episode's chains from its key (worker processes build their own)."""

    def __init__(self, tokenizer_dir):
        if 'broken' in tokenizer_dir:
            raise OSError(f'no tokenizer at {tokenizer_dir}')
        self.tokenizer = FakeTokenizer()

    def chains(self, item):
        n = int(item['key'][1:])
        if n % 7 == 3:
            raise ValueError(f'cannot render {item["key"]}')
        rows, prompt = [], np.arange(5, dtype=np.int32)
        for i in range(1 + n % 3):         # each call's prompt extends the previous call's text: one chain
            sample = np.array([7, CLOSE, n, i], np.int32)
            rows.append({'id': f'{item["key"]}#{i}', 'invocation': 'root', 'index': i, 'prompt': prompt,
                         'sample': sample, 'weight': np.ones(4, np.float32)})
            prompt = np.concatenate([prompt, sample, np.full(3, 2, np.int32)])
        return exp0_sft.chains_of([{'task_id': item['task_id'], 'round': item['key'], 'rows': rows}]), self


def test_build_renders_episodes_in_worker_processes_as_it_does_serially():
    records = [{'key': f'e{n}', 'task_id': f't{n % 4}', 'work': '/w', 'name': f'e{n}',
                'render_tokenizer': '/tok-broken' if n == 11 else '/tok'} for n in range(30)]
    serial = list(fr.rendered_sources(records, 1, FakeRenderer))
    parallel = list(fr.rendered_sources(records, 3, FakeRenderer))
    assert len(serial) == len(parallel) == 30
    for n, (a, b) in enumerate(zip(serial, parallel)):
        if n == 11 or n % 7 == 3:
            assert a[0] is b[0] is None and a[2] == b[2] and ('cannot render' in a[2] or 'no tokenizer' in a[2])
            continue
        assert a[1] == b[1] == CLOSE and a[2] is b[2] is None and len(a[0]) == len(b[0]) == 1
        for x, y in zip(a[0], b[0]):
            assert np.array_equal(x['ids'], y['ids']) and len(x['segs']) == len(y['segs'])
            for (s1, l1, r1), (s2, l2, r2) in zip(x['segs'], y['segs']):
                assert (s1, l1) == (s2, l2) and 'prompt' not in r1 and 'prompt' not in r2
                assert r1.keys() == r2.keys() and all(np.array_equal(r1[k], r2[k]) for k in r1)
        calls = fr.calls_of(a[0][0], CLOSE)          # prompts are cut from the chain ids
        assert [len(c['prompt']) for c in calls] == [5 + 7 * i for i in range(1 + n % 3)]
