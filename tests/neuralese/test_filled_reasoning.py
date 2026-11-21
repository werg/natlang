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
    assert sft_sources.replaced_round(part, {'round': 'filled:rl:run:g1'}) is None
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
    actor = types.SimpleNamespace(post=post, acquire=lambda: None, release=lambda: None, policy=lambda: {'p': 1})
    args = types.SimpleNamespace(unhinted=2, hinted=2, hint_horizon=3, hint_response_chars=1500, hint_max_tokens=6000,
                                 score_horizon=1, max_tokens=512, max_len=32000, band=(0.67, 1.5), close_floor=0.01,
                                 min_improvement=0.0, ratio_floor=None, top_p=0.95, seed=0)
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


def test_fill_policies_name_every_policy_that_drew_a_step():
    record = {'policy_start': {'checkpoint_sha256': 'a'}, 'policy_end': {'checkpoint_sha256': 'b'},
              'chains': [{'calls': [{'policy': 'c'}, {'kind': 'no_think'}]}]}
    assert fr.fill_policies(record) == {'a', 'b', 'c'}
    assert fr.fill_policies({'policy_start': None, 'chains': []}) == set()
