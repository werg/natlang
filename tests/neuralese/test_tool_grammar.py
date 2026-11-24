"""The tool-call grammar law (`common.tool_grammar`) and its exact learner replay (`train.lfm3_fast.HeadMask`): the
structural tag accepts the backbone's call forms and refuses values outside their schema, a reply's masks are
recomputed from its tokens, and the learner's masked NLL, nucleus and gradients equal the renormalized law over the
allowed set."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
from natlang_neuralese.common import tool_grammar  # noqa: E402
from natlang_neuralese.train import lfm3_fast  # noqa: E402

xgrammar = pytest.importorskip('xgrammar')

TOOLS = [{'type': 'function', 'function': {'name': 'return_result', 'parameters': {
            'type': 'object', 'required': ['value'], 'properties': {'value': {'anyOf': [
                {'type': 'array', 'prefixItems': [{'type': 'string'}], 'minItems': 1, 'maxItems': 1},
                {'type': 'array', 'prefixItems': [{'type': 'string'}, {'type': 'number'}], 'minItems': 2,
                 'maxItems': 2}]}}}}},
         {'type': 'function', 'function': {'name': 'eval', 'parameters': {
             'type': 'object', 'required': ['code'], 'properties': {'code': {'type': 'string'}}}}}]
SPECIALS = ['<tool_call>', '</tool_call>', '<|im_end|>']


def char_info():
    """A character-level raw vocabulary with the call markers as single tokens; `<|im_end|>` stops."""
    chars = [chr(c) for c in range(32, 127)] + ['\n']
    vocab = chars + SPECIALS
    return xgrammar.TokenizerInfo(vocab, xgrammar.VocabType.RAW, vocab_size=len(vocab) + 5,
                                  stop_token_ids=[vocab.index('<|im_end|>')]), vocab


def encode(text, vocab):
    ids, i = [], 0
    while i < len(text):
        special = next((s for s in SPECIALS if text.startswith(s, i)), None)
        ids.append(vocab.index(special or text[i]))
        i += len(special) if special else 1
    return ids


def call(value):
    return f'<tool_call>\n<function=return_result>\n<parameter=value>\n{value}\n</parameter>\n</function>\n</tool_call>'


def test_structural_tag_follows_the_call_format_and_the_schema():
    from xgrammar.testing import _is_grammar_accept_string
    for fmt, good, bad in (
            ('xml', 'thinking</think>\n\n' + call('["red", 2]') + 'done', call('["red", "x"]')),
            ('json', '<tool_call>\n{"name": "return_result", "arguments": {"value": ["red"]}}\n</tool_call>',
             '<tool_call>\n{"name": "nope", "arguments": {}}\n</tool_call>')):
        grammar = xgrammar.Grammar.from_structural_tag(json.dumps(tool_grammar.structural_tag(TOOLS, fmt)))
        assert _is_grammar_accept_string(grammar, good), fmt
        assert not _is_grammar_accept_string(grammar, bad), fmt
    assert tool_grammar.structural_tag(TOOLS, 'pythonic') is None and tool_grammar.structural_tag([], 'xml') is None


def test_law_declaration_and_validation(monkeypatch):
    info, _ = char_info()
    law = tool_grammar.tool_call_law(TOOLS, 'xml', info)
    assert law['schema'] == tool_grammar.SCHEMA and tool_grammar.validate_law(law) is law
    assert law['tokenizer_info_sha256'] == tool_grammar.tokenizer_info_sha256(info)
    with pytest.raises(ValueError, match='sha256'):
        tool_grammar.validate_law({**law, 'grammar': law['grammar'] + ' '})
    with pytest.raises(ValueError, match='compiled by'):
        tool_grammar.validate_law({**law, 'compiler': 'xgrammar==0.0.1'})
    monkeypatch.setenv(tool_grammar.ENV, '0')
    assert not tool_grammar.enabled()
    monkeypatch.setenv(tool_grammar.ENV, '1')
    assert tool_grammar.enabled()


def test_replay_masks_recompute_the_draw_masks_and_refuse_off_grammar_tokens():
    info, vocab = char_info()
    law = tool_grammar.tool_call_law(TOOLS, 'xml', info)
    reply = encode('ok ' + call('["red"]') + '<|im_end|>', vocab)
    table, index = tool_grammar.replay_masks(law, reply, info)
    allowed = tool_grammar.unpack(torch.from_numpy(np.stack(table)), info.vocab_size)
    assert len(index) == len(reply) and allowed[index, reply].all()
    free = allowed[index[0]]
    assert free[:len(vocab)].all() and not free[len(vocab):].any()       # free text: every real token, no padding
    value_start = len(encode('ok <tool_call>\n<function=return_result>\n<parameter=value>\n', vocab))
    assert not allowed[index[value_start], vocab.index('a')]             # a value of the schema opens with '['
    # A prefix fed before the first sample (a templated call's opening) puts the grammar in the same state.
    head = encode('<tool_call>\n<function=return_result>\n<parameter=value>\n', vocab)
    tail = encode('["red"]\n</parameter>\n</function>\n</tool_call>', vocab)
    t2, i2 = tool_grammar.replay_masks(law, tail, info, prefix=head)
    t1, i1 = tool_grammar.replay_masks(law, head + tail, info)
    assert np.array_equal(np.stack(t2)[i2], np.stack(t1)[i1[len(head):]])
    assert t2[i2[0]] is t1[i1[len(head)]] and not t2[0].flags.writeable    # one shared array per distinct mask
    with pytest.raises(ValueError, match='outside its tool grammar'):
        tool_grammar.replay_masks(law, encode(call('["red", "x"]'), vocab), info)


def test_matcher_sync_follows_a_rewound_reply():
    info, vocab = char_info()
    law = tool_grammar.tool_call_law(TOOLS, 'xml', info)
    matcher = tool_grammar.Matcher(law, info)
    reply = encode('ab<tool_call>\n<function=eval>\n', vocab)
    assert matcher.sync(reply)
    after = matcher.bitmask().clone()
    assert matcher.sync(reply[:3]) and matcher.sync(reply)           # rolled back, then fed again
    assert torch.equal(matcher.bitmask(), after)
    assert not matcher.sync(encode('<tool_call>\n<function=nope>', vocab))


def _dense(h, weight, target, allowed, top_p=None):
    logits = F.linear(h, weight).float().masked_fill(~allowed, float('-inf'))
    logp = torch.log_softmax(logits, -1)
    nll = -logp.gather(1, target[:, None])[:, 0]
    if top_p is None:
        return nll
    values = logp.sort(dim=-1, descending=True).values
    p = values.exp()
    count = ((p.cumsum(-1) - p) <= top_p).sum(-1)
    cut = values.gather(1, (count - 1)[:, None])
    kept = (logp >= cut) | F.one_hot(target, logits.shape[-1]).bool()
    nucleus = torch.logsumexp(logits.masked_fill(~kept, float('-inf')), -1) - logits.gather(1, target[:, None])[:, 0]
    return nll, nucleus


def test_head_mask_is_the_renormalized_law_with_its_gradients():
    torch.manual_seed(0)
    rows, width, vocab = 41, 10, 70
    weight = (torch.randn(vocab, width) * .9).requires_grad_()
    h = torch.randn(rows, width, requires_grad=True)
    masked_rows = torch.tensor(sorted(np.random.default_rng(0).choice(rows, 25, replace=False).tolist()))
    tables = torch.rand(3, vocab) < .4
    which = torch.randint(0, 3, (len(masked_rows),))
    allowed = torch.ones(rows, vocab, dtype=torch.bool)
    allowed[masked_rows] = tables[which]
    target = torch.stack([torch.multinomial(a.float(), 1)[0] for a in allowed])
    packed = torch.zeros(3, (vocab + 31) // 32, dtype=torch.int64)
    for t in range(vocab):
        packed[:, t // 32] |= tables[:, t].long() << (t % 32)
    packed = torch.where(packed >= 2 ** 31, packed - 2 ** 32, packed).to(torch.int32)
    assert torch.equal(tool_grammar.unpack(packed, vocab), tables)
    mask = lfm3_fast.HeadMask(masked_rows, packed, which)
    scale, scale_nucleus = torch.randn(rows), torch.randn(rows)
    want = _dense(h, weight, target, allowed)
    want_grads = torch.autograd.grad((want * scale).sum(), (h, weight))
    got = lfm3_fast.linear_nll(h, weight, target, chunk=8, mask=mask)
    assert torch.allclose(got, want, atol=1e-5)
    grads = torch.autograd.grad((got * scale).sum(), (h, weight))
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(grads, want_grads))
    unmasked = lfm3_fast.linear_nll(h, weight, target, chunk=8)
    assert (unmasked[masked_rows] > got[masked_rows] + 1e-4).any()      # the bias the mask removes
    others = torch.ones(rows, dtype=torch.bool)
    others[masked_rows] = False
    assert torch.equal(unmasked[others], got[others])
    for top_p in (.6, .95):
        want = _dense(h, weight, target, allowed, top_p)
        want_grads = torch.autograd.grad((want[0] * scale).sum() + (want[1] * scale_nucleus).sum(), (h, weight))
        got = lfm3_fast.linear_nll_nucleus(h, weight, target, top_p, chunk=8, top_k=4, mask=mask)
        assert torch.allclose(got[0], want[0], atol=1e-5) and torch.allclose(got[1], want[1], atol=1e-5)
        grads = torch.autograd.grad((got[0] * scale).sum() + (got[1] * scale_nucleus).sum(), (h, weight))
        assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(grads, want_grads))
    part = mask.subset(10, 30)
    assert torch.equal(part.rows, masked_rows[(masked_rows >= 10) & (masked_rows < 30)] - 10)


def test_lowering_and_law_mask_score_each_target_under_its_draw(monkeypatch):
    """A capture declaring a tool grammar lowers to a row carrying it (and its forced opening and guidance bans);
    exp0_sft.law_mask places each row's masks at its targets in packed order, bans cleared from the masks."""
    import collections
    import exp0_sft
    from lower_rollout_turns import sampled_call
    info, vocab = char_info()
    law = tool_grammar.tool_call_law(TOOLS, 'xml', info)
    monkeypatch.setattr(tool_grammar, 'info_from_base', lambda base: (info, tool_grammar.tokenizer_info_sha256(info)))
    monkeypatch.setitem(exp0_sft.LAW_BASE, 'path', 'char-vocab')
    head = encode('<tool_call>\n<function=return_result>\n<parameter=value>\n', vocab)
    reply = encode('["red"]\n</parameter>\n</function>\n</tool_call><|im_end|>', vocab)
    capture = {'prompt': [{'kind': 'tokens', 'token_ids': [1, 2, 3], 'producing': [False] * 3}],
               'samples': [{'index': i, 'token_id': t, 'logp': -.5, 'input_role': 'ordinary', 'terminal': i == len(reply) - 1,
                            'query_own_position': 2 + len(head) + i, **({'banned_token_ids': [vocab.index('[') + 1]}
                                                                          if i == 0 else {})}
                           for i, t in enumerate(reply)],
               'generated_inputs': [{'token_id': t, 'input_role': 'ordinary', 'sample_index': None} for t in head]
               + [{'token_id': t, 'input_role': 'ordinary', 'sample_index': i} for i, t in enumerate(reply[:-1])],
               'policy': {'checkpoint_sha256': 'p'}, 'sampling_law': {'temperature': 1.0, 'probabilities':
                                                                      'post-processing-normalized', 'tool_grammar': law}}
    stats = collections.Counter()
    row = sampled_call('inv', 0, capture, stats)
    assert stats['tool_grammar_calls'] == 1 and row['tool_grammar'] is law
    assert row['grammar_prefix'].tolist() == head and list(row['bans']) == [0]
    plain = {**row, 'tool_grammar': None, 'bans': None, 'sample': np.asarray(encode('hi', vocab), np.int32)}
    plain.pop('_law_masks', None)
    chains = [{'segs': [(3, 2, plain)]}, {'segs': [(3 + len(head), len(reply), row)]}]
    mask = exp0_sft.law_mask(chains, info.vocab_size)
    assert mask.rows.tolist() == [2 + i for i in range(len(reply))]
    allowed = mask.allowed_table(info.vocab_size, 'cpu')[mask.which]
    want, index = tool_grammar.replay_masks(law, reply, info, prefix=head)
    want = tool_grammar.unpack(torch.from_numpy(np.stack(want)), info.vocab_size)[index]
    want[0, vocab.index('[') + 1] = False                         # the guidance ban at the first sample
    assert torch.equal(allowed, want)
    assert exp0_sft.law_mask(chains[:1], info.vocab_size) is None
    other = tool_grammar.tool_call_law(TOOLS, 'xml', info, info_sha256='0' * 64)
    stale = {**row, 'tool_grammar': other}
    stale.pop('_law_masks', None)
    with pytest.raises(ValueError, match='recorded over tokenizer'):
        exp0_sft.law_mask([{'segs': [(0, len(reply), stale)]}], info.vocab_size)


def test_tokenizer_identity_is_hashed_once_per_info(monkeypatch):
    """The SGLang front asks for the export's tokenizer identity on every chat call; hashing a real vocabulary is ~0.26 s
    of GIL-held Python, which starved a 128-stream collection (aei1, 2026-11-24). One hash per TokenizerInfo object."""
    info, _ = char_info()
    calls = []
    real = tool_grammar._tokenizer_info_sha256
    monkeypatch.setattr(tool_grammar, '_tokenizer_info_sha256', lambda i: calls.append(i) or real(i))
    tool_grammar._INFO_SHA.clear()
    first = tool_grammar.tokenizer_info_sha256(info)
    assert all(tool_grammar.tokenizer_info_sha256(info) == first for _ in range(5)) and len(calls) == 1
    other, _ = char_info()                      # an equal vocabulary in another object: hashed, same identity
    assert tool_grammar.tokenizer_info_sha256(other) == first and len(calls) == 2
    monkeypatch.setenv('NATLANG_TOOL_GRAMMAR_INFO_MEMO', '0')
    assert tool_grammar.tokenizer_info_sha256(info) == first and len(calls) == 3


def test_sglang_front_builds_the_export_tokenizer_info_once(monkeypatch):
    from natlang_neuralese.serve.sglang_actor import SglangEngine
    built = []
    monkeypatch.setattr(tool_grammar, 'info_from_export', lambda d: built.append(d) or ('info', 'sha'))
    engine = SglangEngine.__new__(SglangEngine)
    engine.sglang_export_dir = '/exports/a'
    assert [engine._tool_grammar_info() for _ in range(4)] == [('info', 'sha')] * 4 and built == ['/exports/a']
    engine.sglang_export_dir = '/exports/b'      # a rebind to another export directory rebuilds it
    engine._tool_grammar_info()
    assert built == ['/exports/a', '/exports/b']


def test_concurrent_first_requests_build_the_tokenizer_info_once(monkeypatch):
    """After an actor start every collection stream's first chat call asked for the export's tokenizer info at once, and
    each built it (~3 s of GIL-held tokenizer loading; 32 streams of the a034dbf9c front: 4 requests reached SGLang in
    150 s, 2026-11-24). The first caller builds it while the others wait for it."""
    import threading
    import time
    from natlang_neuralese.serve.sglang_actor import SglangEngine
    built = []

    def slow(directory):
        built.append(directory)
        time.sleep(0.2)
        return ('info', 'sha')
    monkeypatch.setattr(tool_grammar, 'info_from_export', slow)
    engine = SglangEngine.__new__(SglangEngine)
    engine.sglang_export_dir = '/exports/a'
    results = []
    threads = [threading.Thread(target=lambda: results.append(engine._tool_grammar_info())) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [('info', 'sha')] * 16 and built == ['/exports/a']
