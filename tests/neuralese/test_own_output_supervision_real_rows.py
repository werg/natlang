"""Own-output objective on real rendered rows of a registered corpus, with the LFM2.5 and LFM3 tokenizers.

Corpus: ``natlang-native-turns-4962-wholetraj-render-20261013-v3`` (training/neuralese_corpora.json), LFM2.5 view.
Role labels from ``chat_roles`` are checked independently against the rendered ChatML template spans
(``<|im_start|>assistant`` regions and ``<think>`` bodies) through the tokenizers' character offsets.
"""
import hashlib
import itertools
import json
import re
from pathlib import Path

import pytest

from natlang_neuralese.common import paths
from natlang_neuralese.data.native_graph import parse_graph
from natlang_neuralese.data.native_graph_conversion import whole_trajectory_text_row_graph
from natlang_neuralese.train import own_output_supervision as own
from natlang_neuralese.train.native_supervision import SCHEMA as POLICY, apply_policy
from natlang_neuralese.train.text_supervision import ROLE_CODES, failed_assistant_mask

CORPUS = paths.resolve('repo', 'runs/natlang-native-turns-4962-wholetraj-render-20261013-v3',
                       'whole-trajectory/lfm25/text-data.jsonl')
LFM25 = Path.home() / ('.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots/'
                       '9e6c6ccf47cd318696e137d381a7ded8fe4df09f')
LFM3 = Path.home() / ('.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/'
                      '40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd')
ROWS = 6
EXPANDED = 'c' * 64
ASSISTANT = {ROLE_CODES.index('assistant_reasoning'), ROLE_CODES.index('assistant_reply')}


def rows():
    if not CORPUS.exists():
        pytest.skip('registered whole-trajectory corpus is not available')
    with CORPUS.open() as handle:
        return [json.loads(line) for line in itertools.islice(handle, ROWS)]


def tokenizer(path):
    if not (path / 'tokenizer.json').exists():
        pytest.skip(f'{path.name} tokenizer is not available')
    from transformers import PreTrainedTokenizerFast
    # The exact tokenizer.json; no repository code is executed.
    tok = PreTrainedTokenizerFast(tokenizer_file=str(path / 'tokenizer.json'))
    tok.chat_template = (path / 'chat_template.jinja').read_text()   # the pinned template, for its generation prompt
    return tok


def token_offsets(tok, ids, text):
    """Character span of each stored token (rows are rendered segment-wise, so re-tokenizing differs).

    Tokens that only decode together (split UTF-8) share their group's span."""
    offsets, position, buffer = [], 0, []
    for token in ids:
        buffer.append(token)
        piece = tok.decode(buffer, skip_special_tokens=False)
        if piece.endswith('\ufffd') or not text.startswith(piece, position):
            continue
        offsets.extend([(position, position + len(piece))] * len(buffer))
        position += len(piece)
        buffer = []
    assert not buffer and position == len(text)
    return offsets


def template_classes(text, offsets):
    """Per-token class from the rendered template text alone: 'context', 'assistant' or 'reasoning'."""
    regions, thinks = [], []
    for match in re.finditer(r'<\|im_start\|>assistant\n', text):
        begin = match.start() + len('<|im_start|>')
        following = text.find('<|im_start|>', match.end())
        regions.append((begin, len(text) if following < 0 else following))
    for match in re.finditer(r'<think>.*?</think>', text, flags=re.S):
        thinks.append((match.start(), match.end()))
    classes = []
    for start, stop in offsets:
        if any(b <= start and stop <= e for b, e in thinks) and any(b <= start < e for b, e in regions):
            classes.append('reasoning')
        elif any(b <= start < e for b, e in regions):
            classes.append('assistant')
        else:
            classes.append('context')
    return classes


def check_row(source, tok, fingerprint, share=.1, *, full_header=True):
    ids = source['token_ids']
    assert tok.decode(ids, skip_special_tokens=False) == source['text']
    tokens = own.role_tokens_from_tokenizer(tok, tokenizer_fingerprint=EXPANDED)
    objective = own.resolve({'name': own.NAME, 'reasoning_share': share, 'role_tokens': tokens})
    graph = whole_trajectory_text_row_graph(source, source_tokenizer_fingerprint=fingerprint, tokenizer=tok,
                                            expanded_tokenizer_fingerprint=EXPANDED, objective=objective)
    weights = graph['nodes'][0]['target_weights']
    labels = own.label_roles(ids, tokens)
    # Role labels agree with the rendered template spans, token by token.
    classes = template_classes(source['text'], token_offsets(tok, ids, source['text']))
    for label, cls in zip(labels, classes):
        if label == ROLE_CODES.index('assistant_reasoning'):
            assert cls == 'reasoning'
        elif label == ROLE_CODES.index('assistant_reply'):
            assert cls == 'assistant'
        else:
            assert cls == 'context'
    # Zero context, failed turns and runtime-opening turns excluded, per-decision reasoning share.
    assert all(w == 0 for w, label in zip(weights, labels) if label not in ASSISTANT)
    failed = failed_assistant_mask(ids, labels, lambda values: tok.decode(values, skip_special_tokens=False))
    assert all(w == 0 for w, f in zip(weights, failed) if f)
    runs = own.assistant_runs(labels)
    for begin, end in runs[:source.get('runtime_opening_assistant_turns', 0)]:
        assert sum(weights[begin:end]) == 0
    # Template-forced generation prompt tokens are prompt; <|im_end|> is an emitted target; template text after it
    # is not. With the template's own rendering every header must match the generation prompt exactly.
    prompt, report = tokens['generation_prompt'], {}
    own.generated_eligibility(labels, ids, [True] * len(ids), tokens, report=report)
    assert full_header is False or report == {}
    targeted_turns = 0
    for begin, end in runs:
        if not sum(weights[begin:end]):
            continue
        targeted_turns += 1
        forced = 2
        while forced < len(prompt) and ids[begin - 1 + forced] == prompt[forced]:
            forced += 1
        assert ids[begin - 1:begin - 1 + forced] == prompt[:forced] and forced >= 2
        assert all(weights[i] == 0 for i in range(begin, begin - 1 + forced))
        closing = [i for i in range(begin, end) if ids[i] == tokens['end']]
        assert closing and weights[closing[0]] > 0
        assert all(weights[i] == 0 for i in range(closing[0] + 1, end))
    assert targeted_turns
    think = {tokens['think_open'], tokens['think_close']}
    masses = []
    for begin, end in runs:
        mass = sum(weights[begin:end])
        if mass:
            reasoning = sum(weights[i] for i in range(begin, end)
                            if labels[i] == ROLE_CODES.index('assistant_reasoning') and ids[i] not in think)
            has_reasoning = any(labels[i] == ROLE_CODES.index('assistant_reasoning') and ids[i] not in think
                                for i in range(begin, end))
            assert reasoning == pytest.approx(share * mass if has_reasoning else 0.)
            masses.append(mass)
    assert masses and sum(weights) == pytest.approx(1.)
    assert max(masses) == pytest.approx(min(masses))
    # The load-time objective on the historical (legacy-weighted) conversion gives identical weights.
    legacy = whole_trajectory_text_row_graph(source, source_tokenizer_fingerprint=fingerprint, tokenizer=tok,
                                             expanded_tokenizer_fingerprint=EXPANDED)
    loaded = apply_policy(parse_graph(legacy, tokenizer_fingerprint=EXPANDED), legacy, source_sha256='a' * 64,
                          policy={'schema': POLICY, 'sources': [], 'objective': objective})
    assert loaded.target_weights == pytest.approx(tuple(weights), abs=1e-12)
    reply = tok.decode([t for t, c in zip(ids, loaded.objective_categories) if c == own.REPLY],
                       skip_special_tokens=False)
    return graph, reply


def test_real_lfm25_rows_follow_the_template_and_the_objective():
    tok = tokenizer(LFM25)
    seen = 0
    for source in rows():
        check_row(source, tok, source['tokenizer_sha256'])
        seen += 1
    assert seen == ROWS


def test_real_rows_rerendered_with_the_lfm3_tokenizer_supervise_the_same_actions():
    tok25, tok3 = tokenizer(LFM25), tokenizer(LFM3)
    fingerprint = hashlib.sha256((LFM3 / 'tokenizer.json').read_bytes()).hexdigest()
    for source in rows():
        _, reply25 = check_row(source, tok25, source['tokenizer_sha256'])
        text = source['text'].removeprefix('<|startoftext|>')
        ids = tok3(text, add_special_tokens=False)['input_ids']
        lfm3 = {**source, 'text': text, 'token_ids': ids, 'tokenizer_sha256': fingerprint}
        # LFM2.5-rendered text lacks LFM3's forced <think> on turns without reasoning, so headers may be partial.
        _, reply3 = check_row(lfm3, tok3, fingerprint, full_header=False)
        # The same supervised action text under both tokenizers, except LFM3's template-forced opening <think>.
        assert reply3.replace('<think>', '') == reply25.replace('<think>', '')
