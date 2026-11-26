"""A request's `chat_template_kwargs` on the Python engine and the SGLang actor front: a learned operator's call renders
with `enable_thinking: false` (its prompt ends in `<think></think>`, the operator rows' form), unknown switches are
refused, and the in-context capture of such a reply starts at its first token."""
import itertools
import queue
from pathlib import Path
from types import SimpleNamespace

import pytest

from natlang_neuralese.serve.chat import RequestError, bind_history_reasoning
from natlang_neuralese.serve.engine import Engine, GenerationRequest

LFM3 = Path.home() / '.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd'
SITE = [{'role': 'system', 'content': 'You compact values.'}, {'role': 'user', 'content': 'value:\n"a long value"'}]
OFF = {'enable_thinking': False}


@pytest.fixture(scope='module')
def tokenizer():
    if not (LFM3 / 'tokenizer.json').is_file():
        pytest.skip('LFM3-1.7B snapshot not cached')
    from transformers import AutoTokenizer
    return bind_history_reasoning(AutoTokenizer.from_pretrained(str(LFM3), local_files_only=True,
                                                                trust_remote_code=False), model_type='lfm3')


def engine_for(tokenizer, cls=Engine):
    engine = object.__new__(cls)
    engine.tokenizer, engine.prompt_rendering, engine.tool_call_format = tokenizer, 'full', 'xml'
    return engine


def test_enable_thinking_false_ends_the_prompt_in_an_empty_think_block(tokenizer):
    engine = engine_for(tokenizer)
    on = ''.join(engine.render_request(SITE, []).segments)
    off = ''.join(engine.render_request(SITE, [], template_kwargs=OFF).segments)
    assert on.endswith('<|im_start|>assistant\n<think>') and off == on + '</think>'
    # token level: the operator rows' prompt (ending `<think>`) and their first, zero-weight reply token `</think>`
    think_open, think_close = tokenizer.convert_tokens_to_ids(['<think>', '</think>'])
    plain, closed = engine._prompt_plan(SITE, []), engine._prompt_plan(SITE, [], OFF)
    assert closed[-1][1] == plain[-1][1] + [think_close] and plain[-1][1][-1] == think_open
    with pytest.raises(ValueError):
        engine.render_request(SITE, [], template=engine._template, template_kwargs=OFF)


def test_template_kwargs_refuse_unknown_and_contrary_switches(tokenizer):
    engine = engine_for(tokenizer)
    assert engine.template_kwargs(None) is None and engine.template_kwargs({}) is None
    assert engine.template_kwargs(OFF) == OFF
    # LFM3's history-reasoning policy pins preserve_thinking=True: the same value passes, the contrary one is refused
    assert engine.template_kwargs({'preserve_thinking': True, 'enable_thinking': True}) == {
        'preserve_thinking': True, 'enable_thinking': True}
    for bad, code in (({'reasoning_effort': 'low'}, 'unsupported-request-field'), ({'enable_thinking': 'no'}, 'bad-request'),
                      ({'preserve_thinking': False}, 'bad-request'), (['enable_thinking'], 'bad-request')):
        with pytest.raises(RequestError) as error:
            engine.template_kwargs(bad)
        assert error.value.code == code


def test_engine_and_sglang_front_render_the_request_with_its_switches(tokenizer):
    think_close = tokenizer.convert_tokens_to_ids('</think>')
    engine = engine_for(tokenizer)
    engine._incoming, engine._ids, engine.actor_export_default = queue.Queue(), itertools.count(), None
    engine.submit(GenerationRequest(messages=SITE, chat_template_kwargs=OFF))
    assert engine._incoming.get_nowait().prompt_plan[-1][1][-1] == think_close
    from natlang_neuralese.serve.sglang_actor import SglangEngine
    front = engine_for(tokenizer, SglangEngine)
    front._ids = itertools.count()
    front._routes_native = lambda request, plan: False
    front._append_only = lambda request, plan: plan
    front._spawn = lambda run: run()
    front._sglang_run = lambda sequence: sequence.prompt_plan
    plan = front.submit(GenerationRequest(messages=SITE, chat_template_kwargs=OFF)).result()
    assert plan[-1][1][-1] == think_close
    assert front.submit(GenerationRequest(messages=SITE)).result()[-1][1][-1] != think_close


def test_capture_of_a_reply_after_an_empty_think_block_starts_at_its_first_token(tokenizer):
    engine = engine_for(tokenizer)
    roles = SimpleNamespace(ids=(10 ** 6, 10 ** 6 + 1, 10 ** 6 + 2, 10 ** 6 + 3), produce_open=10 ** 6)
    engine.role_tokens, engine.context = roles, 1 << 17
    engine.stop_ids = {tokenizer.convert_tokens_to_ids('<|im_end|>')}
    prompt = engine._prompt_plan(SITE, [], OFF)[-1][1]
    content = tokenizer('Ten lines, all "line of data".', add_special_tokens=False)['input_ids']
    prefix, span = engine.capture_plan(prompt, content + sorted(engine.stop_ids))
    assert span == content and prefix == [*prompt, roles.produce_open]
