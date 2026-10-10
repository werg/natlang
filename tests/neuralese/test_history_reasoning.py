"""History reasoning is a declared per-backbone policy, asserted against each template, shared by training and serving."""
from glob import glob
from pathlib import Path

import pytest

from natlang_neuralese.serve.chat import (
    BACKBONE_HISTORY_REASONING, bind_history_reasoning, history_reasoning_kwargs, probe_history_reasoning,
)

LFM = sorted(glob(str(Path.home() / '.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots/*')))
MELLUM = Path('/home/werg/data/models/mellum21-12b-a2.5b-thinking')
HISTORY = [{'role': 'user', 'content': 'task'},
           {'role': 'assistant', 'reasoning_content': 'EARLIER-THOUGHT', 'content': 'step'},
           {'role': 'tool', 'content': 'result'},
           {'role': 'assistant', 'reasoning_content': 'LATEST-THOUGHT', 'content': 'done'}]


class Template:
    """Keeps only the latest assistant reasoning unless told `preserve_thinking` (as LFM2.5's template)."""
    def __init__(self, chat_template):
        self.chat_template = chat_template

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, **kwargs):
        keep = kwargs.get('preserve_thinking', False)
        last = max(i for i, m in enumerate(messages) if m['role'] == 'assistant')
        return ''.join((f"<think>{m['reasoning_content']}</think>" if m.get('reasoning_content') and (keep or i == last)
                        else '') + m['content'] for i, m in enumerate(messages))


def test_declarations_and_switches():
    assert BACKBONE_HISTORY_REASONING == {'lfm2': 'last_turn_only', 'mellum': 'keep'}
    switch = 'set preserve_thinking = preserve_thinking | default(false)'
    assert history_reasoning_kwargs(switch, 'keep') == {'preserve_thinking': True}
    assert history_reasoning_kwargs(switch, 'last_turn_only') == {'preserve_thinking': False}
    assert history_reasoning_kwargs('{{ messages }}', 'keep') == {}
    with pytest.raises(ValueError):
        history_reasoning_kwargs('', 'drop')


def test_bound_policy_is_pinned_and_asserted():
    tokenizer = bind_history_reasoning(Template('preserve_thinking'), model_type='lfm2')
    assert tokenizer.natlang_history_reasoning == {'policy': 'last_turn_only', 'backbone': 'lfm2',
                                                   'template_kwargs': {'preserve_thinking': False}, 'probe': 'drops-history'}
    text = tokenizer.apply_chat_template(HISTORY, tokenize=False)
    assert 'EARLIER-THOUGHT' not in text and 'LATEST-THOUGHT' in text
    with pytest.raises(ValueError):
        tokenizer.apply_chat_template(HISTORY, tokenize=False, preserve_thinking=True)
    assert bind_history_reasoning(tokenizer) is tokenizer
    with pytest.raises(ValueError):
        bind_history_reasoning(tokenizer, 'keep')
    kept = bind_history_reasoning(Template('preserve_thinking'), 'keep')
    assert 'EARLIER-THOUGHT' in kept.apply_chat_template(HISTORY, tokenize=False)


def test_template_contradicting_its_declaration_fails_loudly():
    with pytest.raises(ValueError, match='not as its declared'):
        bind_history_reasoning(Template('no switch'), model_type='mellum')
    with pytest.raises(ValueError, match='declares no history_reasoning'):
        bind_history_reasoning(Template('no switch'), model_type='other', require_declared=True)
    undeclared = bind_history_reasoning(Template('no switch'), model_type='other')
    assert undeclared.natlang_history_reasoning['policy'] == 'undeclared'


@pytest.mark.skipif(not LFM, reason='LFM2.5-350M tokenizer snapshot not cached')
def test_lfm_renders_last_turn_reasoning_only():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(LFM[-1], local_files_only=True)
    assert probe_history_reasoning(lambda m: tokenizer.apply_chat_template(m, tokenize=False)) == 'drops-history'
    bind_history_reasoning(tokenizer, require_declared=True)
    assert tokenizer.natlang_history_reasoning == {'policy': 'last_turn_only', 'backbone': 'lfm2',
                                                   'template_kwargs': {'preserve_thinking': False}, 'probe': 'drops-history'}


@pytest.mark.skipif(not (MELLUM / 'tokenizer.json').is_file(), reason='Mellum tokenizer not present')
def test_mellum_keeps_history_reasoning():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MELLUM), local_files_only=True)
    bind_history_reasoning(tokenizer, require_declared=True)
    assert tokenizer.natlang_history_reasoning == {'policy': 'keep', 'backbone': 'mellum',
                                                   'template_kwargs': {}, 'probe': 'keep'}
    assert 'EARLIER-THOUGHT' in tokenizer.apply_chat_template(HISTORY, tokenize=False)


def test_recipe_twins_declare_their_backbone_policy():
    import json
    recipe = json.loads((Path(__file__).parents[2] / 'training/neuralese/recipes/raw-recurrence-v3.json').read_text())
    twins = recipe['overrides']['cohorts']['harness_bench']['twins']
    backbone = {'lfm2.5-350m': 'lfm2', 'mellum2.1-12b-a2.5b-thinking': 'mellum'}
    assert {name: twin['history_reasoning'] for name, twin in twins.items()} == \
        {name: BACKBONE_HISTORY_REASONING[backbone[name]] for name in twins}
