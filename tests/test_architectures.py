"""Architecture examples run through native tools, grammar and independent oracles."""
import random

import pytest

from natlang import js
from natlang.gen.architectures import ARCHITECTURES
from scripts.generate import run_program


@pytest.mark.parametrize('family', sorted(ARCHITECTURES))
def test_architecture_generated_programs_match_state_and_effect_oracles(family):
    for seed in range(8):
        samples, episodes = run_program(ARCHITECTURES[family](random.Random(seed)))
        assert samples and episodes >= 1
        assert all(s['messages'][0]['role'] == 'system' for s in samples)


def test_sandbox_scope_preserves_reserved_dictionary_keys():
    scope = {'args': {'orders': {'__proto__': 'paid', 'constructor': 'reserved'}}}
    result = js.run('return args.orders;', scope, None, body=True, path='test')
    assert result == scope['args']['orders']
    assert js.run('return Object.keys(args.orders).sort();', scope, None, body=True, path='test') == ['__proto__', 'constructor']


def test_saga_training_contains_resume_after_lost_acknowledgement():
    samples, _ = run_program(ARCHITECTURES['cb_order_saga'](random.Random(0)))
    resumes = [s for s in samples for c in s['target'].get('tool_calls') or []
               if c['function']['name'] == 'call' and '"function": "dispatch"' in c['function']['arguments']
               and '"inputs"' not in c['function']['arguments']]
    assert resumes
    assert any(any(m['role'] == 'tool' and 'quiesced' in m['content'] for m in s['messages']) for s in samples)
