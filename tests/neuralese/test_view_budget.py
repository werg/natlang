import json
import threading
import urllib.request
from concurrent.futures import Future

import pytest

from natlang_neuralese.view import ViewBudget, ViewBudgetExhausted
from natlang_neuralese.serve.http import serve


def test_budget_counts_actual_usage_and_bounds_every_write():
    budget = ViewBudget({'max_requests': 2, 'max_tokens': 5})
    assert budget.begin(4) == 4
    budget.finish({'usage': {'completion_tokens': 3}}, 4)
    assert budget.begin(4) == 2
    budget.finish({'usage': {'completion_tokens': 2}}, 2)
    with pytest.raises(ViewBudgetExhausted):
        budget.begin(4)
    assert budget.usage() == {'requests': 2, 'tokens': 5}


@pytest.mark.parametrize('bad', [None, {}, {'max_requests': True, 'max_tokens': 1},
                                {'max_requests': -1, 'max_tokens': None}])
def test_budget_rejects_malformed_allowance(bad):
    with pytest.raises(ValueError):
        ViewBudget(bad)


@pytest.mark.parametrize(('max_requests', 'max_tokens', 'expected_limit', 'engine_failure'),
                         [(1, 7, 7, False), (3, 2, 2, False), (3, 7, 7, True)])
def test_http_partial_view_retains_real_turns_and_exact_budget_usage(max_requests, max_tokens, expected_limit, engine_failure):
    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {'input_ids': text.split()}
        def decode(self, ids):
            return ' '.join(ids)
    class Store:
        def hold(self, owner, ids):
            pass
    class Engine:
        context, max_block, tokenizer, store = 1024, 8, Tokenizer(), Store()
        requests = []
        def prompt_positions(self, messages, tools):
            return 4
        def submit(self, request):
            self.requests.append(request)
            result = Future()
            if engine_failure and len(self.requests) == 2:
                result.set_exception(RuntimeError('second chunk failed'))
                return result
            result.set_result({'usage': {'completion_tokens': 2},
                'x_natlang_rl_capture': {'schema': 'natlang.native-rl-sampling/2'},
                'native_state': {'blocks': [{'id': 'nzs1_' + 'a' * 52}]}})
            return result
    engine = Engine()
    server = serve(engine, '127.0.0.1', 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = {'value': 'a b c', 'window': 1, 'budget': {'max_requests': max_requests, 'max_tokens': max_tokens}}
        request = urllib.request.Request(f'http://127.0.0.1:{server.server_address[1]}/v1/neuralese/view',
            data=json.dumps(body).encode(), headers={'content-type': 'application/json'})
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
        if engine_failure:
            assert result['budget_error'] == 'second chunk failed'
            assert 'budget_usage' not in result  # the failed request's token work is unknown
        else:
            assert result['budget_exhausted'] is True
            assert result['budget_usage'] == {'requests': 1, 'tokens': 2}
        assert len(result['source_turns']) == 1
        assert engine.requests[0].max_tokens == expected_limit
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
