from natlang_neuralese.eval.trajectory_execution import decoded


def response(arguments):
    return {'choices': [{'message': {'tool_calls': [{'function': {'name': 'return_result', 'arguments': arguments}}]}}]}


def test_valid_structured_result():
    assert decoded(response('{"status":"success","value":[{"source_id":"a"}]}')) == (True, [{'source_id': 'a'}])


def test_malformed_or_nonobject_arguments_are_failed_decisions():
    for arguments in ['{', '[]', 'null', '"text"']:
        assert decoded(response(arguments)) == (False, None)


def test_decode_prefix_preserves_merged_value_boundary():
    from natlang_neuralese.serve.engine import decode_prefix_tokens
    def tokenize(text):
        return [text[:-1], '='] if text.endswith('=') else [text.split('=')[0], text[text.index('='):]]
    assert decode_prefix_tokens(tokenize, 'return_result(value=') == ['return_result(value']
