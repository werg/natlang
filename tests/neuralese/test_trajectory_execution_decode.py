from natlang_neuralese.eval.trajectory_execution import decoded


def response(arguments):
    return {'choices': [{'message': {'tool_calls': [{'function': {'name': 'return_result', 'arguments': arguments}}]}}]}


def test_valid_structured_result():
    assert decoded(response('{"status":"success","value":[{"source_id":"a"}]}')) == (True, [{'source_id': 'a'}])


def test_malformed_or_nonobject_arguments_are_failed_decisions():
    for arguments in ['{', '[]', 'null', '"text"']:
        assert decoded(response(arguments)) == (False, None)
