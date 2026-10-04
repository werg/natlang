import pytest

from scripts import student_serving
from scripts.model_response import parse_response


class FakeTokenizer:
    pad_token_id = 2

    def __init__(self, encoded, decoded):
        self.encoded = encoded
        self.decoded = decoded

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return self.encoded[text]

    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        return self.decoded[ids[0]]


def test_assistant_terminator_comes_from_closed_template_and_must_be_one_token(monkeypatch):
    monkeypatch.setattr(student_serving, "_assistant_end_token", lambda tokenizer: "<|im_end|>")
    tokenizer = FakeTokenizer({"<|im_end|>": [130073]}, {130073: "<|im_end|>"})
    assert student_serving.assistant_end_token_id(tokenizer) == 130073

    tokenizer = FakeTokenizer({"<|im_end|>": [13, 73]}, {13: "<", 73: "|im_end|>"})
    with pytest.raises(ValueError, match="exactly one token ID"):
        student_serving.assistant_end_token_id(tokenizer)


def test_only_final_assistant_terminator_is_removed_before_tool_parsing(monkeypatch):
    monkeypatch.setattr(student_serving, "_assistant_end_token", lambda tokenizer: "<|im_end|>")
    token_text = {
        1: "<think>Use the tool.</think>",
        2: '<|tool_call_start|>[eval(code="return 7;")]<|tool_call_end|>',
        130073: "<|im_end|>",
    }

    class Tokenizer(FakeTokenizer):
        def decode(self, ids, skip_special_tokens=False):
            assert skip_special_tokens is False
            return "".join(token_text[token_id] for token_id in ids)

    tokenizer = Tokenizer({"<|im_end|>": [130073]}, token_text)
    end_id = student_serving.assistant_end_token_id(tokenizer)
    reply, terminated = student_serving.strip_final_assistant_terminator([1, 2, end_id], end_id)
    decoded = tokenizer.decode(reply, skip_special_tokens=False)
    parsed = parse_response(decoded)
    assert terminated is True
    assert parsed["reasoning_content"] == "Use the tool."
    assert parsed["content"] == ""
    assert parsed["tool_calls"][0]["function"]["name"] == "eval"
    assert parsed["tool_calls"][0]["function"]["arguments"] == '{"code": "return 7;"}'
    assert student_serving.response_finish_reason(
        terminated=terminated, token_count=3, output_limit=3, has_tool_calls=True
    ) == "tool_calls"


def test_nonterminal_output_at_limit_keeps_length_finish_reason():
    tokens, terminated = student_serving.strip_final_assistant_terminator([4, 5], 130073)
    assert tokens == [4, 5]
    assert terminated is False
    assert student_serving.response_finish_reason(
        terminated=terminated, token_count=2, output_limit=2, has_tool_calls=False
    ) == "length"


def test_terminal_output_finish_reason_distinguishes_tool_calls_from_text():
    assert student_serving.response_finish_reason(
        terminated=True, token_count=2, output_limit=2, has_tool_calls=False
    ) == "stop"
    assert student_serving.response_finish_reason(
        terminated=False, token_count=1, output_limit=5, has_tool_calls=True
    ) == "tool_calls"


def test_server_output_ceiling_bounds_client_request_and_validates_inputs():
    assert student_serving.bounded_output_limit({"max_tokens": 2048}, 512) == 512
    assert student_serving.bounded_output_limit({"max_tokens": 128}, 512) == 128
    assert student_serving.bounded_output_limit({}, 512) == 512
    with pytest.raises(ValueError, match="requested output token limit"):
        student_serving.bounded_output_limit({"max_tokens": 0}, 512)
    with pytest.raises(ValueError, match="output token ceiling"):
        student_serving.bounded_output_limit({}, 0)


def test_rendered_chat_prompt_does_not_add_a_second_bos():
    from scripts.student_serving import tokenize_chat_prompt
    class Tokenizer:
        def __call__(self, text, *, add_special_tokens, return_tensors):
            assert text.startswith("<bos>")
            assert return_tensors == "pt"
            return [1, 1, 2] if add_special_tokens else [1, 2]
    assert tokenize_chat_prompt(Tokenizer(), "<bos>prompt", return_tensors="pt") == [1, 2]
