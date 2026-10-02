"""Small, framework-independent helpers for generated assistant responses."""

try:
    from .render_training_corpus import _assistant_end_token
except ImportError:
    from render_training_corpus import _assistant_end_token


def assistant_end_token_id(tokenizer):
    """Return the single token emitted at the end of a closed assistant turn."""
    token = _assistant_end_token(tokenizer)
    encoder = getattr(tokenizer, "encode", None)
    decoder = getattr(tokenizer, "decode", None)
    if not callable(encoder) or not callable(decoder):
        raise ValueError("tokenizer must encode and decode the assistant end token")
    ids = encoder(token, add_special_tokens=False)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if not isinstance(ids, (list, tuple)) or len(ids) != 1:
        raise ValueError("assistant end token must encode to exactly one token ID")
    token_id = int(ids[0])
    if decoder([token_id], skip_special_tokens=False) != token:
        raise ValueError("assistant end token ID does not decode back to the template token")
    return token_id


def strip_final_assistant_terminator(token_ids, end_token_id):
    """Strip only a final assistant terminator; preserve all other special tokens."""
    if len(token_ids) and int(token_ids[-1]) == int(end_token_id):
        return token_ids[:-1], True
    return token_ids, False


def response_finish_reason(*, terminated, token_count, output_limit, has_tool_calls):
    """Use generated length before terminator trimming to classify completion."""
    if not terminated and token_count >= output_limit:
        return "length"
    return "tool_calls" if has_tool_calls else "stop"
