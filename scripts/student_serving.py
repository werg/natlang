"""Small, framework-independent helpers for generated assistant responses."""
import math
import hashlib
from pathlib import Path

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


def bounded_output_limit(request, ceiling):
    """Apply a server-side generation ceiling even when a client requests more tokens."""
    if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling < 1:
        raise ValueError("output token ceiling must be a positive integer")
    requested = request.get("max_tokens", request.get("max_completion_tokens", ceiling))
    if not isinstance(requested, int) or isinstance(requested, bool) or requested < 1:
        raise ValueError("requested output token limit must be a positive integer")
    return min(requested, ceiling)


def sampling_options(request):
    """Greedy unless explicitly requested; validate stochastic rewrite controls."""
    temperature = request.get('temperature', 0)
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise ValueError('temperature must be finite and between zero and two')
    seed = request.get('seed')
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)
                             or not 0 <= seed < 2**63):
        raise ValueError('seed must be a nonnegative 63-bit integer')
    options = {'do_sample': temperature > 0}
    if temperature > 0:
        options.update(temperature=temperature, top_k=0, top_p=1.0)
    return options, seed


def checkpoint_file_hashes(directory):
    if not directory:
        return {}
    result = {}
    for path in sorted(Path(directory).resolve().rglob('*')):
        if path.is_file():
            sha = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    sha.update(block)
            result[str(path)] = sha.hexdigest()
    if not result:
        raise ValueError('checkpoint directory contains no files')
    return result


def tokenize_chat_prompt(tokenizer, rendered, **kwargs):
    """The chat template owns BOS/control tokens; never add them again."""
    return tokenizer(rendered, add_special_tokens=False, **kwargs)
