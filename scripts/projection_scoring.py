"""Exact closed-assistant likelihoods for fixed-student projection search.

Proposal probabilities use temperature only: no top-k, top-p or repetition
penalty. The final assistant terminator is part of both probability measures.
"""
import math
import torch
try:
    from .render_training_corpus import _call_template
except ImportError:
    from render_training_corpus import _call_template


def completion_ids(tokenizer, messages, tools, assistant, end_id):
    prompt = _call_template(tokenizer, messages, tools, True)
    closed = _call_template(tokenizer, messages + [{**assistant, 'role': 'assistant'}], tools, False)
    if not closed.startswith(prompt):
        raise ValueError('assistant rendering changed prompt prefix')
    ids = tokenizer.encode(closed[len(prompt):], add_special_tokens=False)
    validate_completion(ids, end_id, len(tokenizer))
    return ids


def validate_completion(ids, end_id, vocab_size):
    if not isinstance(ids, list) or not ids or any(type(i) is not int or not 0 <= i < vocab_size for i in ids):
        raise ValueError('invalid completion token IDs')
    if ids[-1] != end_id or end_id in ids[:-1]:
        raise ValueError('projection needs exactly one final assistant terminator')


def score_completion(model, tokenizer, messages, tools, ids, temperature, max_context, end_id):
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('proposal temperature must be finite and positive')
    validate_completion(ids, end_id, len(tokenizer))
    prompt = _call_template(tokenizer, messages, tools, True)
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    if not prefix or len(prefix) + len(ids) > max_context:
        raise ValueError('scoring context exceeded; no truncation')
    tokens = torch.tensor([prefix + ids], device=model.device)
    # Score bounded chunks of vocabulary logits rather than creating a second
    # full-sequence fp32 logits tensor. LFM's output itself is still bounded by
    # the experiment context cap.
    with torch.inference_mode():
        logits = model(input_ids=tokens, use_cache=False).logits[0, len(prefix)-1:-1]
        target = torch.tensor(ids, device=model.device)
        values = []
        for start in range(0, len(ids), 64):
            chunk = logits[start:start+64].float() / temperature
            selected = chunk.gather(1, target[start:start+64, None]).squeeze(1)
            values.extend((selected - torch.logsumexp(chunk, dim=-1)).cpu().tolist())
    total = math.fsum(values)
    if not math.isfinite(total):
        raise ValueError('nonfinite likelihood')
    return {'sum_logprob': total, 'mean_logprob': total / len(ids),
            'token_count': len(ids), 'token_logprobs': values,
            'temperature': temperature, 'includes_assistant_terminator': True}


def validate_sampling_config(config):
    """Reject hidden generation constraints not represented by temperature scores."""
    forbidden=('forced_bos_token_id','forced_eos_token_id','bad_words_ids','force_words_ids',
               'suppress_tokens','begin_suppress_tokens','sequence_bias','constraints')
    for field in forbidden:
        if getattr(config,field,None):
            raise ValueError('projection cannot score generation constraint: '+field)
    for field,default in [('num_beams',1),('num_beam_groups',1),('min_length',0),
                          ('min_new_tokens',0),('no_repeat_ngram_size',0)]:
        value=getattr(config,field,None)
        if value is not None and value!=default:
            raise ValueError('projection cannot score generation constraint: '+field)
