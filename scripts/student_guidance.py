"""HF backtracking adapter for the shared Natlang Guide (ordinary decoding only).

HF owns cached forward passes between rejections. On rollback we rebuild the
cache from the retained prefix; this also works with mutable hybrid/SSM caches.
It does not implement a scored MH proposal distribution.
"""
from pathlib import Path
import importlib.util
import sys

import torch
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList


def load_guide(path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[1] / 'training/neuralese/natlang_neuralese/serve/guidance.py'
    spec = importlib.util.spec_from_file_location('natlang_shared_student_guide', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PrefixBans(LogitsProcessor):
    def __init__(self, prompt_length, bans):
        self.prompt_length, self.bans = prompt_length, bans

    def __call__(self, input_ids, scores):
        prefix = tuple(input_ids[0, self.prompt_length:].tolist())
        denied = self.bans.get(prefix, ())
        if denied:
            scores[0, list(denied)] = -torch.inf
        return scores


class CheckReply(StoppingCriteria):
    def __init__(self, tokenizer, guide, prompt_length, eos):
        self.tokenizer, self.guide, self.prompt_length, self.eos = tokenizer, guide, prompt_length, eos
        self.verdict = None

    def __call__(self, input_ids, scores, **kwargs):
        ids = input_ids[0, self.prompt_length:].tolist()
        if ids and ids[-1] == self.eos:
            ids = ids[:-1]
        reply = self.tokenizer.decode(ids, skip_special_tokens=False)
        self.verdict = self.guide.check(reply)
        return torch.tensor([self.verdict is not None], device=input_ids.device)


def rollback_token(tokenizer, ids, offset):
    """Find the first token touching the rejected character, using whole prefixes.

    Independent token decoding is unsafe for Unicode byte-fragment tokens.
    """
    for index in range(len(ids)):
        text = tokenizer.decode(ids[:index + 1], skip_special_tokens=False)
        if len(text) > offset:
            return index
    return len(ids)


def guided_generate(model, tokenizer, inputs, settings, guide_module, *, output_limit,
                    sampling, eos_token_id, pad_token_id, disconnect, attempt_multiplier=3):
    if output_limit < 1 or not 1 <= attempt_multiplier <= 8:
        raise ValueError('invalid guidance resource budget')
    if settings.retries < 0 or settings.retries > 16 or min(settings.repeat, settings.run) < 0:
        raise ValueError('invalid guidance settings')
    if settings.syntax:
        guide_module.syntax_errors('')  # fail explicitly before generation if parser unavailable
    guide = guide_module.Guide(settings)
    prompt = inputs.input_ids
    prompt_length = int(prompt.shape[1])
    forced = tokenizer.encode(guide.forced_prefix(), add_special_tokens=False)
    if len(forced) >= output_limit:
        raise ValueError('output allowance cannot fit guidance prefix and completion')
    ids, bans, retries = list(forced), {}, {}
    attempts = 0
    prefill_tokens = 0
    budget = output_limit * attempt_multiplier
    rejections = []
    budget_exhausted = False
    while len(ids) < output_limit and not disconnect.poll():
        allowance = min(output_limit - len(ids), budget - attempts)
        if allowance <= 0:
            budget_exhausted = True
            break
        suffix = torch.tensor([ids], dtype=prompt.dtype, device=prompt.device)
        prefix = torch.cat((prompt, suffix), dim=1)
        attention = torch.cat((inputs.attention_mask, torch.ones_like(suffix)), dim=1)
        check = CheckReply(tokenizer, guide, prompt_length, eos_token_id)
        prefill_tokens += int(prefix.shape[1])
        output = model.generate(input_ids=prefix, attention_mask=attention,
            max_new_tokens=allowance, **sampling, repetition_penalty=1.0,
            pad_token_id=pad_token_id, eos_token_id=eos_token_id,
            logits_processor=LogitsProcessorList([PrefixBans(prompt_length, bans)]),
            stopping_criteria=StoppingCriteriaList([disconnect, check]))
        new_ids = output[0, prompt_length:].tolist()
        attempts += len(new_ids) - len(ids)
        ids = new_ids
        if disconnect.disconnected:
            break
        if check.verdict is None:
            if ids and ids[-1] == eos_token_id:
                break
            continue
        reason, offset = check.verdict
        target = rollback_token(tokenizer, ids, offset)
        key = tuple(ids[:target])
        tries = retries.get(key, 0)
        accepted = tries >= settings.retries or target < len(forced) or target >= len(ids)
        rejections.append({'reason': reason, 'offset': offset, 'token_index': target,
                           'accepted': accepted, 'retry': tries, 'attempted_tokens': attempts})
        if accepted:
            guide.accept(offset)
            if ids[-1] == eos_token_id:
                break
            continue
        retries[key] = tries + 1
        bans.setdefault(key, set()).add(ids[target])
        ids = ids[:target]
        guide.rewind(len(tokenizer.decode(ids, skip_special_tokens=False)))
    if attempts >= budget and (not ids or ids[-1] != eos_token_id):
        budget_exhausted = True
    return torch.cat((prompt, torch.tensor([ids], dtype=prompt.dtype, device=prompt.device)), dim=1), {
        'method': 'shared-guide-hf-backtracking/1', 'forced_prefix_tokens': len(forced),
        'attempted_tokens': attempts, 'attempt_token_budget': budget,
        'prefill_tokens_processed': prefill_tokens,
        'discarded_generated_tokens': attempts + len(forced) - len(ids),
        'budget_exhausted': budget_exhausted, 'rejections': rejections,
        'mh_proposal_supported': False,
    }
