"""Learned-operator cases: the shared pieces of turning a case into training rows (scripts/neuralese_data/operator_rows.py).

The operator site and the blocks a consumer sees are the runtime's own rendering (ts-host native/operator-site.ts,
called through ts-host/scripts/operator-sites.mjs); this module only tokenizes messages with the serving chat template
(``TokenRenderer``, native-state placeholders spliced as token runs) and picks each source's seed: its supplied
gold/checked span (``sources[i].meta.operator_seed``) or the shared deterministic mechanical seed
(data/operator_seeds.py), clipped to the operator's span cap.
"""
from __future__ import annotations

import base64
import hashlib
import json

from ..model.learned_operators import operator
from .operator_seeds import seed_for

IMITATION_LABELS = {'gold', 'checked'}


class OperatorCaseError(ValueError):
    """A case that cannot become operator rows; the message is its omission reason."""


def _placeholder(tag: str) -> str:
    digest = hashlib.sha256(tag.encode()).digest()
    return 'nzs1_' + base64.b32encode(digest).decode('ascii').lower().rstrip('=')


def source_text(source: dict) -> str:
    if isinstance(source.get('text'), str):
        return source['text']
    if isinstance(source.get('messages'), list):
        return json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    raise OperatorCaseError('source-without-text')


class TokenRenderer:
    """Token IDs of serving messages under one expanded tokenizer, and the seeds of operator sources."""

    def __init__(self, tokenizer, *, role_ids):
        from ..serve.chat import bind_history_reasoning
        self.tokenizer, self.role_ids = tokenizer, tuple(role_ids)
        bound = getattr(tokenizer, 'natlang_history_reasoning', {})
        bind_history_reasoning(tokenizer, policy=bound.get('policy'))
        self.specials = tuple(tokenizer.all_special_tokens)

    def _ids(self, text, escaped=False):
        ids = self.tokenizer(text, add_special_tokens=False, split_special_tokens=escaped)['input_ids']
        return list(ids[0] if ids and isinstance(ids[0], list) else ids)

    def _render(self, messages, tools, splices, *, generation):
        """Token IDs of messages whose native-state placeholder parts are replaced by ``splices[id]``."""
        from ..serve.chat import render_messages, split_escaped
        rendered = render_messages(messages, tools, lambda turns, schemas: self.tokenizer.apply_chat_template(
            turns, tools=schemas or None, tokenize=False, add_generation_prompt=generation), specials=self.specials)
        if sorted(rendered.blocks) != sorted(splices) or len(set(rendered.blocks)) != len(rendered.blocks):
            raise OperatorCaseError('render-placeholders: the template dropped or duplicated a placeholder')
        ids, spans = [], {}
        for segment in rendered.segments:
            if isinstance(segment, int):
                block = rendered.blocks[segment]
                spans[block] = (len(ids), len(ids) + len(splices[block]))
                ids.extend(splices[block])
            else:
                for run, escaped in split_escaped(segment, rendered.escape_nonce):
                    ids.extend(self._ids(run, escaped))
        return ids, spans

    def seed(self, name, record, source):
        """(seed token IDs, seed record) of one source: its supplied seed, else the shared mechanical seed."""
        spec = operator(name)
        seed = seed_for(name, record, source)
        ids = [token for token in self._ids(seed['text'], escaped=True) if token not in self.role_ids]
        if not ids:
            raise OperatorCaseError('seed-empty')
        clipped = len(ids) > spec.seed_cap
        return ids[:spec.seed_cap], {'provenance': seed['provenance'], 'origin': seed.get('origin'),
                                     'tokens': min(len(ids), spec.seed_cap), 'clipped': clipped,
                                     'text_sha256': hashlib.sha256(seed['text'].encode()).hexdigest()}
