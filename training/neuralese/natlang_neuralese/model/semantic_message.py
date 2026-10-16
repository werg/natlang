"""Compact learned dialect using the existing native artifact lifecycle."""
from dataclasses import dataclass
import re
import torch

from .state_block import NativeStateBlock

SCHEMA = 'neuralese.semantic-state/1'
OFFSET_POLICIES = {'exact-tokenizer-offsets', 'actual-token-decoder-alignment',
                   'actual-token-decoder-text-alignment'}


def retained_source_offsets(tokenizer, text, ids):
    """Retain sampled IDs, including byte fragments and noncanonical BPE splits."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded['input_ids']) == list(ids):
        return [list(pair) for pair in encoded['offset_mapping']], 'exact-tokenizer-offsets'
    from difflib import SequenceMatcher
    def decode(values):
        return tokenizer.decode(values, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    decoded = decode(ids)
    boundaries = [0]
    for stop in range(1, len(ids)+1):
        prefix = decode(ids[:stop])
        common = 0
        for left, right in zip(prefix, decoded):
            if left != right:
                break
            common += 1
        boundaries.append(max(boundaries[-1], common))
    boundaries[-1] = len(decoded)
    mapping = list(range(len(decoded)+1))
    policy = 'actual-token-decoder-alignment'
    if decoded != text:
        policy = 'actual-token-decoder-text-alignment'
        for _, a, b, c, d in SequenceMatcher(None, decoded, text, autojunk=False).get_opcodes():
            if a == b:
                mapping[a] = d
            else:
                for index in range(a, b+1):
                    mapping[index] = c + (index-a)*(d-c)//(b-a)
    return [[mapping[a], mapping[b]] for a,b in zip(boundaries, boundaries[1:])], policy


@dataclass
class SemanticStateMessage(NativeStateBlock):
    code: torch.Tensor
    offsets: list
    source_model: str
    adapter_sha256: str
    offset_policy: str = 'exact-tokenizer-offsets'

    def validate(self):
        count = self._validate_token_fields()
        if self.layers:
            raise ValueError('semantic messages carry codes, not raw backbone layer state')
        self._float_tensor('code', self.code, count, 2)
        if not isinstance(self.source_model, str) or not self.source_model:
            raise ValueError('semantic message requires its adapter source model')
        if not isinstance(self.adapter_sha256, str) or not re.fullmatch('[0-9a-f]{64}', self.adapter_sha256):
            raise ValueError('semantic message requires its exact adapter SHA-256')
        if self.offset_policy not in OFFSET_POLICIES:
            raise ValueError('semantic message requires an explicit retained source offset policy')
        if (not isinstance(self.offsets, list) or len(self.offsets) != count
                or any(not isinstance(pair, (list, tuple)) or len(pair) != 2
                       or any(type(x) is not int for x in pair)
                       or not 0 <= pair[0] <= pair[1] <= len(self.text) for pair in self.offsets)):
            raise ValueError('semantic message offsets differ from its actual source tokens/text')
        return self

    def require_native(self, **kwargs):
        raise ValueError('semantic codes require decoding with their matching adapter before native KV splicing')

    def metadata(self):
        return {**super().metadata(), 'schema': SCHEMA, 'source_model': self.source_model,
                'adapter_sha256': self.adapter_sha256, 'offsets': [list(pair) for pair in self.offsets],
                'posterior': 'mean', 'offset_policy': self.offset_policy}

    def tensors(self):
        return {'token_ids': self.token_ids, 'positions': self.positions,
                'embeddings': self.embeddings, 'code': self.code}

    @classmethod
    def from_tensors(cls, metadata, tensors):
        if (metadata.get('schema') != SCHEMA or metadata.get('posterior') != 'mean'
                or metadata.get('layer_kinds') != []
                or set(tensors) != {'token_ids', 'positions', 'embeddings', 'code'}):
            raise ValueError('semantic message schema or tensor inventory differs')
        return cls(**{name: metadata[name] for name in ('text', 'architecture', 'tokenizer_fingerprint',
            'checkpoint_sha256', 'embedding_weight_sha256', 'source_model', 'adapter_sha256', 'offsets', 'offset_policy')},
            **tensors, layers=(),adapter_binding=metadata.get("adapter_binding"),
            derivation=metadata.get("derivation")).validate()
