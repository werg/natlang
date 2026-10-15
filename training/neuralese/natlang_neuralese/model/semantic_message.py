"""Compact learned dialect using the existing native artifact lifecycle."""
from dataclasses import dataclass
import re
import torch

from .state_block import NativeStateBlock

SCHEMA = 'neuralese.semantic-state/1'


@dataclass
class SemanticStateMessage(NativeStateBlock):
    code: torch.Tensor
    offsets: list
    source_model: str
    adapter_sha256: str

    def validate(self):
        count = self._validate_token_fields()
        if self.layers:
            raise ValueError('semantic messages carry codes, not raw backbone layer state')
        self._float_tensor('code', self.code, count, 2)
        if not isinstance(self.source_model, str) or not self.source_model:
            raise ValueError('semantic message requires its adapter source model')
        if not isinstance(self.adapter_sha256, str) or not re.fullmatch('[0-9a-f]{64}', self.adapter_sha256):
            raise ValueError('semantic message requires its exact adapter SHA-256')
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
                'posterior': 'mean'}

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
            'checkpoint_sha256', 'embedding_weight_sha256', 'source_model', 'adapter_sha256', 'offsets')},
            **tensors, layers=()).validate()
