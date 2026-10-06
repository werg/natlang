"""Diagnostic token traces of exact raw embedding payloads; never approximate decoding."""
import hashlib

import torch


class RawWriterTrace:
    def __init__(self, embedding):
        self.rows = embedding.detach().float().cpu().contiguous()
        self.index = {}
        for i, row in enumerate(self.rows):
            self.index.setdefault(self.key(row), []).append(i)

    @staticmethod
    def key(row):
        return hashlib.sha256(row.contiguous().numpy().tobytes()).digest()

    def decode(self, payload, tokenizer):
        value = payload.detach().float().cpu().contiguous()
        ids, unknown, ambiguous = [], [], []
        for i, row in enumerate(value):
            matches = [j for j in self.index.get(self.key(row), []) if torch.equal(row, self.rows[j])]
            if not matches:
                unknown.append(i)
            elif len(matches) > 1:
                ambiguous.append(i)
            ids.append(matches[0] if len(matches) == 1 else None)
        return {'schema': 'natlang.raw-writer-token-trace/1', 'vectors': len(ids),
                'exact_unique_embeddings': len(ids) - len(unknown) - len(ambiguous),
                'unknown_positions': unknown, 'ambiguous_positions': ambiguous,
                'token_ids': ids,
                'decoded': tokenizer.decode(ids, skip_special_tokens=False) if not unknown and not ambiguous else None,
                'approximate_decoding': False}
