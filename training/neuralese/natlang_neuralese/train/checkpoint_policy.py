"""Read a native training checkpoint (full state, shared-participant manifests expanded).

``CheckpointPolicy.load`` is the one loader every reader uses (serving, export, fine-tuning scripts);
``CheckpointPolicy.preload`` owns one authenticated decode for a later exact restore (``PreloadedCheckpoint``).
"""
from __future__ import annotations

from pathlib import Path

import torch


class CheckpointPolicy:
    """Checkpoint reading (the rolling-slot writer of the retired native trainer is in git history)."""

    @staticmethod
    def preload(path, *, expected_sha256, shared_participant=None):
        """Own one ordinary CPU decode for a later exact native restore."""
        return PreloadedCheckpoint._load(path, expected_sha256=expected_sha256,
                                         shared_participant=shared_participant)

    @staticmethod
    def load(path, map_location="cpu", *, shared_participant=None, semantic_only=False,
             expand_shared=True) -> dict:
        """Load a full training state while allowlisting TE's version metadata only."""
        from torch.torch_version import TorchVersion

        from .checkpoint_readahead import enabled, advise_checkpoint
        sequential = enabled()
        with torch.serialization.safe_globals([TorchVersion]):
            state = torch.load(path, map_location=map_location, mmap=True, weights_only=True)
        if sequential:
            advise_checkpoint(path)
        if expand_shared:
            from ..model.native_shared_checkpoint import SHARDED_SCHEMA, expand_shared as expand
            if state.get('schema') == SHARDED_SCHEMA:
                return expand(state, path, participant=shared_participant,
                              semantic_only=semantic_only, map_location=map_location)
        return state


class PreloadedCheckpoint:
    """One-use full state produced only by the ordinary checkpoint policy."""
    def __init__(self, *args, **kwargs):
        raise TypeError('use CheckpointPolicy.preload for an authenticated full decode')

    @staticmethod
    def _stat(path):
        value = path.stat()
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    @classmethod
    def _load(cls, path, *, expected_sha256, shared_participant):
        from ..common.hashing import sha256_file_hex_cached
        path = Path(path).resolve(strict=True)
        before = cls._stat(path)
        if sha256_file_hex_cached(path) != expected_sha256:
            raise ValueError('preloaded checkpoint source SHA differs')
        state = CheckpointPolicy.load(path, shared_participant=shared_participant)
        if cls._stat(path) != before:
            raise ValueError('preloaded checkpoint changed during ordinary decode')
        value = object.__new__(cls)
        value.path, value.sha256, value.stat = path, expected_sha256, before
        value.participant, value._state = shared_participant, state
        return value

    def consume(self, path, *, shared_participant=None):
        from ..common.hashing import sha256_file_hex_cached
        if self._state is None:
            raise ValueError('preloaded checkpoint was already consumed')
        path = Path(path).resolve(strict=True)
        if path != self.path or shared_participant != self.participant:
            raise ValueError('preloaded checkpoint path or participant differs')
        if self._stat(path) != self.stat or sha256_file_hex_cached(path) != self.sha256:
            raise ValueError('preloaded checkpoint source changed before consumption')
        state, self._state = self._state, None
        return state
