"""Reuse immutable validated source trees within one CPU preparation only."""
from pathlib import Path
from ..common.hashing import sha256_file_hex_cached


class NativeSourcePreparation:
    def __init__(self, tokenizer_fingerprint, role_ids):
        from ..data.native_sources import NativeGraphSource
        self.source_type = NativeGraphSource
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.role_ids = tuple(role_ids)
        self.sources = {}
        self.loaded_roots = 0
        self.reused_roots = 0

    def _key(self, path, digest, split, pair_model=None):
        return (str(Path(path).resolve(strict=True)), digest, self.tokenizer_fingerprint,
                split, pair_model, self.role_ids)

    def _remember(self, source):
        pending = [source]
        while pending:
            value = pending.pop()
            key = self._key(value.path, value.file_sha256, value.required_split, value.pair_model)
            self.sources[key] = value
            pending.extend(getattr(value, 'children', ()))

    def _verify(self, source):
        pending = [source]
        while pending:
            value = pending.pop()
            state = value.path.stat()
            if ((state.st_size, state.st_mtime_ns, state.st_ino) != value.file_binding
                    or sha256_file_hex_cached(value.path) != value.file_sha256):
                raise ValueError('reused preparation source changed after validation: ' + str(value.path))
            pending.extend(getattr(value, 'children', ()))

    def load(self, binding, *, split, pair_model=None):
        # Use the existing exact bound-file verifier. Its import is deferred so
        # this mechanism does not introduce a module-initialization cycle.
        from .native_resident import bound
        path = bound(binding)
        key = self._key(path, binding['sha256'], split, pair_model)
        source = self.sources.get(key)
        if source is not None:
            self._verify(source)
            self.reused_roots += 1
            return source
        source = self.source_type(path, self.tokenizer_fingerprint, required_split=split,
                                  pair_model=pair_model, role_ids=self.role_ids)
        if source.file_sha256 != binding['sha256']:
            raise ValueError('preparation source changed during validation')
        self._remember(source)
        self.loaded_roots += 1
        return source
