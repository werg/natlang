"""Reuse immutable validated source trees within one preparation or adoption."""
from pathlib import Path
from ..common.hashing import sha256_file_hex_cached


class NativeSourcePreparation:
    def __init__(self, tokenizer_fingerprint, role_ids, *, cache_only=False):
        from ..data.native_sources import NativeGraphSource
        self.source_type = NativeGraphSource
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.role_ids = tuple(role_ids)
        self.cache_only = bool(cache_only)
        self.sources = {}
        self.loaded_roots = 0
        self.reused_roots = 0
        self.loaded_sources = self.reused_sources = 0
        self._group_strings = {}
        self._group_bundles = {}


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

    def retain(self, source):
        """Seed from an already loaded tree; reverify its bytes before reuse."""
        pending = [source]
        while pending:
            value = pending.pop()
            if (value.tokenizer_fingerprint != self.tokenizer_fingerprint
                    or tuple(value.role_ids) != self.role_ids):
                raise ValueError('retained source changed tokenizer or role contract')
            pending.extend(getattr(value, 'children', ()))
        self._remember(source)

    def _verify(self, source):
        pending = [source]
        while pending:
            value = pending.pop()
            state = value.path.stat()
            checker = getattr(value, '_checked_file_state', None)
            if checker is not None:
                checker(value.file_binding, state, context='reused preparation source changed after validation')
            elif ((state.st_size, state.st_mtime_ns, state.st_ino) != value.file_binding
                    or sha256_file_hex_cached(value.path) != value.file_sha256):
                # A deliberately retained old frozen source keeps its original
                # strict contract; no unqualified verifier is monkeypatched in.
                raise ValueError('reused preparation source changed after validation: ' + str(value.path))
            pending.extend(getattr(value, 'children', ()))

    def _intern_groups(self, source):
        # Group metadata is immutable. Equal objects retain identical values,
        # sequence types and hashes while avoiding one copy per graph/leaf.
        for entry in source.entries:
            groups = entry['groups']
            values = tuple(self._group_strings.setdefault(value, value) for value in groups)
            key = (type(groups), values)
            if key not in self._group_bundles:
                self._group_bundles[key] = type(groups)(values)
            entry['groups'] = self._group_bundles[key]
        source.groups = {self._group_strings.setdefault(value, value) for value in source.groups}

    def _require_cached_leaf(self, path, source_sha256, split, pair_model):
        """Fail closed unless the existing parser-policy-bound leaf index is warm."""
        from ..common import row_index
        from ..common.hashing import canonical_json_sha256_hex
        package = Path(__file__).resolve().parents[1]
        names = ('data/native_sources.py', 'data/native_graph_schema.py', 'data/decision_graph.py', 'model/native_graph_plan.py',
                 'data/native_pair_schema.py', 'common/row_index.py', 'common/jsonio.py', 'common/hashing.py')
        policy = canonical_json_sha256_hex({name: sha256_file_hex_cached(package/name) for name in names})
        key = {'source_sha256': source_sha256, 'tokenizer': self.tokenizer_fingerprint,
               'split': split, 'policy_sha256': policy,
               'kind': 'semantic-pairs' if pair_model is not None else 'native-graphs',
               'pair_model': pair_model, 'role_ids': list(self.role_ids)}
        cache = row_index.entry_path('native-graphs', canonical_json_sha256_hex(key))
        value = row_index.load(cache)
        if (not value or value.get('key') != key or not isinstance(value.get('entries'), list)
                or not value['entries']
                or value.get('metadata_sha256') != canonical_json_sha256_hex(value['entries'])):
            raise ValueError('resident prefix integrity requires the exact completed source index: ' + str(path))

    def _load_child(self, path, fingerprint, *, required_split, pair_model, role_ids, _parents):
        if fingerprint != self.tokenizer_fingerprint or tuple(role_ids) != self.role_ids:
            raise ValueError('preparation child changed tokenizer or role contract')
        return self.load({'path': str(path), 'sha256': sha256_file_hex_cached(path)},
                         split=required_split, pair_model=pair_model, _parents=_parents)

    def load(self, binding, *, split, pair_model=None, _parents=()):
        # Use the existing bound-file verifier and unchanged leaf parser/cache.
        # Explicit recursion also works beside an old frozen source class;
        # no global class or numerical module is replaced.
        from .native_resident import bound
        path = bound(binding)
        if path.resolve() in _parents:
            raise ValueError('native graph source manifests cannot contain cycles')
        key = self._key(path, binding['sha256'], split, pair_model)
        source = self.sources.get(key)
        if source is not None:
            self._verify(source)
            self.reused_roots += not bool(_parents)
            self.reused_sources += 1
            return source
        if self.cache_only and not (path.suffix == '.json' and pair_model is None):
            self._require_cached_leaf(path, binding['sha256'], split, pair_model)
        if path.suffix == '.json' and pair_model is None:
            from ..data.native_source_tree import initialize_manifest, _source_identity
            source = self.source_type.__new__(self.source_type)
            source.path, source.kind, source.pair_model = path, 'native-graphs', None
            source.source_identity_path = _source_identity(path)
            source.role_ids = self.role_ids
            source.tokenizer_fingerprint, source.required_split = self.tokenizer_fingerprint, split
            initialize_manifest(source, _parents, load_child=self._load_child)
        else:
            source_type = self.source_type
            if self.cache_only:
                # Close the check/use window: the unchanged loader may not
                # fall back to streaming the complete source if the index is
                # removed after preflight. This subclass changes no parser
                # bytes, cache key, or accepted cached metadata.
                class CacheOnlyNativeGraphSource(source_type):
                    def _parse(self, row):
                        raise ValueError('resident prefix source index disappeared before cache-only load: '
                                         + str(path))
                source_type = CacheOnlyNativeGraphSource
            source = source_type(path, self.tokenizer_fingerprint, required_split=split,
                                      pair_model=pair_model, role_ids=self.role_ids, _parents=_parents)
            self._intern_groups(source)
        if source.file_sha256 != binding['sha256']:
            raise ValueError('preparation source changed during validation')
        self._remember(source)
        self.loaded_roots += not bool(_parents)
        self.loaded_sources += 1
        return source
