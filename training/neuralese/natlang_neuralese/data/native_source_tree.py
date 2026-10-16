"""Exact manifest metadata plumbing shared by ordinary loading and CPU preparation."""
import json
from pathlib import Path
from ..common.hashing import sha256_file_hex_cached


def initialize_manifest(source, parents, *, load_child):
    # Manifest assembly is always verified; only leaf row indexes are cached.
    before = source.path.stat()
    manifest = json.loads(source.path.read_text())
    if manifest.get('schema') != 'natlang.native-graph-sources/1' or not manifest.get('sources'):
        raise ValueError('native graph source manifest requires its schema and bound sources')
    source.kind = 'manifest'
    source.file_sha256 = sha256_file_hex_cached(source.path)
    source.children, source.entries, source.bindings, source.groups = [], [], {}, set()
    for binding in manifest['sources']:
        priority = binding.get('training_priority', 0)
        if type(priority) is not int or priority < 0:
            raise ValueError('native source training_priority must be a nonnegative integer')
        kind = binding.get('kind')
        if kind not in {'native-graphs', 'semantic-pairs'}:
            raise ValueError('native graph manifest source kind must be explicit')
        if kind == 'semantic-pairs' and not isinstance(binding.get('model'), str):
            raise ValueError('semantic graph source must select its exact model view')
        path = Path(binding['path'])
        if not path.is_absolute():
            path = source.path.parent / path
        if sha256_file_hex_cached(path) != binding.get('sha256'):
            raise ValueError('native graph manifest source SHA-256 mismatch: ' + str(path))
        child = load_child(path, source.tokenizer_fingerprint, required_split=source.required_split,
            pair_model=binding.get('model') if kind == 'semantic-pairs' else None,
            role_ids=source.role_ids, _parents=(*parents, source.path.resolve()))
        if child.file_sha256 != binding['sha256']:
            raise ValueError('native graph manifest dependency changed during validation')
        if source.bindings.keys() & child.bindings.keys():
            raise ValueError('native graph manifest repeats a graph identity across sources')
        source.children.append(child)
        # Scheduling policy never changes physical source order or row
        # binding. An omitted priority inherits a nested declaration.
        for entry in child.entries:
            effective = priority if 'training_priority' in binding else entry.get('training_priority', 0)
            source.entries.append(entry if 'training_priority' in entry and entry['training_priority'] == effective
                              else {**entry, 'training_priority': effective})
        source.bindings.update(child.bindings)
        source.groups.update(child.groups)
    state = source.path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (state.st_size, state.st_mtime_ns, state.st_ino):
        raise ValueError('native graph manifest changed during validation')
    source.file_binding = (state.st_size, state.st_mtime_ns, state.st_ino)
