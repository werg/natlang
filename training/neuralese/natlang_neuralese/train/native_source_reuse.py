"""Reuse a validated source only for the exact same declared foundation input."""
import json
import os
from pathlib import Path

def matching_validated_source(source, path, tokenizer_fingerprint, *, required_split, pair_model=None, role_ids=None):
    """Reuse an identical validated tree only while its files and staging bindings remain current."""
    from ..data.native_source_tree import _source_identity, _staged_path
    path = Path(path)
    # Manifest-relative children are resolved from the lexical parent; two
    # symlinks to the same file can therefore name different source trees.
    if (os.path.abspath(path) != os.path.abspath(source.path) or source.tokenizer_fingerprint != tokenizer_fingerprint
            or source.required_split != required_split or source.pair_model != pair_model
            or source.role_ids != tuple(role_ids or ())):
        return None

    def current(item):
        if _source_identity(item.path).resolve() != item.source_identity_path.resolve():
            return False
        item._checked_file_state(item.file_binding, item.path.stat(), context='source changed after index validation')
        if item.kind != 'manifest':
            return True
        manifest = json.loads(item.path.read_text())
        bindings = manifest.get('sources') or []
        if len(bindings) != len(item.children): return False
        for binding, child in zip(bindings, item.children):
            selected = _staged_path(item.path, binding['path'])
            if not selected.is_absolute(): selected = item.path.parent / selected
            if (os.path.abspath(selected) != os.path.abspath(child.path) or binding.get('sha256') != child.file_sha256
                    or not current(child)):
                return False
        item._checked_file_state(item.file_binding, item.path.stat(), context='source changed after index validation')
        return True
    return source if current(source) else None

