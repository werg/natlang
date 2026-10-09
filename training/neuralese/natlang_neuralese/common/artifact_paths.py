"""Role-bound artifact paths at the checkpoint boundary (ARCHITECTURE_IMPROVEMENT C1).

A training checkpoint names its inputs by the path the launcher typed (``options.heads``, ``options.pieces`` ...), which
may be relative to the launch directory, and pins their bytes by absolute path in ``identity.files`` (or ``identity.inputs``
for the text warm-up). Loading it from another directory then failed, or silently read a different file. This module
binds each input to its role once, and every loader, exporter and diagnostic resolves roles through it:

* New checkpoints carry ``artifact_refs`` = ``{role: {logical, resolved, sha256}}`` (``artifact_refs``).
* ``ArtifactResolver.path(role)`` returns a path whose bytes match the recorded sha256. Order: the caller's explicit
  override; the recorded resolved path; for a legacy checkpoint without refs, the unique ``identity.files`` key that ends
  with the normalized option path. Nothing else is guessed: a missing file, an ambiguous suffix or changed bytes raise
  ``ArtifactResolutionError`` naming the role and asking for a path override.
* The process working directory is never read or changed. Roles that are not pinned files (a model id, a local model
  directory) come back as the recorded value; machine-independent roots remain C4.
"""
from __future__ import annotations

import posixpath
from pathlib import Path

from .hashing import sha256_file_hex


# Roles that are always files: when one cannot be bound, loading stops instead of falling back to the typed path.
FILE_ROLES = frozenset({'heads', 'records', 'pieces', 'bank', 'soft_init', 'text_data', 'student_checkpoint',
                        'continue_from'})


class ArtifactResolutionError(ValueError):
    """A role cannot be bound to a file with the recorded bytes; the message says how to override it."""


def artifact_refs(options, files, roles, *, cwd=None):
    """The role table to store in a new checkpoint.

    ``files`` is the identity's ``{absolute path: sha256}`` pin table. ``cwd`` is the launch directory used to resolve
    the options once, at the time the checkpoint is written (default: the current one).
    """
    base = Path(cwd) if cwd is not None else Path.cwd()
    refs = {}
    for role in roles:
        value = options.get(role)
        if value in (None, ''):
            continue
        path = Path(value)
        resolved = path if path.is_absolute() else base / path
        resolved = resolved.resolve()  # the same symlink-free form identity.files is keyed by
        # A role the identity does not pin (a model directory, a model id) records no digest.
        refs[role] = {'logical': str(value), 'resolved': str(resolved) if resolved.exists() else None,
                      'sha256': files.get(str(resolved))}
    return refs


def _tail(option):
    """The option path without a leading ``./`` or ``../`` so it can be matched against absolute keys."""
    parts = [part for part in posixpath.normpath(str(option)).split('/') if part not in ('', '.', '..')]
    return '/'.join(parts)


class ArtifactResolver:
    def __init__(self, identity, refs=None, *, overrides=None, verify=True):
        self.options = dict((identity or {}).get('options') or {})
        self.files = dict((identity or {}).get('files') or (identity or {}).get('inputs') or {})
        self.refs = dict(refs or {})
        self.overrides = {role: Path(path) for role, path in (overrides or {}).items()}
        self.verify = verify
        self._digests = {}

    @classmethod
    def from_state(cls, state, *, overrides=None, verify=True):
        """From a loaded checkpoint (new ones carry ``artifact_refs``; legacy ones do not)."""
        return cls(state.get('identity'), state.get('artifact_refs'), overrides=overrides, verify=verify)

    def _digest(self, path):
        key = str(path)
        if key not in self._digests:
            self._digests[key] = sha256_file_hex(path)
        return self._digests[key]

    def _pinned(self, role):
        """The digest the checkpoint pinned for this role (recorded, or its one legacy match), if any."""
        ref = self.refs.get(role) or {}
        if ref.get('sha256'):
            return ref['sha256']
        logical = ref.get('logical', self.options.get(role))
        candidates = self._legacy_candidates(logical) if logical not in (None, '') else []
        return self.files[candidates[0]] if len(candidates) == 1 else None

    def _accept(self, role, path, expected, *, how):
        path = Path(path)
        if not path.is_file():
            raise ArtifactResolutionError(
                f'{role}: {how} {path} is not a file; pass an explicit path override for role {role!r}')
        if self.verify and expected is not None and self._digest(path) != expected:
            raise ArtifactResolutionError(
                f'{role}: {path} has changed since the checkpoint pinned it (sha256 differs); '
                f'pass a path override for role {role!r} only if that file is the intended replacement')
        return str(path)

    def _legacy_candidates(self, option):
        value = str(option)
        if posixpath.isabs(value):
            exact = posixpath.normpath(value)
            return [key for key in self.files if posixpath.normpath(key) == exact]
        tail = _tail(value)
        if not tail:
            return []
        return [key for key in self.files if key == tail or key.endswith('/' + tail)]

    def path(self, role, *, required=False):
        """The file bound to ``role``, verified; or the recorded option unchanged when it is not a pinned file
        (a model id); or None when the checkpoint never had that role (an error when ``required``)."""
        if role in self.overrides:
            override = self.overrides[role]
            return self._accept(role, override, self._pinned(role) or self.files.get(str(override.resolve())),
                                how='override')
        ref = self.refs.get(role)
        option = self.options.get(role)
        if ref and ref.get('resolved') and Path(ref['resolved']).is_file():
            return self._accept(role, ref['resolved'], ref.get('sha256'), how='recorded path')
        if ref and ref.get('resolved') and role not in FILE_ROLES and Path(ref['resolved']).exists():
            return ref['resolved']  # a local model directory, bound at launch
        logical = (ref or {}).get('logical', option)
        if logical in (None, ''):
            if required:
                raise ArtifactResolutionError(f'the checkpoint records no {role!r}; pass a path override')
            return None
        candidates = self._legacy_candidates(logical)
        if len(candidates) == 1:
            return self._accept(role, candidates[0], self.files[candidates[0]], how='pinned path')
        if len(candidates) > 1:
            raise ArtifactResolutionError(
                f'{role}: option {logical!r} matches {len(candidates)} pinned files ({sorted(candidates)[:3]}); '
                f'pass an explicit path override for role {role!r}')
        if role in FILE_ROLES or (ref and ref.get('sha256')):
            raise ArtifactResolutionError(
                f'{role}: {(ref or {}).get("resolved") or logical} is not available and no pinned file matches '
                f'{logical!r}; pass an explicit path override for role {role!r}')
        return str(logical)
