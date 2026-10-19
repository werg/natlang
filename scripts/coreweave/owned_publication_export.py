"""Verify and pin the declared owned learner's publication without tensor decode."""
from __future__ import annotations

import json
import inspect
import os
import shutil
from pathlib import Path
import subprocess
import uuid

from .cursor_checkpoint_export import bound_output_checkpoint


def remote_code(spec, *, known_sha256=None, known_step=-1):
    """Render the same verifier against the owner's explicitly frozen runtime."""
    return (
        'import hashlib,json,os,pathlib,re,shutil,subprocess,sys,uuid\nfrom pathlib import Path\n'
        + 'SPEC=' + repr(spec) + '\n'
        + "sys.path[:0]=[str(Path(SPEC['runtime'])/'training/neuralese'),SPEC['runtime']]\n"
        + 'from natlang_neuralese.train.native_resident import code_inventory,digest\n'
        + "assert digest(code_inventory(Path(SPEC['runtime'])/'training/neuralese/natlang_neuralese'))==SPEC['runtime_code_sha256']\n"
        + inspect.getsource(bound_output_checkpoint) + '\n'
        + inspect.getsource(observe_publication) + '\n'
        + 'chosen=observe_publication(SPEC,known_sha256=' + repr(known_sha256)
        + ',known_step=' + repr(known_step) + ')\n'
        + "print(json.dumps({'chosen':chosen,'observations':{}}))\n"
    )


def observe_publication(spec, *, known_sha256=None, known_step=-1, verify=None):
    """Return an export candidate only for this exact owner and saved parent.

    The caller persists pending candidates with the existing private exporter.
    Its state directory belongs to this continuation, never an older trajectory.
    """
    from natlang_neuralese.model.native_checkpoint_control import (
        checked_file, source_stat, validate_publication_selection,
        verify_checkpoint_publication,
    )
    if spec.get('schema') != 'natlang.owned-publication-export/1':
        raise ValueError('owned publication export needs an explicit selection')
    selected = validate_publication_selection(spec['publication'])
    verify = verify or verify_checkpoint_publication
    source = bound_output_checkpoint(spec['effective_argv'], spec['role'],
        production_root=spec['production_root'], control_root=spec['control_root'])
    status = source.parent/'checkpoint-publication-status.json'
    if not status.exists():
        return None
    status_before = source_stat(status)
    published = json.loads(status.read_bytes())
    if published.get('passed') is not True:
        return None
    reference, checkpoint = published['control'], published['checkpoint']
    if Path(checkpoint['path']).resolve() != source.resolve():
        raise ValueError('publication belongs to another learner output')
    directory = source.parent/'checkpoint-controls'/checkpoint['sha256']
    control = checked_file(reference)
    if not control.is_relative_to(directory.resolve()):
        raise ValueError('publication control escaped its owned source directory')
    kwargs = dict(producer=selected['producer'],
        dependency_producer=selected['dependency_producer'],
        namespace=selected['namespace'], readonly_input_roots=selected['readonly_input_roots'])
    source_before = source_stat(source)
    actual = verify(source, reference, spec['parent'], **kwargs)
    if actual['checkpoint'] != checkpoint:
        raise ValueError('publication status and verified checkpoint differ')
    if (source_stat(status) != status_before or source_stat(source) != source_before):
        raise ValueError('publication changed during selection; retry latest committed source')
    if checkpoint['sha256'] == known_sha256:
        return None
    if actual['step'] <= known_step:
        raise ValueError('owned continuation publication regressed or changed an already exported step')
    pins = Path(spec['pin_directory']).resolve()
    pins.mkdir(parents=True, exist_ok=True)
    target = pins/(spec['role']+'-'+checkpoint['sha256']+'.pt')
    reserve = spec['reserve_bytes']
    if type(reserve) is not int or reserve < 0:
        raise ValueError('publication pin needs a declared nonnegative storage reserve')
    if not target.exists() and shutil.disk_usage(pins).free < checkpoint['bytes'] + reserve:
        raise ValueError('insufficient storage for independent full publication pin and reserve')
    temporary = pins/('.owned-publication-'+uuid.uuid4().hex+'.pt')
    try:
        if not target.exists():
            subprocess.run(['cp', '--reflink=auto', '--preserve=mode,timestamps',
                            '--', str(source), str(temporary)], check=True)
            if source_stat(source) != source_before:
                raise ValueError('source advanced while copying publication; retry')
            if source_stat(temporary)[:2] == source_before[:2]:
                raise ValueError('publication export requires an independent inode')
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            # A later atomic replacement of the learner slot cannot change this pin.
            pinned = verify(temporary, reference, spec['parent'], **kwargs)
            if pinned != actual:
                raise ValueError('copied publication differs from verified source')
            # Exclusive publish: never replace an existing immutable export pin.
            try:
                os.link(temporary, target)
            except FileExistsError:
                pass
        if target.is_symlink() or source_stat(target)[:2] == source_before[:2]:
            raise ValueError('publication pin must be an independent regular file')
        if verify(target, reference, spec['parent'], **kwargs) != actual:
            raise ValueError('existing publication pin differs from selected source')
        descriptor = os.open(pins, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return {'source': str(source), 'pin': str(target), 'step': actual['step'],
        'bytes': checkpoint['bytes'], 'sha256': checkpoint['sha256'],
        'checkpoint_control': reference, 'publication_selection': selected,
        'owned_parent': spec['parent'], 'tensor_decodes': 0}
