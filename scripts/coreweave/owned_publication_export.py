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


def remote_code(spec, *, known_sha256=None, known_step=-1, hash_cache_path=None):
    """Render the same verifier against the owner's explicitly frozen runtime."""
    return (
        'import hashlib,json,os,pathlib,re,shutil,subprocess,sys,uuid\nfrom pathlib import Path\n'
        + 'HASH_CACHE=' + repr(None if hash_cache_path is None else str(hash_cache_path)) + '\n'
        + 'SPEC=' + repr(spec) + '\n'
        + "if HASH_CACHE is not None:\n"
        + " p=Path(HASH_CACHE);root=Path(SPEC['production_root'])\n"
        + " if not p.is_absolute() or str(p)!=os.path.normpath(str(p)) or p.is_symlink() or p.parent.is_symlink() or p.parent.resolve()!= (root/'cache').resolve():raise ValueError('owned hash cache escaped production/cache')\n"
        + ' os.environ["NATLANG_HASH_CACHE"]=HASH_CACHE\n'
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
    if status.is_symlink():
        raise ValueError('publication status may not be a symlink')
    if not status.exists():
        return None
    if source.is_symlink() or not source.is_file():
        raise ValueError('owned learner checkpoint must be a regular nonsymlink file')
    status_before = source_stat(status)
    status_raw = status.read_bytes()
    # Bind one complete status record before doing any checkpoint I/O. A
    # concurrent atomic status replacement is fine after this capture: the
    # captured control below decides whether the copied bytes are its save.
    if source_stat(status) != status_before:
        raise ValueError('publication status changed while being captured; retry')
    published = json.loads(status_raw)
    if published.get('passed') is not True:
        # A failed derivation must remain observable: treating it like an
        # absent first save would leave quality evaluation waiting forever.
        kind = published.get('error_type')
        if not isinstance(kind, str) or not kind.isidentifier():
            kind = 'InvalidPublicationStatus'
        raise ValueError('owned checkpoint publication failed: ' + kind)
    reference, checkpoint = published['control'], published['checkpoint']
    if Path(checkpoint['path']).resolve() != source.resolve():
        raise ValueError('publication belongs to another learner output')
    directory = source.parent/'checkpoint-controls'/checkpoint['sha256']
    control = checked_file(reference)
    if not control.is_relative_to(directory.resolve()):
        raise ValueError('publication control escaped its owned source directory')
    if known_sha256 is not None and checkpoint.get('sha256') == known_sha256:
        # This exact content identity was already durably accepted. The local
        # pin may since have been pruned; authenticate the captured control's
        # source/context/parent bindings and return without recopying the full
        # checkpoint merely to report no newly observed content.
        value = json.loads(control.read_bytes())
        context = {'namespace': str(Path(selected['namespace']).absolute()),
                   'readonly_input_roots': sorted(selected['readonly_input_roots'])}
        publication = value.get('publication') or {}
        if (value.get('schema') != 'natlang.native-checkpoint-control/1'
                or value.get('source') != checkpoint
                or value.get('producer') != selected['producer']
                or value.get('dependency_producer') != selected['dependency_producer']
                or value.get('dependency_context') != context
                or publication.get('schema') != 'natlang.native-checkpoint-publication/1'
                or publication.get('full_optimizer_rng_validated') is not True
                or publication.get('parent_checkpoint_sha256') != spec['parent']['sha256']
                or type(value.get('step')) is not int or value['step'] < 0):
            raise ValueError('known publication hash does not match its captured control/context/parent')
        return None
    kwargs = dict(producer=selected['producer'],
        dependency_producer=selected['dependency_producer'],
        namespace=selected['namespace'], readonly_input_roots=selected['readonly_input_roots'])
    # The learner writes checkpoint.pt by atomic replacement. Hashing that
    # mutable path first can race a 10s-of-GB write; snapshot first, then let
    # the immutable control authenticate exactly which save was copied.
    source_before = source_stat(source)
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
        created = False
        actual = None
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise ValueError('publication pin must be a regular nonsymlink file')
            if source_stat(target)[:2] in (source_before[:2], source_stat(source)[:2]):
                raise ValueError('publication pin must be an independent regular file')
            os.chmod(target, target.stat().st_mode & ~0o222)
        if not target.exists():
            subprocess.run(['cp', '--reflink=auto', '--preserve=mode,timestamps',
                            '--', str(source), str(temporary)], check=True)
            temporary_identity = source_stat(temporary)[:2]
            if temporary_identity in (source_before[:2], source_stat(source)[:2]):
                raise ValueError('publication export requires an independent inode')
            os.chmod(temporary, temporary.stat().st_mode & ~0o222)
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            # The temporary is now the immutable candidate. Its content must
            # match the captured status and its exact content-addressed
            # control; the source slot may already have advanced.
            actual = verify(temporary, reference, spec['parent'], **kwargs)
            if actual['checkpoint'] != checkpoint:
                raise ValueError('copied publication differs from captured status')
            if actual['step'] <= known_step:
                raise ValueError('owned continuation publication regressed or changed an already exported step')
            if known_sha256 is not None and actual['checkpoint']['sha256'] == known_sha256:
                return None
            # Exclusive publish: never replace an existing immutable export pin.
            try:
                os.link(temporary, target)
                created = True
            except FileExistsError:
                pass
        if not created:
            # Reverify an extant or concurrently-created pin. A pin created by
            # this call is the same inode as the just-verified snapshot.
            actual = verify(target, reference, spec['parent'], **kwargs)
        if actual['checkpoint'] != checkpoint:
            raise ValueError('existing publication pin differs from captured status')
        if known_sha256 is not None and actual['checkpoint']['sha256'] == known_sha256:
            return None
        if actual['step'] <= known_step:
            raise ValueError('owned continuation publication regressed or changed an already exported step')
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


def verify_owner_pointer(pointer_path, *, role, production_root='/work/native-production',
                         control_root='/work/native-production/control'):
    """Resolve one exact controller-selected owner manifest and detect pointer races."""
    import hashlib
    pointer = Path(pointer_path)
    if not pointer.is_absolute() or pointer.is_symlink():
        raise ValueError('owned publication pointer must be an absolute nonsymlink file')
    control = Path(control_root).resolve()
    if not pointer.resolve(strict=True).is_relative_to(control):
        raise ValueError('owned publication pointer escaped its controller namespace')
    pointer_stat = pointer.stat()
    pointer_raw = pointer.read_bytes()
    selected = json.loads(pointer_raw)
    if (selected.get('schema') != 'natlang.owned-publication-owner-current/1'
            or selected.get('role') != role):
        raise ValueError('owned publication pointer belongs to another role/schema')
    reference = selected.get('owner') or {}
    raw_manifest_path = Path(reference.get('path', ''))
    if raw_manifest_path.is_symlink():
        raise ValueError('owned publication manifest may not be a symlink')
    manifest_path = raw_manifest_path.resolve(strict=True)
    production = Path(production_root).resolve()
    if (not manifest_path.is_relative_to(production)
            or manifest_path.stat().st_size != reference.get('bytes')
            or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != reference.get('sha256')):
        raise ValueError('owned publication manifest escaped or changed its bound production artifact')
    manifest = json.loads(manifest_path.read_bytes())
    generation = manifest.pop('generation_sha256', None)
    raw = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    if (hashlib.sha256(raw).hexdigest() != generation
            or generation != selected.get('generation_sha256')
            or manifest.get('schema') != 'natlang.owned-publication-owner/1'
            or manifest.get('role') != role):
        raise ValueError('owned publication manifest generation or identity differs')
    final_pointer_stat = pointer.stat()
    if ((final_pointer_stat.st_dev, final_pointer_stat.st_ino,
            final_pointer_stat.st_size, final_pointer_stat.st_mtime_ns)
            != (pointer_stat.st_dev, pointer_stat.st_ino,
                pointer_stat.st_size, pointer_stat.st_mtime_ns)
            or pointer.read_bytes() != pointer_raw):
        raise ValueError('owned publication pointer changed during read')
    manifest['generation_sha256'] = generation
    return manifest, {'path': str(manifest_path), 'sha256': reference['sha256'],
        'bytes': reference['bytes']}, (pointer_stat.st_dev, pointer_stat.st_ino,
        pointer_stat.st_size, pointer_stat.st_mtime_ns, hashlib.sha256(pointer_raw).hexdigest())


def verify_owned_publication_command(command, spec, trainer_invocation):
    """Verify production's exact env-prefixed trainer argv and bounded writer."""
    module,environment,_=trainer_invocation(command)
    if spec.get('effective_argv',{}).get('runtime_from_command_env'):
        if Path(environment.get('NATLANG_REPO','')).resolve()!=Path(spec['runtime']).resolve():
            raise ValueError('owned publication runtime differs from actual trainer environment')
    parent=spec['parent']
    if command.count('--continue-from')!=1 or Path(command[command.index('--continue-from')+1]).resolve()!=Path(parent['path']).resolve():
        raise ValueError('owned publication command does not continue its selected parent')
    effective=spec['effective_argv'];allowed=Path(effective['allowed_output_root']).resolve()
    expected=Path(effective.get('writer_output',str(allowed/'learner'))).resolve()
    if (not expected.is_relative_to(allowed) or command.count('--out')!=1
            or Path(command[command.index('--out')+1]).resolve()!=expected):
        raise ValueError('owned publication command does not use its bounded writer output')
    selected=spec['publication']
    if ('--checkpoint-publication' not in command or command.count('--checkpoint-publication-readonly-input-roots')!=1
            or json.loads(command[command.index('--checkpoint-publication-readonly-input-roots')+1])!=selected['readonly_input_roots']
            or command.count('--checkpoint-publication-namespace')!=1
            or command[command.index('--checkpoint-publication-namespace')+1]!=selected['namespace']):
        raise ValueError('owned publication command differs from its sealed producer selection')
    return module


def owner_remote_code(pointer_path, *, role, known_generation=None, known_sha256=None,
                      known_step=-1, production_root='/work/native-production',
                      control_root='/work/native-production/control', hash_cache_path=None):
    """Observe whatever exact learner the supervisor has most recently selected."""
    # The source is serialized and executed on the remote machine. Never embed
    # a host-side PosixPath repr (which is not defined in that source).
    pointer_path = str(pointer_path)
    production_root = str(production_root)
    control_root = str(control_root)
    helper_source = inspect.getsource(verify_owner_pointer)
    command_helper = inspect.getsource(verify_owned_publication_command)
    from natlang_neuralese.rl import native_publication_owner
    trainer_parser = inspect.getsource(native_publication_owner._trainer_invocation)
    source = (
        'import hashlib,json,os,pathlib,re,shutil,subprocess,sys,uuid\nfrom pathlib import Path\n'
        + 'HASH_CACHE=' + repr(None if hash_cache_path is None else str(hash_cache_path)) + '\n'
        + 'POINTER=' + repr(str(pointer_path)) + '\nROLE=' + repr(role) + '\n'
        + 'PRODUCTION_ROOT=' + repr(production_root) + '\nCONTROL_ROOT=' + repr(control_root) + '\n'
        + "if HASH_CACHE is not None:\n"
        + " p=Path(HASH_CACHE);root=Path(PRODUCTION_ROOT)\n"
        + " if not p.is_absolute() or str(p)!=os.path.normpath(str(p)) or p.is_symlink() or p.parent.is_symlink() or p.parent.resolve()!= (root/'cache').resolve():raise ValueError('owned hash cache escaped production/cache')\n"
        + ' os.environ["NATLANG_HASH_CACHE"]=HASH_CACHE\n'
        + 'TRAINER_MODULES=' + repr(tuple(native_publication_owner.TRAINER_MODULES)) + '\n'
        + helper_source + '\n' + trainer_parser + '\n' + command_helper + '\n'
        + "if Path(POINTER).is_symlink():raise ValueError('owned publication pointer may not be a symlink')\n"
        + "if not Path(POINTER).exists():print(json.dumps({'owner':None,'chosen':None,'observations':{}}));sys.exit(0)\n"
        + 'manifest,manifest_ref,pointer_signature=verify_owner_pointer(POINTER,role=ROLE,production_root=PRODUCTION_ROOT,control_root=CONTROL_ROOT)\n'
        + "SPEC=manifest.get('publication_spec') or {}\n"
        + "if SPEC.get('schema')!='natlang.owned-publication-export/1' or SPEC.get('role')!=ROLE:raise ValueError('owned publication spec role/schema differs')\n"
        + "runtime=Path(SPEC['runtime']).resolve(strict=True)\n"
        + "if not runtime.is_relative_to(Path(PRODUCTION_ROOT).resolve()) and not str(runtime).startswith('/work/native-runtime/'):raise ValueError('owned publication runtime escaped qualified runtime roots')\n"
        + "sys.path[:0]=[str(runtime/'training/neuralese'),str(runtime)]\n"
        + 'from natlang_neuralese.train.native_resident import code_inventory,digest\n'
        + "package=runtime/'training/neuralese/natlang_neuralese'\n"
        + "if digest(code_inventory(package))!=SPEC['runtime_code_sha256'] or manifest.get('runtime_code_sha256')!=SPEC['runtime_code_sha256'] or manifest.get('package')!={'path':str(package),'code':code_inventory(package)}:raise ValueError('owned publication runtime inventory changed')\n"
        + "from natlang_neuralese.model.native_checkpoint_control import validate_publication_selection\n"
        + "selected=validate_publication_selection(SPEC['publication'])\n"
        + "if selected['namespace']!=SPEC['production_root'] or Path(SPEC['control_root']).resolve()!=Path(CONTROL_ROOT).resolve():raise ValueError('owned publication roots differ from the current controller namespace')\n"
        + "if any(not Path(selected[k]['path']).resolve(strict=True).is_relative_to(runtime) for k in ('producer','dependency_producer')):raise ValueError('owned publication producer escaped its runtime')\n"
        + inspect.getsource(bound_output_checkpoint) + '\n'
        + inspect.getsource(observe_publication) + '\n'
        + "command_ref=SPEC['effective_argv'];command_path=Path(command_ref['path']).resolve(strict=True);command_raw=command_path.read_bytes()\n"
        + "if len(command_raw)!=command_ref.get('bytes') or hashlib.sha256(command_raw).hexdigest()!=command_ref['sha256']:raise ValueError('owned publication effective command changed')\n"
        + "command=json.loads(command_raw);verify_owned_publication_command(command,SPEC,_trainer_invocation)\n"
        + "kwargs=dict(producer=selected['producer'],dependency_producer=selected['dependency_producer'],namespace=selected['namespace'],readonly_input_roots=selected['readonly_input_roots'])\n"
        + 'known_generation=' + repr(known_generation) + '\n'
        + 'known_sha256=' + repr(known_sha256) + '\nknown_step=' + repr(known_step) + '\n'
        + 'if known_generation!=manifest.get("generation_sha256"):known_sha256=None;known_step=-1\n'
        + "chosen=observe_publication(SPEC,known_sha256=known_sha256,known_step=known_step)\n"
        + "again,again_ref,again_signature=verify_owner_pointer(POINTER,role=ROLE,production_root=PRODUCTION_ROOT,control_root=CONTROL_ROOT)\n"
        + "if again_signature!=pointer_signature or again_ref!=manifest_ref or again.get('generation_sha256')!=manifest.get('generation_sha256'):raise ValueError('owned publication owner changed while its checkpoint was verified')\n"
        + "print(json.dumps({'owner':{'generation_sha256':manifest['generation_sha256'],'manifest':manifest_ref},'chosen':chosen,'observations':{}}))\n"
    )
    return source
