"""Actual in-process native trainer dispatch; no synthetic child or old owner."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from ..common.jsonio import write_json_durable
from ..train.native_readiness import _bound
from .native_raw_handoff import observe, process_argv, same_process, binding

SCHEMA = 'natlang.native-inprocess-trainer-dispatch/1'
MODULES = ('natlang_neuralese.train.native_state_sft', 'natlang_neuralese.train.native_preference')


def capture(wrapper, effective_argv, continuation, directory, *, environment):
    """Call in the actual wrapper before trainer.main, while this owner lives."""
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    owner = observe(os.getpid()); wrapper = binding(Path(wrapper).resolve(strict=True))
    if owner is None or wrapper['path'] not in process_argv(owner):
        raise ValueError('in-process dispatch needs its actual executing wrapper')
    _bound(continuation)
    command = list(effective_argv)
    if (len(command) < 4 or command[1:3] != ['-u', '-m'] or command[3] not in MODULES
            or not all(isinstance(item, str) for item in command)):
        raise ValueError('in-process dispatch needs an explicit existing native trainer command')
    value = {'schema': SCHEMA, 'at': time.time(), 'owner': owner, 'wrapper': wrapper,
        'effective_argv': command, 'continuation': continuation, 'env': dict(environment)}
    dispatch = directory / 'dispatch.json'
    if dispatch.exists():
        raise ValueError('in-process launch already captured; reconcile its actual owner')
    write_json_durable(dispatch, value)
    launch = directory / 'launch.json'
    write_json_durable(launch, {'at': time.time(), 'pid': owner['pid'], 'identity': owner,
        'argv': process_argv(owner), 'inprocess_dispatch': binding(dispatch)})
    return binding(launch)


def verify(captured, observed, snapshots=None):
    """Replay a genuine capture, optionally using its immutable control snapshots."""
    reference = observed['inprocess_dispatch']
    chosen = (snapshots or {}).get('intent', reference)
    if chosen['sha256'] != reference['sha256']:
        raise ValueError('in-process dispatch snapshot changed its captured authority')
    dispatch = json.loads(_bound(chosen).read_text())
    owner = captured['owner']; argv = process_argv(owner)
    wrapper = dispatch['wrapper']; selected = (snapshots or {}).get('supervisor', wrapper)
    continuation = dispatch['continuation']; parent_ref = (snapshots or {}).get('continuation', continuation)
    if (selected['sha256'] != wrapper['sha256'] or parent_ref['sha256'] != continuation['sha256']):
        raise ValueError('in-process dispatch changed its wrapper or continuation snapshot')
    _bound(selected)
    command = dispatch.get('effective_argv')
    if (dispatch.get('schema') != SCHEMA or not same_process(dispatch.get('owner') or {}, owner)
            or dispatch['owner'].get('argv_hex') != owner.get('argv_hex')
            or observed.get('argv') != argv or wrapper['path'] not in argv
            or not isinstance(command, list) or len(command) < 4
            or not all(isinstance(item, str) for item in command)
            or command[1:3] != ['-u', '-m'] or command[3] not in MODULES
            or not isinstance(dispatch.get('env'), dict)):
        raise ValueError('in-process dispatch lost its actual wrapper/trainer/process identity')
    return json.loads(_bound(parent_ref).read_text()), command, {
        'intent': reference, 'continuation': continuation, 'supervisor': wrapper}
