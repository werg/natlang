"""Actual in-process native trainer dispatch; no synthetic child or old owner."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from ..common.jsonio import write_json_durable, write_bytes_durable
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


COMPLETION = 'natlang.native-inprocess-completion-contract/1'


def capture_completion_contract(launch, terminal_path, out):
    """Predeclare a reviewed wrapper's normal trainer.main-return marker.

    Call while its genuine optimizer owner lives, before the terminal exists.
    The caller must review the captured wrapper's normal-return marker semantics;
    this receipt does not assert an observed operating-system exit status.
    """
    source = json.loads(_bound(launch).read_text())
    owner = observe(source['pid'])
    if (owner is None or owner['state'] in ('Z', 'X')
            or not same_process(source.get('identity') or {}, owner)
            or process_argv(owner) != source.get('argv')):
        raise ValueError('completion contract requires the actual live captured owner')
    _, _, refs = verify({'owner': owner}, source)
    terminal = Path(terminal_path).resolve()
    if terminal != Path(refs['supervisor']['path']).parent / 'entry-terminal.json':
        raise ValueError('completion contract needs the reviewed wrapper terminal path')
    if terminal.exists() or terminal.with_name('entry-failed.json').exists():
        raise ValueError('completion contract must precede terminal or failure evidence')
    out = Path(out)
    if out.exists():
        raise ValueError('completion contract already exists; preserve its original capture')
    directory = out.parent / (out.stem + '-controls')
    directory.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    for name, reference in {'launch': launch, **refs}.items():
        target = directory / (name + '.json')
        write_bytes_durable(target, _bound(reference).read_bytes())
        if binding(target)['sha256'] != reference['sha256']:
            raise ValueError('completion control changed during capture')
        snapshots[name] = binding(target)
    again = observe(owner['pid'])
    if (not same_process(owner, again) or again['state'] in ('Z', 'X')
            or again['argv_hex'] != owner['argv_hex'] or terminal.exists()
            or terminal.with_name('entry-failed.json').exists()):
        raise ValueError('completion owner or terminal changed during capture')
    write_json_durable(out, {'schema': COMPLETION, 'at': time.time(),
        'owner': owner, 'source_launch': launch, 'controls': snapshots,
        'terminal_path': str(terminal), 'success_kind': 'normal-trainer-main-return',
        'qualification_credit': False})
    return binding(out)


def verify_completion_contract(reference, intent, terminal_reference, terminal):
    """Authenticate a genuine wrapper return without inventing a process exit."""
    contract = json.loads(_bound(reference).read_text())
    controls = contract.get('controls') or {}
    launch = json.loads(_bound(controls['launch']).read_text())
    verify(intent, launch, controls)
    owner = intent['owner']
    times = (launch.get('at'), contract.get('at'), terminal.get('at'))
    if (contract.get('schema') != COMPLETION
            or contract.get('success_kind') != 'normal-trainer-main-return'
            or contract.get('qualification_credit') is not False
            or contract.get('source_launch') != intent.get('source_launch', intent['launch'])
            or controls['launch']['sha256'] != intent['launch']['sha256']
            or not same_process(contract.get('owner') or {}, owner)
            or contract['owner'].get('argv_hex') != owner['argv_hex']
            or contract.get('terminal_path') != terminal_reference['path']
            or Path(contract['terminal_path']) != Path(json.loads(
                _bound(controls['intent']).read_text())['wrapper']['path']).parent / 'entry-terminal.json'
            or terminal.get('passed') is not True
            or type(terminal.get('raw_completion_credit')) is not int
            or terminal['raw_completion_credit'] != 0 or 'returncode' in terminal
            or any(type(value) not in (int, float) for value in times)
            or not times[0] <= times[1] <= times[2]):
        raise ValueError('in-process completion lost its predeclared successful-return authority')
    return contract
