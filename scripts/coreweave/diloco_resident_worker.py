#!/usr/bin/env python3
"""Host-side immutable data staging and CAS publication for a native resident learner."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.common.jsonio import write_json_durable


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def install(staged, target, expected, size):
    staged, target = Path(staged), Path(target)
    if staged.stat().st_size != size or sha(staged) != expected:
        raise ValueError('staged content differs from declared size or hash')
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(staged, target)  # no clobber: target is immutable and has never been read yet
    except FileExistsError:
        if target.stat().st_size != size or sha(target) != expected:
            raise ValueError('existing immutable target differs: ' + str(target))
    finally:
        staged.unlink(missing_ok=True)
    return {'path': str(target), 'sha256': expected, 'bytes': size}


def require_owner(directory):
    initial = json.loads((Path(directory) / 'initial.json').read_text())
    if not resident.owner_alive(initial.get('owner') or {}):
        raise ValueError('resident optimizer owner is unavailable; resume the worker before publishing')
    return initial


def prepare(directory, recipe, stage, tokenizer, roles, geometry=None):
    require_owner(directory)
    evidence = json.loads(Path(geometry).read_text()) if geometry else None
    request = resident.prepare(directory, recipe, stage, tokenizer_fingerprint=tokenizer,
                               role_ids=roles, geometry_evidence=evidence, publish=False)
    value = json.loads(resident.bound(request).read_text())
    payload = value['payload']
    ce = payload['plan']['stage_inputs'][stage].get('ce_supervision_policy')
    return {'request': request, 'generation': value['generation'],
            'source_identity': {'train_sha256': payload['train']['sha256'], 'held_sha256': payload['held']['sha256'],
                                'recipe_sha256': payload['recipe']['sha256'],
                                'ce_sha256': ce['sha256'] if ce else None, 'stage': stage,
                                'tokenizer': tokenizer, 'roles': roles}}


def publish(directory, request):
    directory = Path(directory)
    with resident.locked(directory):
        require_owner(directory)
        path = resident.bound(request)
        value = json.loads(path.read_text())
        if value.get('schema') != resident.SCHEMA or resident.digest(value['payload']) != value['generation']:
            raise ValueError('invalid resident request')
        state = resident.current(directory)
        predecessor = state['generation'] if state else resident.digest(json.loads(
            (directory.parent / 'recipe-plan.json').read_text()))
        if value['payload']['predecessor'] != predecessor:
            raise ValueError('resident predecessor advanced before publish')
        pointer = directory / 'pending.json'
        if pointer.exists():
            previous = json.loads(pointer.read_text())
            if previous != request:
                previous_value = json.loads(resident.bound(previous).read_text())
                if not state or previous_value.get('generation') != state['generation']:
                    raise ValueError('another resident request is pending; do not overwrite it')
        write_json_durable(pointer, request)
        return {'request': request, 'generation': value['generation'], 'predecessor': predecessor}


def status(directory):
    directory = Path(directory)
    with resident.locked(directory):
        state = resident.current(directory)
        pending = directory / 'pending.json'
        pending_ref = json.loads(pending.read_text()) if pending.exists() else None
        pending_generation = (json.loads(resident.bound(pending_ref).read_text())['generation'] if pending_ref else None)
        rejected = []
        if (directory / 'rejections').exists():
            rejected = [json.loads(p.read_text()) for p in sorted((directory / 'rejections').glob('*.json'))]
        initial = json.loads((directory / 'initial.json').read_text()) if (directory / 'initial.json').exists() else {}
        payload = state['payload'] if state else {}
        return {'owner_alive': resident.owner_alive(initial.get('owner') or {}),
                'active': ({'train': payload['train'], 'held': payload['held'], 'recipe': payload['recipe'],
                            'geometry_qualification': payload.get('geometry_qualification'),
                            'ce_supervision_policy': payload['plan']['stage_inputs'][payload['stage_id']].get('ce_supervision_policy')}
                           if state else None),
                'generation': state['generation'] if state else None,
                'at_global_step': state.get('at_global_step') if state else None,
                'pending': pending_ref, 'pending_generation': pending_generation,
                'publication_state': ('adopted' if state and pending_generation == state['generation'] else
                                      'pending' if pending_ref else 'idle'), 'rejections': rejected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    cmd = sub.add_parser('install')
    cmd.add_argument('staged'); cmd.add_argument('target'); cmd.add_argument('sha256'); cmd.add_argument('size', type=int)
    cmd = sub.add_parser('prepare')
    cmd.add_argument('directory'); cmd.add_argument('recipe'); cmd.add_argument('stage'); cmd.add_argument('tokenizer')
    cmd.add_argument('roles', nargs=4, type=int); cmd.add_argument('--geometry')
    cmd = sub.add_parser('publish'); cmd.add_argument('directory'); cmd.add_argument('request')
    cmd = sub.add_parser('status'); cmd.add_argument('directory')
    args = parser.parse_args()
    if args.action == 'install': result = install(args.staged, args.target, args.sha256, args.size)
    elif args.action == 'prepare': result = prepare(args.directory, args.recipe, args.stage, args.tokenizer, args.roles, args.geometry)
    elif args.action == 'publish': result = publish(args.directory, json.loads(args.request))
    else: result = status(args.directory)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
