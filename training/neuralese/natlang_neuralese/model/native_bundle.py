"""Portable immutable native checkpoint dependencies, with no identity rewriting.

HF model source files remain separately pinned by the native checkpoint. This
bundle transports the updated actor state, parameter/optimizer artifacts and
training RNG; the standard loader accepts an explicit identical HF base path.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from ..common.hashing import sha256_file_hex_cached
from ..common.jsonio import write_json_durable
from .native_composite import SCHEMA, resolve_composite_artifact

BUNDLE_SCHEMA = 'natlang.native-checkpoint-bundle/1'
ACTOR_VIEW_SCHEMA = 'natlang.native-actor-view/1'


def _file_binding(path):
    path = Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':sha256_file_hex_cached(path),'bytes':path.stat().st_size}


def write_actor_view(checkpoint, out, *, validated_state, decoded_source_stat=None):
    """Extract exact actor tensors from the exporter's already-validated decode.

    The caller must pass the actual state decoded from this immutable source,
    after its normal full-state validation. Unsupported source forms retain the
    existing full-checkpoint path by returning None, without writing a view.
    """
    from .native_model import CHECKPOINT_SCHEMA
    from ..train.checkpoint_policy import CheckpointPolicy
    from ..train.checkpoint_offload import _same_values
    from ..train.trajectory_state import atomic_checkpoint
    from .native_composite import _durable_immutable
    source = Path(checkpoint).resolve(strict=True)
    if validated_state.get('schema') != CHECKPOINT_SCHEMA:
        return None
    with source.open('rb') as stream:
        if stream.read(1) == b'{':
            return None  # Existing composite/shared closure remains the full actor path.
    if validated_state.get('optimizer_pruned') or validated_state.get('inference_only'):
        raise ValueError('actor view requires its complete resumable native source')
    training = validated_state.get('training') or {}
    if not all(training.get(key) for key in ('optimizer','rng','source_receipts')):
        raise ValueError('actor view source lacks its actual optimizer/RNG/source validation')
    before = source.stat()
    signature = (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
    if signature != decoded_source_stat:
        raise ValueError('actor view source is not the unchanged file used for the validated decode')
    source_binding = _file_binding(source)
    selected = {key:validated_state[key] for key in
                ('schema','model_config','backbone_state','interface_state','step')}
    if type(selected['step']) is not int or selected['step'] < 0:
        raise ValueError('actor view source has no actual saved integer step')
    selected.update(inference_only=True,optimizer_pruned=True,
        optimizer_pruned_keys=['training'],inference_source={
            'schema':'natlang.native-inference-source/1',
            'training_checkpoint_sha256':source_binding['sha256'],
            'training_checkpoint_bytes':source_binding['bytes'],'step':selected['step']})
    mechanism = _file_binding(__file__)
    out = Path(out).resolve()/source_binding['sha256']
    out.mkdir(parents=True,exist_ok=True)
    fd,name = tempfile.mkstemp(prefix='.actor-view-',suffix='.pt',dir=out)
    os.close(fd)
    pending = Path(name)
    try:
        atomic_checkpoint(pending,selected)
        written = CheckpointPolicy.load(pending)
        _same_values(selected,written)
        del written
        digest = sha256_file_hex_cached(pending)
        target = out/(digest+'.pt')
        try:os.link(pending,target)
        except FileExistsError:
            if sha256_file_hex_cached(target) != digest:
                raise ValueError('immutable actor view collision')
        directory = os.open(out,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        pending.unlink(missing_ok=True)
    after = source.stat()
    if (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) != signature:
        raise ValueError('actor view source changed during extraction')
    receipt = {'schema':ACTOR_VIEW_SCHEMA,'source':source_binding,'view':{
        **_file_binding(target),'path':target.name},'step':selected['step'],
        'model_config':selected['model_config'],'mechanism':mechanism,
        'verification':'safe reread and bitwise comparison of every selected value',
        'training_resume':False,'logical_identity':'original full source checkpoint'}
    raw = (json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode()
    import hashlib
    proof = out/(hashlib.sha256(raw).hexdigest()+'.json')
    _durable_immutable(proof,raw)
    return _file_binding(proof)


def load_actor_view(checkpoint, reference):
    """Verify an explicitly bound actor view while keeping full-source identity."""
    from ..train.checkpoint_policy import CheckpointPolicy
    from .native_model import CHECKPOINT_SCHEMA
    if not isinstance(reference,dict) or set(reference) != {'path','sha256','bytes'}:
        raise ValueError('actor view needs an explicit exact receipt binding')
    if type(reference['bytes']) is not int or reference['bytes'] <= 0:
        raise ValueError('actor view receipt requires its exact integer byte count')
    proof = Path(reference['path']).resolve(strict=True)
    if _file_binding(proof) != reference:
        raise ValueError('actor view receipt binding changed')
    receipt = json.loads(proof.read_text())
    if (receipt.get('schema') != ACTOR_VIEW_SCHEMA or receipt.get('training_resume') is not False
            or receipt.get('logical_identity') != 'original full source checkpoint'
            or type(receipt.get('step')) is not int
            or type(receipt.get('source',{}).get('bytes')) is not int
            or type(receipt.get('view',{}).get('bytes')) is not int):
        raise ValueError('actor view receipt is not a read-only full-source binding')
    source = _file_binding(checkpoint)
    if any(source[key] != receipt['source'].get(key) for key in ('sha256','bytes')):
        raise ValueError('actor view belongs to another complete checkpoint')
    view = resolve_composite_artifact(receipt['view'],proof,sibling=True)
    if view.stat().st_size != receipt['view']['bytes']:
        raise ValueError('actor view byte length differs from receipt')
    state = CheckpointPolicy.load(view)
    if (state.get('schema') != CHECKPOINT_SCHEMA or state.get('inference_only') is not True
            or state.get('optimizer_pruned') is not True or 'training' in state
            or state.get('model_config') != receipt['model_config']
            or state.get('step') != receipt['step']
            or state.get('inference_source') != {
                'schema':'natlang.native-inference-source/1',
                'training_checkpoint_sha256':source['sha256'],
                'training_checkpoint_bytes':source['bytes'],'step':receipt['step']}):
        raise ValueError('actor view payload differs from its exact source receipt')
    state['actor_view'] = {'receipt':reference,'view':_file_binding(view),'source':source,
                           'training_resume':False}
    return state


def actor_view_dependencies(reference):
    """Exact explicitly consumed proof/view closure; full source stays separate."""
    proof = Path(reference['path']).resolve(strict=True)
    if _file_binding(proof) != reference:
        raise ValueError('actor view receipt binding changed')
    receipt = json.loads(proof.read_text())
    state = load_actor_view(receipt['source']['path'],reference)
    return [reference,state['actor_view']['view'],state['actor_view']['source']]


def composite_dependencies(checkpoint, *, participant=None, semantic_only=False, checkpoint_loader=None):
    """Verify every parent/parameter/optimizer/training role without loading weights."""
    current = Path(checkpoint).resolve(strict=True)
    result, seen = {}, set()
    while current not in seen:
        seen.add(current)
        digest = sha256_file_hex_cached(current)
        result[digest] = current
        with current.open('rb') as stream:
            prefix = stream.read(1)
        if prefix != b'{':
            from .native_shared_checkpoint import shared_manifest, shared_dependencies
            manifest = shared_manifest(current,load=checkpoint_loader)
            if manifest is not None:
                for source in shared_dependencies(manifest, current, participant=participant,
                                                  semantic_only=semantic_only).values():
                    result[sha256_file_hex_cached(source)] = source
            return result
        manifest = json.loads(current.read_text())
        if manifest.get('schema') != SCHEMA:
            raise ValueError('native bundle JSON checkpoint must be an exact native composite')
        for role in ('parameter_artifact','optimizer_state_artifact','training_state'):
            binding = manifest.get(role)
            if binding is not None:
                artifact = resolve_composite_artifact(binding,current,sibling=True)
                result[binding['sha256']] = artifact
        selected = manifest['base_checkpoint'].get('participant')
        if selected is not None:
            if participant is not None and participant != selected:
                raise ValueError('native composite selected participant differs from requested worker')
            participant = selected
        current = resolve_composite_artifact(manifest['base_checkpoint'],current)
    raise ValueError('native bundle checkpoint ancestry contains a cycle')


def _copy_immutable(source,destination,digest):
    """Bounded-memory durable transport copy; never hardlink a mutable source."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if sha256_file_hex_cached(destination) != digest:
            raise ValueError('native bundle immutable destination contains different bytes')
        return
    fd,pending = tempfile.mkstemp(prefix='.native-bundle-',dir=destination.parent)
    try:
        with os.fdopen(fd,'wb') as target,Path(source).open('rb') as original:
            shutil.copyfileobj(original,target,length=8*1024*1024)
            target.flush();os.fsync(target.fileno())
        if sha256_file_hex_cached(pending) != digest:
            raise ValueError('native bundle source changed while copying')
        try:
            os.link(pending,destination)
        except FileExistsError:
            if sha256_file_hex_cached(destination) != digest:
                raise ValueError('native bundle immutable destination collision')
        descriptor = os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:
        Path(pending).unlink(missing_ok=True)


def _semantic_dependencies(adapter,checkpoint_sha256):
    adapter = Path(adapter).resolve(strict=True)
    receipt = json.loads(adapter.read_text())
    if (receipt.get('schema') != 'natlang.native-semantic-adapter/1' or not receipt.get('models')
            or receipt['checkpoint']['sha256'] != checkpoint_sha256
            or set(receipt.get('tokenizers',{})) != set(receipt['models'])
            or any(value['native_identity']['checkpoint_sha256'] != checkpoint_sha256 for value in receipt['models'].values())):
        raise ValueError('portable semantic adapter must bind the exact bundled participants/checkpoint')
    resolve_composite_artifact(receipt['checkpoint'],adapter)
    result = {}
    for binding in receipt['tokenizers'].values():
        directory = (adapter.parent/binding['path']).resolve(strict=True)
        if adapter.parent not in directory.parents:
            raise ValueError('semantic tokenizer escaped its exact export directory')
        for filename,digest in binding['files'].items():
            source = (directory/filename).resolve(strict=True)
            if directory not in source.parents or sha256_file_hex_cached(source) != digest:
                raise ValueError('semantic tokenizer content differs from its adapter export')
            logical = str(source.relative_to(adapter.parent))
            result[logical] = (source,digest)
    return result


def export_bundle(checkpoint,out,*,semantic_adapter=None,participant=None):
    checkpoint,out = Path(checkpoint).resolve(strict=True),Path(out).resolve()
    dependencies = composite_dependencies(checkpoint, participant=participant)
    digest = sha256_file_hex_cached(checkpoint)
    out.mkdir(parents=True,exist_ok=True)
    with checkpoint.open('rb') as stream:
        filename = 'checkpoint.json' if stream.read(1) == b'{' else 'objects/'+digest
    files = []
    for expected,source in sorted(dependencies.items()):
        target = out/filename if expected == digest else out/'objects'/expected
        _copy_immutable(source,target,expected)
        if target != out/filename:
            files.append({'path':str(target.relative_to(out)),'sha256':expected,'bytes':target.stat().st_size})
    entry = out/filename
    _copy_immutable(checkpoint,entry,digest)
    semantic_binding,semantic_files = None,[]
    if semantic_adapter is not None:
        adapter = Path(semantic_adapter).resolve(strict=True)
        for logical,(source,expected) in sorted(_semantic_dependencies(adapter,digest).items()):
            _copy_immutable(source,out/logical,expected)
            semantic_files.append({'path':logical,'sha256':expected,'bytes':(out/logical).stat().st_size})
        adapter_sha = sha256_file_hex_cached(adapter)
        _copy_immutable(adapter,out/'semantic-adapter.json',adapter_sha)
        semantic_binding = {'path':'semantic-adapter.json','sha256':adapter_sha}
    receipt = {'schema':BUNDLE_SCHEMA,'semantic_adapter':semantic_binding,'semantic_files':semantic_files,
        'participant':participant,
        'checkpoint':{'path':filename,'sha256':digest},
        'files':files,'hf_base':'separately pinned; standard loader --base may relocate identical files',
        'identity_rewritten':False}
    write_json_durable(out/'bundle.json',receipt)
    return verify_bundle(out)


def verify_bundle(path):
    root = Path(path).resolve(strict=True)
    receipt = json.loads((root/'bundle.json').read_text())
    if receipt.get('schema') != BUNDLE_SCHEMA or receipt.get('identity_rewritten') is not False:
        raise ValueError('unsupported native checkpoint bundle')
    native_entries = [receipt['checkpoint'],*receipt['files']]
    entries = [*native_entries,*receipt.get('semantic_files',[]),
               *([receipt['semantic_adapter']] if receipt.get('semantic_adapter') else [])]
    seen = set()
    for binding in entries:
        artifact = (root/binding['path']).resolve(strict=True)
        if root not in artifact.parents or str(artifact) in seen or sha256_file_hex_cached(artifact) != binding['sha256']:
            raise ValueError('native bundle inventory changed, repeated or escaped its root')
        if 'bytes' in binding and artifact.stat().st_size != binding['bytes']:
            raise ValueError('native bundle artifact length changed')
        seen.add(str(artifact))
    checkpoint = root/receipt['checkpoint']['path']
    # Verify declared closure uses supplied objects even if original absolute
    # paths happen to remain available on this machine.
    actual = composite_dependencies(checkpoint, participant=receipt.get('participant'))
    declared = {value['sha256'] for value in native_entries}
    if set(actual) != declared:
        raise ValueError('native bundle inventory differs from complete checkpoint dependency closure')
    semantic = None
    if receipt.get('semantic_adapter'):
        adapter = root/receipt['semantic_adapter']['path']
        dependencies = _semantic_dependencies(adapter,receipt['checkpoint']['sha256'])
        declared_semantic = {value['path']:value['sha256'] for value in receipt.get('semantic_files',[])}
        if {key:value[1] for key,value in dependencies.items()} != declared_semantic:
            raise ValueError('native bundle semantic inventory differs from its exact tokenizer closure')
        semantic = {**receipt['semantic_adapter'],'resolved':str(adapter)}
    elif receipt.get('semantic_files'):
        raise ValueError('native bundle tokenizer files require their semantic adapter binding')
    return {**receipt,'checkpoint':{**receipt['checkpoint'],'resolved':str(checkpoint)},'semantic_adapter':semantic}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action',required=True)
    export = commands.add_parser('export');export.add_argument('--checkpoint',required=True);export.add_argument('--out',required=True);export.add_argument('--semantic-adapter');export.add_argument('--participant')
    verify = commands.add_parser('verify');verify.add_argument('--bundle',required=True)
    args = parser.parse_args()
    print(json.dumps(export_bundle(args.checkpoint,args.out,semantic_adapter=args.semantic_adapter,participant=args.participant) if args.action=='export' else verify_bundle(args.bundle),sort_keys=True))


if __name__ == '__main__':
    main()
