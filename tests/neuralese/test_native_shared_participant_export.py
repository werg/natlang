import hashlib
import json
from pathlib import Path
import sys

import pytest

from natlang_neuralese.model.native_shared_checkpoint import shared_manifest, shared_dependencies
from natlang_neuralese.model.native_bundle import verify_bundle
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from test_native_shared_checkpoint import actual_shared

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/coreweave'))
import native_role_exchange_io as transport
from native_shared_participant_export import (prepare_bundle, export_bundle_files,
    prepare_current_from_owner, export_with_runfile_exporter, INVENTORY_SCHEMA)


def test_selected_shared_participant_export_is_full_and_dynamically_custodied(tmp_path):
    policy,participants,channel,optimizer,adapter=actual_shared(tmp_path)
    manifest=shared_manifest(policy.slot)
    owner_state=tmp_path/'owner-state.json'
    adapter_path=policy.out/'semantic-adapter.json'
    for role in ('lfm','mellum'):
        checkpoint=tmp_path/'private'/role/'checkpoint.pt'
        owner_state.write_text(json.dumps({'checkpoint':str(policy.slot),
            'semantic_adapter':{'path':str(adapter_path),'sha256':hashlib.sha256(adapter_path.read_bytes()).hexdigest()}}))
        selected=prepare_current_from_owner(owner_state,role,checkpoint.parent)
        assert selected['owner_state']['checkpoint']['sha256']==selected['source_checkpoint']['sha256']
        verified=verify_bundle(selected['bundle_root'])
        assert verified['participant']==role and verified['semantic_adapter']
        restored=CheckpointPolicy.load(verified['checkpoint']['resolved'],shared_participant=role)
        assert set(restored['participants'])=={role}
        assert restored['training']['step']==2
        selected_dependencies=shared_dependencies(manifest,policy.slot,participant=role)
        assert set(selected_dependencies)=={'participant:'+role,'semantic','training'}

        inventory=tmp_path/'private'/role/'private-export-inventory.json'
        object_store={}
        def independent_export(source,identity):
            raw=Path(source).read_bytes()
            assert len(raw)==identity['bytes']
            assert hashlib.sha256(raw).hexdigest()==identity['sha256']
            object_store[identity['sha256']]=raw
            downloaded=object_store[identity['sha256']]
            assert hashlib.sha256(downloaded).hexdigest()==identity['sha256']
            assert len(downloaded)==identity['bytes']
            return {'private':True,'role':role,'source':source,'identity':dict(identity),
                'independent_verification':dict(identity),'run':'fixture/natlang/shared-closure',
                'name':identity['sha256']}
        custody=export_bundle_files(selected,inventory,role=role,export_file=independent_export)
        assert custody['all_bundle_files_independently_verified'] is True
        value=json.loads(inventory.read_text())
        assert value['schema']==INVENTORY_SCHEMA and value['role']==role
        assert {row['identity']['sha256'] for row in value['entries']}=={
            row['sha256'] for row in selected['files']}

        runtime=tmp_path/'runtime';runtime.mkdir(exist_ok=True)
        local_root=tmp_path/'private'
        provider={'sandboxes':{'cloud':{'runtime':str(runtime)}},'receipts':str(tmp_path/'receipts'),
            'local_source_host':{'source_host_id':'local','runtime':str(runtime),
                'python':sys.executable,'read_roots':[str(local_root)],'write_roots':[str(local_root)],
                'private_export_role':role,'private_export_inventory':str(inventory)}}
        io=transport.NativeRoleExchangeIO(provider)
        for entry in value['entries']:
            assert io._require_private_export({'path':entry['source_path'],**entry['identity']})
        unexported={'path':str(tmp_path/'new-large-file'),'sha256':'f'*64,'bytes':99}
        assert io._require_private_export(unexported) is False
        assert len(object_store)==len(selected['files'])


def test_dynamic_private_inventory_rejects_wrong_role_and_false_readback(tmp_path):
    root=tmp_path/'local';root.mkdir();runtime=tmp_path/'runtime';runtime.mkdir()
    source=root/'object';source.write_bytes(b'actual bytes')
    identity={'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'bytes':source.stat().st_size}
    inventory=root/'inventory.json'
    receipt={'private':True,'role':'lfm','source':str(source),'identity':identity,
        'independent_verification':{'sha256':'0'*64,'bytes':identity['bytes']},
        'run':'fixture/natlang','name':'object'}
    inventory.write_text(json.dumps({'schema':INVENTORY_SCHEMA,'role':'lfm','entries':[
        {'source_path':str(source),'identity':identity,'receipt':receipt}]}))
    io=transport.NativeRoleExchangeIO({'sandboxes':{'cloud':{'runtime':str(runtime)}},
        'receipts':str(tmp_path/'receipts'),'local_source_host':{'source_host_id':'local',
        'runtime':str(runtime),'python':sys.executable,'read_roots':[str(root)],
        'write_roots':[str(root)],'private_export_role':'mellum',
        'private_export_inventory':str(inventory)}})
    with pytest.raises(ValueError,match='another schema or owner'):
        io._require_private_export({'path':str(source),**identity})


def test_existing_runfile_exporter_cli_is_invoked_with_bound_local_bytes(tmp_path):
    policy,participants,channel,optimizer,adapter=actual_shared(tmp_path)
    adapter_path=policy.out/'semantic-adapter.json'
    selected=prepare_bundle(policy.slot,'lfm',tmp_path/'bundle',semantic_adapter=adapter_path)
    exporter=tmp_path/'fake-private-exporter.py'
    exporter.write_text('''#!/usr/bin/env python3
import hashlib,json,os,sys
from pathlib import Path
role,source=sys.argv[1:]
raw=Path(source).read_bytes();identity={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
assert identity=={'sha256':os.environ['CW_EXPORT_EXPECTED_SHA256'],'bytes':int(os.environ['CW_EXPORT_EXPECTED_BYTES'])}
receipt={'private':True,'role':role,'source':source,'identity':identity,
 'independent_verification':dict(identity),'run':'fixture/natlang','name':identity['sha256']}
Path(os.environ['CW_EXPORT_RECEIPT_FILE']).write_text(json.dumps(receipt))
''')
    exporter.chmod(0o755)
    inventory=tmp_path/'inventory.json'
    result=export_with_runfile_exporter(selected,inventory,role='lfm',
        exporter=exporter,source_mode='local')
    assert result['all_bundle_files_independently_verified'] is True
    assert len(result['entries'])==len(selected['files'])
    assert json.loads(inventory.read_text())['role']=='lfm'


def test_sandbox_shared_bundle_exports_exact_remote_members_without_local_tensor_copy(tmp_path):
    policy,participants,channel,optimizer,adapter=actual_shared(tmp_path)
    selected=prepare_bundle(policy.slot,'mellum',tmp_path/'bundle',
        semantic_adapter=policy.out/'semantic-adapter.json')
    remote_root='/work/native-production/control/native-bundle'
    selected['bundle_root']=remote_root
    for index,row in enumerate(selected['files']):
        row['path']=remote_root+('/bundle.json' if row['path'].endswith('/bundle.json')
            else f'/objects/{index:04d}-'+Path(row['path']).name)
    bundle_ref=next(row for row in selected['files']if row['path'].endswith('/bundle.json'))
    selected['bundle']={'path':bundle_ref['path'],'sha256':bundle_ref['sha256'],
        'bytes':bundle_ref['bytes']}
    exporter=tmp_path/'fake-remote-private-exporter.py'
    exporter.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
role,source=sys.argv[1:]
assert os.environ['CW_EXPORT_LOCAL']=='0' and os.environ['CW_EXPORT_SANDBOX_ID']=='mellum-box'
identity={'sha256':os.environ['CW_EXPORT_EXPECTED_SHA256'],'bytes':int(os.environ['CW_EXPORT_EXPECTED_BYTES'])}
Path(os.environ['CW_EXPORT_RECEIPT_FILE']).write_text(json.dumps({'private':True,'role':role,
 'source':source,'identity':identity,'independent_verification':dict(identity),
 'run':'fixture/natlang','name':identity['sha256']}))
''')
    exporter.chmod(0o755)
    inventory=tmp_path/'remote-inventory.json'
    result=export_with_runfile_exporter(selected,inventory,role='mellum',
        exporter=exporter,source_mode='sandbox',sandbox_id='mellum-box')
    assert result['all_bundle_files_independently_verified'] is True
    assert {row['source_path']for row in result['entries']}=={row['path']for row in selected['files']}
    assert all(not Path(row['source_path']).is_file() for row in result['entries'])
