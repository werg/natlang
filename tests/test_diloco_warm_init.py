import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from scripts.coreweave import diloco_warm_init as warm


def test_preparation_verifies_identity_mirrors_and_child_environment(tmp_path,monkeypatch):
    work=tmp_path/'work';root=tmp_path/'original';repo=root/'natlang';runtime=tmp_path/'runtime'
    target=work/'mirror'/str(root).lstrip('/')/'natlang';(target/'.coreweave-prebuilt').mkdir(parents=True)
    archive=target/'.coreweave-prebuilt/wheelhouse.tar.gz';archive.write_bytes(b'frozen wheelhouse')
    digest=hashlib.sha256(archive.read_bytes()).hexdigest();image='image@sha256:'+'a'*64
    (target/'.coreweave-prebuilt/wheelhouse.json').write_text(json.dumps({'kind':'wheelhouse','sha256':digest,'image':image}))
    (runtime/'pkg').mkdir(parents=True);(runtime/'pkg/code.py').write_text('bound runtime')
    config={'work':str(work),'repo':str(repo),'runtime':str(runtime),'mirror_roots':[str(root)],'image':image,
            'wheelhouse_sha256':digest,'runtime_file':'pkg/code.py','runtime_sha256':warm._sha(runtime/'pkg/code.py'),
            'ready_path':str(work/'ready.json'),'imports':['json']}
    calls=[];monkeypatch.setattr(warm.shutil,'which',lambda name:'/usr/bin/zstd')
    monkeypatch.setattr(warm.subprocess,'run',lambda argv,**kwargs:calls.append((argv,kwargs)))
    monkeypatch.setattr(warm.os,'environ',warm.os.environ.copy())
    receipt=warm.prepare(config)
    assert receipt['passed'] and root.resolve()==(work/'mirror'/str(root).lstrip('/')).resolve()
    assert len(calls)==2 and calls[1][1]['env']['PYTHONPATH'].startswith(str(runtime))
    assert json.loads((work/'ready.json').read_text())['wheelhouse_sha256']==digest
    archive.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='archive identity'):warm.prepare(config)


def test_create_compiles_before_allocation_and_persists_token(tmp_path):
    token=tmp_path/'token';calls=[]
    class CW:
        def throttled_create(self,fn,**kw):return fn()
        def sdk_auth(self):return 'auth'
        def keep_share_token(self,box):token.write_text('secret');return token
    def run(*args,**kwargs):calls.append((args,kwargs));return SimpleNamespace(sandbox_id='sandbox')
    image='image@sha256:'+'a'*64
    spec={'image':image,'snapshot_id':'snapshot','resources':{'gpu':{'count':1}},'max_lifetime_seconds':10,
          'volume_size':'10Gi','tags':[],'services':[{'endpoint':{'auth':'share_token'}}],
          'initialization':{'image':image,'runtime':'/work/runtime','runtime_file':'pkg/code.py',
                            'runtime_sha256':'b'*64,'wheelhouse_sha256':'c'*64}}
    with pytest.raises(SyntaxError):warm.create(spec,'this is not python !',tmp_path/'bad.json',sandbox_run=run,cwctl=CW())
    assert not calls and not (tmp_path/'bad.json').exists()
    _,value=warm.create(spec,'print("owner")',tmp_path/'intent.json',sandbox_run=run,cwctl=CW())
    assert value['phase']=='token-persisted' and token.read_text()=='secret'
    assert json.loads((tmp_path/'intent.json').read_text())['sandbox_id']=='sandbox'
    assert calls[0][0][:3]==('python','-u','-c')
    compile(calls[0][0][3],'<entrypoint>','exec')
    with pytest.raises(FileExistsError):warm.create(spec,'pass',tmp_path/'intent.json',sandbox_run=run,cwctl=CW())


def test_compiled_owner_initializes_before_supervisor(tmp_path):
    work=tmp_path/'work';root=tmp_path/'hostroot';repo=root/'natlang';runtime=tmp_path/'runtime'
    mirrored=work/'mirror'/str(root).lstrip('/')/'natlang'
    prebuilt=mirrored/'.coreweave-prebuilt';prebuilt.mkdir(parents=True)
    archive=prebuilt/'wheelhouse.tar.gz';archive.write_bytes(b'wheelhouse')
    digest=warm._sha(archive);image='image@sha256:'+'a'*64
    (prebuilt/'wheelhouse.json').write_text(json.dumps({'kind':'wheelhouse','sha256':digest,'image':image}))
    installer=mirrored/'scripts/coreweave/prebuilt.py';installer.parent.mkdir(parents=True);installer.write_text('pass\n')
    (runtime/'pkg').mkdir(parents=True);(runtime/'pkg/code.py').write_text('runtime')
    zstd=tmp_path/'zstd';zstd.write_text('#!/bin/sh\nexit 0\n');zstd.chmod(0o755)
    config=warm.defaults({'work':str(work),'repo':str(repo),'runtime':str(runtime),'mirror_roots':[str(root)],
        'image':image,'wheelhouse_sha256':digest,'runtime_file':'pkg/code.py','runtime_sha256':warm._sha(runtime/'pkg/code.py'),
        'imports':['json'],'input_wait_seconds':1})
    source="from pathlib import Path\nassert Path(%r).exists()\nprint('owner-started')\n" % config['ready_path']
    run=subprocess.run([sys.executable,'-c',warm.owner_program(source,config)],
                       env={**os.environ,'PATH':str(tmp_path)+':'+os.environ['PATH']},capture_output=True,text=True,timeout=15)
    assert run.returncode==0,run.stderr
    assert 'owner-started' in run.stdout
    assert json.loads(Path(config['ready_path']).read_text())['pid']>0


def test_readiness_requires_same_boot_and_actual_pid1_owner():
    init={'image':'image','wheelhouse_sha256':'wheel','runtime_sha256':'runtime',
          'runtime':'/work/runtime','runtime_file':'pkg/code.py'}
    spec={'image':'image','initialization':init};config=warm.defaults(init)
    receipt={'schema':'natlang.diloco-warm-initialization/1','passed':True,'boot_id':'boot','pid':1,
             'image':'image','wheelhouse_sha256':'wheel','runtime_sha256':'runtime',
             'runtime_file':'/work/runtime/pkg/code.py','mirror_roots':config['mirror_roots'],
             'environment':{'NATLANG_ROW_INDEX_CACHE':'/work/lfm3-row-index','HF_HUB_CACHE':'/work/hf',
                            'PYTHONPATH':'/work/runtime/training/neuralese:/work/runtime','PYTHONUNBUFFERED':'1'},
             'zstd':'/usr/bin/zstd','imports':['torch','transformers','safetensors','zstandard','natlang_neuralese.train.native_diloco']}
    owner={'schema':'natlang.diloco-role-supervisor/1','boot_id':'boot','supervisor_pid':1,
           'supervisor_start_ticks':'12','owner_id':'owner'}
    assert warm.verify_readiness(receipt,owner,spec,boot_id='boot',pid1_start_ticks='12')['ready']
    with pytest.raises(ValueError,match='owner'):warm.verify_readiness(receipt,{**owner,'supervisor_pid':2},spec,boot_id='boot',pid1_start_ticks='12')
    with pytest.raises(ValueError,match='receipt'):warm.verify_readiness(receipt,owner,spec,boot_id='new-boot',pid1_start_ticks='12')
    actual={'entrypoint_sha256':'program','boot_id':'boot','start_ticks':'12','receipt':receipt,'owner':owner}
    class Future:
        def result(self,timeout):return SimpleNamespace(returncode=0,stdout=json.dumps(actual))
    class Sandbox:
        def exec(self,*args,**kwargs):return Future()
    # A ready claim requires the observed entrypoint as well as the PID/boot receipt.
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        intent={'entrypoint_sha256':'program'}
        assert warm.wait_ready(Sandbox(),spec,Path(directory)/'intent.json',intent,timeout=1)['ready']
        assert intent['phase']=='ready'
