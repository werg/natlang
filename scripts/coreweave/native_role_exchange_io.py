"""Bounded local-DGX plus CoreWeave source-host transport for role exchange.

The semantic consumer remains the existing cloud host. One source owner may
live on the DGX; local Python runs against its explicitly frozen runtime, while
cross-host artifacts use the SDK's streaming file API and exact SHA/size
verification. No cloud credential is exposed to the local child or vice versa.
"""
from __future__ import annotations

import hashlib
import json
import os
import contextlib
from pathlib import Path
import shutil
import subprocess
import uuid

from natlang_neuralese.common.jsonio import write_bytes_durable
from provider_worker import SDKIO
from provider_bridge import checked_binding

MAX_LOCAL_METADATA=16*1024**2


class NativeRoleExchangeIO:
    def __init__(self,provider):
        if not isinstance(provider,dict)or not isinstance(provider.get('sandboxes'),dict):
            raise ValueError('role exchange provider needs its existing cloud sandbox allowlist')
        self.config=provider;self.sdk=SDKIO(provider)
        local=provider.get('local_source_host')
        if local is not None:
            required={'source_host_id','runtime','python','read_roots','write_roots'}
            if (not isinstance(local,dict)or set(local)-required-{'role_controls','reserve_bytes','cpu_helpers','cpu_helper_loader','private_exports','private_export_inventory','private_export_role','readonly_input_roots','exporter_python'}
                    or not required<=set(local)
                    or not all(isinstance(local.get(key),str)and local[key] for key in
                        ('source_host_id','runtime','python'))
                    or not Path(local['runtime']).is_absolute()or not Path(local['python']).is_absolute()
                    or not Path(local['runtime']).is_dir()or not Path(local['python']).is_file()):
                raise ValueError('local source host must bind its exact runtime, interpreter and path roots')
            if ('exporter_python' in local and (not isinstance(local['exporter_python'],str)
                    or not Path(local['exporter_python']).is_absolute()
                    or not Path(local['exporter_python']).is_file())):
                raise ValueError('local shared-bundle exporter must bind its exact SDK interpreter')
            for name in ('read_roots','write_roots'):
                roots=local[name]
                if not isinstance(roots,list)or not roots or any(not isinstance(root,str)
                        or not Path(root).is_absolute()or '..'in Path(root).parts for root in roots):
                    raise ValueError('local source host path roots must be explicit absolute directories')
            if ('reserve_bytes'in local and(type(local['reserve_bytes'])is not int or local['reserve_bytes']<0)):
                raise ValueError('local source host disk reserve must be a nonnegative byte count')
            if 'private_exports'in local and not isinstance(local['private_exports'],list):
                raise ValueError('local source host private export inventory must be a list')
            if 'private_export_inventory'in local:
                inventory=Path(local['private_export_inventory'])
                if (not inventory.is_absolute()or'..'in inventory.parts or str(inventory)!=os.path.normpath(str(inventory))
                        or (inventory.exists()and inventory.is_symlink())):
                    raise ValueError('dynamic private export inventory must be an absolute canonical nonsymlink path')
                if not any(inventory==Path(root)or Path(root)in inventory.parents for root in local['read_roots']):
                    raise ValueError('dynamic private export inventory must be under a declared local read root')
            if 'private_export_role'in local and local['private_export_role']not in ('lfm','mellum'):
                raise ValueError('local private export inventory role is unsupported')
            if local['source_host_id'] in provider['sandboxes']:
                raise ValueError('local source host identity collides with a CoreWeave sandbox')
        self.local=local

    def has_host(self,sid):
        return sid in self.config['sandboxes']or(self.local is not None and sid==self.local['source_host_id'])

    def _local_path(self,path,*,write=False):
        if self.local is None:raise ValueError('no local source host is configured')
        candidate=Path(path)
        if not candidate.is_absolute()or '..'in candidate.parts:raise ValueError('local source path is not absolute/canonical')
        cursor=candidate
        while cursor!=cursor.parent:
            if cursor.is_symlink():raise ValueError('local source paths may not traverse symlinks')
            cursor=cursor.parent
        roots=self.local['write_roots' if write else 'read_roots']
        resolved=candidate.resolve(strict=False)
        for root in roots:
            root_path=Path(root).resolve(strict=False)
            if resolved==root_path or root_path in resolved.parents:return resolved
        raise ValueError('local source path escaped its declared roots')

    def _local_file(self,path,*,optional=False):
        selected=self._local_path(path)
        if not selected.exists():
            if optional:return None
            raise FileNotFoundError(str(selected))
        if selected.is_symlink()or not selected.is_file():raise ValueError('local source artifact is not a regular file')
        size=selected.stat().st_size
        if size>MAX_LOCAL_METADATA:raise ValueError('local role metadata exceeds its bounded read limit')
        raw=selected.read_bytes()
        after=selected.stat()
        if len(raw)!=size or after.st_size!=size:raise ValueError('local source metadata changed during read')
        return raw

    def remote(self,sid,code,*,timeout=120):
        if self.local is None or sid!=self.local['source_host_id']:
            return self.sdk.remote(sid,code,timeout=timeout)
        runtime=self.local['runtime'];python=self.local['python']
        env=dict(os.environ);env['NATLANG_REPO']=runtime
        env['PYTHONPATH']=runtime+'/training/neuralese'+os.pathsep+runtime
        prefix='import sys;sys.path[:0]='+repr([runtime+'/training/neuralese',runtime])+';'
        try:
            result=subprocess.run([python,'-c',prefix+code],cwd=runtime,env=env,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,check=False)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError('local source-host metadata command timed out')from error
        if result.returncode:
            raise RuntimeError('local source-host metadata command failed: '+str(result.returncode))
        output=result.stdout.decode('utf-8')
        # Keep the local adapter's value contract identical to SDKIO.remote:
        # source-host expressions print JSON, and the SDK adapter returns the
        # decoded value rather than its serialized stdout representation.
        return json.loads(output)if output.strip()else None

    def source_host_controls(self,sid):
        if sid in self.config['sandboxes']:return self.config['sandboxes'][sid].get('role_controls')
        if self.local is not None and sid==self.local['source_host_id']:return self.local.get('role_controls')
        raise ValueError('role controller requested an unknown source host')

    def dependency_code(self,sid):
        if sid in self.config['sandboxes']:return self.sdk.dependency_code(sid)
        if self.local is None or sid!=self.local['source_host_id']:
            raise ValueError('provider dependency closure requested an unknown source host')
        runtime=self.local['runtime'];path=Path(__file__).with_name('provider_dependencies.py')
        raw=path.read_text()
        code=('import sys,json;sys.path.insert(0,'+repr(runtime+'/training/neuralese')+');'
            'exec(compile('+repr(raw)+','+repr(str(path))+',"exec"));')
        if self.local.get('cpu_helpers')or self.local.get('cpu_helper_loader'):
            if not self.local.get('cpu_helpers')or not self.local.get('cpu_helper_loader'):
                raise ValueError('local CPU composition needs both loader and manifest')
            from provider_bridge import checked_binding
            loader=checked_binding(self.local['cpu_helper_loader']);modules=checked_binding(self.local['cpu_helpers'])
            code+=('import pathlib,hashlib;p=pathlib.Path('+repr(loader['path'])+');raw=p.read_bytes();'
                'assert len(raw)=='+str(loader['bytes'])+' and hashlib.sha256(raw).hexdigest()=='+repr(loader['sha256'])+';'
                'namespace={"__file__":str(p),"__name__":"provider_cpu_loader"};'
                'exec(compile(raw,str(p),"exec"),namespace);namespace["load"]('+repr(modules)+');')
        return code

    def read(self,sid,path):
        if self.local is not None and sid==self.local['source_host_id']:
            return self._local_file(path)
        return self.sdk.read(sid,path)

    def read_optional(self,sid,path):
        if self.local is not None and sid==self.local['source_host_id']:
            return self._local_file(path,optional=True)
        return self.sdk.read_optional(sid,path)

    def write_atomic(self,sid,path,data):
        if self.local is not None and sid==self.local['source_host_id']:
            target=self._local_path(path,write=True)
            target.parent.mkdir(parents=True,exist_ok=True)
            write_bytes_durable(target,bytes(data))
            if target.read_bytes()!=bytes(data):raise ValueError('local metadata write changed its bytes')
            return
        return self.sdk.write_atomic(sid,path,data)

    def dependency_closure(self,sid,request):
        if self.local is None or sid!=self.local['source_host_id']:
            return self.sdk.dependency_closure(sid,request)
        roots=self.local.get('readonly_input_roots',self.local['read_roots'])
        code=(self.dependency_code(sid)+'print(json.dumps(dependency_closure('+repr(request)
            +',readonly_input_roots='+repr(roots)+')))')
        return json.loads(self.remote(sid,code,timeout=900))

    def verify_assets(self,sid,assets):
        if self.local is None or sid!=self.local['source_host_id']:
            return self.sdk.verify_assets(sid,assets)
        roots=self.local.get('readonly_input_roots',self.local['read_roots'])
        code=self.dependency_code(sid)+'verify_assets('+repr(assets)+',readonly_input_roots='+repr(roots)+')'
        code+='\nimport subprocess,shutil\nnode=shutil.which("node")\n'
        code+='if not node: raise RuntimeError("local task runtime needs its declared node executable")\n'
        code+='\nfor asset in '+repr(assets)+':\n'
        code+=(' if asset["kind"]=="task-runtime":\n'
            '  result=subprocess.run([node,asset["path"]+"/dist/teacher/cli.js","--help"],capture_output=True,timeout=30)\n'
            '  if result.returncode: raise RuntimeError("local compiled task CLI/npm import preflight failed")\n')
        self.remote(sid,code,timeout=900)

    def _cloud_has(self,sid,item):
        runtime=self.config['sandboxes'][sid]['runtime']
        code=('import pathlib,json,sys;sys.path.insert(0,'+repr(runtime+'/training/neuralese')+');'
            'from natlang_neuralese.common.hashing import sha256_file_hex_cached;'
            'p=pathlib.Path('+repr(item['path'])+');print(json.dumps(p.is_file() and '
            'p.stat().st_size=='+str(item['bytes'])+' and sha256_file_hex_cached(p)=='+repr(item['sha256'])+'))')
        return self.sdk.remote(sid,code,timeout=600)

    def _cloud_target_state(self,sid,item):
        runtime=self.config['sandboxes'][sid]['runtime']
        code=('import pathlib,json,sys;sys.path.insert(0,'+repr(runtime+'/training/neuralese')+');'
            'from natlang_neuralese.common.hashing import sha256_file_hex_cached;'
            'p=pathlib.Path('+repr(item['path'])+');state=("missing" if not p.exists() else '
            '("match" if p.is_file() and p.stat().st_size=='+str(item['bytes'])+
            ' and sha256_file_hex_cached(p)=='+repr(item['sha256'])+' else "different"));'
            'print(json.dumps(state))')
        return json.loads(self.sdk.remote(sid,code,timeout=600))

    def _local_has(self,item):
        path=self._local_path(item['path']);
        if not path.is_file()or path.is_symlink()or path.stat().st_size!=item['bytes']:return False
        h=hashlib.sha256()
        with path.open('rb')as stream:
            while block:=stream.read(8*1024**2):h.update(block)
        return h.hexdigest()==item['sha256']

    def _stream_local_to_cloud(self,source,target,item,*,target_path=None):
        local_path=self._local_path(item['path']);before=local_path.stat()
        if before.st_size!=item['bytes']or self._file_sha(local_path)!=item['sha256']:
            raise ValueError('local source bytes differ from the exact role inventory')
        if item['bytes']>=32*1024**2 and not self._require_private_export(item):return False
        target_path=target_path or item['path']
        target_host=self.config['sandboxes'].get(target,{})
        write_roots=target_host.get('write_roots')or ['/work/native-production']
        read_roots=[*target_host.get('read_roots',[]),*target_host.get('readonly_input_roots',[])]
        roots=[*write_roots,*read_roots]
        if not Path(target_path).is_absolute()or '..'in Path(target_path).parts \
                or not any(Path(target_path)==Path(root)or Path(root)in Path(target_path).parents for root in roots):
            raise ValueError('cloud staged role target escaped configured readable/writable roots')
        remote_item={**item,'path':target_path}
        target_state=self._cloud_target_state(target,remote_item)
        if target_state=='match':
            after=local_path.stat()
            if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                raise ValueError('local source file changed while verifying existing cloud target')
            return
        if target_state!='missing':
            raise ValueError('cloud prepared target already exists with different bytes')
        if not any(Path(target_path)==Path(root)or Path(root)in Path(target_path).parents for root in write_roots):
            raise ValueError('missing cloud prepared target is not under a writable root')
        temporary=target_path+'.local-transfer-'+uuid.uuid4().hex
        with local_path.open('rb')as stream:
            self.sdk.sandbox(target).write_file_streaming(temporary,stream,timeout_seconds=3600).result(timeout=3660)
        runtime=self.config['sandboxes'][target]['runtime']
        code=('import pathlib,os,json,sys;sys.path.insert(0,'+repr(runtime+'/training/neuralese')+');'
            'from natlang_neuralese.common.hashing import sha256_file_hex_cached;'
            'p=pathlib.Path('+repr(temporary)+');assert p.is_file() and p.stat().st_size=='+str(item['bytes'])+
            ' and sha256_file_hex_cached(p)=='+repr(item['sha256'])+');'
            'os.replace('+repr(temporary)+','+repr(target_path)+');print(json.dumps(True))')
        self.sdk.remote(target,code,timeout=600)
        after=local_path.stat()
        if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(
                after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('local source file changed during cross-host streaming')

    def stage_prepared_files(self,entries,*,source_host_id):
        """Stage a complete prepared-file inventory before either owner is launched."""
        if self.local is None or source_host_id!=self.local['source_host_id']:
            raise ValueError('prepared files must stream from the explicitly configured local source host')
        staged=[]
        for entry in entries:
            if not isinstance(entry,dict)or not {'source_host_id','target_host_id','target','source'}<=set(entry):
                raise ValueError('prepared file entry omitted its exact source host, binding or target')
            if entry['source_host_id']!=source_host_id:
                raise ValueError('prepared file source host changed from the declared local owner')
            target_host=entry['target_host_id']
            if not self.has_host(target_host):raise ValueError('prepared file targets an unknown source host')
            source=checked_binding(entry['source'])
            if target_host==source_host_id:
                local_source=self._local_path(source['path'])
                local_target=self._local_path(entry['target'],write=True)
                if local_source==local_target:
                    if not self._local_has(source):raise ValueError('local prepared source differs from its binding')
                elif local_target.is_file():
                    if local_target.is_symlink()or local_target.stat().st_size!=source['bytes'] \
                            or self._file_sha(local_target)!=source['sha256']:
                        raise ValueError('local prepared target already exists with different bytes')
                else:
                    local_target.parent.mkdir(parents=True,exist_ok=True)
                    temp=local_target.with_name(local_target.name+'.stage-'+uuid.uuid4().hex)
                    try:
                        digest=hashlib.sha256();size=0;before=local_source.stat()
                        with local_source.open('rb')as inp,temp.open('xb')as out:
                            while block:=inp.read(8*1024**2):out.write(block);digest.update(block);size+=len(block)
                            out.flush();os.fsync(out.fileno())
                        after=local_source.stat()
                        if (size!=source['bytes']or digest.hexdigest()!=source['sha256']
                                or (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=
                                (after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)):
                            raise ValueError('local prepared source changed during atomic staging')
                        os.replace(temp,local_target)
                    finally:
                        if temp.exists():temp.unlink()
            else:
                copied=self._stream_local_to_cloud(source_host_id,target_host,source,target_path=entry['target'])
                if copied is False:return False
            staged.append({'source_host_id':source_host_id,'target_host_id':target_host,
                'target':entry['target'],'sha256':source['sha256'],'bytes':source['bytes']})
        return staged

    def _require_private_export(self,item):
        from private_export_guard import validate_receipt
        for entry in self.local.get('private_exports',[]):
            identity=entry.get('identity')or{}
            if identity!={'sha256':item['sha256'],'bytes':item['bytes']}:continue
            ref=entry.get('receipt')
            if not isinstance(ref,dict)or set(ref)-{'path','sha256','bytes'}or not {'path','sha256'}<=set(ref):
                raise ValueError('large local role transfer has malformed independent private receipt')
            path=self._local_path(ref['path']);raw=self._local_file(path)
            if (hashlib.sha256(raw).hexdigest()!=ref['sha256']
                    or('bytes'in ref and len(raw)!=ref['bytes'])):
                raise ValueError('large local role transfer private receipt bytes changed')
            validate_receipt(json.loads(raw),identity)
            return True
        inventory_reference=self.local.get('private_export_inventory')
        if inventory_reference is not None:
            inventory_path=self._local_path(inventory_reference)
            if not inventory_path.exists():return False
            raw=self._local_file(inventory_path)
            value=__import__('json').loads(raw)
            if (value.get('schema')!='natlang.native-private-export-inventory/1'
                    or value.get('role')!=self.local.get('private_export_role')
                    or not isinstance(value.get('entries'),list)):
                raise ValueError('dynamic private export inventory has another schema or owner')
            for entry in value['entries']:
                if (not isinstance(entry,dict)or not isinstance(entry.get('identity'),dict)
                        or not isinstance(entry.get('source_path'),str)):
                    raise ValueError('dynamic private export inventory entry is malformed')
                identity=entry['identity']
                if identity!={'sha256':item['sha256'],'bytes':item['bytes']}:
                    continue
                source=Path(entry['source_path'])
                if source.resolve(strict=True)!=self._local_path(item['path']).resolve(strict=True):
                    continue
                validate_receipt(entry.get('receipt'),identity)
                if (entry['receipt'].get('source')!=entry['source_path']
                        or entry['receipt'].get('role')!=self.local.get('private_export_role')):
                    raise ValueError('dynamic private receipt names another bundle member')
                return True
            return False
        raise ValueError('large local role transfer lacks its independently verified private export receipt')

    def _stream_cloud_to_local(self,source,target,item):
        path=self._local_path(item['path'],write=True)
        if path.is_file():
            if self._local_has(item):return
            raise ValueError('local role target already exists with different content')
        from recovery_inputs import write_verified
        sandbox=self.sdk.sandbox(source)
        path.parent.mkdir(parents=True,exist_ok=True)
        with contextlib.closing(sandbox.read_file_streaming(item['path'],timeout_seconds=3600)) as chunks:
            write_verified(chunks,path,{'bytes':item['bytes'],'sha256':item['sha256']},
                reserve_bytes=self.local.get('reserve_bytes',0))

    @staticmethod
    def _file_sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb')as stream:
            while block:=stream.read(8*1024**2):h.update(block)
        return h.hexdigest()

    def mirror_many(self,source,target,items,*,should_stop=lambda:False):
        if source in self.config['sandboxes']and target in self.config['sandboxes']:
            return self.sdk.mirror_many(source,target,items,should_stop=should_stop)
        unique={}
        for value in items:
            item=checked_binding(value)
            if item['path']in unique and unique[item['path']]!=item:
                raise ValueError('cross-host role transfer contains conflicting path bindings')
            unique[item['path']]=item
        for item in unique.values():
            if should_stop():return False
            if source==target:
                if source==self.local['source_host_id']:
                    if not self._local_has(item):raise ValueError('local role source is not its exact bound file')
                elif not self._cloud_has(source,item):raise ValueError('cloud role source is not its exact bound file')
            elif source==self.local['source_host_id']:
                self._stream_local_to_cloud(source,target,item)
            elif target==self.local['source_host_id']:
                self._stream_cloud_to_local(source,target,item)
            else:raise ValueError('role transfer names an unconfigured source host')
        return True
