"""Authenticated streaming transfer of ordered replica journal transactions."""
from __future__ import annotations
import base64
import hmac
import hashlib
import http.client
import json
import os
import re
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from .replica_journal import CHUNK


def serve(journal, host, port, token, *, max_bytes, stop, machine=None, prepared=None, store_root=None, allow_unmaterialized_for_test=False, token_path=None):
    if machine is None and not allow_unmaterialized_for_test:
        raise ValueError('replica service requires a recovered state machine')
    guard = threading.Lock()
    failure = []
    active = {'journal':journal,'machine':machine,'prepared':prepared,'token':token}
    store_root=Path(store_root) if store_root is not None else journal.root.parent
    credential_receipt=store_root/'credential-rotation.json'
    if credential_receipt.exists():
        from ...common.jsonio import write_json_durable
        previous=json.loads(credential_receipt.read_text());actual=hashlib.sha256(token.encode()).hexdigest()
        if previous.get('state')=='prepared' and actual==previous.get('new_token_sha256'):
            previous['state']='committed'
            historical=store_root/'credential-history'/(previous['nonce']+'.json')
            if not historical.is_file():raise ValueError('credential transition history is missing')
            write_json_durable(historical,previous);write_json_durable(credential_receipt,previous)
        elif actual not in (previous.get('old_token_sha256'),previous.get('new_token_sha256')):
            raise ValueError('replica token does not match credential transition')
        elif previous.get('state')=='committed' and actual!=previous.get('new_token_sha256'):
            raise ValueError('replica token rolled back after credential transition')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+active['token']):self.send_error(403);return
            if self.path!='/v1/head':self.send_error(404);return
            with guard:
                if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+active['token']):self.send_error(403);return
                current=active['journal']
                if failure or current.sequence%2:self.send_error(409);return
                value={'run_id':current.identity['run_identity'],'baseline_sha256':current.identity['baseline_sha256'],
                       'sequence':current.sequence,'head':current.head}
            data=json.dumps(value,sort_keys=True).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_PUT(self):
            if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer '+active['token']):
                self.send_error(403); return
            if self.path.startswith('/v1/file/'):
                self._receive_file();return
            if self.path != '/v1/transaction':
                self.send_error(404); return
            temp = None
            try:
                length = int(self.headers['Content-Length'])
                if length < 0 or length > max_bytes: raise ValueError('invalid transaction length')
                header = json.loads(base64.b64decode(self.headers['X-Transaction'], validate=True))
                if set(header) != {'sequence','previous','event'}: raise ValueError('invalid transaction metadata')
                fd, name = tempfile.mkstemp(prefix='.incoming-', dir=active['journal'].root); temp = Path(name)
                with os.fdopen(fd, 'wb') as out:
                    left = length
                    while left:
                        block = self.rfile.read(min(CHUNK,left))
                        if not block: raise ValueError('truncated transaction')
                        out.write(block);left -= len(block)
                    out.flush();os.fsync(out.fileno())
                with guard:
                    if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+active['token']):raise ValueError('replica credential changed')
                    if failure:raise ValueError('replica materialization is fenced')
                    current=active['journal'];replica_machine=active['machine']
                    duplicate = header['sequence'] == current.sequence
                    ack = current.append(header['event'],temp,expected_sequence=header['sequence'],expected_previous=header['previous'])
                    event=header['event']
                    if replica_machine is not None and event.get('kind') == 'seal' and not duplicate:
                        sequence=header['sequence']
                        if sequence % 2 or event.get('operation_sequence') != sequence-1:
                            failure.append('seal sequence mismatch');raise ValueError(failure[-1])
                        opdir=current.root/f'{sequence-1:020d}'
                        op=json.loads((opdir/'record.json').read_text())
                        try:
                            actual=replica_machine.apply(op['event'],opdir/'payload')
                            if actual!=event.get('state'):raise ValueError('replica materialization differs from seal')
                        except Exception as exc:
                            failure.append(str(exc));raise ValueError('replica materialization failed') from exc
                payload=json.dumps(ack,sort_keys=True).encode();self.send_response(200);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
            except (KeyError,ValueError,OSError,TypeError):
                self.close_connection=True;self.send_error(400)
            finally:
                if temp is not None:temp.unlink(missing_ok=True)
        def _receive_file(self):
            name=self.path[len('/v1/file/'):];temp=None
            try:
                if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]{0,127}',name) or name.endswith('.sha256'):
                    raise ValueError('invalid replica file name')
                expected=self.headers['X-Sha256'];length=int(self.headers['Content-Length'])
                if not re.fullmatch('[0-9a-f]{64}',expected) or not 0<=length<=64*(1<<30):raise ValueError('invalid replica file declaration')
                directory=store_root/'replica-uploads';directory.mkdir(parents=True,exist_ok=True)
                fd,temp=tempfile.mkstemp(prefix='.incoming-file-',dir=directory);digest=hashlib.sha256()
                with os.fdopen(fd,'wb') as out:
                    left=length
                    while left:
                        block=self.rfile.read(min(CHUNK,left))
                        if not block:raise ValueError('truncated replica file')
                        out.write(block);digest.update(block);left-=len(block)
                    out.flush();os.fsync(out.fileno())
                if digest.hexdigest()!=expected:raise ValueError('replica file hash differs')
                with guard:
                    if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+active['token']):raise ValueError('replica credential changed')
                    if (store_root/'replica-files-frozen.json').exists():raise ValueError('replica file publication is frozen')
                    target=directory/name;sidecar=directory/(name+'.sha256')
                    from ...common.jsonio import write_bytes_durable
                    if target.exists() or sidecar.exists():
                        if not target.is_file() or target.stat().st_size!=length:raise ValueError('replica file name conflicts')
                        if sidecar.exists() and (not sidecar.is_file() or sidecar.read_text().strip()!=expected):raise ValueError('replica file name conflicts')
                        with target.open('rb') as check:
                            if hashlib.file_digest(check,'sha256').hexdigest()!=expected:raise ValueError('existing replica file bytes differ')
                        if not sidecar.exists():write_bytes_durable(sidecar,(expected+'\n').encode())
                    else:
                        os.replace(temp,target);temp=None;write_bytes_durable(sidecar,(expected+'\n').encode())
                    reply={'name':name,'sha256':expected,'bytes':length}
                data=json.dumps(reply,sort_keys=True).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            except (KeyError,ValueError,OSError,TypeError):self.close_connection=True;self.send_error(400)
            finally:
                if temp is not None and os.path.exists(temp):os.unlink(temp)
        def do_POST(self):
            if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer '+active['token']):
                self.send_error(403);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=65536:raise ValueError('invalid generation request')
                value=json.loads(self.rfile.read(length))
                with guard:
                    if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+active['token']):raise ValueError('replica credential changed')
                    if failure or active['machine'] is None:raise ValueError('replica cannot rotate')
                    current=active['journal']
                    if self.path=='/v1/file/freeze':
                        if set(value)!={'nonce'} or not isinstance(value['nonce'],str) or not re.fullmatch('[A-Za-z0-9_-]{32,128}',value['nonce']):raise ValueError('invalid file freeze nonce')
                        from ...common.jsonio import write_json_durable
                        marker=store_root/'replica-files-frozen.json';declaration={'schema':'natlang.diloco-replica-file-freeze/1','nonce':value['nonce']}
                        if marker.exists() and json.loads(marker.read_text())!=declaration:raise ValueError('another file freeze exists')
                        if not marker.exists():write_json_durable(marker,declaration)
                        reply={'nonce':value['nonce'],'frozen':True}
                    elif self.path=='/v1/credential/rotate':
                        import hashlib
                        if (token_path is None or set(value)!={'sequence','head','new_token','nonce'} or
                                value['sequence']!=current.sequence or value['head']!=current.head or current.sequence%2 or
                                not isinstance(value['nonce'],str) or not re.fullmatch('[A-Za-z0-9_-]{32,128}',value['nonce']) or
                                not isinstance(value['new_token'],str) or len(value['new_token'])<24 or value['new_token']==active['token']):
                            raise ValueError('invalid replica credential transition')
                        from ...common.jsonio import write_json_durable
                        credential={'schema':'natlang.diloco-replica-credential-rotation/1','state':'prepared',
                            'sequence':current.sequence,'head':current.head,'nonce':value['nonce'],
                            'old_token_sha256':hashlib.sha256(active['token'].encode()).hexdigest(),
                            'new_token_sha256':hashlib.sha256(value['new_token'].encode()).hexdigest()}
                        history=store_root/'credential-history';history.mkdir(exist_ok=True)
                        historical=history/(value['nonce']+'.json')
                        if historical.exists():raise ValueError('credential rotation nonce was already used')
                        write_json_durable(historical,credential)
                        write_json_durable(credential_receipt,credential)
                        target=Path(token_path)
                        fd,name=tempfile.mkstemp(prefix='.replica-token-',dir=target.parent)
                        try:
                            with os.fdopen(fd,'wb') as stream:
                                stream.write(value['new_token'].encode());stream.flush();os.fsync(stream.fileno())
                            os.replace(name,target);active['token']=value['new_token'];directory_fd=os.open(target.parent,os.O_DIRECTORY)
                            try:os.fsync(directory_fd)
                            finally:os.close(directory_fd)
                        finally:
                            if os.path.exists(name):os.unlink(name)
                        credential['state']='committed';write_json_durable(historical,credential);write_json_durable(credential_receipt,credential)
                        reply={'sequence':current.sequence,'head':current.head,'new_token_sha256':hashlib.sha256(active['token'].encode()).hexdigest(),'nonce':value['nonce']}
                    elif self.path=='/v1/generation/prepare':
                        if set(value)!={'id','previous','captured_monotonic','captured_wall'} or not re.fullmatch('[0-9a-f]{32}',value['id']):
                            raise ValueError('invalid generation identity')
                        prepared=active['prepared']
                        if prepared is not None:
                            if prepared['request']!=value:raise ValueError('another generation is prepared')
                            reply=prepared['reply']
                        else:
                            from .replica_generation import prepare_peer_generation
                            from ...common.jsonio import write_json_durable
                            root=store_root/('generation-'+value['id'])
                            baseline,next_journal=prepare_peer_generation(active['machine'],current,root/'baseline',root/'journal',
                                previous=value['previous'],captured_monotonic=value['captured_monotonic'],captured_wall=value['captured_wall'])
                            reply={'baseline_sha256':baseline['sha256'],'new_head':next_journal.head}
                            active['prepared']={'request':value,'reply':reply,'journal':next_journal,'baseline':str(root/'baseline')}
                            write_json_durable(store_root/'replica-prepared.json',{'schema':'natlang.diloco-replica-prepared-generation/1',
                                'request':value,'reply':reply,'baseline':str(root/'baseline'),'journal':str(next_journal.root)})
                    elif self.path=='/v1/generation/commit':
                        prepared=active['prepared']
                        if set(value)!={'id','previous','baseline_sha256','new_head'}:
                            raise ValueError('invalid generation commit')
                        if prepared is None:
                            pointer=store_root/'replica-active.json'
                            current_pointer=json.loads(pointer.read_text()) if pointer.exists() else {}
                            if (current_pointer.get('id')!=value['id'] or current_pointer.get('previous')!=value['previous'] or
                                    current_pointer.get('baseline_sha256')!=value['baseline_sha256'] or
                                    current_pointer.get('head')!=value['new_head'] or current.sequence!=0 or current.head!=value['new_head']):
                                raise ValueError('no exact prepared generation')
                            reply={'baseline_sha256':value['baseline_sha256'],'new_head':value['new_head']}
                        else:
                            reply=None
                        request=prepared['request'] if prepared else None
                        if prepared:reply=prepared['reply']
                        if prepared:
                            if (value['id']!=request['id'] or value['previous']!=request['previous'] or
                                    value['baseline_sha256']!=reply['baseline_sha256'] or value['new_head']!=reply['new_head'] or
                                    {'sequence':current.sequence,'head':current.head}!=request['previous']):
                                raise ValueError('generation changed before commit')
                            from ...common.jsonio import write_json_durable
                            write_json_durable(store_root/'replica-active.json',{'schema':'natlang.diloco-replica-active-generation/1',
                                'id':value['id'],'previous':value['previous'],'baseline':prepared['baseline'],
                                'baseline_sha256':reply['baseline_sha256'],'journal':str(prepared['journal'].root),'head':reply['new_head'],
                                'max_payload_bytes':prepared['journal'].max_payload_bytes,'max_transactions':prepared['journal'].max_transactions})
                            active['journal']=prepared['journal'];active['prepared']=None;current.close()
                            (store_root/'replica-prepared.json').unlink(missing_ok=True)
                    elif self.path=='/v1/generation/retain':
                        if set(value)!={'id','previous','baseline_sha256','new_head'}:raise ValueError('invalid retention confirmation')
                        pointer=store_root/'replica-active.json';pointer_value=json.loads(pointer.read_text())
                        if (pointer_value.get('id')!=value['id'] or pointer_value.get('previous')!=value['previous'] or
                                pointer_value.get('baseline_sha256')!=value['baseline_sha256'] or pointer_value.get('head')!=value['new_head']):
                            raise ValueError('peer active generation differs')
                        if current.sequence!=0 or current.head!=value['new_head']:raise ValueError('retention boundary changed')
                        from .replica_retention import Retention
                        retention=Retention(store_root);prior=retention.latest()
                        if prior==value['id']:
                            entries=retention._read()['generations'];prior=entries[-1]['previous']
                        reply={'baseline_sha256':value['baseline_sha256'],'new_head':value['new_head']}
                        retention.register(generation=value['id'],baseline_sha256=value['baseline_sha256'],head=value['new_head'],
                            previous=prior,replica_confirmation=reply,active_pointer=pointer)
                        retention.collect()
                    else:raise ValueError('unknown generation operation')
                payload=json.dumps(reply,sort_keys=True).encode();self.send_response(200);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
            except (KeyError,ValueError,OSError,TypeError):
                self.close_connection=True;self.send_error(400)
    server=ThreadingHTTPServer((host,port),Handler);server.timeout=.2
    try:
        while not stop.is_set():server.handle_request()
    finally:server.server_close()


def transmit(url, token, *, sequence, previous, event, payload, timeout=30, allow_http=False, gateway=None):
    address=urlsplit(url)
    if address.scheme not in ('https','http') or (address.scheme=='http' and not allow_http) or not address.hostname or address.path not in ('','/'):
        raise ValueError('replica endpoint must be an HTTPS origin')
    header=base64.b64encode(json.dumps({'sequence':sequence,'previous':previous,'event':event},sort_keys=True,separators=(',',':')).encode()).decode()
    if len(header)>65536:raise ValueError('replica event metadata too large')
    path=Path(payload);size=path.stat().st_size
    cls=http.client.HTTPSConnection if address.scheme=='https' else http.client.HTTPConnection
    conn=cls(address.hostname,address.port,timeout=timeout)
    try:
        conn.putrequest('PUT','/v1/transaction');conn.putheader('Authorization','Bearer '+token);conn.putheader('Content-Length',str(size));conn.putheader('X-Transaction',header)
        if gateway:conn.putheader('X-Sandbox-Share-Token',gateway)
        conn.endheaders()
        with path.open('rb') as source:
            while block:=source.read(CHUNK):conn.send(block)
        response=conn.getresponse();data=response.read(65537)
        if response.status!=200 or len(data)>65536:raise ValueError('replica did not durably acknowledge transaction (HTTP %d)'%response.status)
        ack=json.loads(data)
        if set(ack)!={'sequence','head'} or ack['sequence']!=sequence:raise ValueError('replica acknowledgement differs')
        return ack
    finally:conn.close()


def generation_request(url, token, operation, body, *, timeout=3600, allow_http=False, gateway=None):
    if operation not in ('prepare','commit','retain'):raise ValueError('invalid generation operation')
    address=urlsplit(url)
    if address.scheme not in ('https','http') or (address.scheme=='http' and not allow_http) or not address.hostname or address.path not in ('','/'):
        raise ValueError('replica endpoint must be an HTTPS origin')
    raw=json.dumps(body,sort_keys=True,separators=(',',':')).encode()
    if len(raw)>65536:raise ValueError('generation request too large')
    cls=http.client.HTTPSConnection if address.scheme=='https' else http.client.HTTPConnection
    for attempt in range(3):
        conn=cls(address.hostname,address.port,timeout=timeout)
        try:
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'}
            if gateway:headers['X-Sandbox-Share-Token']=gateway
            conn.request('POST','/v1/generation/'+operation,body=raw,headers=headers)
            response=conn.getresponse();data=response.read(65537)
            if response.status!=200 or len(data)>65536:raise ValueError('replica generation was not confirmed')
            value=json.loads(data)
            if set(value)!={'baseline_sha256','new_head'}:raise ValueError('invalid generation confirmation')
            return value
        except (OSError,http.client.HTTPException):
            if attempt==2:raise
        finally:conn.close()


def witness_head(url,token,*,timeout=30,allow_http=False,gateway=None):
    from .worker import Transport
    result=Transport(url,token,timeout,allow_http=allow_http,gateway=gateway or '').json('GET','/v1/head')
    if set(result)!={'run_id','baseline_sha256','sequence','head'} or type(result['sequence']) is not int or result['sequence']%2:
        raise ValueError('peer has no sealed witnessed head')
    return result


def rotate_replica_credential(url,old_token,new_token,*,sequence,head,nonce,allow_http=False,gateway=None,timeout=30):
    import hashlib
    address=urlsplit(url)
    if address.scheme not in ('https','http') or (address.scheme=='http' and not allow_http) or not address.hostname or address.path not in ('','/'):
        raise ValueError('replica credential endpoint must be an HTTPS origin')
    body=json.dumps({'sequence':sequence,'head':head,'new_token':new_token,'nonce':nonce},sort_keys=True,separators=(',',':')).encode()
    cls=http.client.HTTPSConnection if address.scheme=='https' else http.client.HTTPConnection
    conn=cls(address.hostname,address.port,timeout=timeout)
    try:
        headers={'Authorization':'Bearer '+old_token,'Content-Type':'application/json'}
        if gateway:headers['X-Sandbox-Share-Token']=gateway
        conn.request('POST','/v1/credential/rotate',body=body,headers=headers)
        response=conn.getresponse();raw=response.read(65537)
        if response.status!=200 or len(raw)>65536:raise ValueError('peer credential rotation not confirmed')
        result=json.loads(raw);expected={'sequence':sequence,'head':head,'nonce':nonce,'new_token_sha256':hashlib.sha256(new_token.encode()).hexdigest()}
        if result!=expected:raise ValueError('peer credential rotation binding differs')
        return result
    finally:conn.close()


def transmit_file(url,token,name,path,sha256,*,timeout=3600,allow_http=False,gateway=None):
    address=urlsplit(url)
    if address.scheme not in ('https','http') or (address.scheme=='http' and not allow_http) or not address.hostname or address.path not in ('','/'):
        raise ValueError('replica file endpoint must be an HTTPS origin')
    source=Path(path);size=source.stat().st_size;cls=http.client.HTTPSConnection if address.scheme=='https' else http.client.HTTPConnection
    conn=cls(address.hostname,address.port,timeout=timeout)
    try:
        conn.putrequest('PUT','/v1/file/'+name);conn.putheader('Authorization','Bearer '+token);conn.putheader('Content-Length',str(size));conn.putheader('X-Sha256',sha256)
        if gateway:conn.putheader('X-Sandbox-Share-Token',gateway)
        conn.endheaders()
        with source.open('rb') as stream:
            while block:=stream.read(CHUNK):conn.send(block)
        response=conn.getresponse();raw=response.read(65537)
        if response.status!=200 or len(raw)>65536:raise ValueError('replica file was not durably acknowledged')
        reply=json.loads(raw)
        if reply!={'name':name,'sha256':sha256,'bytes':size}:raise ValueError('replica file acknowledgement differs')
        return reply
    finally:conn.close()


def freeze_replica_files(url,token,nonce,*,timeout=30,allow_http=False,gateway=None):
    address=urlsplit(url)
    if address.scheme not in ('http','https') or not address.hostname or address.path not in ('','/') or (address.scheme=='http' and not allow_http):raise ValueError('invalid replica endpoint')
    conn=(http.client.HTTPSConnection if address.scheme=='https' else http.client.HTTPConnection)(address.hostname,address.port,timeout=timeout)
    try:
        headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'}
        if gateway:headers['X-Sandbox-Share-Token']=gateway
        conn.request('POST','/v1/file/freeze',body=json.dumps({'nonce':nonce}).encode(),headers=headers)
        response=conn.getresponse();raw=response.read(65537)
        if response.status!=200 or json.loads(raw)!={'nonce':nonce,'frozen':True}:raise ValueError('replica file freeze was not confirmed')
        return {'nonce':nonce,'frozen':True}
    finally:conn.close()
