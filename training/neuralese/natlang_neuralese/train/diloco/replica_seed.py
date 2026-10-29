"""Sealed replica seed publication and verification for a replacement peer."""
from __future__ import annotations
import hashlib,json,os,re
from pathlib import Path
from ...common.jsonio import write_json_durable
from .replica_baseline import verify
from .replica_journal import ReplicaJournal,_identity
from .replica_retention import Retention


def _digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def publish_seed(store, destination, *, expected_sequence, expected_head, lease_id):
    """Pin exact immutable baseline and sealed WAL; lease remains until transfer receipt."""
    store=Path(store).resolve();pointer=json.loads((store/'replica-active.json').read_text())
    generation=pointer['id'];baseline=Path(pointer['baseline']).resolve();journal=Path(pointer['journal']).resolve()
    if not journal.is_relative_to(store):raise ValueError('replica journal escapes store')
    lease=None
    if generation!='initial':
        if not re.fullmatch('[0-9a-f]{32}',generation):raise ValueError('invalid active generation')
        retention=Retention(store);retention.acquire(generation,lease_id);lease={'generation':generation,'id':lease_id}
    try:
        value=verify(baseline,pointer['baseline_sha256']);identity=json.loads((journal/'identity.json').read_text())
        if identity['run_identity']!=value['run_id'] or identity['baseline_sha256']!=pointer['baseline_sha256']:
            raise ValueError('active journal does not bind verified baseline')
        if type(expected_sequence) is not int or expected_sequence<0 or expected_sequence%2:raise ValueError('seed head is not sealed')
        previous=_identity(identity)
        for seq in range(1,expected_sequence+1):
            root=journal/f'{seq:020d}';record=json.loads((root/'record.json').read_text())
            if record.get('sequence')!=seq or record.get('previous')!=previous or record.get('payload_bytes')!=(root/'payload').stat().st_size or record.get('payload_sha256')!=_digest(root/'payload'):
                raise ValueError('seed WAL has a gap or invalid payload')
            previous=_identity(record)
        if previous!=expected_head:raise ValueError('seed WAL differs from witnessed head')
        destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
        files=[]
        sources=[(baseline/'replica-baseline.json',Path('baseline/replica-baseline.json'))]
        sources.extend((baseline/item['path'],Path('baseline')/item['path']) for item in value['files'])
        sources.append((journal/'identity.json',Path('journal/identity.json')))
        for seq in range(1,expected_sequence+1):
            for name in ('record.json','payload'):sources.append((journal/f'{seq:020d}'/name,Path('journal')/f'{seq:020d}'/name))
        for source,relative in sources:
            target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
            files.append({'path':str(relative),'bytes':target.stat().st_size,'sha256':_digest(target)})
        record={'schema':'natlang.diloco-replica-seed/1','run_id':value['run_id'],'baseline_sha256':pointer['baseline_sha256'],
                'sequence':expected_sequence,'head':expected_head,'lease':lease,'files':files}
        write_json_durable(destination/'seed-receipt.json',record)
        return record
    except BaseException:
        if lease:retention.release(generation,lease_id)
        raise


def verify_seed(root, *, expected_receipt_sha256):
    root=Path(root).resolve();receipt=root/'seed-receipt.json'
    if _digest(receipt)!=expected_receipt_sha256:raise ValueError('seed receipt hash differs')
    record=json.loads(receipt.read_text())
    if record.get('schema')!='natlang.diloco-replica-seed/1':raise ValueError('unsupported replica seed')
    seen=set()
    for item in record['files']:
        relative=Path(item['path']);path=root/relative
        if relative.is_absolute() or '..' in relative.parts or str(relative) in seen or path.is_symlink() or not path.resolve().is_relative_to(root):raise ValueError('invalid seed file path')
        seen.add(str(relative))
        if not path.is_file() or path.stat().st_size!=item['bytes'] or _digest(path)!=item['sha256']:raise ValueError('seed file differs')
    if {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=receipt and p!=root/'journal'/'.writer.lock'}!=seen:raise ValueError('seed inventory differs')
    value=verify(root/'baseline',record['baseline_sha256']);identity=json.loads((root/'journal'/'identity.json').read_text())
    journal=ReplicaJournal(root/'journal',run_identity=identity['run_identity'],baseline_sha256=identity['baseline_sha256'],
        max_payload_bytes=identity['max_payload_bytes'],max_transactions=identity['max_transactions'],previous_generation=identity.get('previous_generation'))
    try:journal.verify(expected_sequence=record['sequence'],expected_head=record['head'])
    finally:journal.close()
    if value['run_id']!=record['run_id'] or identity['run_identity']!=record['run_id']:raise ValueError('seed run identity differs')
    return record


def install_seed(seed, store, *, expected_receipt_sha256):
    """Install an independently verified seed as a fresh peer active generation."""
    seed=Path(seed).resolve();record=verify_seed(seed,expected_receipt_sha256=expected_receipt_sha256)
    store=Path(store);store.mkdir(parents=True,exist_ok=False)
    for item in record['files']:
        relative=Path(item['path']);target=store/relative;target.parent.mkdir(parents=True,exist_ok=True);os.link(seed/relative,target)
    identity=json.loads((store/'journal'/'identity.json').read_text())
    pointer={'schema':'natlang.diloco-replica-active-generation/1','id':'seeded',
             'previous':identity.get('previous_generation'),'baseline':str((store/'baseline').resolve()),
             'baseline_sha256':record['baseline_sha256'],'journal':str((store/'journal').resolve()),
             'head':_identity(identity),'max_payload_bytes':identity['max_payload_bytes'],
             'max_transactions':identity['max_transactions']}
    write_json_durable(store/'replica-active.json',pointer)
    write_json_durable(store/'seed-install.json',{'schema':'natlang.diloco-replica-seed-install/1',
        'seed_receipt_sha256':expected_receipt_sha256,'run_id':record['run_id'],'sequence':record['sequence'],'head':record['head']})
    return record


def release_seed_lease(store, record, *, witnessed):
    """Release only after the replacement peer reports the exact sealed head."""
    expected={'run_id':record['run_id'],'baseline_sha256':record['baseline_sha256'],'sequence':record['sequence'],'head':record['head']}
    if witnessed!=expected:raise ValueError('replacement peer did not witness the exact seed')
    lease=record.get('lease')
    if lease:Retention(store).release(lease['id'],lease['generation'])


def serve_seed(root,host,port,token,*,stop):
    """Serve only the published immutable seed over an authenticated gateway."""
    import hmac
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    from urllib.parse import unquote,urlsplit
    root=Path(root).resolve();receipt=root/'seed-receipt.json';record=json.loads(receipt.read_text())
    allowed={item['path'] for item in record['files']};allowed.add('seed-receipt.json')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def do_GET(self):
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):self.send_error(403);return
            name=unquote(urlsplit(self.path).path.lstrip('/'))
            if name not in allowed:self.send_error(404);return
            path=root/name
            if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():self.send_error(404);return
            with path.open('rb') as source:
                size=os.fstat(source.fileno()).st_size;self.send_response(200);self.send_header('Content-Length',str(size));self.end_headers()
                while block:=source.read(1<<20):self.wfile.write(block)
    server=ThreadingHTTPServer((host,port),Handler);server.timeout=.2
    try:
        while not stop.is_set():server.handle_request()
    finally:server.server_close()


def download_seed(url,token,destination,*,expected_receipt_sha256,gateway=None,allow_http=False,timeout=120):
    """Stream a receipt-bound seed directly over the peer gateway; never use DGX as relay."""
    import urllib.parse,urllib.request
    from .worker import _SameOriginRedirects
    address=urllib.parse.urlsplit(url)
    if address.scheme not in ('http','https') or not address.netloc or address.path not in ('','/') or (address.scheme=='http' and not allow_http):
        raise ValueError('seed transfer requires an HTTPS origin')
    opener=urllib.request.build_opener(_SameOriginRedirects);headers={'Authorization':'Bearer '+token}
    if gateway:headers['X-Sandbox-Share-Token']=gateway
    def get(name):return opener.open(urllib.request.Request(url.rstrip('/')+'/'+urllib.parse.quote(name),headers=headers),timeout=timeout)
    with get('seed-receipt.json') as response:raw=response.read(8<<20)
    if hashlib.sha256(raw).hexdigest()!=expected_receipt_sha256:raise ValueError('seed receipt transport hash differs')
    record=json.loads(raw)
    if record.get('schema')!='natlang.diloco-replica-seed/1':raise ValueError('invalid seed receipt')
    destination=Path(destination);intent=destination/'download-intent.json'
    expected_intent={'schema':'natlang.diloco-replica-seed-download/1','receipt_sha256':expected_receipt_sha256}
    if destination.exists():
        completed=destination/'seed-receipt.json'
        if not intent.exists() and completed.is_file():return verify_seed(destination,expected_receipt_sha256=expected_receipt_sha256)
        if destination.is_symlink() or not intent.is_file() or json.loads(intent.read_text())!=expected_intent:
            raise ValueError('seed destination belongs to another transfer')
    else:
        destination.mkdir(parents=True,exist_ok=False);write_json_durable(intent,expected_intent)
    seen=set()
    for item in record['files']:
        relative=Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in seen:raise ValueError('invalid seed transport path')
        seen.add(str(relative));target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256();total=0
        if target.is_symlink() or not target.resolve().is_relative_to(destination.resolve()):raise ValueError('seed transport path escapes destination')
        if target.exists():
            if target.is_file() and target.stat().st_size==item['bytes'] and _digest(target)==item['sha256']:continue
            if not target.is_file():raise ValueError('invalid seed transport target')
            target.unlink()
        with get(str(relative)) as response,target.open('xb') as out:
            while block:=response.read(1<<20):
                total+=len(block)
                if total>item['bytes']:raise ValueError('seed response exceeds declared size')
                digest.update(block);out.write(block)
            out.flush();os.fsync(out.fileno())
        if total!=item['bytes'] or digest.hexdigest()!=item['sha256']:raise ValueError('seed response differs')
    from ...common.jsonio import write_bytes_durable
    write_bytes_durable(destination/'seed-receipt.json',raw)
    intent.unlink();fd=os.open(destination,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return verify_seed(destination,expected_receipt_sha256=expected_receipt_sha256)
