"""Install a privately exported full parent using only a scoped read capability."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener
from natlang_neuralese.common.jsonio import write_json_durable


class _ObjectRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target=urlsplit(newurl)
        if target.scheme!='https' or target.hostname!='storage.googleapis.com':
            raise ValueError('private parent redirect left scoped object storage')
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def _open(url, timeout):
    return build_opener(_ObjectRedirect()).open(url,timeout=timeout)


def ensure_base(base, *, download=None):
    """Fetch a public, commit-pinned base on a fresh machine without owner credentials."""
    revision=base['revision'];cache=Path(base['cache']);expected=Path(base['path'])
    if (len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision)
            or not cache.is_absolute() or not expected.is_absolute()
            or not expected.is_relative_to(cache) or expected.name!=revision):
        raise ValueError('invalid commit-pinned base descriptor')
    if expected.is_dir():return {'repo_id':base['repo_id'],'revision':revision,'path':str(expected),'source':'warm-cache'}
    if download is None:
        from huggingface_hub import snapshot_download
        download=snapshot_download
    resolved=Path(download(repo_id=base['repo_id'],revision=revision,cache_dir=str(cache),token=False))
    if resolved != expected or not resolved.is_dir():raise ValueError('base download resolved to a different snapshot')
    return {'repo_id':base['repo_id'],'revision':revision,'path':str(expected)}


def fetch(capability, parent, *, opener=_open):
    cap=json.loads(Path(capability).read_text()); parent=Path(parent)
    sha,size,step=cap['sha256'],cap['bytes'],cap['global_step']
    if (len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha) or type(size) is not int or size<=0
            or type(step) is not int or step<1 or not cap.get('full_optimizer_rng_exposure')):
        raise ValueError('invalid full-parent read capability')
    url=urlsplit(cap['url'])
    if url.scheme!='https' or url.hostname!='storage.googleapis.com' or url.username or url.password or url.fragment:
        raise ValueError('parent read capability is not scoped HTTPS object storage')
    parent.parent.mkdir(parents=True,exist_ok=True)
    def valid():
        if not parent.is_file() or parent.stat().st_size!=size:return False
        with parent.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()==sha
    if parent.is_symlink():raise ValueError('parent target is a symlink')
    if parent.exists():
        if not valid():raise ValueError('existing parent differs from exact private pin')
    else:
        fd,name=tempfile.mkstemp(prefix='.'+parent.name+'.',suffix='.part',dir=parent.parent)
        try:
            h=hashlib.sha256();count=0
            try:
                with os.fdopen(fd,'wb') as target,opener(cap['url'],timeout=1800) as response:
                    while block:=response.read(8<<20):
                        count+=len(block)
                        if count>size:raise ValueError('private parent exceeds pinned bytes')
                        h.update(block);target.write(block)
                    target.flush();os.fsync(target.fileno())
            except OSError:
                raise RuntimeError('private parent GET failed') from None
            if count!=size or h.hexdigest()!=sha:raise ValueError('private parent GET differs from exact pin')
            os.link(name,parent);os.unlink(name)
        finally:
            Path(name).unlink(missing_ok=True)
    receipt={'sha256':sha,'bytes':size,'global_step':step,'requested_stop':cap['requested_stop'],
             'full_optimizer_rng_exposure':True,'source':'private-full-object-get'}
    if not 0<receipt['requested_stop']<=step:raise ValueError('invalid parent stop step')
    path=Path(str(parent)+'.seed.json')
    if path.is_symlink():raise ValueError('parent receipt is a symlink')
    if path.exists():
        existing=json.loads(path.read_text())
        if any(existing.get(key)!=receipt[key] for key in ('sha256','bytes','global_step','requested_stop','full_optimizer_rng_exposure')):
            raise ValueError('existing parent receipt differs')
    if not path.exists():write_json_durable(path,receipt)
    if cap.get('base'):ensure_base(cap['base'])
    return {'sha256':sha,'bytes':size,'global_step':step}


if __name__=='__main__':
    result=fetch(Path(os.environ['NATLANG_CW_SECRETS'])/'parent-read',sys.argv[1])
    print(json.dumps({'event':'verified-private-parent-get',**result}))
