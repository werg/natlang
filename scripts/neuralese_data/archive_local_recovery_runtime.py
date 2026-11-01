#!/usr/bin/env python3
"""Seal an exact frozen runtime as a cache-bounded independently checked tar."""
from __future__ import annotations
import argparse,hashlib,io,json,os,tarfile
from pathlib import Path
from scripts.coreweave.cache_bounded_archive import open_recovery_tar
from scripts.coreweave.private_export_guard import CacheBoundedReader
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.train.checkpoint_offload import file_sha256


def seal(runtime, output, receipt, *, allow_external_file_links=False):
    runtime=Path(runtime).resolve(strict=True);output=Path(output).resolve();receipt=Path(receipt).resolve()
    if output.exists() or receipt.exists():raise ValueError('runtime archive output already exists')
    output.parent.mkdir(parents=True,exist_ok=True);pending=output.with_name(output.name+'.pending')
    if pending.exists():raise ValueError('runtime archive has an existing pending file')
    entries=[]
    def identity(path):
        s=path.lstat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    try:
        with open_recovery_tar(pending) as archive:
            for base,dirs,files in os.walk(runtime,followlinks=False):
                base=Path(base);dirs.sort();files.sort()
                for name in [*dirs,*files]:
                    path=base/name;member='runtime/'+path.relative_to(runtime).as_posix();before=identity(path)
                    if path.is_symlink():
                        target=path.resolve(strict=True)
                        if not target.is_relative_to(runtime):
                            if not allow_external_file_links or not target.is_file():raise ValueError('runtime symlink escapes frozen runtime')
                            target_before=identity(target);digest=file_sha256(target);info=archive.gettarinfo(str(target),member)
                            if info.islnk():info.type=tarfile.REGTYPE;info.linkname='';info.size=target.stat().st_size
                            with target.open('rb') as source:archive.addfile(info,source)
                            if identity(target)!=target_before:raise ValueError('external link target changed during archive')
                            entries.append({'name':member,'sha256':digest,'bytes':info.size,'original_link':os.readlink(path)})
                        else:
                            info=tarfile.TarInfo(member);info.type=tarfile.SYMTYPE;info.linkname=os.readlink(path);info.mode=0o777
                            archive.addfile(info);entries.append({'name':member,'link':info.linkname})
                    elif path.is_dir():
                        info=archive.gettarinfo(str(path),member);archive.addfile(info)
                        entries.append({'name':member,'directory':True,'mode':info.mode & 0o7777})
                    elif path.is_file():
                        digest=file_sha256(path);info=archive.gettarinfo(str(path),member)
                        if info.islnk():info.type=tarfile.REGTYPE;info.linkname='';info.size=path.stat().st_size
                        with path.open('rb') as source:archive.addfile(info,source)
                        entries.append({'name':member,'sha256':digest,'bytes':info.size})
                    else:raise ValueError('unsupported frozen runtime entry')
                    if identity(path)!=before:raise ValueError('frozen runtime changed during archive: '+str(path))
            manifest={'schema':'natlang.local-recovery-runtime-archive/1','original':str(runtime),'entries':entries}
            data=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode();info=tarfile.TarInfo('runtime-manifest.json');info.size=len(data);archive.addfile(info,io.BytesIO(data))
        expected={entry['name']:entry for entry in entries};seen=set();observed_manifest=None
        with CacheBoundedReader(str(pending)) as source:
            with tarfile.open(fileobj=source,mode='r|') as archive:
                for member in archive:
                    if member.name=='runtime-manifest.json':observed_manifest=json.load(archive.extractfile(member));continue
                    item=expected.get(member.name)
                    if item is None or member.name in seen:raise ValueError('unexpected runtime archive member')
                    seen.add(member.name)
                    if 'link' in item:
                        if not member.issym() or member.linkname!=item['link']:raise ValueError('runtime link differs')
                    elif item.get('directory'):
                        if not member.isdir() or member.mode!=item['mode']:raise ValueError('runtime directory differs')
                    elif (not member.isfile() or member.size!=item['bytes']
                          or hashlib.file_digest(archive.extractfile(member),'sha256').hexdigest()!=item['sha256']):
                        raise ValueError('runtime payload differs')
        if seen!=set(expected) or observed_manifest!=manifest:raise ValueError('runtime archive inventory incomplete')
        digest=file_sha256(pending);os.replace(pending,output);fd=os.open(output.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        result={'schema':'natlang.local-recovery-runtime-archive-receipt/1','source':str(runtime),
                'archive':{'path':str(output),'sha256':digest,'bytes':output.stat().st_size},
                'entries':len(entries),'all_members_independently_reread':True,
                'builder':{'path':str(Path(__file__).resolve()),'sha256':file_sha256(Path(__file__))},
                'external_file_links_dereferenced':allow_external_file_links,
                'scope':'frozen runtime bytes only; model, environment, data and actual restore are separate'}
        write_json_durable(receipt,result);return result
    except BaseException:
        pending.unlink(missing_ok=True)
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',required=True);parser.add_argument('--output',required=True);parser.add_argument('--receipt',required=True);parser.add_argument('--allow-external-file-links',action='store_true')
    args=parser.parse_args();print(json.dumps(seal(args.runtime,args.output,args.receipt,allow_external_file_links=args.allow_external_file_links)),flush=True)

if __name__=='__main__':main()
