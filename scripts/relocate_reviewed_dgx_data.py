#!/usr/bin/env python3
"""Copy two explicitly authorized inactive DGX trees, verify, and preserve their paths."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess

SOURCES = ('/home/werg/sdkb-runs', '/home/werg/bgkit-data-nvme')
DESTINATION = Path('/mnt/external/natlang-storage-20261002')
STATE = Path('/home/werg/natlang-remote/storage-relocation-20261002')

def writable_descriptors(source):
    found = []
    prefix = str(source) + '/'
    for process in Path('/proc').glob('[0-9]*'):
        try:
            for fd in (process/'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    if not target.startswith(prefix): continue
                    flags = next(line.split()[1] for line in (process/'fdinfo'/fd.name).read_text().splitlines() if line.startswith('flags:'))
                    if int(flags, 8) & os.O_ACCMODE: found.append(dict(pid=int(process.name), path=target))
                except (OSError, StopIteration): pass
        except OSError: pass
    return found

def main():
    STATE.mkdir(exist_ok=True)
    def status(state, **extra):
        row = dict(state=state, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), pid=os.getpid(), **extra)
        tmp = STATE/'status.tmp'; tmp.write_text(json.dumps(row, indent=2)+'\n'); tmp.replace(STATE/'status.json')
        print(json.dumps(row), flush=True)
    with (STATE/'move.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not Path('/mnt/external').is_mount(): raise ValueError('External filesystem is not mounted')
        DESTINATION.mkdir(exist_ok=True)
        completed = []
        for name in SOURCES:
            source = Path(name); dest = DESTINATION/source.name
            backup = source.with_name(source.name+'.before-external-move-20261002')
            if source.is_symlink() and source.resolve() == dest:
                completed.append(dict(source=str(source),destination=str(dest),already_moved=True)); continue
            if not source.is_dir() or source.is_symlink() or backup.exists(): raise ValueError('Source or backup requires review: '+name)
            writers = writable_descriptors(source)
            if writers:
                status('paused_source_writers', source=name, writers=writers); return
            dest.mkdir(exist_ok=True)
            status('copying', source=name, destination=str(dest), completed=completed, cache_move=False)
            with (STATE/(source.name+'.rsync.log')).open('ab') as log:
                subprocess.run(['rsync','-aH','--stats',str(source)+'/',str(dest)+'/'],stdout=log,stderr=subprocess.STDOUT,check=True)
            status('verifying', source=name, destination=str(dest), completed=completed, cache_move=False)
            comparison = subprocess.run(['rsync','-aHnc','--delete','--itemize-changes',str(source)+'/',str(dest)+'/'],capture_output=True,text=True,check=True)
            (STATE/(source.name+'.verification.txt')).write_text(comparison.stdout)
            if comparison.stdout.strip() or comparison.stderr.strip():
                status('paused_verification_mismatch', source=name, completed=completed); return
            writers = writable_descriptors(source)
            if writers:
                status('paused_source_writers', source=name, writers=writers, completed=completed); return
            source.rename(backup)
            try:
                source.symlink_to(dest, target_is_directory=True)
            except BaseException:
                backup.rename(source); raise
            if source.resolve() != dest: raise ValueError('Path switch verification failed')
            # Only remove the original after its verified external copy is live at the original path.
            status('releasing_verified_original', source=name, destination=str(dest), completed=completed, cache_move=False)
            shutil.rmtree(backup)
            receipt = dict(source=name,destination=str(dest),byte_verification='rsync checksum dry-run; no differences',original_path_preserved=True)
            (STATE/(source.name+'.receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');completed.append(receipt)
        status('complete', completed=completed, cache_move=False, internal_free_bytes=shutil.disk_usage('/').free)

if __name__ == '__main__':
    try: main()
    except Exception as error:
        STATE.mkdir(exist_ok=True)
        (STATE/'error.json').write_text(json.dumps(dict(error_type=type(error).__name__,message=str(error)),indent=2)+'\n')
        raise
