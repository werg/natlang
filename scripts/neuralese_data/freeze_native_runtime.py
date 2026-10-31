#!/usr/bin/env python3
"""Freeze a committed runtime with its mandatory protected data closure."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from natlang_neuralese.common.runtime_inventory import assemble_protected_runtime_inputs


def freeze_node_dependencies(source, destination):
    """Copy and bind the collector's actual Node package closure."""
    source, destination = Path(source), Path(destination)
    if not (source / 'just-bash' / 'package.json').is_file():
        raise ValueError('Node dependency closure lacks just-bash')
    shutil.copytree(source, destination, symlinks=True)
    digest = hashlib.sha256()
    count = size = 0
    for path in sorted(destination.rglob('*')):
        relative = path.relative_to(destination).as_posix()
        if path.is_symlink():
            digest.update(f'{relative}\0link\0{path.readlink()}\n'.encode())
        elif path.is_file():
            file_digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    file_digest.update(block)
            digest.update(f'{relative}\0{file_digest.hexdigest()}\n'.encode())
            count += 1
            size += path.stat().st_size
    return {'source': str(source.resolve()), 'sha256': digest.hexdigest(),
            'files': count, 'bytes': size,
            'just_bash_version': json.loads((destination / 'just-bash/package.json').read_text())['version']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--predecessor',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--control-bindings',type=Path,required=True,
                        help='JSON list of exact source, control and prompt fixture path/SHA bindings')
    parser.add_argument('--node-dependencies', type=Path, default=Path('node_modules'),
                        help='installed Node dependency closure used by the frozen collector')
    args=parser.parse_args()
    bindings=json.loads(args.control_bindings.read_text())
    if not isinstance(bindings,list) or not bindings:
        raise ValueError('runtime assembly requires declared actual control and prompt bindings')
    args.out.mkdir(parents=True,exist_ok=False)
    archive=subprocess.Popen(['git','archive',args.commit],stdout=subprocess.PIPE)
    try:
        subprocess.run(['tar','-x','-C',str(args.out)],stdin=archive.stdout,check=True)
    finally:
        if archive.stdout:archive.stdout.close()
    if archive.wait()!=0:raise RuntimeError('git archive failed')
    shutil.copytree(args.predecessor/'ts-host',args.out/'ts-host',dirs_exist_ok=True,symlinks=True)
    node = freeze_node_dependencies(args.node_dependencies, args.out/'node_modules')
    collector = (args.out/'ts-host/dist/teacher/collector.js').resolve().as_uri()
    subprocess.run(['node', '--input-type=module', '-e', 'await import(process.argv[1])', collector], check=True)
    receipt=assemble_protected_runtime_inputs(args.predecessor,args.out,bound_files=bindings)
    receipt['node_dependencies'] = {**node, 'collector_module_import': 'passed'}
    receipt['commit']=subprocess.check_output(['git','rev-parse',args.commit],text=True).strip()
    (args.out/'runtime-input-closure.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'runtime':str(args.out.resolve()),'receipt':str((args.out/'runtime-input-closure.json').resolve()),
                      'protected_worlds_sha256':receipt['protected_worlds_sha256']}))


if __name__=='__main__':main()
