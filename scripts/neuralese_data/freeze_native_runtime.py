#!/usr/bin/env python3
"""Freeze a committed runtime with its mandatory protected data closure."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from natlang_neuralese.common.runtime_inventory import assemble_protected_runtime_inputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--predecessor',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--control-bindings',type=Path,required=True,
                        help='JSON list of exact source, control and prompt fixture path/SHA bindings')
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
    receipt=assemble_protected_runtime_inputs(args.predecessor,args.out,bound_files=bindings)
    receipt['commit']=subprocess.check_output(['git','rev-parse',args.commit],text=True).strip()
    (args.out/'runtime-input-closure.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'runtime':str(args.out.resolve()),'receipt':str((args.out/'runtime-input-closure.json').resolve()),
                      'protected_worlds_sha256':receipt['protected_worlds_sha256']}))


if __name__=='__main__':main()
