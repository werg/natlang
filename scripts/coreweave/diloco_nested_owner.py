"""Construct a hash-bound inline nested owner for an existing role supervisor."""
from __future__ import annotations
import hashlib
from pathlib import Path


def prepare(*, source, expected_sha256, python, control_directory, runtime):
    path=Path(source);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('nested owner source differs')
    if not all(isinstance(x,str) and x.startswith('/') for x in (python,control_directory,runtime)):
        raise ValueError('nested owner paths must be absolute')
    code=('import sys\nfrom pathlib import Path\n'
          'sys.path.insert(0,'+repr(runtime+'/training/neuralese')+')\n'
          'sys.argv=["role-supervisor",'+repr(control_directory)+']\n'
          'exec(compile('+repr(raw.decode('utf-8'))+',"<nested-role-owner>","exec"),{"__name__":"__main__"})\n')
    argv=[python,'-c',code];cmdline=b'\0'.join(x.encode() for x in argv)+b'\0'
    return {'argv':argv,'source_sha256':expected_sha256,'entrypoint_sha256':hashlib.sha256(code.encode()).hexdigest(),
            'supervisor_cmdline_sha256':hashlib.sha256(cmdline).hexdigest(),'control_directory':control_directory}
