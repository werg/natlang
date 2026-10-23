"""Authenticated CPU replay of historical recipe and invocation metadata."""
from pathlib import Path
import json


def package_metadata_replay(package, code, operation, request, references):
    """Replay exact historical metadata code in an isolated CPU import context.

    Reader dependencies are part of the original metadata plan. Changing their
    implementation is legitimate future work, never an excuse to reinterpret
    an earned historical plan with today's implementation.
    """
    import os
    import subprocess
    import sys
    from ..train.native_resident import code_inventory
    from ..rl.native_raw_handoff import _bound
    package = Path(package).resolve(strict=True)
    if code_inventory(package) != code:
        raise ValueError('historical recipe runtime differs from declared code')
    root = package.resolve().parents[2]
    if operation not in ('recipe-plans', 'completed-future-config'):
        raise ValueError('unsupported historical metadata operation')
    program = '''import sys,json
from pathlib import Path
package=Path(sys.argv[1]);sys.path[:0]=[str(package.parent),str(package.parents[2])]
from natlang_neuralese.train.native_readiness import _bound
request=json.load(sys.stdin)
if sys.argv[2]=='recipe-plans':
 from natlang_neuralese.rl.native_raw_handoff import _recipe_plan
 value=[_recipe_plan(_bound(ref),request['device'])for ref in request['recipes']]
elif sys.argv[2]=='completed-future-config':
 from natlang_neuralese.rl.native_completed_supervised import _future_config
 value=_future_config(*request['args'],**request['kwargs'])
else:raise ValueError('unsupported historical metadata operation')
print(json.dumps(value))
'''
    references=[{'path':ref['path'],'sha256':ref.get('file_sha256',ref['sha256']),
        **({'bytes':ref['bytes']}if'bytes'in ref else{})}for ref in references]
    def check_inputs():
        for ref in references:
            path=_bound(ref)
            if 'bytes'in ref and path.stat().st_size!=ref['bytes']:
                raise ValueError('historical recipe byte count changed')
    check_inputs()
    result = subprocess.run([sys.executable, '-c', program, str(package), operation],
        input=json.dumps(request), capture_output=True, text=True,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'NATLANG_REPO': str(root),
             'PYTHONPATH': str(package.parent)+':'+str(root)})
    if result.returncode:
        raise ValueError('historical recipe reconstruction failed: '+result.stderr[-2000:])
    if code_inventory(package) != code:
        raise ValueError('historical recipe runtime changed during reconstruction')
    check_inputs()
    return json.loads(result.stdout)

