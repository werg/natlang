"""Observe an already-running external bridge without inventing its startup records.

The supervisor chooses which live bridge to recover. These exact OS/request
checks authenticate its actual owner, child, executable source and import path.
They grant no training credit, quality qualification or completion.
"""
from pathlib import Path
import json
import time

from .native_raw_handoff import observe, same_process, process_argv
from ..train.native_resident import binding, code_inventory
from ..train.native_readiness import _bound
from ..common.jsonio import write_json_durable

SCHEMA = 'natlang.native-observed-bridge-dispatch/1'
MODULES = ('natlang_neuralese.train.native_state_sft', 'natlang_neuralese.train.native_preference')


def _argument(argv, flag):
    if argv.count(flag) != 1 or argv.index(flag)+1 >= len(argv):
        raise ValueError('observed bridge requires one actual '+flag)
    return argv[argv.index(flag)+1]


def capture(owner, child, request, original_launch, entry, out, *, historical_package):
    """Capture current live evidence, explicitly after the original bridge started."""
    from .native_bridge import read_request
    actual, learner = observe(owner['pid']), observe(child['pid'])
    if (not same_process(owner, actual) or not same_process(child, learner)
            or actual['state'] in ('Z','X') or learner['state'] in ('Z','X')
            or learner['ppid'] != actual['pid']):
        raise ValueError('observed bridge needs its actual live direct optimizer child')
    original = json.loads(_bound(original_launch).read_text())
    if not same_process(original, actual):
        raise ValueError('observed bridge differs from its original launch identity')
    entry = binding(_bound(entry))
    if entry['path'] not in process_argv(actual) or original.get('source') != entry['path']:
        raise ValueError('observed bridge entry differs from its executing source')
    invocation = read_request(_bound(request))
    argv = process_argv(learner)
    if len(argv)<3 or argv[1]!='-m' or argv[2] not in MODULES:
        raise ValueError('observed bridge child is not an actual native trainer')
    config = json.loads(_bound(invocation['config']).read_text())
    parent = invocation['parent_checkpoint']
    if (invocation.get('native_actor_mode')!='raw-state' or invocation.get('kind')!='round'
            or Path(_argument(argv,'--out')).resolve()!=Path(invocation['out']).resolve()/'learner'
            or Path(_argument(argv,'--continue-from')).resolve()!=_bound(parent).resolve()
            or _argument(argv,'--stage-id') != config['trainer_parameters']['stage_id']
            or json.loads(_argument(argv,'--recipe-binding')) != config['trainer_parameters']['recipe_binding']):
        raise ValueError('observed trainer differs from its actual bridge request/configuration')
    env = dict(part.split('=',1) for part in Path('/proc',str(child['pid']),'environ').read_text().split('\0')
               if '=' in part)
    env = {key:env[key] for key in ('PYTHONPATH','NATLANG_REPO') if key in env}
    directory = Path(_argument(argv,'--resident-data-directory'))
    state = json.loads((directory/'current.json').read_text())
    package = Path(historical_package).resolve(strict=True)
    if code_inventory(package) != state['payload']['plan']['code']:
        raise ValueError('observed bridge historical package differs from its adopted plan')
    if not same_process(actual,observe(actual['pid'])) or not same_process(learner,observe(learner['pid'])):
        raise ValueError('observed bridge changed during its live capture')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    value={'schema':SCHEMA,'observed_at':time.time(),'owner':actual,'child':learner,'argv':argv,
           'request':request,'config':invocation['config'],'parent':parent,'original_launch':original_launch,
           'supervisor':entry,'env':env,'historical_package':str(package),
           'historical_code':state['payload']['plan']['code'],'startup_capture_claimed':False}
    dispatch=out/'dispatch.json';write_json_durable(dispatch,value)
    launch=out/'launch.json';write_json_durable(launch,{'at':time.time(),'pid':actual['pid'],
        'identity':actual,'argv':process_argv(actual),'observed_bridge_dispatch':binding(dispatch)})
    return binding(launch)


def verify(captured, observed, snapshots=None):
    """Verify captured byte identities and actual process-tree/request bindings."""
    original_ref=observed['observed_bridge_dispatch']
    selected=(snapshots or {}).get('intent',original_ref)
    if selected['sha256'] != original_ref['sha256']:
        raise ValueError('observed bridge dispatch snapshot changed')
    value=json.loads(_bound(selected).read_text())
    def selected_ref(name, reference):
        chosen=(snapshots or {}).get(name,reference)
        if chosen['sha256']!=reference['sha256']:
            raise ValueError('observed bridge snapshot changed '+name)
        _bound(chosen)
        return chosen
    entry=selected_ref('supervisor',value['supervisor'])
    request=selected_ref('continuation',value['request'])
    original=selected_ref('original',value['original_launch'])
    config_ref=selected_ref('config',value['config'])
    launch=json.loads(_bound(original).read_text())
    if (value.get('schema')!=SCHEMA or value.get('startup_capture_claimed') is not False
            or not same_process(value['owner'],captured['owner'])
            or process_argv(value['owner'])!=process_argv(captured['owner'])
            or observed.get('argv')!=process_argv(captured['owner'])
            or value['supervisor']['path'] not in process_argv(captured['owner'])
            or not same_process(launch,captured['owner']) or launch.get('source')!=value['supervisor']['path']
            or not any(same_process(child,value['child']) and process_argv(child)==value['argv']
                       and child['ppid']==captured['owner']['pid'] for child in captured['children'])):
        raise ValueError('observed bridge lost its actual captured owner/child/source')
    from .native_bridge import read_request
    invocation=read_request(_bound(request))
    config=json.loads(_bound(config_ref).read_text())
    argv=value['argv']
    if (len(argv)<3 or argv[1]!='-m' or argv[2] not in MODULES
            or invocation.get('native_actor_mode')!='raw-state' or invocation.get('kind')!='round'
            or invocation['config']!=value['config'] or invocation['parent_checkpoint']!=value['parent']
            or Path(_argument(argv,'--out')).resolve()!=Path(invocation['out']).resolve()/'learner'
            or Path(_argument(argv,'--continue-from')).resolve()!=_bound(value['parent']).resolve()
            or _argument(argv,'--stage-id')!=config['trainer_parameters']['stage_id']
            or json.loads(_argument(argv,'--recipe-binding'))!=config['trainer_parameters']['recipe_binding']
            or code_inventory(value['historical_package'])!=value['historical_code']):
        raise ValueError('observed bridge request/recipe/parent/historical runtime changed')
    refs={'intent':selected,'supervisor':entry,'continuation':request,'original':original,'config':config_ref}
    return value,argv,refs
