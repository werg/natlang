"""Host-owned bootstrap verification receipts; no tensor or global state cache."""
import json
from pathlib import Path
from ..common.hashing import canonical_json_sha256_hex as digest
from ..train.native_readiness import _bound
from ..train.native_resident import code_inventory,binding


def evidence(plan_path,declaration,handoff,config):
    from ..common.runtime_inventory import task_runtime_inventory
    try:collector=task_runtime_inventory(Path(declaration['runtime']))
    except ValueError:return None  # No cache for incomplete collector fixtures/configuration.
    files={};packages={}
    from .native_raw_rounds import _inputs
    active={**declaration,'production_recipe':binding(Path(config['production_recipe']))}
    for key in ('seed_native_graphs','held_native_graphs'):
        if config.get(key):active[key]=binding(Path(config[key]))
    inputs=_inputs(active)  # Retain exact TRAIN/source/protection checks on every reuse.
    files.update({str(path):binding(path) for path in inputs.values()})
    def add(reference):
        path=_bound(reference);files[str(path)]=binding(path);return path
    def plan(value):
        for path,sha in value.get('inputs',{}).items():add({'path':path,'sha256':sha})
    record=json.loads(add(handoff).read_text())
    intent=json.loads(add(record['intent']).read_text());add(intent['launch']);add(record['terminal']);add(record['checkpoint'])
    resident=record.get('resident') or {}
    for key in ('initial','current','plan'):
        if resident.get(key):
            value=json.loads(add(resident[key]).read_text())
            if key=='plan':plan(value)
            if key=='current':plan(value['payload']['plan'])
    transition=declaration.get('drained_recipe_handoff')
    if transition:
        value=json.loads(add(transition).read_text())
        for key in ('old_plan','new_plan'):
            plan(json.loads(add(value[key]).read_text()))
        for key in ('recipe','retirement'):
            if value.get(key):add(value[key])
        for key in ('old_package','new_package'):
            packages[key]=code_inventory(Path(value[key]))
            if packages[key]!=value[key.replace('_package','_code')]:
                raise ValueError('bootstrap proof package identity changed')
    cursor_path=Path(config['out']).parent/'cycle.json'
    if cursor_path.exists():
        cursor=json.loads(cursor_path.read_text())
        if cursor.get('resident_current'):
            value=json.loads(add(cursor['resident_current']).read_text());plan(value['payload']['plan'])
    from ..common.runtime_inventory import task_runtime_inventory
    from ..train.recipe import native_recipe_binding
    from .native_supervised import recipe_lineage
    lineage=recipe_lineage(config)
    return {'schema':'natlang.native-bootstrap-metadata/1','plan':binding(plan_path),
        'declaration_sha256':digest(declaration),'handoff':handoff,'files':files,'packages':packages,
        'verifier_code':code_inventory(Path(__file__).parents[1]),
        'collector':collector,
        'current_recipe':native_recipe_binding(Path(config['production_recipe']),lineage),
        'lineage':lineage,'raw_model_id':config.get('raw_model_id')}


def unchanged(path,plan_path,declaration,handoff,config):
    """Only reuse a local verifier-produced receipt after fresh owner/evidence checks."""
    path=Path(path)
    if not path.exists():return False
    value=json.loads(path.read_text())
    if value.get('config')!=binding(Path(config['out']).parent/'config.json'):return False
    from .native_raw_handoff import verify_handoff_metadata
    verify_handoff_metadata(handoff,requirement=declaration['handoff_requirement'])
    expected=declaration['handoff_requirement']['model_id']
    if config.get('raw_model_id')!=expected:return False
    actual=evidence(plan_path,declaration,handoff,config)
    return actual is not None and value.get('evidence')==actual


def save(path,plan_path,declaration,handoff,config):
    from ..common.jsonio import write_json_durable
    actual=evidence(plan_path,declaration,handoff,config)
    if actual is not None:
        write_json_durable(path,{'config':binding(Path(config['out']).parent/'config.json'),'evidence':actual})
