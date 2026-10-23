"""Exact native derivative-program and optimized-state provenance contracts."""
from __future__ import annotations

import hashlib
import json
import math
import re

from .native_gated_delta import PAYLOAD_FIELDS


_HISTORY_FIELD_PATTERN = '|'.join(('raw_k', 'v', 'gated_inputs', *(re.escape(name) for name in PAYLOAD_FIELDS)))

PROGRAM='natlang.native-derivative-program/1'
EXECUTION='natlang.native-derivative-execution/1'
DERIVATION='natlang.native-state-derivation/1'
STATE_OPT_ARITHMETIC='natlang.native-state-optimizer-torch-cpu/1'
_ID=re.compile(r'^(nzs1_|nnt1_)[a-z2-7]{52}$')
_SHA=re.compile(r'^[0-9a-f]{64}$')
_OPT_KEYS=('learning_rate','weight_decay','beta1','beta2','eps','momentum')


def state_optimizer_arithmetic():
    """Exact supported CPU executor, not a universal cross-CPU bitwise claim."""
    import inspect,platform,torch
    from ..train.optim import named_service_optimizer,functional_named_service_step
    from ..common.native_higher_order import adam_zero_moment_root
    implementation='\n'.join(inspect.getsource(value) for value in
        (named_service_optimizer,functional_named_service_step,adam_zero_moment_root))
    return {'schema':STATE_OPT_ARITHMETIC,'implementation_sha256':hashlib.sha256(implementation.encode()).hexdigest(),
        'torch_version':str(torch.__version__),'torch_git_version':str(torch.version.git_version),
        'torch_build_sha256':hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        'machine':platform.machine(),'cpu_capability':torch.backends.cpu.get_cpu_capability(),
        'python_version':platform.python_version(),'libc_version':'/'.join(platform.libc_ver()),
        'float32_subnormal_hex':torch.tensor([float.fromhex('0x1p-149')],dtype=torch.float32).item().hex()}


def validate_state_optimizer_arithmetic(value):
    required={'schema','implementation_sha256','torch_version','torch_git_version','torch_build_sha256',
        'machine','cpu_capability','python_version','libc_version','float32_subnormal_hex'}
    if (not isinstance(value,dict) or set(value)!=required or value['schema']!=STATE_OPT_ARITHMETIC
            or any(not isinstance(item,str) or not item for item in value.values())
            or any(_SHA.fullmatch(value[key]) is None for key in ('implementation_sha256','torch_build_sha256'))):
        raise ValueError('native state optimizer arithmetic needs its exact qualified CPU implementation/runtime binding')


def native_id(value,prefix=None):
    return isinstance(value,str) and _ID.fullmatch(value) is not None and (prefix is None or value.startswith(prefix))


def optimizer_binding(options):
    if (not isinstance(options,dict) or options.get('name') not in ('sgd','adam') or set(options)-{'name',*_OPT_KEYS}
            or any(type(value) not in (int,float) for key,value in options.items() if key!='name')):
        raise ValueError('native optimizer options require finite numeric settings')
    defaults={'learning_rate':1e-3,'weight_decay':0.,'beta1':.9,'beta2':.999,'eps':1e-8,'momentum':0.}
    result={'name':options['name'],**{key+'_hex':float(options.get(key,default)).hex() for key,default in defaults.items()}}
    optimizer_options(result)
    return result


def optimizer_options(binding):
    if not isinstance(binding,dict) or set(binding)!={'name',*(key+'_hex' for key in _OPT_KEYS)}:
        raise ValueError('native state optimizer binding needs exact named hexadecimal options')
    result={'name':binding['name']}
    for key in _OPT_KEYS:
        text=binding[key+'_hex']
        try:value=float.fromhex(text)
        except (ValueError,TypeError):raise ValueError('native state optimizer binding has invalid hexadecimal options') from None
        if not math.isfinite(value) or value.hex()!=text:
            raise ValueError('native state optimizer binding requires canonical finite hexadecimal options')
        result[key]=value
    if (result['name'] not in ('sgd','adam') or result['learning_rate']<=0 or result['weight_decay']<0
            or not 0<=result['beta1']<1 or not 0<=result['beta2']<1 or result['eps']<=0 or result['momentum']<0):
        raise ValueError('native state optimizer binding has out-of-range options')
    return result


def validate_derivation(value):
    if value is None:return
    if not isinstance(value,dict) or value.get('schema')!=DERIVATION:
        raise ValueError('native state derivation needs its versioned exact contract')
    kind=value.get('kind')
    if kind=='optimizer':
        required={'schema','kind','origin_state_ids','parent_state_id','gradient_id','update_artifact_id',
                  'program_id','optimizer_state_parent_id','step','optimizer','updated_tensor_names','arithmetic'}
        if (set(value)!=required or not native_id(value['parent_state_id'],'nzs1_')
                or not isinstance(value['origin_state_ids'],list) or not value['origin_state_ids']
                or any(not native_id(ident,'nzs1_') for ident in value['origin_state_ids'])
                or value['origin_state_ids']!=sorted(set(value['origin_state_ids']))
                or any(not native_id(value[key],'nnt1_') for key in ('gradient_id','update_artifact_id'))
                or any(value[key] is not None and not native_id(value[key],'nnt1_') for key in ('program_id','optimizer_state_parent_id'))
                or type(value['step']) is not int or value['step']<1
                or not isinstance(value['updated_tensor_names'],list) or not value['updated_tensor_names']
                or any(not isinstance(key,str) or not re.fullmatch(r'layers\.[0-9]+\.(' + _HISTORY_FIELD_PATTERN + ')',key)
                       for key in value['updated_tensor_names'])
                or value['updated_tensor_names']!=sorted(set(value['updated_tensor_names']))):
            raise ValueError('native optimized state has invalid exact origin, update or history bindings')
        optimizer_options(value['optimizer'])
        validate_state_optimizer_arithmetic(value['arithmetic'])
    elif kind in ('semantic-encode','semantic-adapt'):
        required={'schema','kind','origin_state_ids','parent_state_id','adapter_sha256','source_native_identity','target_native_identity'}
        if 'training_execution' in value or 'source_adapter_sha256' in value:
            required|={'training_execution','source_adapter_sha256'}
            execution=value.get('training_execution')
            if (kind!='semantic-adapt' or not isinstance(execution,dict)
                    or set(execution)!={'schema','parent_checkpoint_sha256','parent_adapter_sha256','parent_step','step','behavior_adapter_sha256'}
                    or execution['schema']!='natlang.native-semantic-policy-execution/1'
                    or any(not isinstance(execution[key],str) or _SHA.fullmatch(execution[key]) is None
                           for key in ('parent_checkpoint_sha256','parent_adapter_sha256','behavior_adapter_sha256'))
                    or any(type(execution[key]) is not int or execution[key]<0 for key in ('parent_step','step'))
                    or execution['step']<execution['parent_step']
                    or value.get('source_adapter_sha256')!=execution['behavior_adapter_sha256']
                    or value.get('adapter_sha256')!=execution['parent_adapter_sha256']
                    or (value.get('target_native_identity') or {}).get('checkpoint_sha256')!=execution['parent_checkpoint_sha256']):
                raise ValueError('training semantic derivation needs exact immutable behavior and current decoder execution')
        if (set(value)!=required or not native_id(value['parent_state_id'],'nzs1_')
                or not isinstance(value['origin_state_ids'],list) or not value['origin_state_ids']
                or any(not native_id(item,'nzs1_') for item in value['origin_state_ids'])
                or value['origin_state_ids']!=sorted(set(value['origin_state_ids']))
                or not isinstance(value['adapter_sha256'],str) or _SHA.fullmatch(value['adapter_sha256']) is None):
            raise ValueError('native semantic derivation needs exact source, origins and adapter')
        for key in ('source_native_identity','target_native_identity'):
            identity=value[key]
            if (not isinstance(identity,dict) or set(identity)!={'checkpoint_sha256','tokenizer_fingerprint','embedding_weight_sha256','architecture'}
                    or any(not isinstance(identity[name],str) or _SHA.fullmatch(identity[name]) is None
                        for name in ('checkpoint_sha256','tokenizer_fingerprint','embedding_weight_sha256'))
                    or not isinstance(identity['architecture'],str) or not identity['architecture']):
                raise ValueError('native semantic derivation needs exact source/target model identities')
    else:raise ValueError('unknown native state derivation kind')


def derivation_dependencies(value):
    validate_derivation(value)
    if value is None:return []
    keys=('origin_state_ids','parent_state_id','gradient_id','update_artifact_id','program_id','optimizer_state_parent_id')
    result=set(value.get('origin_state_ids',[]))
    result.update(value[key] for key in keys if key!='origin_state_ids' and value.get(key) is not None)
    return sorted(result)


def validate_finite_program(program,*,bound=False):
    required={'schema','inputs','nodes','outer','wrt','arithmetic'}
    if not isinstance(program,dict) or set(program)!=required or program.get('schema')!=PROGRAM:
        raise ValueError('native derivative program needs its exact versioned finite input/node/outer contract')
    arithmetic=program['arithmetic']
    if (not isinstance(arithmetic,dict) or set(arithmetic)!={'schema','query_chunk_tokens'}
            or arithmetic['schema']!='natlang.native-reference-arithmetic/1'
            or type(arithmetic['query_chunk_tokens']) is not int or arithmetic['query_chunk_tokens']<1):
        raise ValueError('native derivative program requires explicit reference arithmetic')
    inputs,nodes,wrt=program['inputs'],program['nodes'],program['wrt']
    if (not isinstance(inputs,dict) or not inputs or any(not isinstance(key,str) or not key for key in inputs)
            or not isinstance(nodes,list) or not isinstance(wrt,list) or not wrt
            or any(not isinstance(key,str) or key not in inputs for key in wrt) or len(set(wrt))!=len(wrt)):
        raise ValueError('native derivative program needs unique roots and original differentiation inputs')
    kinds={}
    for name,target in inputs.items():
        if not isinstance(target,dict) or set(target)-{'kind','id','names'}:
            raise ValueError('native derivative inputs need an exact typed target')
        kind=target.get('kind');ident=target.get('id')
        if kind not in ('model-parameters','adapter','native-state'):
            raise ValueError('native derivative input kind is model-parameters, adapter or native-state')
        if (kind!='model-parameters' or ident is not None or bound) and not native_id(ident,'nzs1_' if kind=='native-state' else 'nnt1_'):
            raise ValueError('native derivative input needs its exact native artifact ID')
        names=target.get('names')
        if (names is not None or kind=='model-parameters') and (not isinstance(names,list) or not names
                or any(not isinstance(key,str) or not key for key in names) or len(set(names))!=len(names)):
            raise ValueError('native derivative named targets must be explicit and unique')
        kinds[name]=kind
    values=dict(kinds);gradients={};all_names=set(inputs)
    def objective(value):
        if not isinstance(value,dict) or set(value)!={'request','bindings'}:
            raise ValueError('native derivative objective needs its exact request and input bindings')
        bindings=value['bindings'];request=value['request']
        if (not isinstance(bindings,dict) or any(key not in inputs or not isinstance(item,str) or item not in values
                or values[item]!=kinds[key] for key,item in bindings.items())):
            raise ValueError('native derivative bindings must select available values for exact declared roots')
        if (not isinstance(request,dict) or request.get('schema')!='natlang.native-gradient-request/1'
                or any(key in request for key in ('target','order','derived','create_graph'))
                or not isinstance(request.get('graph'),dict)):
            raise ValueError('native derivative objectives need a complete shared native graph request without a separate target')
        if isinstance(request.get('objective'),dict) and request['objective'].get('kind')!='native-query':
            raise ValueError('native derivative registered-policy objectives require portable bound objective assets')
        query_readout_dependencies(request.get('objective'))
    for node in nodes:
        if (not isinstance(node,dict) or not isinstance(node.get('name'),str) or not node['name']
                or node['name'] in all_names):raise ValueError('native derivative nodes require unique ordered names')
        name=node['name'];kind=node.get('kind')
        if kind=='gradient':
            if set(node)!={'kind','name','value','request','bindings'} or node['value'] not in values:
                raise ValueError('native derivative gradient references an unavailable earlier value')
            objective({key:node[key] for key in ('request','bindings')})
            if not any(node['bindings'].get(key,key)==node['value'] for key in inputs):
                raise ValueError('native derivative gradient value must be bound in the actual objective')
            gradients[name]=node['value']
        elif kind=='optimizer':
            if (set(node)!={'kind','name','value','gradient','optimizer','optimizer_state'}
                    or node['value'] not in values or gradients.get(node['gradient'])!=node['value']):
                raise ValueError('native derivative update gradient belongs to another exact earlier value')
            optimizer_binding(node['optimizer'])
            history=node['optimizer_state']
            if history is not None and history!=node['value'] and not native_id(history,'nnt1_'):
                raise ValueError('native derivative optimizer needs explicit new or exact earlier history')
            if history==node['value'] and history in inputs:
                raise ValueError('native derivative root has no implicit optimizer history')
            values[name]=values[node['value']]
        else:raise ValueError('native derivative node kind must be gradient or optimizer')
        all_names.add(name)
    objective(program['outer'])


def validate_program_artifact(provenance):
    if not isinstance(provenance,dict) or provenance.get('schema') not in (PROGRAM,EXECUTION):
        raise ValueError('native derivative artifact requires a declared program or execution receipt')
    if provenance['schema']==PROGRAM:
        text=provenance.get('program_json')
        try:parsed=json.loads(text) if isinstance(text,str) else None
        except (ValueError,TypeError):parsed=None
        if (not isinstance(parsed,dict) or parsed.get('schema')!=PROGRAM
                or hashlib.sha256(text.encode()).hexdigest()!=provenance.get('program_sha256')):
            raise ValueError('native derivative program JSON differs from its immutable content binding')
        validate_finite_program(parsed,bound=True)
        if provenance.get('dependencies')!=program_dependencies(parsed):
            raise ValueError('native derivative declaration omitted or changed its exact executable dependencies')
    elif not native_id(provenance.get('program_id'),'nnt1_'):
        raise ValueError('native derivative execution requires its exact program artifact')
    dependencies=provenance.get('dependencies')
    if (not isinstance(dependencies,list) or any(not native_id(value) for value in dependencies)
            or dependencies!=sorted(set(dependencies))):
        raise ValueError('native derivative artifact needs its exact sorted dependency inventory')


def validate_derivative_target(artifact):
    value=(artifact.provenance or {}).get('native_derivative_target')
    if value is None:return None
    provenance=artifact.provenance
    expected={'adapter-gradient':'adapter','state-gradient':'native-state','parameter-gradient':'model-parameters'}.get(artifact.kind)
    if (not isinstance(value,dict) or set(value)!={'schema','stage','variable','value_id','kind','names'}
            or value.get('schema')!='natlang.native-derivative-target/1' or value.get('stage') not in ('inner','outer')
            or not isinstance(value.get('variable'),str) or not value['variable'] or value.get('kind')!=expected
            or not native_id(provenance.get('derivative_program_id'),'nnt1_')
            or value['stage']=='outer' and not native_id(provenance.get('derivative_execution_id'),'nnt1_')
            or value.get('value_id')!=(provenance.get('parameter_value_id') if expected=='model-parameters' else artifact.source_artifact_id)
            or not native_id(value.get('value_id'),'nzs1_' if expected=='native-state' else 'nnt1_')
            or value.get('names')!=sorted(artifact.tensors)):
        raise ValueError('native derivative gradient needs its exact differentiation origin distinct from evaluated actors')
    return value


def validate_functional_actor(artifact):
    provenance=artifact.provenance or {};actor=provenance.get('functional_actor')
    if actor is None:return None
    from ..common.hashing import canonical_json_sha256_hex
    from ..common.native_higher_order import ReferenceArithmetic
    required={'schema','base_native_identity','parameter_values','uniform_adapters','invocation_adapters','arithmetic'}
    if (not isinstance(actor,dict) or set(actor)!=required or actor['schema']!='natlang.native-functional-actor/1'
            or actor['base_native_identity']!=artifact.native_identity
            or provenance.get('functional_actor_sha256')!=canonical_json_sha256_hex(actor)):
        raise ValueError('native functional actor needs its exact base, overlays and arithmetic identity')
    values=actor['parameter_values'];names=set();ids=[]
    if not isinstance(values,list):raise ValueError('native functional actor parameter overlays must be ordered exact artifacts')
    for entry in values:
        if (not isinstance(entry,dict) or set(entry)!={'id','names'} or not native_id(entry['id'],'nnt1_')
                or not isinstance(entry['names'],list) or not entry['names']
                or any(not isinstance(key,str) or not key for key in entry['names'])
                or entry['names']!=sorted(set(entry['names'])) or names.intersection(entry['names'])):
            raise ValueError('native functional actor has overlapping or unbound parameter overlays')
        names.update(entry['names']);ids.append(entry['id'])
    if ids!=sorted(set(ids)):raise ValueError('native functional actor parameter identities must be unique and ordered')
    arithmetic=actor['arithmetic'];chunk=arithmetic.get('query_chunk_tokens') if isinstance(arithmetic,dict) else None
    if type(chunk) is not int or chunk<1 or arithmetic!=ReferenceArithmetic(query_chunk_tokens=chunk).binding():
        raise ValueError('native functional actor requires the complete exact reference arithmetic contract')
    uniform=actor['uniform_adapters'];invocations=actor['invocation_adapters']
    if (not isinstance(uniform,list) or invocations is not None and (not isinstance(invocations,dict) or not invocations or uniform)):
        raise ValueError('native functional actor needs one exact uniform or per-invocation coefficient declaration')
    for entries in [uniform,*(invocations or {}).values()]:
        if not isinstance(entries,list):raise ValueError('native functional coefficients must be ordered declarations')
        for entry in entries:
            if (not isinstance(entry,dict) or set(entry)!={'id','scale_hex'} or not native_id(entry['id'],'nnt1_')):
                raise ValueError('native functional coefficients need exact artifact IDs and scales')
            try:scale=float.fromhex(entry['scale_hex'])
            except (ValueError,TypeError):raise ValueError('native functional coefficient scale is invalid') from None
            if not math.isfinite(scale) or scale.hex()!=entry['scale_hex']:raise ValueError('native functional coefficient scale is not canonical finite hexadecimal')
    if not values:
        if uniform!=(provenance.get('adapter_binding') or {}).get('adapters',[]):
            raise ValueError('native functional actor differs from its evaluated uniform coefficients')
        receipt=provenance.get('invocation_actors')
        actual=({name:(item.get('adapter_binding') or {}).get('adapters',[]) for name,item in receipt['invocations'].items()}
                if receipt is not None else None)
        if invocations!=actual:raise ValueError('native functional actor differs from its evaluated per-invocation coefficients')
    return actor


def verify_derivative_gradient_origin(store,artifact):
    """Exact program/actual-evaluation links, without a false numerical rerender claim."""
    target=validate_derivative_target(artifact)
    if target is None:return
    actor=validate_functional_actor(artifact)
    if actor is None:raise ValueError('native derivative gradient lacks its complete actual functional actor')
    provenance=artifact.provenance
    program=store.get_native_tensor_artifact(provenance['derivative_program_id'])
    if program is None or program.artifact.kind!='derivative-program' or program.artifact.provenance.get('schema')!=PROGRAM:
        raise ValueError('native derivative gradient needs its immutable executable program declaration')
    parsed=json.loads(program.artifact.provenance['program_json']);validate_finite_program(parsed,bound=True)
    if program.artifact.native_identity!=artifact.native_identity:raise ValueError('native derivative program belongs to another exact base')
    inputs=provenance.get('evaluation_inputs')
    if not isinstance(inputs,dict) or set(inputs)!=set(parsed['inputs']):raise ValueError('native derivative evaluation lost its complete root binding inventory')
    for name,value in inputs.items():
        if (not isinstance(value,dict) or set(value)!={'kind','value_id','names'} or value['kind']!=parsed['inputs'][name]['kind']
                or not native_id(value['value_id'],'nzs1_' if value['kind']=='native-state' else 'nnt1_')
                or not isinstance(value['names'],list) or any(not isinstance(key,str) or not key for key in value['names'])
                or value['names']!=sorted(set(value['names']))):
            raise ValueError('native derivative evaluation has invalid exact bound root values')
        record=(store.get_native_state(value['value_id']) if value['kind']=='native-state'
                else store.get_native_tensor_artifact(value['value_id']))
        if record is None:raise ValueError('native derivative evaluation lacks an exact bound root artifact')
        if value['kind']=='native-state':
            from .native_state_derivation import state_values
            available=state_values(record.block)
        else:
            expected_kind='adapter-coefficients' if value['kind']=='adapter' else 'parameter-set'
            if record.artifact.kind!=expected_kind or record.artifact.native_identity!=artifact.native_identity:
                raise ValueError('native derivative evaluation root kind or exact base differs')
            available=record.artifact.tensors
        if not value['names'] or not set(value['names'])<=available.keys():
            raise ValueError('native derivative evaluation root names differ from its exact artifact')
    if target['stage']=='inner':
        node=next((node for node in parsed['nodes'] if node['name']==provenance.get('node')),None)
        if node is None or node['kind']!='gradient' or node['value']!=target['variable']:
            raise ValueError('native derivative gradient changed its declared inner objective')
        owners=[key for key in parsed['inputs'] if node['bindings'].get(key,key)==target['variable']]
        if not owners or any(inputs[key]['value_id']!=target['value_id'] or inputs[key]['names']!=target['names'] for key in owners):
            raise ValueError('native derivative inner gradient belongs to another exact evaluated value')
    else:
        root=parsed['inputs'].get(target['variable'])
        execution=store.get_native_tensor_artifact(provenance['derivative_execution_id'])
        if (root is None or root['id']!=target['value_id'] or execution is None
                or execution.artifact.kind!='derivative-program' or execution.artifact.native_identity!=artifact.native_identity
                or execution.artifact.provenance.get('schema')!=EXECUTION
                or execution.artifact.provenance.get('program_id')!=program.id
                or execution.artifact.provenance.get('outer_graph_sha256')!=provenance.get('graph_sha256')
                or execution.artifact.provenance.get('outer_evaluation_inputs')!=inputs
                or execution.artifact.provenance.get('outer_functional_actor')!=actor):
            raise ValueError('native derivative outer gradient differs from its exact original input and actual execution')
    objective=node if target['stage']=='inner' else parsed['outer']
    request=objective['request']
    # Bind all evaluated roots, not only the one whose adjoint is returned.
    for name,value in inputs.items():
        variable=objective['bindings'].get(name,name)
        if variable in parsed['inputs']:
            if value['value_id']!=parsed['inputs'][variable]['id']:
                raise ValueError('native derivative objective changed an unchanged declared root')
        else:
            update=next((item for item in parsed['nodes'] if item['name']==variable),None)
            if update is None or update['kind']!='optimizer':raise ValueError('native derivative objective lacks its declared updated value')
            if value['kind']=='native-state':
                state=store.get_native_state(value['value_id']).block
                derivation=state.derivation or {}
                gradient=store.get_native_tensor_artifact(derivation.get('gradient_id'))
                if (derivation.get('program_id')!=program.id or gradient is None or
                        gradient.artifact.provenance.get('node')!=update['gradient'] or
                        derivation.get('optimizer')!=optimizer_binding(update['optimizer'])):
                    raise ValueError('native derivative objective changed its declared state update')
            else:
                current=store.get_native_tensor_artifact(value['value_id']).artifact
                if current.provenance.get('derivative_program_id')!=program.id or current.provenance.get('node')!=variable:
                    raise ValueError('native derivative objective changed its declared parameter update')
    import copy
    expected_graph=copy.deepcopy(request['graph'])
    replacements={root['id']:inputs[name]['value_id'] for name,root in parsed['inputs'].items() if root['kind']=='native-state'}
    for value in (expected_graph.get('external_states') or {}).values():value['id']=replacements.get(value['id'],value['id'])
    from ..common.hashing import canonical_json_sha256_hex
    text=provenance.get('graph_json')
    if (not isinstance(text,str) or hashlib.sha256(text.encode()).hexdigest()!=provenance.get('graph_sha256') or
            json.loads(text)!=expected_graph or canonical_json_sha256_hex(expected_graph)!=provenance.get('graph_sha256')):
        raise ValueError('native derivative gradient changed its declared objective graph')
    expected_objective=copy.deepcopy(request.get('objective'))
    if isinstance(expected_objective,dict) and expected_objective.get('kind')=='native-query':
        expected_objective['graph_sha256']=provenance['graph_sha256']
        text=provenance.get('query_objective_json')
        if (not isinstance(text,str) or hashlib.sha256(text.encode()).hexdigest()!=provenance.get('query_objective_sha256') or
                json.loads(text)!=expected_objective):
            raise ValueError('native derivative gradient changed its declared query objective')
    elif provenance.get('query_objective_json') is not None:
        raise ValueError('native derivative gradient introduced an undeclared query objective')
    for entries in [actor['uniform_adapters'],*(actor['invocation_adapters'] or {}).values()]:
        for entry in entries:
            record=store.get_native_tensor_artifact(entry['id'])
            if record is None or record.artifact.kind!='adapter-coefficients' or record.artifact.native_identity!=artifact.native_identity:
                raise ValueError('native functional actor lacks exact evaluated coefficients')
    for entry in actor['parameter_values']:
        record=store.get_native_tensor_artifact(entry['id'])
        if (record is None or record.artifact.kind!='parameter-set' or record.artifact.native_identity!=artifact.native_identity
                or sorted(record.artifact.tensors)!=entry['names']):
            raise ValueError('native functional actor lacks an exact evaluated parameter set')


def program_dependencies(program):
    """All native roots/static inputs required to execute this immutable program."""
    result={target['id'] for target in program['inputs'].values() if target.get('id') is not None}
    requests=[*(node['request'] for node in program['nodes'] if node['kind']=='gradient'),program['outer']['request']]
    for request in requests:
        for entries in [request.get('adapters') or [],*(request.get('invocation_adapters') or {}).values()]:
            result.update(entry if isinstance(entry,str) else entry['id'] for entry in entries)
        result.update(value['id'] for value in (request['graph'].get('external_states') or {}).values())
        result.update(query_readout_dependencies(request.get('objective')))
    return sorted(result)


def query_readout_dependencies(spec):
    if not isinstance(spec,dict) or spec.get('kind')!='native-query':return []
    terms=spec.get('terms')
    if not isinstance(terms,list) or any(not isinstance(term,dict) for term in terms):
        raise ValueError('native query needs its actual declared term inventory')
    result=set()
    for term in terms:
        declared=term.get('readouts',[])
        teachers=term.get('teacher_readouts',[])
        if not isinstance(declared,list) or not isinstance(teachers,list):raise ValueError('native query readouts must be an ordered list')
        for ident in [term.get('readout'),term.get('teacher_readout'),*declared,*teachers]:
            if ident is None:continue
            if not native_id(ident,'nnt1_'):raise ValueError('native query readout needs its exact native tensor artifact ID')
            result.add(ident)
    return sorted(result)
