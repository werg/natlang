"""Exact source-curated offline preference eligibility; no numerical/capacity claim."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import torch

from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.data.native_graph_schema import parse_graph
from natlang_neuralese.data.native_preference import pair_graph
from natlang_neuralese.model.native_model import tokenizer_inventory
from natlang_neuralese.train.native_preference import NativePreferenceObjective,SCHEMA,KIND
from scripts.neuralese_data.acquired_preferences import normalized_preference_records
from scripts.neuralese_data.build_acquired_full_session_graphs import sessions,completed_shards,prepare_tokenizer
from scripts.neuralese_data.build_acquired_preference_graphs import (
    SHARD_SCHEMA,ADDITIONAL,protected_pair,_runtime,_bound,ROOT)
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
from scripts.neuralese_data.decision_holds import load_decision_holds
from scripts.neuralese_data.prepare_acquired_session_admission import bound,verify_global_source_closure,decision_contract

CODE_ROOT=Path(__file__).resolve().parents[2]

# D24's durable six-view conversion was produced before three shared helpers
# changed. Those views are reusable only through the full source/eligibility
# and per-pair current-tokenizer rerender below; every other runtime byte must
# still match. The transition is exact and cannot authorize a future version.
_D24_RENDER_REQUALIFICATION={
    'from_fingerprint':'469eec820c8947e9302a25f861435d7e22ec0372b642815f6b758142ed62484a',
    'to_fingerprint':'45902d4663905c5d0d261cabd90b0ba126272e9b4cb94aeb34976128d35e7e8a',
    'changed_files':{
        'preference_builder':('99641ba3ae8e3887e63a4ea69f8cd304d86a39ba06208fa4fd918119714ec2e5',
                              'd8f120e0ad0d00e11fd277812ef0556fdb438f50de141f798c9a62100ad15810'),
        'serve_chat':('436ad40940e17a6c79c7a898bdcd89e069271164dc019ae77f68b443f9f2f5c8',
                      '12e9dfca075a1159b72fc5f2be1b7b376c65e5e8780a766adc62ad78a4ceea36'),
        'source_admission':('221c77789a01cc2c8588e406cac7e30eb0a69bc2c05429689e1b2c28768d83e6',
                            '618beef2bacb26995cdf391ce07e621f6ab43ab146ca91fa5f6033625d8fcda9'),
    }}

# The model loader also owns tokenizer inventory. D24's old renderer predates
# unrelated attention-loading changes to native_model.py, so a whole-module
# hash is too broad for that one dependency. This witness pins the two exact
# functions and the imports/globals they use. The module's complete old/new
# hashes remain in the receipt for provenance; any change to these functions,
# their imports, or a rebinding of their globals still rejects the transition.
_TOKENIZER_INVENTORY_WITNESS={
    'module':'training/neuralese/natlang_neuralese/model/native_model.py',
    'functions':{
        '_input_inventory':'9deb4993237bbe0d853ca4912fda519e3b60aa68a05bacd2598efaa82eaf2821',
        'tokenizer_inventory':'7d1f7639865c403de69295eae79c7ff84c6363e0607a4192a4f685f6327fec8c',
    },
    'imports':'7c5568b938f4cc9d2cb8264ee8e05304adb23cb4298c2c04277529ed204d1321',
    'bindings':['Path','sha256_file_hex_cached','_input_inventory','tokenizer_inventory'],
}


def _tokenizer_inventory_ast_witness(path):
    source=Path(path).read_text()
    tree=ast.parse(source)
    wanted=set(_TOKENIZER_INVENTORY_WITNESS['functions'])
    definitions=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))]
    functions={node.name:node for node in definitions}
    if not wanted.issubset(functions) or any(sum(node.name==name for node in definitions)!=1 for name in wanted):
        raise ValueError('tokenizer inventory function witness is incomplete')
    actual_functions={name:hashlib.sha256(ast.dump(functions[name],include_attributes=False).encode()).hexdigest()
                      for name in sorted(wanted)}
    # Freeze only imports relevant to the inventory functions. The definitions
    # are still exact ASTs, and all names they can resolve are checked below.
    relevant_imports=[]
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and node.module=='pathlib':
            if any(alias.name=='Path' for alias in node.names):relevant_imports.append(node)
        elif isinstance(node,ast.ImportFrom) and node.module and node.module.endswith('common.hashing'):
            if any(alias.name=='sha256_file_hex_cached' for alias in node.names):relevant_imports.append(node)
    import_sha=hashlib.sha256('\n'.join(ast.dump(node,include_attributes=False)
                                            for node in relevant_imports).encode()).hexdigest()
    if actual_functions!=_TOKENIZER_INVENTORY_WITNESS['functions'] or import_sha!=_TOKENIZER_INVENTORY_WITNESS['imports']:
        raise ValueError('tokenizer inventory implementation differs from its qualified function witness')
    bindings=set(_TOKENIZER_INVENTORY_WITNESS['bindings'])
    defining_nodes=[]
    for node in tree.body:
        names=set()
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node,(ast.Import,ast.ImportFrom)):
            names.update(alias.asname or alias.name.split('.')[0] for alias in node.names)
        else:
            for child in ast.walk(node):
                if isinstance(child,ast.Name) and isinstance(child.ctx,ast.Store):names.add(child.id)
        overlap=names & bindings
        if overlap:
            allowed=((isinstance(node,ast.ImportFrom) and node.module=='pathlib' and overlap=={'Path'})
                     or (isinstance(node,ast.ImportFrom) and node.module and node.module.endswith('common.hashing')
                         and overlap=={'sha256_file_hex_cached'})
                     or (isinstance(node,ast.FunctionDef) and node.name in wanted))
            if not allowed:
                raise ValueError('tokenizer inventory dependency global is rebound: '+','.join(sorted(overlap)))
    for function in (node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))):
        declared={name for child in ast.walk(function) if isinstance(child,ast.Global) for name in child.names}
        rebound={child.id for child in ast.walk(function)
                 if isinstance(child,ast.Name) and isinstance(child.ctx,ast.Store)
                 and child.id in bindings and child.id in declared}
        if rebound:
            raise ValueError('tokenizer inventory dependency global is rebound: '+','.join(sorted(rebound)))
    return {'module_path':_TOKENIZER_INVENTORY_WITNESS['module'],
            'module_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'function_ast_sha256':actual_functions,'relevant_imports_ast_sha256':import_sha,
            'forbidden_rebindings':sorted(bindings)}


def _conversion_runtime_transition(conversion_runtime,verification_runtime=None):
    verification_runtime=_runtime() if verification_runtime is None else verification_runtime
    if conversion_runtime==verification_runtime:
        return {'kind':'identical-runtime','conversion_fingerprint':conversion_runtime.get('fingerprint'),
                'verification_fingerprint':verification_runtime.get('fingerprint'),'changed_files':{}}
    expected=_D24_RENDER_REQUALIFICATION
    old_files=conversion_runtime.get('files') or {};new_files=verification_runtime.get('files') or {}
    differences={key:(old_files.get(key) or {}).get('sha256') for key in sorted(set(old_files)|set(new_files))
        if (old_files.get(key) or {}).get('sha256')!=(new_files.get(key) or {}).get('sha256')}
    named={key:pair[0] for key,pair in expected['changed_files'].items()}
    inventory_key='tokenizer_inventory'
    inventory_old='99eee6d5b2bbe87b8957b3df486951fdf37e8a102d73eaf0cbd91f7274924e9d'
    inventory_proof=None
    inventory_transition=(inventory_key in differences
                          and (old_files.get(inventory_key) or {}).get('sha256')==inventory_old)
    if inventory_transition:
        inventory_proof=_tokenizer_inventory_ast_witness(
            CODE_ROOT/_TOKENIZER_INVENTORY_WITNESS['module'])
    allowed_differences=set(named)|({inventory_key} if inventory_transition else set())
    if (conversion_runtime.get('fingerprint')!=expected['from_fingerprint']
            or conversion_runtime.get('fingerprint')!=canonical_json_sha256_hex(old_files)
            or verification_runtime.get('fingerprint')!=canonical_json_sha256_hex(new_files)
            or set(differences)!=allowed_differences
            or set(old_files)!=set(new_files)
            or (inventory_transition and (new_files.get(inventory_key) or {}).get('sha256')!=inventory_proof['module_sha256'])
            or any((new_files.get(key) or {}).get('sha256')!=pair[1]
                   for key,pair in expected['changed_files'].items())):
        raise ValueError('offline conversion runtime differs outside its exact all-pair rerender continuation')
    proof={'kind':'exact-all-pair-rerender-continuation',
        'conversion_fingerprint':conversion_runtime['fingerprint'],
        'verification_fingerprint':verification_runtime['fingerprint'],
        'changed_files':{key:{'conversion_sha256':pair[0],'verification_sha256':pair[1]}
                         for key,pair in expected['changed_files'].items()},
        'required_requalification':['full-current-source-eligibility','all-current-tokenizer-pair-renders',
                                    'exact-graph-and-objective-hashes','global-split-and-held-joins']}
    if inventory_proof is not None:
        proof['function_scoped_tokenizer_inventory_transition']={
            'conversion_module_sha256':inventory_old,
            'verification_module_sha256':inventory_proof['module_sha256'],
            'qualified_function_ast_sha256':inventory_proof['function_ast_sha256'],
            'qualified_imports_ast_sha256':inventory_proof['relevant_imports_ast_sha256'],
            'forbidden_rebindings':inventory_proof['forbidden_rebindings'],
            'scope':'tokenizer_inventory and _input_inventory are byte-equivalent ASTs with qualified imports; unrelated module changes are provenance-bound only'}
    elif old_files.get(inventory_key)!=new_files.get(inventory_key):
        raise ValueError('tokenizer inventory module drift has no exact function-scoped witness')
    return proof


def _verification_dependencies():
    """Code that can change the meaning of a cached source/render verification."""
    root=CODE_ROOT
    paths={
        'admission':Path(__file__),
        'preference_builder':root/'scripts/neuralese_data/build_acquired_preference_graphs.py',
        'preference_reader':root/'scripts/neuralese_data/acquired_preferences.py',
        'session_reader':root/'scripts/neuralese_data/build_acquired_full_session_graphs.py',
        'source_admission':root/'scripts/neuralese_data/prepare_acquired_session_admission.py',
        'source_policy':root/'scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py',
        'decision_holds':root/'scripts/neuralese_data/decision_holds.py',
        'graph_parser':root/'training/neuralese/natlang_neuralese/data/native_graph_schema.py',
        'pair_renderer':root/'training/neuralese/natlang_neuralese/data/native_preference.py',
        'pair_objective':root/'training/neuralese/natlang_neuralese/train/native_preference.py',
    }
    return {name:{'path':path.resolve(strict=True).relative_to(root).as_posix(),
                  'bytes':path.stat().st_size,'sha256':sha256_file_hex(path)}
            for name,path in sorted(paths.items())}


def _cached_verification(path,identity):
    if not path.is_file():return None
    record=json.loads(path.read_text())
    payload=record.get('payload')
    if (record.get('schema')!='natlang.offline-preference-view-verification/1'
            or not isinstance(payload,dict)
            or record.get('payload_sha256')!=canonical_json_sha256_hex(payload)
            or payload.get('identity_sha256')!=canonical_json_sha256_hex(identity)
            or canonical_json_sha256_hex(payload.get('identity'))!=canonical_json_sha256_hex(identity)):
        return None
    return payload.get('result')


def _write_owned_json(path,value):
    """Publish an entrypoint idempotently, accepting only exact prior bytes."""
    path=Path(path)
    if path.exists():
        try:existing=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError) as error:
            raise ValueError(f'partial offline proposal file is not valid JSON: {path.name}') from error
        if canonical_json_sha256_hex(existing)!=canonical_json_sha256_hex(value):
            raise ValueError(f'offline proposal file belongs to different content: {path.name}')
        return
    write_json_durable(path,value)


SOURCE_INDEX_SCHEMA='natlang.offline-preference-source-eligibility-index/1'
SOURCE_INDEX_RECORD_SCHEMA='natlang.offline-preference-source-eligibility-record/1'


def _source_index_implementation():
    """Bind the exact parser, pair-join and source policy used by this index."""
    root=CODE_ROOT
    paths={
        'admission':Path(__file__),
        'preference_reader':root/'scripts/neuralese_data/acquired_preferences.py',
        'session_reader':root/'scripts/neuralese_data/build_acquired_full_session_graphs.py',
        'source_policy':root/'scripts/neuralese_data/build_acquired_preference_graphs.py',
        'decision_holds':root/'scripts/neuralese_data/decision_holds.py',
        'source_admission':root/'scripts/neuralese_data/prepare_acquired_session_admission.py',
    }
    return {name:{'path':path.resolve(strict=True).relative_to(root).as_posix(),
                  'bytes':path.stat().st_size,'sha256':sha256_file_hex(path)}
            for name,path in sorted(paths.items())}


def _read_source_index(index_path,identity,*,state=None):
    """Read and check a committed source-index prefix, truncating only torn/uncommitted tail bytes."""
    index_path=Path(index_path)
    identity_sha=canonical_json_sha256_hex(identity)
    state_path=index_path.with_suffix('.state.json')
    if state is None:
        state=json.loads(state_path.read_text()) if state_path.is_file() else None
    committed_offset=0;committed_units=0;source_frontier=0;rows=[]
    if state is not None:
        if (state.get('schema')!=SOURCE_INDEX_SCHEMA or state.get('identity')!=identity
                or state.get('identity_sha256')!=identity_sha
                or type(state.get('byte_offset')) is not int or type(state.get('units')) is not int
                or type(state.get('source_frontier')) is not int):
            raise ValueError('offline preference source-index checkpoint has a different input binding')
        committed_offset=state['byte_offset'];committed_units=state['units'];source_frontier=state['source_frontier']
    if not index_path.exists():
        if committed_offset:raise ValueError('offline preference source-index frontier has no data file')
        return [],0,0,0
    with index_path.open('r+b') as stream:
        size=stream.seek(0,os.SEEK_END)
        if committed_offset>size:raise ValueError('offline preference source-index checkpoint exceeds its data file')
        if size!=committed_offset:
            stream.truncate(committed_offset);stream.flush();os.fsync(stream.fileno())
        stream.seek(0)
        previous=0
        for line in stream:
            if not line.endswith(b'\n'):
                raise ValueError('committed offline preference source-index has a torn line')
            item=json.loads(line)
            indices=item.get('normalized_source_rows')
            if (item.get('schema')!=SOURCE_INDEX_RECORD_SCHEMA
                    or item.get('identity_sha256')!=identity_sha
                    or not isinstance(indices,list) or not indices
                    or indices!=list(range(previous,previous+len(indices)))):
                raise ValueError('offline preference source-index records are not an exact ordered partition')
            previous+=len(indices);rows.append(item)
        if len(rows)!=committed_units or previous!=source_frontier:
            raise ValueError('offline preference source-index frontier differs from its committed records')
    return rows,committed_units,source_frontier,committed_offset


def _load_complete_source_index(index_path,identity):
    """Authenticate the immutable finished source eligibility index."""
    index_path=Path(index_path);receipt_path=index_path.with_suffix('.receipt.json')
    receipt=json.loads(receipt_path.read_text())
    if (receipt.get('schema')!=SOURCE_INDEX_SCHEMA or receipt.get('identity')!=identity
            or receipt.get('identity_sha256')!=canonical_json_sha256_hex(identity)
            or receipt.get('index')!={'path':str(index_path.resolve()),'bytes':index_path.stat().st_size,
                                      'sha256':sha256_file_hex(index_path)}):
        raise ValueError('completed offline preference source-index receipt or bytes differ')
    rows,units,frontier,offset=_read_source_index(index_path,identity,
        state={'schema':SOURCE_INDEX_SCHEMA,'identity':identity,
               'identity_sha256':canonical_json_sha256_hex(identity),
               'byte_offset':index_path.stat().st_size,'units':receipt['units'],
               'source_frontier':receipt['source_frontier']})
    if (units!=receipt['units'] or frontier!=receipt['source_frontier']
            or frontier!=receipt['normalized_sessions'] or offset!=index_path.stat().st_size):
        raise ValueError('completed offline preference source-index accounting differs')
    return rows,receipt


def _source_eligibility_index(source,export_path,export,source_binding,closure_binding,
                              protection,decision_ids,decision_bindings,index_root,*,
                              expected_units,batch_units=128):
    """Build/reuse one resumable source-only eligibility index shared by model views."""
    source_contract={'source_file':export['contract']['source'],
                     'source_revision':export['contract']['revision']}
    identity={'schema':SOURCE_INDEX_SCHEMA,'source':source_binding,
        'export_receipt':_bound(export_path),'source_contract':source_contract,
        'source_closure_receipt':closure_binding,
        'protection':protection['proof'],
        'protection_sets':{key:sorted(protection[key]) for key in ('held','quality','pending')},
        'decision_holds':decision_contract(decision_bindings),
        'implementation':_source_index_implementation()}
    identity_sha=canonical_json_sha256_hex(identity)
    root=Path(index_root);root.mkdir(parents=True,exist_ok=True)
    index_path=root/(identity_sha+'.jsonl');receipt_path=index_path.with_suffix('.receipt.json')
    state_path=index_path.with_suffix('.state.json')
    if receipt_path.is_file():
        indexed,receipt=_load_complete_source_index(index_path,identity)
    else:
        indexed,units,source_frontier,offset=_read_source_index(index_path,identity)
        # Verify the complete normalized session source contract once per
        # shared source index, rather than once for each tokenizer view.
        actual_sessions=0
        for value in sessions(source):
            if (value['source_file']!=source_contract['source_file']
                    or value['source_revision']!=source_contract['source_revision']):
                raise ValueError('offline normalized session changed its original source binding')
            actual_sessions+=1
        if actual_sessions!=export['complete_sessions']:
            raise ValueError('offline source session count differs from its complete export')
        iterator=iter(normalized_preference_records(source))
        for _ in range(units):
            try:next(iterator)
            except StopIteration as error:raise ValueError('offline source index frontier exceeds its source') from error
        batch=[];unit_frontier=units;row_frontier=source_frontier
        mode='ab' if offset else 'wb'
        with index_path.open(mode) as stream:
            for indices,record,error in iterator:
                if tuple(indices)!=tuple(range(row_frontier,row_frontier+len(indices))):
                    raise ValueError('offline normalized source rows are omitted, duplicated or reordered')
                row_frontier+=len(indices);unit_frontier+=1
                record,reason=(None,str(error)) if error else protected_pair(record,protection,decision_ids)
                if reason:
                    item={'schema':SOURCE_INDEX_RECORD_SCHEMA,'identity_sha256':identity_sha,
                          'normalized_source_rows':list(indices),'kind':'omission','reason':reason}
                else:
                    source_sha=canonical_json_sha256_hex(record)
                    item={'schema':SOURCE_INDEX_RECORD_SCHEMA,'identity_sha256':identity_sha,
                          'normalized_source_rows':list(indices),'kind':'eligible',
                          'source_record_sha256':source_sha,'split':record['split'],
                          'groups':record['source_groups'],'source':record['source']}
                batch.append(item)
                if len(batch)>=batch_units:
                    for value in batch:
                        stream.write((json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode())
                    stream.flush();os.fsync(stream.fileno());offset=stream.tell()
                    write_json_durable(state_path,{'schema':SOURCE_INDEX_SCHEMA,'identity':identity,
                        'identity_sha256':identity_sha,'byte_offset':offset,'units':unit_frontier,
                        'source_frontier':row_frontier})
                    indexed.extend(batch);batch=[]
            if batch:
                for value in batch:
                    stream.write((json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode())
                stream.flush();os.fsync(stream.fileno());offset=stream.tell()
                write_json_durable(state_path,{'schema':SOURCE_INDEX_SCHEMA,'identity':identity,
                    'identity_sha256':identity_sha,'byte_offset':offset,'units':unit_frontier,
                    'source_frontier':row_frontier})
                indexed.extend(batch)
        if row_frontier!=export['complete_sessions']:
            raise ValueError('offline preference source index lost or added normalized source rows')
        if unit_frontier!=expected_units:
            raise ValueError('offline preference source index changed its normalized pair unit count')
        receipt={'schema':SOURCE_INDEX_SCHEMA,'identity':identity,'identity_sha256':identity_sha,
            'units':unit_frontier,'source_frontier':row_frontier,'normalized_sessions':actual_sessions,
            'index':{'path':str(index_path.resolve()),'bytes':index_path.stat().st_size,
                     'sha256':sha256_file_hex(index_path)}}
        write_json_durable(receipt_path,receipt)
        state_path.unlink(missing_ok=True)
    expected={};omitted=[];frontier=0
    for item in indexed:
        indices=item['normalized_source_rows'];frontier+=len(indices)
        if item['kind']=='omission':
            omitted.append({'normalized_source_rows':indices,'reason':item['reason']})
        elif item['kind']=='eligible':
            source_sha=item['source_record_sha256']
            if source_sha in expected:raise ValueError('offline preference source repeats an immutable source occurrence')
            expected[source_sha]={'split':item['split'],'groups':item['groups'],'source':item['source'],
                                  'normalized_source_rows':indices}
        else:raise ValueError('offline preference source-index record has an unknown kind')
    return expected,omitted,frontier,receipt['units'],receipt


def _proposal_request(models,protection_path,protection,runtime,holds):
    """Bind a resumable proposal to every declared source and decision input."""
    return {'schema':'natlang.acquired-offline-preference-request/1',
        'models':models,
        'protection':protection['proof'],
        'decision_holds':decision_contract(holds),
        'runtime':runtime,
        'verifier_dependencies':_verification_dependencies(),
        'protection_path':str(Path(protection_path).resolve())}


def verify_conversion(directory,*,protection_path,base,prepared=None,
                      verified_closures=None,verification_receipt_path=None,
                      source_index_root=None):
    directory=Path(directory);receipt_path=directory/'receipt.json'
    receipt=json.loads(receipt_path.read_text());binding=receipt.get('binding') or {}
    if (receipt.get('schema')!='natlang.acquired-offline-preference-graphs/1'
            or receipt.get('training_admission') is not False or receipt.get('capacity_qualified') is not False):
        raise ValueError('offline admission requires its exact qualified renderer and complete conversion receipt')
    runtime_transition=_conversion_runtime_transition(binding.get('runtime') or {})
    protection=current_protection(protection_path)
    closure_binding=binding.get('source_closure_receipt')
    closure_path=bound(closure_binding['path'],closure_binding) if closure_binding else None
    closure_key=(str(closure_path.resolve()) if closure_path else str(Path(protection_path).resolve()),
                 sha256_file_hex(closure_path) if closure_path else protection['proof']['sha256'])
    if verified_closures is None or closure_key not in verified_closures:
        if closure_path is None:
            verify_global_source_closure(protection_path,protection)
        else:
            verify_global_source_closure(protection_path,protection,receipt_path=closure_path)
        if verified_closures is not None:verified_closures.add(closure_key)
    if binding.get('protection')!=protection['proof']:
        raise ValueError('offline preference conversion uses different current source protection')
    holds=sorted((ROOT/'training/decision-holds').glob('*.jsonl'))
    decision_ids,decision_bindings=load_decision_holds(holds)
    if decision_contract(binding['decision_holds'])!=decision_contract(decision_bindings):
        raise ValueError('offline preference conversion uses different decision holds')
    if binding['prepared_tokenizer']['base_files']!=tokenizer_inventory(base):
        raise ValueError('offline model selector differs from its exact tokenizer base inventory')
    prepared=prepare_tokenizer(base) if prepared is None else prepared
    prepared.validate(base)
    if prepared.binding!=binding['prepared_tokenizer']:
        raise ValueError('offline admission tokenizer rendering contract differs from conversion')
    export_path=bound(binding['export_receipt']['path'],binding['export_receipt'])
    export=json.loads(export_path.read_text())
    source=bound(binding['source']['path'],binding['source'])
    if (source.resolve()!= (export_path.parent/'full-trajectories.jsonl.zst').resolve()
            or export['outputs'][source.name]!=binding['source']['sha256']):
        raise ValueError('offline source differs from its complete normalized source contract')
    # A completed view receipt is reusable only after all of its byte-bound
    # products and durable shard outputs still match.  The durable attestation
    # is per model/view; a stopped sibling view can be recomputed independently.
    source_artifacts=[binding['export_receipt'],binding['source'],*receipt.get('outputs',[]),
                      receipt.get('objective',{}),receipt.get('row_proof',{}),receipt.get('omissions',{})]
    if closure_binding:source_artifacts.append(closure_binding)
    for item in source_artifacts:
        if item and item.get('path'):
            bound(item['path'],item)
    units=binding.get('source_units')
    if type(units) is not int or units<1:raise ValueError('offline verification receipt lacks exact source-unit count')
    end,shards=completed_shards(directory/'pair-shards',binding,units,
                                schema=SHARD_SCHEMA,additional_outputs=ADDITIONAL)
    if end!=units or shards!=receipt.get('shards'):
        raise ValueError('offline conversion lacks its exact completed durable source frontier')
    for item in receipt.get('outputs',[]):
        path=Path(item['path'])
        if path.parent.name not in {'train','test'} or path.resolve()!=(directory/path.parent.name/'native-graphs.jsonl.zst').resolve():
            raise ValueError('offline admission consumes aggregate outputs only')
        bound(item['path'],item)
    for key in ('objective','row_proof','omissions'):
        bound(receipt[key]['path'],receipt[key])
    units=binding.get('source_units')
    if type(units) is not int or units<1:raise ValueError('offline verification receipt lacks exact source-unit count')
    index_root=(Path(source_index_root) if source_index_root is not None
                else directory/'source-eligibility-indexes')
    expected,omitted,frontier,index_units,index_receipt=_source_eligibility_index(
        source,export_path,export,binding['source'],closure_binding,protection,decision_ids,
        decision_bindings,index_root,expected_units=units)
    if (frontier!=export['complete_sessions'] or frontier!=binding['normalized_sessions']
            or index_units!=binding['source_units']):
        raise ValueError('offline preference complete source accounting differs')
    if list(sessions(receipt['omissions']['path']))!=omitted:
        raise ValueError('offline omission ledger differs from current exact source eligibility')
    identity={'conversion_receipt':{'path':str(receipt_path.resolve()),'bytes':receipt_path.stat().st_size,
                                    'sha256':sha256_file_hex(receipt_path)},
        'conversion_binding':binding,'verified_shards':shards,
        'runtime':_runtime(),'conversion_runtime_transition':runtime_transition,
        'verifier_dependencies':_verification_dependencies(),
        'source_eligibility_index':index_receipt,
        'protection':protection['proof'],'decision_holds':decision_contract(decision_bindings),
        'tokenizer_base':tokenizer_inventory(base),'prepared_tokenizer':prepared.binding,
        'source_artifacts':[{'path':str(Path(item['path']).resolve()),'bytes':item.get('bytes',Path(item['path']).stat().st_size),
                             'sha256':item['sha256']} for item in source_artifacts if item and item.get('path')]}
    identity_sha=canonical_json_sha256_hex(identity)
    if verification_receipt_path is not None:
        cached=_cached_verification(Path(verification_receipt_path),identity)
        if cached is not None:return cached
    render_checkpoint=None;verified_renders={}
    if verification_receipt_path is not None:
        render_checkpoint=Path(verification_receipt_path).with_name(
            Path(verification_receipt_path).stem+'-'+identity_sha+'.renders.jsonl')
        if render_checkpoint.exists():
            # Only newline-terminated, individually fsynced records are
            # committed. Recover a torn final append without losing prior rows.
            valid_end=0
            with render_checkpoint.open('r+b') as stream:
                while True:
                    line=stream.readline()
                    if not line:break
                    if not line.endswith(b'\n'):
                        stream.seek(valid_end);stream.truncate();stream.flush();os.fsync(stream.fileno());break
                    record=json.loads(line)
                    if (record.get('schema')!='natlang.offline-preference-render-checkpoint/1'
                            or record.get('identity_sha256')!=identity_sha
                            or not isinstance(record.get('source_record_sha256'),str)
                            or not isinstance(record.get('id'),str)
                            or not isinstance(record.get('graph_sha256'),str)
                            or not isinstance(record.get('objective_sha256'),str)):
                        raise ValueError('offline preference render checkpoint has a different or malformed binding')
                    key=record['source_record_sha256']
                    value={'id':record['id'],'graph_sha256':record['graph_sha256'],
                           'objective_sha256':record['objective_sha256']}
                    if key in verified_renders and verified_renders[key]!=value:
                        raise ValueError('offline preference render checkpoint repeats a source with different output')
                    verified_renders[key]=value;valid_end=stream.tell()
    objective=NativePreferenceObjective(receipt['objective']['path'])
    if {'kind':objective.kind,'beta':objective.beta,'margin':objective.margin}!=binding['objective']:
        raise ValueError('offline objective differs from declared conversion objective')
    rendered={};proofs={};groups={'train':set(),'test':set()};counts=Counter();seen_source=set();seen_ids=set();outputs=[]
    for output in receipt['outputs']:
        path=Path(output['path']);split=path.parent.name
        if split not in groups or path.resolve()!=(directory/split/'native-graphs.jsonl.zst').resolve():
            raise ValueError('offline admission consumes aggregate outputs only')
        rows=0;tokens=0;maximum=0
        for row in sessions(path):
            example=parse_graph(row,tokenizer_fingerprint=binding['prepared_tokenizer']['expanded_fingerprint'])
            conversion=row.get('conversion') or {};source_sha=conversion.get('source_record_sha256')
            original=expected.get(source_sha)
            if (original is None or source_sha in seen_source or example.id in seen_ids
                    or row.get('required_objective')!=KIND or conversion.get('ordinary_ce_targets')!=0
                    or conversion.get('native_on_policy') is not False or example.split!=split
                    or split!=original['split'] or set(example.source_groups)!=set(original['groups'])
                    or conversion.get('source_preference')!=original['source']):
                raise ValueError('offline graph source, split, alternatives or objective admission differs')
            with torch.no_grad():objective.forward_inputs({'examples':[example]})
            spec=objective.rows[example.id]
            if spec['source_record_sha256']!=source_sha or spec['source_preference']!=original['source']:
                raise ValueError('offline objective changed original preference provenance')
            rendered[source_sha]={'id':example.id,'graph_sha256':canonical_json_sha256_hex(row),
                'objective_sha256':canonical_json_sha256_hex(spec)}
            seen_source.add(source_sha);seen_ids.add(example.id);groups[split].update(example.source_groups)
            proofs[example.id]={'id':example.id,'graph_sha256':canonical_json_sha256_hex(row),
                'source_groups':list(example.source_groups),'split':split,
                'normalized_source_rows':original['normalized_source_rows'],'source_record_sha256':source_sha}
            rows+=1;tokens+=example.plan.token_count;maximum=max(maximum,example.plan.token_count)
        if rows!=output['rows']:raise ValueError('offline aggregate row count differs from its conversion receipt')
        counts['pairs:'+split]+=rows;counts['tokens:'+split]+=tokens;counts['max_tokens']=max(counts['max_tokens'],maximum)
        outputs.append({**output,'split':split,'tokens':tokens,'max_tokens':maximum})
    if seen_source!=set(expected) or seen_ids!=set(objective.rows):
        raise ValueError('offline admission lost or added jointly bound source preferences')
    # Independently derive BOTH alternatives from the immutable normalized source.
    # Keep only compact hashes between passes; full histories exist for one pair
    # at a time. Self-consistent forged graph/objective/proof files are not a
    # substitute for this exact token and suffix-mask correspondence.
    checked=set()
    for indices,record,error in normalized_preference_records(source):
        record,reason=(None,str(error)) if error else protected_pair(record,protection,decision_ids)
        if reason:continue
        source_sha=canonical_json_sha256_hex(record)
        if source_sha in checked:
            raise ValueError('offline normalized source repeats an immutable preference occurrence')
        cached_render=verified_renders.get(source_sha)
        if cached_render is None:
            graph,spec=pair_graph(record,prepared.tokenizer,
                tokenizer_fingerprint=prepared.expanded_fingerprint)
            expected_render={'id':graph['id'],'graph_sha256':canonical_json_sha256_hex(graph),
                'objective_sha256':canonical_json_sha256_hex(spec)}
            if rendered.get(source_sha)!=expected_render:
                raise ValueError('offline stored graph/objective differs from exact original-alternative tokenizer rendering')
            cached_render={'id':expected_render['id'],'graph_sha256':expected_render['graph_sha256'],
                           'objective_sha256':expected_render['objective_sha256']}
            if render_checkpoint is not None:
                line={'schema':'natlang.offline-preference-render-checkpoint/1',
                    'identity_sha256':identity_sha,'source_record_sha256':source_sha,**cached_render}
                with render_checkpoint.open('ab') as stream:
                    stream.write((json.dumps(line,sort_keys=True,separators=(',',':'))+'\n').encode())
                    stream.flush();os.fsync(stream.fileno())
            verified_renders[source_sha]=cached_render
        elif rendered.get(source_sha)!=cached_render:
            raise ValueError('offline stored graph/objective differs from its exact durable render checkpoint')
        checked.add(source_sha)
    if checked!=seen_source:
        raise ValueError('offline original-alternative rendering inventory differs from admitted graphs')
    prepared.validate(base)
    row_proofs=list(sessions(receipt['row_proof']['path']))
    if len(row_proofs)!=len(proofs) or {p['id'] for p in row_proofs}!=set(proofs) or any(proofs[p['id']]!=p for p in row_proofs):
        raise ValueError('offline row proof inventory differs from complete exact graph/source joins')
    if groups['train'] & groups['test']:raise ValueError('offline train/held groups overlap')
    if current_protection(protection_path)!=protection or decision_contract(load_decision_holds(holds)[1])!=decision_contract(decision_bindings):
        raise ValueError('offline protections changed during admission')
    for item in [binding['export_receipt'],binding['source'],*receipt['outputs'],receipt['objective'],receipt['row_proof'],receipt['omissions']]:
        bound(item['path'],item)
    result={'receipt':_bound(receipt_path),'objective':receipt['objective'],'source':binding['source'],
        'conversion_runtime_transition':runtime_transition,
        'outputs':outputs,'counts':dict(counts),'groups':{key:sorted(value) for key,value in groups.items()},
        'proofs':proofs,'source_membership':sorted(expected),'source_units':units,'normalized_sessions':frontier,
        'source_eligibility_index':index_receipt,
        'exact_original_alternative_renders':len(checked),'source_render_verified':True}
    if verification_receipt_path is not None:
        payload={'identity':identity,'identity_sha256':identity_sha,'result':result}
        write_json_durable(Path(verification_receipt_path),{
            'schema':'natlang.offline-preference-view-verification/1','payload':payload,
            'payload_sha256':canonical_json_sha256_hex(payload)})
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',required=True,help='JSON model->{base,conversions:[directory],ordinary_held:{path,sha256}}')
    parser.add_argument('--protection',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    from scripts.neuralese_data.conversion_output_lease import acquire_output_lease
    with acquire_output_lease(args.output):return prepare(args)


def prepare(args):
    models=json.loads(args.models)
    initial_protection=current_protection(args.protection)
    initial_runtime=_runtime()
    initial_holds=load_decision_holds(sorted((ROOT/'training/decision-holds').glob('*.jsonl')))[1]
    if not isinstance(models,dict) or len(models)<2:raise ValueError('offline admission requires both production model views')
    request=_proposal_request(models,args.protection,initial_protection,initial_runtime,initial_holds)
    request_path=args.output/'.proposal-request.json'
    if request_path.exists():
        if json.loads(request_path.read_text())!=request:
            raise ValueError('offline admission output belongs to a different proposal request')
    else:
        if any(path.name not in {'.conversion.lock'} for path in args.output.iterdir()):
            raise ValueError('offline admission output has files without its durable proposal request')
        write_json_durable(request_path,request)
    owned={'.conversion.lock','.proposal-request.json','conversion-verifications','audit.json'}
    for name in models:
        owned.update({f'{name}-{split}-sources.json' for split in ('train','test')})
        owned.add(f'{name}-objective.json')
    if any(path.name not in owned for path in args.output.iterdir()):
        raise ValueError('offline admission output contains an unrelated file')
    cache_root=args.output/'conversion-verifications';cache_root.mkdir(exist_ok=True)
    verified_closures=set();verified={};reference=None;all_groups={'train':set(),'test':set()}
    for name,declaration in models.items():
        prepared=prepare_tokenizer(declaration['base'])
        views=[]
        for index,path in enumerate(declaration['conversions']):
            cache_path=cache_root/f'{name}-{index:04d}.json'
            views.append(verify_conversion(path,protection_path=args.protection,base=declaration['base'],
                prepared=prepared,verified_closures=verified_closures,
                verification_receipt_path=cache_path,
                source_index_root=cache_root/'source-eligibility-indexes'))
        membership=[source for view in views for source in view['source_membership']]
        if len(set(membership))!=len(membership):raise ValueError('offline admission repeats source pairs across conversions')
        if reference is not None and set(membership)!=reference:raise ValueError('offline model views differ in exact source pair membership')
        reference=set(membership);verified[name]=views
        for split in all_groups:
            all_groups[split].update(group for view in views for group in view['groups'][split])
        held=bound(declaration['ordinary_held']['path'],declaration['ordinary_held'])
        from natlang_neuralese.data.native_sources import NativeGraphSource
        first=json.loads(Path(views[0]['receipt']['path']).read_text())
        fingerprint=first['binding']['prepared_tokenizer']['expanded_fingerprint']
        held_source=NativeGraphSource(held,fingerprint,required_split='test')
        all_groups['test'].update(held_source.groups)
    if all_groups['train']&all_groups['test']:raise ValueError('offline cohort global train/held overlap')
    if (current_protection(args.protection)!=initial_protection or _runtime()!=initial_runtime
            or decision_contract(load_decision_holds(sorted((ROOT/'training/decision-holds').glob('*.jsonl')))[1])!=decision_contract(initial_holds)):
        raise ValueError('offline source protection/holds/runtime changed across model verification')
    entrypoints={}
    for name,views in verified.items():
        rows={};specification=None;sources={'train':[],'test':[]}
        for view in views:
            objective=json.loads(Path(view['objective']['path']).read_text())
            declaration={key:objective[key] for key in ('schema','kind','beta','margin')}
            if specification is not None and declaration!=specification:raise ValueError('offline objective declarations differ')
            specification=declaration
            if set(rows)&objective['rows'].keys():raise ValueError('offline aggregate objective repeats a pair graph')
            rows.update(objective['rows'])
            for output in view['outputs']:
                if output['rows']:
                    sources[output['split']].append({'kind':'native-graphs','path':os.path.relpath(output['path'],args.output),
                        'sha256':output['sha256'],'weight':1.,'epochs':1})
        if not sources['train']:raise ValueError('offline cohort has no eligible training pairs')
        ordinary=models[name]['ordinary_held']
        sources['test'].insert(0,{'kind':'native-graphs','path':os.path.relpath(ordinary['path'],args.output),
                                'sha256':ordinary['sha256'],'weight':1.,'epochs':1})
        for split in sources:
            _write_owned_json(args.output/f'{name}-{split}-sources.json',{'schema':'natlang.native-graph-sources/1','sources':sources[split]})
        _write_owned_json(args.output/f'{name}-objective.json',{**specification,'offline':True,'native_on_policy':False,'rows':rows})
        entrypoints[name]={'kind':'native_offline_preference','native_graphs':_bound(args.output/f'{name}-train-sources.json'),
            'held_native_graphs':_bound(args.output/f'{name}-test-sources.json'),'objective_spec':_bound(args.output/f'{name}-objective.json'),
            'requires_current_shared_participant_and_capacity_qualification':True,'earned_pair_visits':0}
    result={'schema':'natlang.acquired-offline-preference-admission/1','status':'passed','training_admission':False,
        'admission_class':'source-curated-offline-pairwise-preferences','oracle_verified':False,
        'numerical_rerender_claimed':False,'all_pair_token_render_verified':True,
        'native_execution_verified':False,'capacity_qualified':False,
        'source_eligibility_scope':'byte-bound qualified conversion plus all-row original-alternative tokenizer re-render, exact graph/suffix masks and source/split joins',
        'protection':current_protection(args.protection)['proof'],'models':verified,'training_entrypoints':entrypoints,
        'training_sources':'aggregate graph outputs only; durability shards and original chosen SFT are not extra preference visits'}
    _write_owned_json(args.output/'audit.json',result)
    return result


if __name__=='__main__':main()
