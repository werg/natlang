/** Exact provenance/retention contract, shared by native derivative artifacts. */
import { hexDigest } from './hash.js';

const ID=/^(?:nzs1_|nnt1_)[a-z2-7]{52}$/;
const TENSOR=/^nnt1_[a-z2-7]{52}$/;
const STATE=/^nzs1_[a-z2-7]{52}$/;
const DERIVATION='natlang.native-state-derivation/1';
export type NativeStateOptimizationDerivation={schema:typeof DERIVATION;kind:'optimizer';origin_state_ids:string[];
  parent_state_id:string;gradient_id:string;update_artifact_id:string;program_id:string|null;
  optimizer_state_parent_id:string|null;step:number;optimizer:Record<string,string>;updated_tensor_names:string[];arithmetic:Record<string,string>};
export type NativeStateSemanticDerivation={schema:typeof DERIVATION;kind:'semantic-encode'|'semantic-adapt';origin_state_ids:string[];
  parent_state_id:string;adapter_sha256:string;source_native_identity:Record<string,string>;target_native_identity:Record<string,string>};
export type NativeStateDerivation=NativeStateOptimizationDerivation|NativeStateSemanticDerivation;
const keys=(value:object)=>Object.keys(value).sort().join(',');
const sortedUnique=(value:unknown,pattern:RegExp):value is string[]=>Array.isArray(value)&&value.every(item=>typeof item==='string'&&pattern.test(item))&&
  value.join(',')===[...new Set(value)].sort().join(',');

function hexNumber(text:unknown):number {
  if(typeof text!=='string')throw new RangeError('native optimizer options require canonical hexadecimal values');
  if(/^-?0x0\.0p\+0$/.test(text))return text.startsWith('-')?-0:0;
  const match=/^(-?)0x([01])\.([0-9a-f]{13})p([+-]\d+)$/.exec(text);
  if(!match)throw new RangeError('native optimizer options require canonical hexadecimal values');
  const exponent=Number(match[4]);
  if(match[4]!==`${exponent>=0?'+':''}${exponent}`||exponent < -1022||exponent>1023||match[2]==='0'&&(exponent!==-1022||Number.parseInt(match[3]!,16)===0))
    throw new RangeError('native optimizer options require canonical finite hexadecimal values');
  return (match[1]?-1:1)*(Number(match[2])+Number.parseInt(match[3]!,16)/2**52)*2**exponent;
}
export function validateNativeOptimizerBinding(value:Record<string,string>):void {
  if(!value||keys(value)!=='beta1_hex,beta2_hex,eps_hex,learning_rate_hex,momentum_hex,name,weight_decay_hex'||!['adam','sgd'].includes(value.name!))
    throw new RangeError('native optimizer needs its exact named algorithm/settings binding');
  const lr=hexNumber(value.learning_rate_hex),decay=hexNumber(value.weight_decay_hex),b1=hexNumber(value.beta1_hex),
    b2=hexNumber(value.beta2_hex),eps=hexNumber(value.eps_hex),momentum=hexNumber(value.momentum_hex);
  if(!(lr>0&&decay>=0&&b1>=0&&b1<1&&b2>=0&&b2<1&&eps>0&&momentum>=0))throw new RangeError('native optimizer settings are out of range');
}
export function validateNativeStateDerivation(value:NativeStateDerivation|undefined):void {
  if(value===undefined)return;
  if(value&&['semantic-encode','semantic-adapt'].includes(value.kind)){
    const item=value as NativeStateSemanticDerivation;
    if(keys(item)!=='adapter_sha256,kind,origin_state_ids,parent_state_id,schema,source_native_identity,target_native_identity'||item.schema!==DERIVATION||
      !STATE.test(item.parent_state_id)||!sortedUnique(item.origin_state_ids,STATE)||!item.origin_state_ids.length||!/^[0-9a-f]{64}$/.test(item.adapter_sha256))
      throw new RangeError('native semantic derivation needs exact origins, source and adapter');
    for(const identity of [item.source_native_identity,item.target_native_identity])if(!identity||keys(identity)!=='architecture,checkpoint_sha256,embedding_weight_sha256,tokenizer_fingerprint'||
      !identity.architecture||['checkpoint_sha256','tokenizer_fingerprint','embedding_weight_sha256'].some(key=>typeof identity[key]!=='string'||!/^[0-9a-f]{64}$/.test(identity[key]!)))
      throw new RangeError('native semantic derivation needs exact source/target model identities');
    return;
  }
  if(!value||value.kind!=='optimizer'||keys(value)!=='arithmetic,gradient_id,kind,optimizer,optimizer_state_parent_id,origin_state_ids,parent_state_id,program_id,schema,step,update_artifact_id,updated_tensor_names'||
    value.schema!==DERIVATION||value.kind!=='optimizer'||!STATE.test(value.parent_state_id)||!TENSOR.test(value.gradient_id)||!TENSOR.test(value.update_artifact_id)||
    value.program_id!==null&&!TENSOR.test(value.program_id)||value.optimizer_state_parent_id!==null&&!TENSOR.test(value.optimizer_state_parent_id)||
    !Number.isSafeInteger(value.step)||value.step<1||!sortedUnique(value.origin_state_ids,STATE)||!value.origin_state_ids.length||
    !sortedUnique(value.updated_tensor_names,/^layers\.[0-9]+\.(?:raw_k|v|gated_inputs)$/)||!value.updated_tensor_names.length)
    throw new RangeError('native optimized state needs its exact origin/update/history contract');
  validateNativeOptimizerBinding(value.optimizer);
  const arithmetic=value.arithmetic;
  if(!arithmetic||keys(arithmetic)!=='cpu_capability,float32_subnormal_hex,implementation_sha256,libc_version,machine,python_version,schema,torch_build_sha256,torch_git_version,torch_version'||
    arithmetic.schema!=='natlang.native-state-optimizer-torch-cpu/1'||Object.values(arithmetic).some(value=>typeof value!=='string'||!value)||
    ['implementation_sha256','torch_build_sha256'].some(key=>!/^[0-9a-f]{64}$/.test(arithmetic[key]!)))
    throw new RangeError('native state optimizer needs its exact qualified CPU implementation/runtime binding');
}
export function nativeStateDerivationDependencies(value:NativeStateDerivation|undefined):string[] {
  validateNativeStateDerivation(value);
  if(value&&value.kind!=='optimizer')return [...new Set([...value.origin_state_ids,value.parent_state_id])].sort();
  return value?[...new Set([...value.origin_state_ids,value.parent_state_id,value.gradient_id,value.update_artifact_id,
    ...[value.program_id,value.optimizer_state_parent_id].filter((id):id is string=>id!==null)])].sort():[];
}
/** Exact finite topology and asset closure, shared by every portable archive. */
function finiteProgramDependencies(value:unknown):string[]{
  type Obj=Record<string,unknown>;
  const object=(item:unknown):item is Obj=>!!item&&typeof item==='object'&&!Array.isArray(item);
  const fail=():never=>{throw new RangeError('native derivative program has invalid finite topology or executable dependencies');};
  if(!object(value)||keys(value)!=='arithmetic,inputs,nodes,outer,schema,wrt'||value.schema!=='natlang.native-derivative-program/1')return fail();
  const arithmetic=value.arithmetic;
  if(!object(arithmetic)||keys(arithmetic)!=='query_chunk_tokens,schema'||arithmetic.schema!=='natlang.native-reference-arithmetic/1'||
      !Number.isSafeInteger(arithmetic.query_chunk_tokens)||(arithmetic.query_chunk_tokens as number)<1)return fail();
  const roots=value.inputs,nodes=value.nodes,wrt=value.wrt;
  if(!object(roots)||!Object.keys(roots).length||Object.keys(roots).some(name=>!name)||!Array.isArray(nodes)||
      !Array.isArray(wrt)||!wrt.length||wrt.some(name=>typeof name!=='string'||!Object.hasOwn(roots,name))||new Set(wrt).size!==wrt.length)return fail();
  const kinds=new Map<string,string>(),available=new Map<string,string>(),names=new Set<string>(),gradients=new Map<string,string>(),deps=new Set<string>();
  for(const [name,target] of Object.entries(roots)){
    if(!object(target)||Object.keys(target).some(key=>!['kind','id','names'].includes(key))||
      !['adapter','native-state','model-parameters'].includes(target.kind as string)||typeof target.id!=='string'||
      !(target.kind==='native-state'?STATE:TENSOR).test(target.id))return fail();
    if(target.names!==undefined||target.kind==='model-parameters'){
      if(!Array.isArray(target.names)||!target.names.length||target.names.some(name=>typeof name!=='string'||!name)||new Set(target.names).size!==target.names.length)return fail();
    }
    kinds.set(name,target.kind as string);available.set(name,target.kind as string);names.add(name);deps.add(target.id);
  }
  const objective=(item:unknown):void=>{
    if(!object(item)||keys(item)!=='bindings,request'||!object(item.bindings)||!object(item.request))return fail();
    for(const [name,current] of Object.entries(item.bindings))if(!kinds.has(name)||typeof current!=='string'||available.get(current)!==kinds.get(name))return fail();
    const request=item.request,graph=request.graph;
    if(request.schema!=='natlang.native-gradient-request/1'||['target','order','derived','create_graph'].some(key=>Object.hasOwn(request,key))||
      !object(graph)||object(request.objective))return fail();
    const invocation=request.invocation_adapters;
    if(invocation!==undefined&&(!object(invocation)))return fail();
    for(const entries of [request.adapters??[],...Object.values((invocation??{}) as Obj)]){
      if(!Array.isArray(entries))return fail();
      for(const entry of entries){const id=typeof entry==='string'?entry:object(entry)?entry.id:undefined;
        if(typeof id!=='string'||!TENSOR.test(id))return fail();deps.add(id);}
    }
    if(graph.external_states!==undefined&&!object(graph.external_states))return fail();
    for(const state of Object.values((graph.external_states??{}) as Obj)){
      if(!object(state)||typeof state.id!=='string'||!STATE.test(state.id))return fail();deps.add(state.id);
    }
  };
  for(const node of nodes){
    if(!object(node)||typeof node.name!=='string'||!node.name||names.has(node.name))return fail();
    if(node.kind==='gradient'){
      if(keys(node)!=='bindings,kind,name,request,value'||typeof node.value!=='string'||!available.has(node.value))return fail();
      objective({request:node.request,bindings:node.bindings});
      if(!Object.keys(roots).some(name=>((node.bindings as Obj)[name]??name)===node.value))return fail();
      gradients.set(node.name,node.value);
    }else if(node.kind==='optimizer'){
      if(keys(node)!=='gradient,kind,name,optimizer,optimizer_state,value'||typeof node.value!=='string'||!available.has(node.value)||
        typeof node.gradient!=='string'||gradients.get(node.gradient)!==node.value||!object(node.optimizer)||
        Object.keys(node.optimizer).some(key=>!['name','learning_rate','weight_decay','beta1','beta2','eps','momentum'].includes(key))||
        !['sgd','adam'].includes(node.optimizer.name as string))return fail();
      const settings={learning_rate:1e-3,weight_decay:0,beta1:.9,beta2:.999,eps:1e-8,momentum:0,...node.optimizer};
      if(Object.entries(settings).some(([key,item])=>key!=='name'&&(typeof item!=='number'||!Number.isFinite(item)))||
        !(settings.learning_rate>0&&settings.weight_decay>=0&&settings.beta1>=0&&settings.beta1<1&&settings.beta2>=0&&settings.beta2<1&&settings.eps>0&&settings.momentum>=0))return fail();
      const history=node.optimizer_state;
      if(history!==null&&(history!==node.value&&(typeof history!=='string'||!TENSOR.test(history))||history===node.value&&Object.hasOwn(roots,history as string)))return fail();
      available.set(node.name,available.get(node.value)!);
    }else return fail();
    names.add(node.name);
  }
  objective(value.outer);return [...deps].sort();
}
export function validateNativeDerivativeArtifact(value:Record<string,unknown>):void {
  if(!value||!['natlang.native-derivative-program/1','natlang.native-derivative-execution/1'].includes(value.schema as string)||
    !sortedUnique(value.dependencies,ID))throw new RangeError('native derivative artifact needs its declared sorted dependency inventory');
  if(value.schema==='natlang.native-derivative-program/1'){
    if(typeof value.program_json!=='string'||hexDigest(value.program_json)!==value.program_sha256||
      JSON.stringify(finiteProgramDependencies(JSON.parse(value.program_json)))!==JSON.stringify(value.dependencies))
      throw new RangeError('native derivative program differs from its exact immutable JSON/dependency binding');
  } else if(typeof value.program_id!=='string'||!TENSOR.test(value.program_id))throw new RangeError('native derivative execution needs its exact program artifact');
}

/** The target of a meta-gradient can differ from the actors evaluated by its outer loss. */
export function validateNativeDerivativeTarget(metadata:{kind:string;native_identity:unknown;source_artifact_id:string|null;provenance:Record<string,unknown>},
    tensors:Record<string,unknown>):boolean{
  const provenance=metadata.provenance,value=provenance.native_derivative_target as Record<string,unknown>|undefined;
  if(value===undefined)return false;
  const kind=({'adapter-gradient':'adapter','state-gradient':'native-state','parameter-gradient':'model-parameters'} as Record<string,string>)[metadata.kind];
  if(!value||keys(value)!=='kind,names,schema,stage,value_id,variable'||value.schema!=='natlang.native-derivative-target/1'||
    !['inner','outer'].includes(value.stage as string)||typeof value.variable!=='string'||!value.variable||value.kind!==kind||
    typeof provenance.derivative_program_id!=='string'||!TENSOR.test(provenance.derivative_program_id)||
    value.stage==='outer'&&(typeof provenance.derivative_execution_id!=='string'||!TENSOR.test(provenance.derivative_execution_id))||
    value.value_id!==(kind==='model-parameters'?provenance.parameter_value_id:metadata.source_artifact_id)||
    typeof value.value_id!=='string'||!(kind==='native-state'?STATE:TENSOR).test(value.value_id)||
    JSON.stringify(value.names)!==JSON.stringify(Object.keys(tensors).sort()))throw new RangeError('native derivative gradient needs its exact differentiation origin');
  return true;
}

export type NativeFunctionalActor={schema:'natlang.native-functional-actor/1';base_native_identity:unknown;
  parameter_values:{id:string;names:string[]}[];uniform_adapters:{id:string;scale_hex:string}[];
  invocation_adapters:Record<string,{id:string;scale_hex:string}[]>|null;arithmetic:Record<string,unknown>};
export function validateNativeFunctionalActor(metadata:{native_identity:unknown;provenance:Record<string,unknown>},
    canonical:(value:unknown)=>string):NativeFunctionalActor|undefined{
  const provenance=metadata.provenance,actor=provenance.functional_actor as NativeFunctionalActor|undefined;
  if(actor===undefined)return;
  if(!actor||keys(actor)!=='arithmetic,base_native_identity,invocation_adapters,parameter_values,schema,uniform_adapters'||
    actor.schema!=='natlang.native-functional-actor/1'||canonical(actor.base_native_identity)!==canonical(metadata.native_identity)||
    hexDigest(canonical(actor))!==provenance.functional_actor_sha256||!Array.isArray(actor.parameter_values))
    throw new RangeError('native functional actor needs exact base, overlays and arithmetic identity');
  const names=new Set<string>(),ids:string[]=[];
  for(const entry of actor.parameter_values){
    if(!entry||keys(entry)!=='id,names'||!TENSOR.test(entry.id)||!Array.isArray(entry.names)||!entry.names.length||
      entry.names.some(name=>typeof name!=='string'||!name||names.has(name))||JSON.stringify(entry.names)!==JSON.stringify([...new Set(entry.names)].sort()))
      throw new RangeError('native functional actor has invalid or overlapping exact parameter overlays');
    ids.push(entry.id);entry.names.forEach(name=>names.add(name));
  }
  if(JSON.stringify(ids)!==JSON.stringify([...new Set(ids)].sort()))throw new RangeError('native functional overlays must be unique ordered artifacts');
  const arith=actor.arithmetic,chunk=arith?.query_chunk_tokens;
  const expected={schema:'natlang.native-reference-arithmetic/1',native_projection_product_dtype:'float64',
    stock_torch_modules:'source-dtype/differentiable',adapter_delta:'stored-base-plus-unrounded-float32',
    optimizer_adam_zero_moment_tangent:'zero-v-boundary/finite-valid-history',activation_storage:'source-native',
    attention:'exact-visible-softmax',expert_dispatch:'source-top-k/differentiable-selected-weights',query_chunk_tokens:chunk};
  if(!Number.isSafeInteger(chunk)||(chunk as number)<1||canonical(arith)!==canonical(expected))throw new RangeError('native functional actor requires complete reference arithmetic');
  if(!Array.isArray(actor.uniform_adapters)||actor.invocation_adapters!==null&&
    (!actor.invocation_adapters||typeof actor.invocation_adapters!=='object'||Array.isArray(actor.invocation_adapters)||
      !Object.keys(actor.invocation_adapters).length||actor.uniform_adapters.length))
    throw new RangeError('native functional actor needs uniform or per-invocation coefficients');
  for(const entries of [actor.uniform_adapters,...Object.values(actor.invocation_adapters??{})]){
    if(!Array.isArray(entries))throw new RangeError('native functional coefficients need ordered declarations');
    for(const entry of entries){if(!entry||keys(entry)!=='id,scale_hex'||!TENSOR.test(entry.id))throw new RangeError('native functional coefficients need exact IDs/scales');hexNumber(entry.scale_hex);}
  }
  if(!actor.parameter_values.length){
    const binding=provenance.adapter_binding as {adapters:unknown}|undefined;
    const receipt=provenance.invocation_actors as {invocations:Record<string,{adapter_binding:{adapters:unknown}|null}>}|undefined;
    const actual=receipt?Object.fromEntries(Object.entries(receipt.invocations).map(([name,value])=>[name,value.adapter_binding?.adapters??[]])):null;
    if(canonical(actor.uniform_adapters)!==canonical(binding?.adapters??[])||canonical(actor.invocation_adapters)!==canonical(actual))
      throw new RangeError('native functional actor differs from actual evaluated coefficient actors');
  }
  return actor;
}
