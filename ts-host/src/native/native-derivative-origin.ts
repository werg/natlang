/** Exact immutable program/evaluation linkage for inner and outer gradients. */
import {nativeCanonicalJson,type NativeTensorArtifact,type NeuraleseStore} from './neuralese-store.js';
import {validateNativeDerivativeTarget,validateNativeFunctionalActor} from './native-derivative-contract.js';
export async function verifyNativeDerivativeGradientOrigin(store:Pick<NeuraleseStore,'getNativeTensorArtifact'|'getNativeState'>,
    artifact:Omit<NativeTensorArtifact,'id'>):Promise<void>{
  if(!validateNativeDerivativeTarget(artifact.metadata,artifact.tensors))return;
  const provenance=artifact.metadata.provenance,target=provenance.native_derivative_target as {
    stage:'inner'|'outer';variable:string;value_id:string;kind:string;names:string[]};
  const actor=validateNativeFunctionalActor(artifact.metadata,nativeCanonicalJson);
  if(!actor)throw new RangeError('native derivative gradient lacks its complete actual functional actor');
  const program=await store.getNativeTensorArtifact?.(provenance.derivative_program_id as string);
  if(!program||program.metadata.kind!=='derivative-program'||program.metadata.provenance.schema!=='natlang.native-derivative-program/1'||
    nativeCanonicalJson(program.metadata.native_identity)!==nativeCanonicalJson(artifact.metadata.native_identity))
    throw new RangeError('native derivative gradient lacks its exact executable program/base');
  const parsed=JSON.parse(program.metadata.provenance.program_json as string) as {
    inputs:Record<string,{id:string;kind:string}>;nodes:{name:string;kind:string;value:string;bindings:Record<string,string>}[]};
  const inputs=provenance.evaluation_inputs as Record<string,{kind:string;value_id:string;names:string[]}>;
  if(!parsed.inputs||!Array.isArray(parsed.nodes)||!inputs||nativeCanonicalJson(Object.keys(inputs).sort())!==nativeCanonicalJson(Object.keys(parsed.inputs).sort()))
    throw new RangeError('native derivative evaluation lost its complete root binding inventory');
  for(const [name,value] of Object.entries(inputs)){
    if(!value||Object.keys(value).sort().join(',')!=='kind,names,value_id'||
    value.kind!==parsed.inputs[name]?.kind||!(value.kind==='native-state'?/^nzs1_[a-z2-7]{52}$/:/^nnt1_[a-z2-7]{52}$/).test(value.value_id)||
    !Array.isArray(value.names)||value.names.some(key=>typeof key!=='string')||nativeCanonicalJson(value.names)!==nativeCanonicalJson([...new Set(value.names)].sort()))
      throw new RangeError('native derivative evaluation has invalid exact root values');
    const state=value.kind==='native-state'?await store.getNativeState?.(value.value_id):undefined;
    const tensor=value.kind!=='native-state'?await store.getNativeTensorArtifact?.(value.value_id):undefined;
    const available=state?Object.keys(state.tensors).filter(name=>/^layers\.[0-9]+\.(?:raw_k|v|gated_inputs)$/.test(name)):
      tensor&&tensor.metadata.kind===(value.kind==='adapter'?'adapter-coefficients':'parameter-set')&&
      nativeCanonicalJson(tensor.metadata.native_identity)===nativeCanonicalJson(artifact.metadata.native_identity)?Object.keys(tensor.tensors):[];
    if(!value.names.length||value.names.some(name=>!available.includes(name)))throw new RangeError('native derivative evaluation lacks exact bound root names/artifacts');
  }
  if(target.stage==='inner'){
    const node=parsed.nodes.find(node=>node.name===provenance.node);
    if(!node||node.kind!=='gradient'||node.value!==target.variable)throw new RangeError('native derivative gradient changed its declared inner objective');
    const owners=Object.keys(parsed.inputs).filter(key=>(node.bindings[key]??key)===target.variable);
    if(!owners.length||owners.some(key=>inputs[key]?.value_id!==target.value_id||nativeCanonicalJson(inputs[key]?.names)!==nativeCanonicalJson(target.names)))
      throw new RangeError('native inner gradient belongs to another evaluated value');
  }else{
    const execution=await store.getNativeTensorArtifact?.(provenance.derivative_execution_id as string);
    if(parsed.inputs[target.variable]?.id!==target.value_id||!execution||execution.metadata.kind!=='derivative-program'||
      nativeCanonicalJson(execution.metadata.native_identity)!==nativeCanonicalJson(artifact.metadata.native_identity)||
      execution.metadata.provenance.schema!=='natlang.native-derivative-execution/1'||execution.metadata.provenance.program_id!==program.id||
      execution.metadata.provenance.outer_graph_sha256!==provenance.graph_sha256||
      nativeCanonicalJson(execution.metadata.provenance.outer_evaluation_inputs)!==nativeCanonicalJson(inputs)||
      nativeCanonicalJson(execution.metadata.provenance.outer_functional_actor)!==nativeCanonicalJson(actor))
      throw new RangeError('native outer gradient differs from its original input and actual execution');
  }
  for(const entries of [actor.uniform_adapters,...Object.values(actor.invocation_adapters??{})])for(const entry of entries){
    const value=await store.getNativeTensorArtifact?.(entry.id);
    if(!value||value.metadata.kind!=='adapter-coefficients'||nativeCanonicalJson(value.metadata.native_identity)!==nativeCanonicalJson(artifact.metadata.native_identity))
      throw new RangeError('native functional actor lacks exact evaluated coefficients');
  }
  for(const entry of actor.parameter_values){
    const value=await store.getNativeTensorArtifact?.(entry.id);
    if(!value||value.metadata.kind!=='parameter-set'||nativeCanonicalJson(value.metadata.native_identity)!==nativeCanonicalJson(artifact.metadata.native_identity)||
      nativeCanonicalJson(Object.keys(value.tensors).sort())!==nativeCanonicalJson(entry.names))throw new RangeError('native functional actor lacks an exact parameter set');
  }
}
