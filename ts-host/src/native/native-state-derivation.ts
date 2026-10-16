/** Immutable optimized-state payload/origin verifier. Numeric optimizer
 * execution is reverified by the Python consumer under its bound Torch CPU
 * runtime; this codec does not claim to implement Torch's Adam arithmetic. */
import {nativeCanonicalJson,type NativeStateArtifact,type NativeTensorArtifact,type NeuraleseStore,type NativeStateTensor} from './neuralese-store.js';
import {validateNativeStateDerivation} from './native-derivative-contract.js';
const equalTensor=(a:NativeStateTensor|undefined,b:NativeStateTensor|undefined)=>!!a&&!!b&&a.dtype===b.dtype&&
  JSON.stringify(a.shape)===JSON.stringify(b.shape)&&a.data.length===b.data.length&&a.data.every((v,i)=>v===b.data[i]);
export async function verifyNativeStateDerivation(store:Pick<NeuraleseStore,'getNativeState'|'getNativeTensorArtifact'>,
    artifact:Omit<NativeStateArtifact,'id'>,seen=new Set<string>()):Promise<void>{
  const value=artifact.metadata.derivation;
  validateNativeStateDerivation(value);if(!value)return;
  if(seen.has(value.parent_state_id))throw new RangeError('native derivation ancestry is cyclic');
  const nested=new Set(seen);nested.add(value.parent_state_id);
  const parent=await store.getNativeState?.(value.parent_state_id);
  if(!parent)throw new RangeError('optimized native state needs its exact archived parent');
  await verifyNativeStateDerivation(store,parent,nested);
  if(value.kind!=='optimizer'){verifyNativeStateDerivationRecords(artifact,parent,undefined,undefined);return;}
  const gradient=await store.getNativeTensorArtifact?.(value.gradient_id),update=await store.getNativeTensorArtifact?.(value.update_artifact_id);
  const prior=value.optimizer_state_parent_id===null?undefined:await store.getNativeTensorArtifact?.(value.optimizer_state_parent_id);
  const program=value.program_id===null?undefined:await store.getNativeTensorArtifact?.(value.program_id);
  verifyNativeStateDerivationRecords(artifact,parent,gradient,update,prior,program);
}

/** Pure payload/provenance check, reused by synchronous portable-file decoding. */
export function verifyNativeStateDerivationRecords(artifact:Omit<NativeStateArtifact,'id'>,parent:NativeStateArtifact,
    gradient:NativeTensorArtifact|undefined,update:NativeTensorArtifact|undefined,prior?:NativeTensorArtifact,program?:NativeTensorArtifact):void {
  const value=artifact.metadata.derivation;
  validateNativeStateDerivation(value);if(!value)return;
  if(parent.id!==value.parent_state_id)throw new RangeError('optimized native state parent binding changed');
  const origin=parent.metadata.derivation?.origin_state_ids??[parent.id];
  if(nativeCanonicalJson(origin)!==nativeCanonicalJson(value.origin_state_ids))throw new RangeError('optimized native state changed its original producer provenance');
  if(value.kind!=='optimizer'){
    const identity=(metadata:NativeStateArtifact['metadata'])=>({checkpoint_sha256:metadata.checkpoint_sha256,tokenizer_fingerprint:metadata.tokenizer_fingerprint,
      embedding_weight_sha256:metadata.embedding_weight_sha256,architecture:metadata.architecture});
    if(artifact.metadata.text!==parent.metadata.text||nativeCanonicalJson(value.source_native_identity)!==nativeCanonicalJson(identity(parent.metadata))||
      nativeCanonicalJson(value.target_native_identity)!==nativeCanonicalJson(identity(artifact.metadata)))throw new RangeError('derived semantic state changed source/target provenance');
    if(value.kind==='semantic-encode'){
      if(artifact.metadata.schema!=='neuralese.semantic-state/1'||artifact.metadata.adapter_sha256!==value.adapter_sha256||
        nativeCanonicalJson(identity(artifact.metadata))!==nativeCanonicalJson(identity(parent.metadata))||
        ['token_ids','positions','embeddings'].some(name=>!equalTensor(artifact.tensors[name],parent.tensors[name])))
        throw new RangeError('derived semantic encoding changed tokens, embeddings or adapter');
    }else if(artifact.metadata.schema!=='neuralese.native-state/1'||parent.metadata.schema==='neuralese.semantic-state/1'&&parent.metadata.adapter_sha256!==value.adapter_sha256)
      throw new RangeError('derived semantic decode changed trained adapter or target format');
    return;
  }
  const {derivation:_,...before}=parent.metadata,{derivation:__,...after}=artifact.metadata;
  if(nativeCanonicalJson(before)!==nativeCanonicalJson(after))throw new RangeError('optimized native state changed text, native basis or adapter provenance');
  const expectedBase=parent.metadata.adapter_binding?.base_native_identity??{
    checkpoint_sha256:parent.metadata.checkpoint_sha256,architecture:parent.metadata.architecture,
    tokenizer_fingerprint:parent.metadata.tokenizer_fingerprint,embedding_weight_sha256:parent.metadata.embedding_weight_sha256};
  if(!gradient||!update||gradient.metadata.kind!=='state-gradient'||gradient.metadata.source_artifact_id!==parent.id||
    update.metadata.kind!=='parameter-update'||update.metadata.source_artifact_id!==gradient.id||
    nativeCanonicalJson(gradient.metadata.native_identity)!==nativeCanonicalJson(expectedBase)||
    nativeCanonicalJson(update.metadata.native_identity)!==nativeCanonicalJson(gradient.metadata.native_identity)||
    nativeCanonicalJson(Object.keys(gradient.tensors).sort())!==nativeCanonicalJson(value.updated_tensor_names)||
    nativeCanonicalJson(Object.keys(update.tensors).sort())!==nativeCanonicalJson(value.updated_tensor_names)||
    (gradient.metadata.provenance.derivative_program_id??null)!==value.program_id||
    nativeCanonicalJson(update.metadata.provenance)!==nativeCanonicalJson({schema:value.schema,parent_state_id:parent.id,
      optimizer:value.optimizer,step:value.step,optimizer_state_parent_id:value.optimizer_state_parent_id,derivative_program_id:value.program_id,arithmetic:value.arithmetic}))
    throw new RangeError('optimized native state update, gradient, optimizer or program binding differs');
  if(nativeCanonicalJson(Object.keys(parent.tensors).sort())!==nativeCanonicalJson(Object.keys(artifact.tensors).sort())||
    Object.entries(parent.tensors).some(([name,tensor])=>!equalTensor(artifact.tensors[name],value.updated_tensor_names.includes(name)?update.tensors[name]:tensor)))
    throw new RangeError('optimized native state changed an unselected tensor or differs from its numeric update');
  if(value.optimizer_state_parent_id===null){if(value.step!==1)throw new RangeError('new native state history must begin at step one');}
  else{
    if(!prior||prior.metadata.kind!=='optimizer-state'||prior.metadata.source_artifact_id!==parent.id||
      nativeCanonicalJson(prior.metadata.native_identity)!==nativeCanonicalJson(gradient.metadata.native_identity)||
      prior.metadata.provenance.scope!==value.schema||prior.metadata.provenance.optimizer!==value.optimizer.name||prior.metadata.provenance.step!==value.step-1)
      throw new RangeError('optimized native state lacks its exact previous optimizer history');
  }
  if(value.program_id!==null&&program?.metadata.kind!=='derivative-program')
    throw new RangeError('optimized native state lacks its executable derivative program');
}
