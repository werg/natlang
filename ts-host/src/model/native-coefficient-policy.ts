/** Exact declared actor bindings for generation, decisions and held execution. */
import {nativeCanonicalJson,nativeModelIdentityEqual,type NativeModelIdentity} from '../native/neuralese-store.js';
export type NativeCoefficientRuntime = {schema:'natlang.native-coefficient-runtime/1';base_native_identity:NativeModelIdentity;
  policy_sha256:string;replacements:Record<string,string>;defaults:{id:string;scale:number}[];
  state_transport:'strict'|'same-base-adapter-state'};
const id=/^nnt1_[a-z2-7]{52}$/;
const invalid=():never=>{throw new Error('neuralese-native-coefficient-policy: restore the exact declared actor policy and its coefficient bundle');};
export function validateCoefficientRuntime(value:unknown):NativeCoefficientRuntime{
  const policy=value as NativeCoefficientRuntime;
  if(!policy || Object.keys(policy).sort().join(',')!=='base_native_identity,defaults,policy_sha256,replacements,schema,state_transport' ||
    policy.schema!=='natlang.native-coefficient-runtime/1' || !/^[0-9a-f]{64}$/.test(policy.policy_sha256) ||
    !['strict','same-base-adapter-state'].includes(policy.state_transport) || !policy.replacements ||
    typeof policy.replacements!=='object' || Array.isArray(policy.replacements) || !Object.keys(policy.replacements).length ||
    Object.entries(policy.replacements).some(([before,after])=>!id.test(before) || typeof after!=='string' || !id.test(after)) ||
    !Array.isArray(policy.defaults) || policy.defaults.some(entry=>!entry || Object.keys(entry).sort().join(',')!=='id,scale' ||
      !(entry.id in policy.replacements) || typeof entry.scale!=='number' || !Number.isFinite(entry.scale)))return invalid();
  const base=policy.base_native_identity;
  if(!base || Object.keys(base).sort().join(',')!=='architecture,checkpoint_sha256,embedding_weight_sha256,tokenizer_fingerprint' ||
    ![base.checkpoint_sha256,base.tokenizer_fingerprint,base.embedding_weight_sha256].every(value=>typeof value==='string' && /^[0-9a-f]{64}$/.test(value)) ||
    typeof base.architecture!=='string' || !base.architecture)return invalid();
  return policy;
}
export function requireCoefficientAttestation(policy:NativeCoefficientRuntime,info:Record<string,unknown>|null):void{
  const receipt=info?.rl_policy_binding as {native_identity?:NativeModelIdentity;coefficient_policy_sha256?:string;
    coefficient_policy?:NativeCoefficientRuntime}|undefined;
  if(!receipt?.native_identity || !nativeModelIdentityEqual(receipt.native_identity,policy.base_native_identity) ||
    receipt.coefficient_policy_sha256!==policy.policy_sha256 || nativeCanonicalJson(receipt.coefficient_policy)!==nativeCanonicalJson(policy))invalid();
}
export function coefficientBindings(policy:NativeCoefficientRuntime|undefined,active:readonly {id:string;scale:number}[],bound:readonly {id:string;scale:number}[]):{id:string;scale:number}[]{
  return [...(policy?.defaults??[]),...active,...bound].map(entry=>({...entry,id:policy?.replacements[entry.id]??entry.id}));
}
export function coefficientStateModes(policy:NativeCoefficientRuntime|undefined,ids:readonly string[],explicit?:Record<string,'same-base-adapter-state'>):Record<string,'same-base-adapter-state'>|undefined{
  if(policy?.state_transport!=='same-base-adapter-state')return explicit;
  return {...Object.fromEntries(ids.map(id=>[id,'same-base-adapter-state' as const])),...explicit};
}
