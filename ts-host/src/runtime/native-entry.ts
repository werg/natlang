/** Trace an authored TypeScript entry as native execution, with its actual NL descendants. */
import {CallCapture,definitionKey,interfaceHash,reportStoreFailure} from '../calls/recorder.js';
import type {DefinitionIdentity} from '../calls/types.js';
import {NativeEffectOwner,recordingServices} from '../native/effects.js';
import {currentFrame,runInFrame} from './context.js';
import {resolveFrame} from './runtime.js';
import {traceFor} from '../native/graph.js';
import {openCrispTrace,capturedArgumentBindings} from './crisp-trace.js';
import type {ModuleRecord} from './loader.js';

export function invokeNativeEntry<T>(record:ModuleRecord, exportName:string, args:unknown[], invoke:()=>T):T {
  const current=currentFrame();
  const frame=current?resolveFrame():undefined;
  // Inside an existing invocation TS remains its native implementation. This
  // boundary is the authored host entry, not a new model call for each helper.
  if(!frame)return invoke();
  // Scope ancestry follows actual TS invocation even without recording. NL
  // recursion and model-depth accounting continue to use frame.chain only.
  const declared=record.exports[exportName];
  const owned = !frame.nativeEffects && !(frame.parentCallId && frame.task.isInvocationActive(frame.parentCallId));
  const nativeEffects = owned ? new NativeEffectOwner(frame.task.runtime.options.settleNativeEffects !== false, declared?.kind!=='function'||declared.async) : frame.nativeEffects;
  const scopedFrame={...frame,...(nativeEffects ? {nativeEffects} : {}),serviceChain:[...(frame.serviceChain??frame.chain),record.id]};
  const settle = (run:()=>T):T => {
    if (!nativeEffects) return run();
    if (!owned) {
      const value=run();
      if(nativeEffects.enabled && value!==null && (typeof value==='object'||typeof value==='function') && typeof (value as {then?:unknown}).then==='function') {
        const pending=Promise.resolve(value);nativeEffects.track(pending);frame.task.track(pending);
      }
      return value;
    }
    try {
      const value=run();
      if(value!==null && (typeof value==='object'||typeof value==='function') && typeof (value as {then?:unknown}).then==='function')
        return Promise.resolve(value).then(async result=>{await nativeEffects.drain();
          return result;},async error=>{await nativeEffects.drain();throw error;}).finally(()=>nativeEffects.close()) as T;
      if(nativeEffects.hasPending) throw new Error('A synchronous native entry started asynchronous Native I/O without returning its promise.');
      nativeEffects.close();return value;
    } catch(error) { nativeEffects.close();throw error; }
  };
  // Suppressing a trace root never suppresses logical effect ownership.
  const scopedInvoke=()=>settle(()=>runInFrame({...scopedFrame,services:owned ?
    recordingServices(frame.services??frame.task.services,()=>{},undefined,{task:frame.task},nativeEffects) : scopedFrame.services},invoke));
  if(!owned || frame.task.runtime.options.recording?.nativeRoots===false)return scopedInvoke();
  const task=frame.task,store=task.runtime.callStore();
  if(!store)return scopedInvoke();
  const spec=declared?.kind==='function'?declared:{args:Object.fromEntries(args.map((_,index)=>[`arg${index}`,'unknown'])),returns:'unknown'};
  const callId=task.nextCallId(),name=exportName==='default'?record.name:exportName;
  const params=Object.entries(spec.args).map(([name,type])=>({name:name.replace(/\?$/,''),type,...(name.endsWith('?')?{optional:true}:{}),
    ...(declared?.kind==='function' && declared.rest===name?{rest:true}:{})}));
  const identity:DefinitionIdentity={id:`${record.id}#${exportName}`,name,source:record.source,
    key:definitionKey({id:`${record.id}#${exportName}`,body:record.text,params,returns:spec.returns,types:record.types,subtype:'function'}),
    interface:interfaceHash(record.codebase),site:'named',subtype:'function',params,returns:spec.returns,types:record.types,
    instructions:{complete:false,reason:'excluded'}};
  let capture:CallCapture;
  try {
    capture=new CallCapture(store,store.settings(),{callId,parentCallId:frame.parentCallId??null,
      parentActionIndex:traceFor(frame.parentCallId)?.events.filter(event=>event.kind==='action').length??null,taskId:task.id,
      programId:task.programView.program?.id??null,buildHash:task.programView.program?.buildHash??null,
      programRoot:task.runtime.options.programRoot??null,definition:identity,model:{id:null,revision:null},
      checkpoints:task.runtime.options.recording?.checkpoints,exclude:task.runtime.options.recording?.exclude});
    capture.executor={kind:'crisp',model_id:null,model_revision:null};
    identity.instructions=capture.ref(record.text);
    capture.setInputs(Object.fromEntries(params.map((param,index)=>[param.name,param.rest?args.slice(index):args[index]]).filter(([,value])=>value!==undefined)));
    capture.announce();
  }catch(error){reportStoreFailure(store,error);return scopedInvoke();}
  task.beginInvocation(callId,true);
  task.registerGraphCall(callId,frame.parentCallId);
  const nativeTrace=openCrispTrace({frame,callId,definition:{id:identity.id,name,revision:record.revision,source:record.source},
    signature:`(${params.map(param=>`${param.name}${param.optional?'?':''}: ${param.type}`).join(', ')}) => ${spec.returns}`,arguments:capturedArgumentBindings(capture)});
  capture.watch(()=>nativeTrace.trace.events);
  const observeEffect=(event:import('../native/effects.js').EffectEvent)=>capture.effect({...event,
    ...(event.phase==='requested'?{graph_order:task.nextGraphEffectOrder()}:{})},'crisp');
  task.setInvocationEffectObserver(callId,observeEffect);
  const services=recordingServices(frame.services ?? task.services,observeEffect,undefined,{task,callId});
  const finish=(outcome:string,value:unknown,error?:unknown)=>{
    try { nativeTrace.finish(outcome);capture.finish({outcome,detail:error instanceof Error?error.message:error===undefined?'':String(error),
      output:value,hasOutput:outcome==='done',events:nativeTrace.trace.events,graphChildren:task.graphChildren(callId)}); }
    finally { task.endInvocation(callId); }
  };
  try {
    const value=settle(()=>runInFrame({...scopedFrame,parentCallId:callId,services},invoke));
    if(value!==null && (typeof value==='object'||typeof value==='function') && typeof (value as {then?:unknown}).then==='function')
      return Promise.resolve(value).then(result=>{finish('done',result);return result;},error=>{finish('error',undefined,error);throw error;}) as T;
    finish('done',value);return value;
  }catch(error){if(owned) nativeEffects?.close();finish('error',undefined,error);throw error;}
}
