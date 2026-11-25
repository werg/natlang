/** Real trace ancestry for native invocations that do not run a model session. */
import {NativeTraceRecorder} from '../native/trace.js';
import {executionOccurrence,occurrenceArgumentHash} from '../native/execution-occurrence.js';
import {FILE_CONTEXT,invocationNodeId,producerScopeForTrace,registerTrace,releaseTrace,traceFor} from '../native/graph.js';
import type {NativeExecutionOccurrence} from '../contracts.js';
import type {CallCapture} from '../calls/recorder.js';
import type {Frame} from './context.js';
/** Read only snapshots already taken by the recorder; never execute argument getters again. */
export function capturedArgumentBindings(capture:CallCapture):Record<string,unknown>{
  return Object.fromEntries(Object.entries(capture.inputs).map(([name,ref])=>{
    const text=ref.complete?capture.blobs.get(ref.hash):undefined;
    return [name,text===undefined?{$unobserved:ref}:JSON.parse(text)];
  }));
}
export function openCrispTrace(input:{frame:Frame;callId:string;definition:{id:string;name:string;revision:string;source?:string};signature:string;arguments:unknown;ordinal?:number}){
  const {frame,callId,definition}=input,parent=traceFor(frame.parentCallId);
  const ordinal=input.ordinal??frame.task.reserveExecutionSiblingOrdinal(frame.parentCallId,definition.id);
  const parentOccurrence=parent?.events.find(e=>e.kind==='invocation'&&e.phase==='start'&&e.call_id===frame.parentCallId)?.native_occurrence as NativeExecutionOccurrence|undefined;
  const occurrence=executionOccurrence({definitionId:definition.id,revision:definition.revision,context:FILE_CONTEXT,signature:input.signature,parent:parentOccurrence,ordinal,argumentHash:occurrenceArgumentHash(input.arguments)});
  const trace=new NativeTraceRecorder({run_id:callId,parent_call_id:frame.parentCallId??null,task_id:frame.task.id,definition_id:definition.id,definition_name:definition.name,
    source_revision:definition.revision,...(definition.source?{definition_source:definition.source}:{}),context_id:FILE_CONTEXT,sibling_ordinal:ordinal,executor:'crisp',coverage:'native-invocation-and-recorded-children'});
  trace.emit('invocation',{phase:'start',call_id:callId,attempt:1,node:invocationNodeId(callId),inputs:frame.parentCallId?[{node:invocationNodeId(frame.parentCallId),port:'caller'}]:[],
    definition:{id:definition.id,revision:definition.revision},context:FILE_CONTEXT,signature:input.signature,captures:{},native_occurrence:occurrence});
  registerTrace(callId,trace,producerScopeForTrace(parent)??trace);
  let open=true;
  return {trace,finish(outcome:string){if(!open)return;open=false;try{trace.emit('invocation',{phase:'end',call_id:callId,outcome});}finally{releaseTrace(callId);}}};
}
