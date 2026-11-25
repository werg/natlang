/**
 * The trace of one combinator run: a record `std:<name>` under the call that ran it, with one `std_stage` event per
 * stage naming the calls the stage made. Child calls run inside this span's own frame (`parentCallId === this.callId`),
 * so `natlang traces` nests them under the combinator and concurrent combinators never bleed call IDs across stages.
 */
import {CallCapture,definitionKey,interfaceHash,reportStoreFailure} from '../calls/recorder.js';
import type {DefinitionIdentity} from '../calls/types.js';
import {traceFor} from '../native/graph.js';
import {openCrispTrace} from '../runtime/crisp-trace.js';
import {hexDigest} from '../native/hash.js';
import {STD_TRACE_SOURCE} from './trace-source.generated.js';
import { currentFrame, runInFrame, type Frame } from '../runtime/context.js';
import { resolveFrame } from '../runtime/runtime.js';

let sequence = 0;

export class StdSpan {
  readonly parentFrame: Frame;
  readonly callId: string;
  readonly frame: Frame;
  private readonly events: Record<string, unknown>[] = [];
  private mark: number;
  private readonly started = Date.now();
  private capture?:CallCapture;
  private open=true;
  private readonly nativeTrace:ReturnType<typeof openCrispTrace>;

  constructor(readonly name: string) {
    this.parentFrame = currentFrame() ?? resolveFrame();
    this.callId = `${this.parentFrame.task.id}/std-${this.name}-${++sequence}`;
    this.frame = { ...this.parentFrame, parentCallId: this.callId };
    this.mark = this.parentFrame.task.traces.length;
    const task=this.parentFrame.task;
    task.beginInvocation(this.callId);
    task.registerGraphCall(this.callId,this.parentFrame.parentCallId);
    this.nativeTrace=openCrispTrace({frame:this.parentFrame,callId:this.callId,definition:{id:`std:${name}`,name:`std ${name}`,revision:hexDigest(STD_TRACE_SOURCE).slice(0,16),source:'@natlang/std/trace.ts'},
      signature:'(operation: string) => unknown',arguments:{operation:name}});
    const store=task.runtime.callStore();
    if(store)try{
      // This is the native execution scope, not a fabricated authored program.
      // Its exact executing implementation and operation label are observed here.
      const body=STD_TRACE_SOURCE;
      const id=`std:${name}`,params=[{name:'operation',type:'string'}];
      const definition:DefinitionIdentity={id,name:`std ${name}`,source:'@natlang/std/trace.ts',
        key:definitionKey({id,body,params,returns:'unknown',types:{},subtype:'function'}),interface:interfaceHash({}),
        site:'internal',subtype:'function',params,returns:'unknown',types:{},instructions:{complete:false,reason:'excluded'}};
      const capture=new CallCapture(store,store.settings(),{callId:this.callId,parentCallId:this.parentFrame.parentCallId??null,
        parentActionIndex:traceFor(this.parentFrame.parentCallId)?.events.filter(event=>event.kind==='action').length??null,
        taskId:task.id,programId:task.programView.program?.id??null,buildHash:task.programView.program?.buildHash??null,
        programRoot:task.runtime.options.programRoot??null,definition,model:{id:null,revision:null},
        checkpoints:task.runtime.options.recording?.checkpoints,exclude:task.runtime.options.recording?.exclude});
      capture.executor={kind:'crisp',model_id:null,model_revision:null};
      definition.instructions=capture.ref(body);capture.setInputs({operation:name});capture.announce();capture.watch(()=>this.nativeTrace.trace.events);
      this.capture=capture;task.markInvocationRecorded(this.callId);
    }catch(error){reportStoreFailure(store,error);}
  }

  /** Run a caller's or a shipped function inside this combinator's span frame. */
  run<T>(fn: () => T): T { return runInFrame(this.frame, fn); }

  /** Emit a custom event directly onto this span's trace (for example `decision_escalate`). */
  emit(kind: string, data: Record<string, unknown> = {}): void {
    this.events.push({ kind, call_id: this.callId, combinator: this.name, ...data });
    this.nativeTrace.trace.emit(kind,{call_id:this.callId,combinator:this.name,...data});
  }

  /** Close a stage: the child calls recorded since the previous stage under this span, and the stage's facts. Returns those calls' IDs. */
  stage(stage: string, data: Record<string, unknown> = {}): string[] {
    const traces = this.parentFrame.task.traces;
    const calls = traces.slice(this.mark).filter(trace => trace.parentCallId === this.callId).map(trace => trace.callId);
    this.mark = traces.length;
    this.emit('std_stage',{stage,call_ids:calls,...data});
    return calls;
  }

  end(outcome: 'done' | 'error', detail: string, value?:unknown): void {
    if(!this.open)return;
    this.open=false;
    const task = this.parentFrame.task;
    const events=[{ kind: 'std', call_id: this.callId, combinator: this.name, wall_ms: Date.now() - this.started }, ...this.events];
    try {
      this.nativeTrace.finish(outcome);
      this.capture?.finish({outcome,detail,output:value,hasOutput:outcome==='done',events:this.nativeTrace.trace.events,graphChildren:task.graphChildren(this.callId)});
      try {task.record({ callId: this.callId, parentCallId: this.parentFrame.parentCallId ?? null, taskId: task.id,
        definitionId: `std:${this.name}`, name: `std ${this.name}`, outcome, detail,events });}
      catch(error){reportStoreFailure(task.runtime,error);}
    } finally {this.nativeTrace.finish(outcome);this.capture=undefined;task.endInvocation(this.callId);}
  }

  /** Run `body` as the combinator, recording the trace whether it returns or throws. */
  async around<T>(body: () => Promise<T>, detail: (value: T) => string): Promise<T> {
    try {
      const value = await body();
      this.end('done', detail(value),value);
      return value;
    } catch (error) {
      let detail='Standard-library operation failed';
      try {const message=typeof error==='string'?error:error&&Object.getOwnPropertyDescriptor(error,'message')?.value;if(typeof message==='string')detail=message;}catch{/* Diagnostics never replace the original thrown value. */}
      this.end('error', detail);
      throw error;
    }
  }
}

/** A `TypeError` with a stable code, the form every std argument error takes. */
export function stdError(code: string, message: string): TypeError {
  return Object.assign(new TypeError(`${code}: ${message}`), { code });
}
