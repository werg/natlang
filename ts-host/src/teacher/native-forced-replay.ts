/** Bind a self-hosted Qwen correction turn to an exact no-hint task replay. */
import { createHash, randomUUID } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { executeProgram, type ExecuteOptions, type ProgramRecord, type ProgramRun } from './collector.js';

type TeacherTurn = { phase?: unknown; invocation_id?: unknown; context?: unknown; tools_offered?: unknown;
  assistant?: { content?: unknown; reasoning?: unknown; calls?: { tool: string; arguments: Record<string, unknown> }[] } };
type TeacherRow = { version?: unknown; id?: unknown; task?: unknown; provenance?:Record<string,unknown>;
  trajectory?: unknown; outcome?: { execution_graph?: unknown; invocation_ledger?:unknown } };
type SourceBinding = { sourcePath: string; sourceFileSha256: string; sourceView: 'unhinted' | 'privileged-hint';
  actorSourceSha256: string; executionIdentitySha256: string; model: 'nvidia/Qwen3.6-35B-A3B-NVFP4';
  revision: '1355db6a052410cfd62085d94b58866fd0f2c3c5'; actorSourcePath:string; executionIdentityPath:string;
  sourceVerificationPath:string; sourceVerificationSha256:string; transferReceiptPath:string; transferReceiptSha256:string };

function canonical(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  return `{${Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b))
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(',')}}`;
}
function sha(value: string): string { return createHash('sha256').update(value).digest('hex'); }
function shaBytes(value: Uint8Array): string { return createHash('sha256').update(value).digest('hex'); }

/** Load one Qwen row only through its canonical tranche and local immutable actor transfer receipts. */
export async function loadVerifiedQwenSource(trancheDirectory: string, rowId: string,
    sourceView: SourceBinding['sourceView']): Promise<{ row: TeacherRow; binding: SourceBinding }> {
  const root = resolve(trancheDirectory);
  const [canonicalBytes, transferBytes] = await Promise.all([
    readFile(resolve(root, 'canonical-source-verification.json')),
    readFile(resolve(root, 'transfer-receipt.json'))]);
  const canonicalSource = JSON.parse(canonicalBytes.toString('utf8')) as Record<string, any>;
  const transfer = JSON.parse(transferBytes.toString('utf8')) as Record<string, any>;
  if (canonicalSource.schema !== 'natlang.teacher-live-tranche-canonical-source-verification/1' ||
      canonicalSource.status !== 'passed' || canonicalSource.training_admission !== false ||
      transfer.schema !== 'natlang.verified-teacher-trajectory-transfer/1' ||
      !Array.isArray(transfer.files) || !canonicalSource.source ||
      typeof canonicalSource.source.path !== 'string' || !/^[0-9a-f]{64}$/.test(canonicalSource.source.sha256))
    throw new Error('Qwen tranche canonical-source or transfer receipt is not an accepted source binding');
  const sourcePath = resolve(canonicalSource.source.path);
  const sourceBytes = await readFile(sourcePath);
  if (shaBytes(sourceBytes) !== canonicalSource.source.sha256 || sourceBytes.length !== canonicalSource.source.bytes)
    throw new Error('Qwen trajectory source bytes do not match the canonical tranche receipt');
  const actorEntry = transfer.files.find((entry: any) => typeof entry.path === 'string' && entry.path.endsWith('/teacher-serving-actor.json'));
  const identityEntry = transfer.files.find((entry: any) => typeof entry.path === 'string' && entry.path.endsWith('/teacher-execution-identity.json'));
  if (!actorEntry || !identityEntry) throw new Error('Qwen transfer receipt omits its serving actor or execution identity');
  const [actorBytes, identityBytes] = await Promise.all([readFile(actorEntry.path), readFile(identityEntry.path)]);
  if (shaBytes(actorBytes) !== actorEntry.sha256 || actorBytes.length !== actorEntry.bytes ||
      shaBytes(identityBytes) !== identityEntry.sha256 || identityBytes.length !== identityEntry.bytes ||
      canonicalSource.actor?.sha256 !== actorEntry.sha256 || canonicalSource.execution_identity?.sha256 !== identityEntry.sha256)
    throw new Error('Qwen actor proof bytes do not match canonical and transfer receipts');
  const actor = JSON.parse(actorBytes.toString('utf8')) as Record<string, any>;
  const execution = JSON.parse(identityBytes.toString('utf8')) as Record<string, any>;
  if (actor.schema !== 'natlang.selected-teacher-actor-source/1' ||
      actor.model !== 'nvidia/Qwen3.6-35B-A3B-NVFP4' || actor.revision !== '1355db6a052410cfd62085d94b58866fd0f2c3c5' ||
      execution.schema !== 'natlang.self-hosted-teacher-execution-identity/1' || execution.model_id !== 'qwen3.6-35b-a3b' ||
      execution.actor?.sha256 !== actorEntry.sha256 || transfer.teacher?.model !== actor.model ||
      transfer.teacher?.revision !== actor.revision || transfer.teacher?.artifact?.sha256 !== actor.download?.sha256)
    throw new Error('Qwen serving actor and actual execution identity are not the selected teacher');
  const lines = sourceBytes.toString('utf8').split(/\r?\n/).filter(line => line.trim());
  if (lines.length !== canonicalSource.source.rows) throw new Error('Qwen trajectory source row count differs from its tranche receipt');
  const candidates = lines.map(line => JSON.parse(line) as TeacherRow).filter(row => row.id === rowId);
  if (candidates.length !== 1) throw new Error('Qwen source row id is missing or ambiguous in the verified tranche');
  if ((candidates[0]!.provenance as Record<string, unknown> | undefined)?.model !== 'qwen3.6-35b-a3b')
    throw new Error('selected Qwen teacher row carries another actor identity');
  return { row: candidates[0]!, binding: { sourcePath, sourceFileSha256: canonicalSource.source.sha256,
    sourceView, actorSourceSha256: actorEntry.sha256, executionIdentitySha256: identityEntry.sha256,
    model: actor.model, revision: actor.revision, actorSourcePath:actorEntry.path,executionIdentityPath:identityEntry.path,
    sourceVerificationPath:resolve(root,'canonical-source-verification.json'),sourceVerificationSha256:shaBytes(canonicalBytes),
    transferReceiptPath:resolve(root,'transfer-receipt.json'),transferReceiptSha256:shaBytes(transferBytes) } };
}

type ForcedTextSpan={path:(string|number)[];marker:string;text:string};
function teacherParts(value: unknown, row: TeacherRow, turn: TeacherTurn, turnOrdinal: number, context: string,
    path:(string|number)[], spans:ForcedTextSpan[]): unknown {
  if (Array.isArray(value)) return value.map((item,index) => teacherParts(item,row,turn,turnOrdinal,context,[...path,index],spans));
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value as Record<string,unknown>)
    .map(([key,item])=>[key,teacherParts(item,row,turn,turnOrdinal,context,[...path,key],spans)]));
  if (typeof value !== 'string') return value;
  const markers=[...value.matchAll(/<\|neuralese\|>([\s\S]*?)<\|\/neuralese\|>/g)];
  if (!markers.length) return value;
  if(context==='return-result'&&(markers.length!==1||markers[0]![0]!==value))return value;
  const graph=row.outcome?.execution_graph as Record<string,any>[];
  const modelTurns=graph.filter(event=>event.kind==='model_turn'&&event.call_id===turn.invocation_id&&event.turn===turnOrdinal+1);
  if (modelTurns.length!==1) return value;
  const modelNode=modelTurns[0]!.node;
  const pieces: unknown[]=[]; let cursor=0;
  for (const marker of markers) {
    const body=marker[1]!;
    const evidence=graph.filter(event=>event.kind==='block_write'&&event.call_id===turn.invocation_id&&
      event.turn===modelNode&&event.marker_context===context&&event.result_type==='Neuralese<string>'&&
      ['typed-text-result','typed-text-result-field'].includes(String(event.source_kind))&&
      event.producer==='text-marker-emulation'&&event.emulation_version==='text-marker-standin/2'&&
      event.text_body_sha256===sha(body)&&event.truncated===false);
    const sameBodyMarkers=markers.filter(candidate=>candidate[1]===body);
    if (evidence.length!==1||sameBodyMarkers.length!==1) continue;
    if (marker.index!>cursor) pieces.push(value.slice(cursor,marker.index));
    if (!body) throw new Error('typed Qwen Neuralese source span is empty');
    const replacement=`NATLANGQWENSPAN${randomUUID().replace(/-/g,'')}END`;
    spans.push({path,marker:replacement,text:body});pieces.push(replacement);cursor=marker.index!+marker[0].length;
  }
  if (!pieces.length) return value;
  if (cursor<value.length) pieces.push(value.slice(cursor));
  return pieces.join('');
}

function teacherAssistant(row: TeacherRow, turn: TeacherTurn, ordinal: number): {assistant:Record<string,unknown>;spans:ForcedTextSpan[]} {
  const source=turn.assistant!;
  const spans:ForcedTextSpan[]=[];
  const content=teacherParts(source.content,row,turn,ordinal,'assistant-content',['content'],spans);
  const calls=source.calls!.map((call,index)=>{
    const args=structuredClone(call.arguments);
    if(call.tool==='eval'&&typeof args.code==='string')args.code=teacherParts(args.code,row,turn,ordinal,'eval-code',
      ['tool_calls',index,'arguments','code'],spans);
    if(call.tool==='return_result'&&Object.hasOwn(args,'value'))args.value=teacherParts(args.value,row,turn,ordinal,'return-result',
      ['tool_calls',index,'arguments','value'],spans);
    return {id:`teacher-${index}`,type:'function',function:{name:call.tool,arguments:JSON.stringify(args)}};
  });
  return {assistant:{role:'assistant',content,
      ...(source.reasoning?{reasoning:source.reasoning}:{}),tool_calls:calls},spans};
}

/**
 * The returned driver requires the replay to reuse the teacher source run ID and exact invocation IDs.
 * It supplies targets as transport metadata; task messages are never copied or edited here.
 */
function qwenTeacherForcedDriver(row: TeacherRow, binding: SourceBinding,
    send: (request: ModelTurnRequest) => Promise<ModelTurn>): { driver: (request: ModelTurnRequest) => Promise<ModelTurn>; runId: string } {
  if (row.version !== 'natlang.teacher_trajectory.native/1' || !Array.isArray(row.trajectory) || !row.task ||
      !row.id || typeof row.id !== 'string' || !/^[0-9a-f]{64}$/.test(binding.sourceFileSha256) ||
      !/^[0-9a-f]{64}$/.test(binding.actorSourceSha256) || !/^[0-9a-f]{64}$/.test(binding.executionIdentitySha256) ||
      binding.model !== 'nvidia/Qwen3.6-35B-A3B-NVFP4' || binding.revision !== '1355db6a052410cfd62085d94b58866fd0f2c3c5')
    throw new Error('Qwen source row or immutable actor/source binding is invalid');
  const graph = row.outcome?.execution_graph;
  if (!Array.isArray(graph)) throw new Error('Qwen source row lacks its actual execution graph for occurrence matching');
  const startEvents = new Map<string, Record<string, any>>();
  for (const event of graph as Record<string, any>[]) if (event.kind === 'invocation' && event.phase === 'start' &&
      typeof event.call_id === 'string') {
    if (startEvents.has(event.call_id)) throw new Error('Qwen source graph has duplicate invocation start bindings');
    startEvents.set(event.call_id, event);
  }
  const ledgerParents=new Map<string,string|null>();
  for(const item of Array.isArray(row.outcome?.invocation_ledger)?row.outcome.invocation_ledger as Record<string,any>[]:[])
    if(typeof item.invocation_id==='string')ledgerParents.set(item.invocation_id,
      typeof item.parent_invocation_id==='string'?item.parent_invocation_id:null);
  const siblingOrdinals=new Map<string,number>(),siblingCounts=new Map<string,number>();
  for(const [callId,event] of [...startEvents].sort((a,b)=>Number(a[1].seq??0)-Number(b[1].seq??0))){
    const definition=event.definition&&typeof event.definition==='object'?event.definition:{};
    const siblingKey=canonical({parent:ledgerParents.get(callId)??null,definition_id:String(definition.id??''),
      revision:String(definition.revision??'')});
    siblingOrdinals.set(callId,siblingCounts.get(siblingKey)??0);siblingCounts.set(siblingKey,(siblingCounts.get(siblingKey)??0)+1);
  }
  const sourceOccurrences=new Map<string,Record<string,any>>(),occurrenceStack=new Set<string>();
  const sourceOccurrence=(callId:string):Record<string,any>=>{
    const cached=sourceOccurrences.get(callId);if(cached)return cached;
    const event=startEvents.get(callId);if(!event)throw new Error(`Qwen source parent invocation ${callId} is absent from its graph`);
    if(occurrenceStack.has(callId))throw new Error('Qwen source invocation parent bindings contain a cycle');
    occurrenceStack.add(callId);
    const definition=event.definition&&typeof event.definition==='object'?event.definition:{};
    const parentId=ledgerParents.get(callId)??null;
    const parentHash=parentId===null?null:sourceOccurrence(parentId).occurrence_sha256;
    let occurrence=event.native_occurrence;
    if(occurrence!==undefined){
      const base={schema:occurrence.schema,definition_id:occurrence.definition_id,revision:occurrence.revision,
        context:occurrence.context,signature:occurrence.signature,parent_occurrence_sha256:occurrence.parent_occurrence_sha256,
        sibling_ordinal:occurrence.sibling_ordinal,argument_binding_sha256:occurrence.argument_binding_sha256};
      if(occurrence.schema!=='natlang.execution-occurrence/1'||occurrence.occurrence_sha256!==sha(canonical(base)))
        throw new Error('Qwen source graph carries a malformed typed execution-occurrence binding');
    }else{
      const base={schema:'natlang.execution-occurrence/1',definition_id:String(definition.id??''),
        revision:String(definition.revision??''),context:String(event.context??''),signature:String(event.signature??''),
        parent_occurrence_sha256:parentHash,sibling_ordinal:siblingOrdinals.get(callId)??0};
      occurrence={...base,occurrence_sha256:sha(canonical(base))};
    }
    occurrenceStack.delete(callId);sourceOccurrences.set(callId,occurrence);return occurrence;
  };
  const turns = new Map<string, { turn: TeacherTurn; ordinal: number }[]>();
  for (const item of row.trajectory as TeacherTurn[]) {
    if (item.phase !== 'action' || typeof item.invocation_id !== 'string' || !item.assistant ||
        !Array.isArray(item.assistant.calls) || !Array.isArray(item.context))
      throw new Error('Qwen source trajectory contains an incomplete action turn');
    if (!startEvents.has(item.invocation_id))
      throw new Error(`Qwen source invocation ${item.invocation_id} has no execution-graph start event`);
    const entries = turns.get(item.invocation_id) ?? [];
    entries.push({ turn: item, ordinal: entries.length }); turns.set(item.invocation_id, entries);
  }
  const byOccurrence = new Map<string, { sourceId: string; entries: { turn: TeacherTurn; ordinal: number }[] }>();
  for (const [sourceId, entries] of turns) {
    const occurrence=sourceOccurrence(sourceId); const key=canonical({definition_id:occurrence.definition_id,
      revision:occurrence.revision,context:occurrence.context,signature:occurrence.signature,
      parent_occurrence_sha256:occurrence.parent_occurrence_sha256,sibling_ordinal:occurrence.sibling_ordinal});
    if (byOccurrence.has(key)) throw new Error('Qwen source has ambiguous duplicate structural execution occurrences');
    byOccurrence.set(key,{sourceId,entries});
  }
  const matched = new Map<string, number>();
  const runId = `qwen-forced-${randomUUID()}`;
  return { runId, driver: async request => {
    const occurrence=request.native_execution_occurrence;
    if (!occurrence || occurrence.schema!=='natlang.execution-occurrence/1' ||
        occurrence.occurrence_sha256!==createHash('sha256').update(canonical({schema:occurrence.schema,
          definition_id:occurrence.definition_id,revision:occurrence.revision,context:occurrence.context,
          signature:occurrence.signature,parent_occurrence_sha256:occurrence.parent_occurrence_sha256,
          sibling_ordinal:occurrence.sibling_ordinal,argument_binding_sha256:occurrence.argument_binding_sha256})).digest('hex'))
      throw new Error('replay request lacks an authenticated structural execution occurrence');
    const structuralKey=canonical({definition_id:occurrence.definition_id,revision:occurrence.revision,
      context:occurrence.context,signature:occurrence.signature,parent_occurrence_sha256:occurrence.parent_occurrence_sha256,
      sibling_ordinal:occurrence.sibling_ordinal});
    const source=byOccurrence.get(structuralKey);
    if (!source) throw new Error('no exact Qwen source execution-graph occurrence matches this replay call');
    const sourceOccurrenceBinding=sourceOccurrence(source.sourceId);
    if(typeof sourceOccurrenceBinding.argument_binding_sha256==='string'&&
        sourceOccurrenceBinding.argument_binding_sha256!==occurrence.argument_binding_sha256)
      throw new Error('Qwen source argument binding differs from the actual replay invocation');
    const ordinal = matched.get(source.sourceId) ?? 0; const entry = source.entries[ordinal];
    if (!entry) throw new Error(`no exact Qwen source action turn remains for occurrence ${source.sourceId}#${ordinal}`);
    matched.set(source.sourceId, ordinal + 1);
    const turn = entry.turn, assistant = turn.assistant!;
    const sourcePrompt = { messages: turn.context, tools: turn.tools_offered ?? [] };
    const target = teacherAssistant(row,turn,entry.ordinal);
    const receipt = { schema: 'natlang.teacher-turn-source/1', source_path: binding.sourcePath,
      source_file_sha256: binding.sourceFileSha256, source_view: binding.sourceView,
      source_verification_path:binding.sourceVerificationPath,source_verification_sha256:binding.sourceVerificationSha256,
      transfer_receipt_path:binding.transferReceiptPath,transfer_receipt_sha256:binding.transferReceiptSha256,
      row_id: row.id, task_sha256: sha(canonical(row.task)), invocation_id: source.sourceId, ordinal: entry.ordinal,
      program_ir_sha256:sha(canonical((row.task as {program_ir?:unknown}).program_ir)),
      turn_sha256: sha(canonical(turn)), prompt_sha256: sha(canonical(sourcePrompt)),
      actor: { model: binding.model, revision: binding.revision, actor_source_sha256: binding.actorSourceSha256,
        actor_source_path:binding.actorSourcePath,execution_identity_sha256: binding.executionIdentitySha256,
        execution_identity_path:binding.executionIdentityPath } };
    return send({ ...request, native_teacher_forced_turn: { schema: 'natlang.native-teacher-forced-turn/1',
      assistant: target.assistant, native_text_spans:target.spans, source_receipt: receipt } });
  } };
}

/** Public factory loads and verifies the source row and self-hosted actor receipts before returning the driver. */
export async function createVerifiedQwenTeacherForcedDriver(trancheDirectory:string,rowId:string,
    sourceView:SourceBinding['sourceView'],send:(request:ModelTurnRequest)=>Promise<ModelTurn>) {
  const {row,binding}=await loadVerifiedQwenSource(trancheDirectory,rowId,sourceView);
  return {row,binding,...qwenTeacherForcedDriver(row,binding,send)};
}

/** Execute the same source program with a fresh runtime ID and exact structural Qwen occurrence bindings. */
export async function executeVerifiedQwenTeacherReplay(trancheDirectory:string,rowId:string,
    sourceView:SourceBinding['sourceView'],record:ProgramRecord,options:Omit<ExecuteOptions,'runId'>,
    send:(request:ModelTurnRequest)=>Promise<ModelTurn>):Promise<ProgramRun> {
  const {row,binding,driver,runId}=await createVerifiedQwenTeacherForcedDriver(trancheDirectory,rowId,sourceView,send);
  const sourceRecord = (row.task as { program_ir?: unknown } | undefined)?.program_ir;
  if (!sourceRecord || canonical(sourceRecord) !== canonical(record))
    throw new Error('Qwen teacher source and failure case do not bind the exact same original task program');
  return executeProgram(record, driver, { ...options, runId });
}
