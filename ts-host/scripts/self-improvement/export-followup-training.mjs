/**
 * Export actual, verified invocation turns of the program improver; keep terminal actions and each invocation's own contract.
 * Rows are keyed by the definition source the trace records (improver-stages.mjs): the stages of the current improver
 * (diagnose, hypothesize, editSource, editSourceStructural, the natural-language search policies) and, for runs recorded
 * earlier, the single rewriteProgram editor and the model-run step. Each row names its stage and generation, and declares
 * whole-trajectory supervision (prompts and inputs trained; tool output and mechanical feedback at the lower feedback weight).
 */
import {readFile,writeFile,mkdir,readdir} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {isDeepStrictEqual} from 'node:util';
import {recordedDriver} from './replay-followup-study.mjs';
import {runtimeModule} from './replay-runtime.mjs';
import {LEGACY,STAGES,legacyServices,definitionSourceOf,invocationFields,isTrainableDefinition,linkExperiments,measuredEditAdmission,omissionOf,stageAdmission,stageFields,stageOf,supervisionOf} from './improver-stages.mjs';
const hash=value=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
const filesOf=folder=>Object.fromEntries(folder.filePaths().map(path=>[path,new TextDecoder().decode(folder.readBytesSync(path))]));
export function observedActions(trace,turn){
 const ended=trace.events.findIndex(event=>event.kind==='model_request'&&event.phase==='end'&&event.turn===turn);
 if(ended<0)return [];
 const next=trace.events.findIndex((event,index)=>index>ended&&event.kind==='model_request'&&event.phase==='start');
 return trace.events.slice(ended+1,next<0?undefined:next).filter(event=>event.kind==='action');
}
export function successfulTurn(trace,exchange,turn){
 const calls=exchange.turn.calls??[];
 const actions=observedActions(trace,turn);
 if(!calls.length)return trace.outcome==='done';
 return calls.length===actions.length&&actions.every((action,index)=>action.name===calls[index][0]&&isDeepStrictEqual(action.arguments,calls[index][1])&&['ok','completed'].includes(action.outcome));
}
/** A completed write can still be part of a rejected source experiment. Keep it as failure context. */
export function sourceEditAdmission(parentTrace,before,after){
 if(isDeepStrictEqual(before,after))return {approved:true,reason:'Honest unchanged editor result.'};
 const state=parentTrace?.events.findLast(event=>event.kind==='state'&&event.phase==='final')?.value?.$lambda?.return;
 const experiment=state?.history?.at(-1);
 if(!experiment)throw Error('Edited source has no independently measured parent experiment.');
 return {approved:experiment.accepted===true,reason:experiment.reason};
}
/** Isolated edit targets must reproduce the candidate the parent actually measured. */
export function verifyMeasuredEditorEffects(parentTrace,before,after){
 if(isDeepStrictEqual(before,after))return;
 const parent=parentTrace?.events.findLast(event=>event.kind==='state'&&event.phase==='final')?.value?.$lambda?.return;
 const files=parent?.lastExperiment?.sourceFiles;
 if(!files||!isDeepStrictEqual(after,Object.fromEntries(files.map(file=>[file.path,file.text]))))throw Error('Editor effects differ from the measured candidate');
}
export async function verifyInvocation(program,replay,callId){
 const runImprovementFixture=(await runtimeModule(replay.runtime,'improvement/teacher.js')).run;
 const ids=new Set([callId]);
 for(let pass=0;pass<replay.traces.length;pass++)for(const trace of replay.traces)if(ids.has(trace.parentCallId))ids.add(trace.callId);
 const exchanges=[...replay.exchanges.filter(exchange=>ids.has(exchange.request.invocation_id)),...replay.targetExchanges];
 const driver=recordedDriver(exchanges,{compareSeed:false});
 const result=await runImprovementFixture(program,driver,{rootSeed:0,runId:'training-replay',systemPrompt:'',contextTokens:16384,signal:AbortSignal.timeout(60000)});
 if(!result.outcome.accepted||!isDeepStrictEqual(result.outcome.value,program.semantics.expected)||!isDeepStrictEqual(result.outcome.source,program.semantics.expected_files))throw Error('Exported invocation contract did not replay: '+program.id);
 return {id:program.id,accepted:true,requests:driver.audit().requestsReplayed,providerCalls:0};
}
/** Rows per exported definition and generation: the manifest shows which definitions the corpus holds. */
export function stageCounts(rows){
 const counts={};
 for(const row of rows){const key=row.improver_stage.definition_source;counts[key]??={generation:row.improver_stage.generation,rows:0};counts[key].rows++;}
 return counts;
}
export async function exportCase(replayPath){
 let replay=JSON.parse(await readFile(replayPath,'utf8'));
 if(!replay.runtime||!replay.verified&&!replay.source?.originalHash)return {rows:[],quarantined:replay.id};
 if(!replay.runtime)throw Error('Reexecute with a pinned current SDK before exporting training');
 const module=path=>runtimeModule(replay.runtime,path);
 const {Folder,createNatlangRuntime,OperationJournal}=await module('index.js');
 const {loadVirtualNatlang}=await module('runtime/virtual-project.js');
 const {EVALUATOR_DECLARATION,PLANS_DECLARATION}=await module('improvement/services.js');
 const directory=replay.source.directory;
 const protocol=JSON.parse(await readFile(join(directory,'protocol.json'),'utf8'));
 const original=JSON.parse(await readFile(join(directory,'native.json'),'utf8'));
 if(!replay.verified){
  // A timed-out target continuation can quarantine a loop without invalidating
  // a fully recorded editor invocation. Verify those exact contexts separately.
  let recorded;try{recorded=JSON.parse(await readFile(join(directory,'runtime','replay-runtime.json'),'utf8'));}catch(error){if(error.code==='ENOENT')return {rows:[],quarantined:replay.id};throw error;}
  if(recorded.identity!==replay.runtime.identity||hash(original)!==replay.source.originalHash)return {rows:[],quarantined:replay.id};
  const traces=(await readFile(join(directory,'native-traces.ndjson'),'utf8')).trim().split('\n').map(JSON.parse);
  const editors=new Set(traces.filter(trace=>trace.outcome==='done'&&isTrainableDefinition(definitionSourceOf(trace))).map(trace=>trace.callId));
  const exchanges=(await readFile(join(directory,'exchanges.ndjson'),'utf8')).trim().split('\n').map(JSON.parse).filter(row=>row.command==='native'&&row.role==='optimizer'&&editors.has(row.request.invocation_id));
  replay={...replay,traces,exchanges,targetExchanges:[],invocationOnly:true,audit:{fullLoopVerified:false,fullLoopError:replay.error,recordedEditorsOnly:true,providerCalls:0}};
 }
 if(original.state.quality<1&&!original.state.history.some(step=>step.accepted))return {rows:[],failureEvidence:replay.id,reason:'No supported source improvement; retain failed optimization as curriculum evidence, not positive SFT.'};
 const journal=new OperationJournal(replay.journalPath);
 const byCall=new Map(replay.traces.map(trace=>[trace.callId,trace]));
 const groups=new Map();
 for(const exchange of replay.exchanges){const id=exchange.request.invocation_id;const group=groups.get(id)??[];group.push(exchange);groups.set(id,group);}
 const rows=[],invocations=[],failureInvocations=[],omitted=[];
 const {callableMeta}=await module('runtime/callable.js');
 const authoredFiles=original.authored.files;
 const filesDigest=files=>Folder.fromFiles(files).snapshot().digest;
 // Pass 1: every recorded stage invocation, replayed and bound to its exact arguments, result and folder effects.
 const records=[];
 for(const [callId,exchanges] of groups){
  const trace=byCall.get(callId);
  if(!trace||trace.outcome!=='done')continue;
  const root=definitionSourceOf(trace);
  if(omissionOf(root)){omitted.push({id:replay.id+':'+callId.split('/').at(-1),root,reason:omissionOf(root)});continue;}
  const stage=stageOf(root);
  const record={callId,exchanges,trace,root,stage};
  if(stage.shape==='step'){
   // The retired model-run lifecycle step: its source states are the journal's committed sources.
   const initial=trace.events.find(event=>event.kind==='state'&&event.phase==='initial').value.$lambda;
   const final=trace.events.findLast(event=>event.kind==='state'&&event.phase==='final').value.$lambda;
   record.args=initial.args;record.value=final.return;
   record.before=journal.read('source:'+initial.args.state.incumbent)?.value;
   record.after=journal.read('source:'+final.return.incumbent)?.value;
   if(!record.before||!record.after)throw Error('Missing actual committed source for '+callId);
  }else{
   const fields=invocationFields(trace);
   record.args=fields.args;record.value=fields.value;
   if(stage.shape==='editor'){
    const request=fields.args.request;
    record.before=Object.fromEntries(request.sourceFiles.map(row=>[row.path,row.text]));
    const folder=Folder.fromFiles(record.before),driver=recordedDriver(exchanges,{compareSeed:false});
    const runtime=createNatlangRuntime({model:{driver,maxTurns:16,maxTokens:24000,turnTokens:2048,maxFailureRepairs:4},seed:{mode:'derived',root:0},codeEdits:'deny',network:false,...(stage.generation===LEGACY?legacyServices(authoredFiles,EVALUATOR_DECLARATION,PLANS_DECLARATION):{}),signal:AbortSignal.timeout(60000)});
    let value;try{value=await runtime.run(()=>folder.apply(loadVirtualNatlang(authoredFiles,root),request));}catch(error){await writeFile('/tmp/natlang-export-request-mismatch.json',JSON.stringify({actual:error.request,expected:error.expected},null,2));throw Error('Standalone child replay failed; request mismatch saved in /tmp/natlang-export-request-mismatch.json');}
    if(!isDeepStrictEqual(value,fields.value)||driver.audit().unconsumedRequests)throw Error('Child invocation failed independent replay: '+callId);
    record.after=filesOf(folder.snapshot());
    if(replay.invocationOnly&&stage.generation===LEGACY)verifyMeasuredEditorEffects(byCall.get(trace.parentCallId),record.before,record.after);
   }else{record.before={};record.after={};}
  }
  records.push(record);
 }
 // Pass 2: the experiments around the stages, and what the run's own measurements say about each edit.
 const stageRecords=records.filter(record=>record.stage.generation===STAGES);
 const experiments=linkExperiments(stageRecords.map(record=>({callId:record.callId,stage:record.stage,args:record.args,value:record.value})));
 const outcomes=new Map(stageRecords.filter(record=>record.stage.shape==='editor').map(record=>[record.callId,
  measuredEditAdmission(original.state.history,filesDigest(record.after),isDeepStrictEqual(record.before,record.after))]));
 const shadowDisagreements=new Set(replay.traces.flatMap(trace=>trace.events??[]).filter(event=>event.kind==='pluggable_shadow'&&event.agree===false).map(event=>event.name));
 // Pass 3: rows.
 for(const record of records){
  const {callId,exchanges,trace,root,stage}=record;
  const invocationId=replay.id+':'+callId.split('/').at(-1);
  // Test cases are sealed research confirmation material and must never enter training fixtures.
  const caseDefinition={...protocol,cases:protocol.cases.filter(row=>row.split!=='test')};
  for(const key of ['referenceFiles','reference','optimizer','executor','separatedSource','baselineEvidenceFrom'])delete caseDefinition[key];
  const populationSources={};
  for(const member of record.args.state?.population??[]){const source=journal.read('source:'+member.source)?.value;if(!source)throw Error('Missing population source');populationSources[member.source]=source;}
  const semantics={root,files:authoredFiles,inputs:record.args,folder_files:record.before,expected:record.value,expected_files:record.after,
   evaluation_fixture:{kind:'flat-program-evaluator',scope:'invocation',caseDefinition,populationSources,executorId:protocol.executor.model,executorTimeoutMs:protocol.executorTimeoutMs}};
  const program={version:'natlang.program/2',id:invocationId,kind:'lambda_source',family:'native-program-improvement',split:'train',source_groups:[...new Set(['improvement-target:'+replay.id,...(protocol.sourceGroups??[])])],source_ids:(protocol.incidents??[]).map(incident=>incident.id),source:'generated-failure-corpus',license:'project-generated',semantics};
  const verification=await verifyInvocation(program,replay,callId);
  let admission={approved:true,reason:null};
  if(stage.generation===LEGACY&&stage.shape==='editor')admission={...sourceEditAdmission(byCall.get(trace.parentCallId),record.before,record.after)};
  else if(stage.generation===STAGES)admission=stageAdmission({stage,invocation:{value:record.value},experiment:experiments.get(callId),outcomes,editApproval:outcomes.get(callId),shadowDisagreements});
  if(!admission.approved){failureInvocations.push({id:invocationId,root,stage:stage.stage,generation:stage.generation,turns:exchanges.length,reason:admission.reason,verification,source:replayPath,disposition:'context-only-'+stage.stage});continue;}
  let admitted=0;
  for(const [index,exchange] of exchanges.entries()){
   if(!successfulTurn(trace,exchange,index+1))continue;
   const target={role:'assistant',content:exchange.turn.text??'',...(exchange.turn.raw_calls?.length?{tool_calls:exchange.turn.raw_calls}:exchange.turn.calls?.length?{tool_calls:exchange.turn.calls.map(([name,args],i)=>({id:'recorded-'+i,type:'function',function:{name,arguments:JSON.stringify(args)}}))}:{})};
   rows.push({version:'natlang.teacher_training_turn.native/1',id:invocationId+':'+index,teacher_trajectory_id:replay.id,teacher_trajectory_digest:hash(replay),owner:'improver',program_id:invocationId,source_groups:program.source_groups,split:'train',family:program.family,task_family:program.family,task_kind:stage.shape==='editor'||stage.shape==='step'?'directory-reducer':'function',task_modality:stage.shape==='editor'||stage.shape==='step'?'program-editing':'program-judgment',
    ...stageFields(stage),
    task:{kind:'whole_program',program_ir:program},messages:exchange.request.messages,tools:exchange.request.tools,target,supervision:supervisionOf(exchange.request.messages),teacher_reasoning:exchange.turn.reasoning??null,teacher_reasoning_trained:exchange.turn.reasoning!==undefined,sourceIncidents:program.source_ids,source_ref:{trajectory_id:replay.id,source_row_sha256:replay.source.originalHash},source:filesDigest(record.before),
    provenance:{collection_role:'teacher',owner:'improver',runtime_api:'native-ordinary-runtime',runtime_compiler:replay.migration.currentCompiler,source_replay:replayPath,source_invocation:callId,verification_scope:replay.invocationOnly?'editor-with-measured-parent':'complete-loop'},
    training_admission:{kind:'exact-native-runtime-oracle',approved:true,reason:replay.invocationOnly?'Exact recorded inputs, result and effects independently reexecuted; the outcome is the run\'s own measurement ('+(admission.reason??'none')+'); full loop remains quarantined':'Full loop reexecuted; invocation inputs, result and folder effects verified; actual action completed'+(admission.reason?' ('+admission.reason+')':'')},trace_admission:{admitted:true},outcome:{accepted:true},evidence:{actions:observedActions(trace,index+1),replayAudit:replay.audit}});
   admitted++;
  }
  invocations.push({id:invocationId,root,stage:stage.stage,generation:stage.generation,turns:exchanges.length,admitted,folderChanged:!isDeepStrictEqual(record.before,record.after),verification});
 }
 return {rows,invocations,failureInvocations,omitted,id:replay.id,...(replay.invocationOnly?{quarantinedLoop:true,verificationScope:'independently-replayed-editors-with-measured-parent'}:{})};
}
if(process.argv[1]===new URL(import.meta.url).pathname){
 const input=resolve(process.argv[2]),output=resolve(process.argv[3]);await mkdir(output,{recursive:true});
 const rows=[],cases=[];
 for(const entry of await readdir(input,{withFileTypes:true})){
  if(!entry.isDirectory())continue;
  const path=join(input,entry.name,'replay.json');
  try{await readFile(path);}catch{continue;}
  try{const result=await exportCase(path);rows.push(...result.rows);const {rows:_,...summary}=result;cases.push(summary);}catch(error){cases.push({id:entry.name,quarantined:true,error:String(error)});}
 }
 const text=rows.map(JSON.stringify).join('\n')+'\n';await writeFile(join(output,'training-turns.jsonl'),text);
 await writeFile(join(output,'manifest.json'),JSON.stringify({schema:'natlang.native-improvement-training/1',rows:rows.length,sha256:createHash('sha256').update(text).digest('hex'),stages:stageCounts(rows),omitted:cases.flatMap(row=>row.omitted??[]),cases,providerCalls:0,source:input,contextMigration:'Provider replies preserved; each exported current-context invocation independently reexecuted against exact value and file oracles. No newly sampled teacher responses.'},null,2)+'\n');
 const {publish}=await import('./publish-optimizer-training.mjs');
 const publication=await publish(resolve(new URL('../../../',import.meta.url).pathname),[join(output,'manifest.json')]);
 console.log(JSON.stringify({rows:rows.length,cases:cases.length,publication}));
}
