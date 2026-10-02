/** Export actual, verified invocation turns; keep terminal actions and each invocation's own contract. */
import {readFile,writeFile,mkdir,readdir} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {isDeepStrictEqual} from 'node:util';
import {recordedDriver} from './replay-followup-study.mjs';
import {runtimeModule} from './replay-runtime.mjs';
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
export async function exportCase(replayPath){
 const replay=JSON.parse(await readFile(replayPath,'utf8'));
 if(!replay.verified)return {rows:[],quarantined:replay.id};
 if(!replay.runtime)throw Error('Reexecute with a pinned current SDK before exporting training');
 const module=path=>runtimeModule(replay.runtime,path);
 const {Folder,createNatlangRuntime,OperationJournal}=await module('index.js');
 const {loadVirtualNatlang}=await module('runtime/virtual-project.js');
 const {EVALUATOR_DECLARATION}=await module('improvement/services.js');
 const directory=replay.source.directory;
 const protocol=JSON.parse(await readFile(join(directory,'protocol.json'),'utf8'));
 const original=JSON.parse(await readFile(join(directory,'native.json'),'utf8'));
 if(original.state.quality<1&&!original.state.history.some(step=>step.accepted))return {rows:[],failureEvidence:replay.id,reason:'No supported source improvement; retain failed optimization as curriculum evidence, not positive SFT.'};
 const journal=new OperationJournal(replay.journalPath);
 const byCall=new Map(replay.traces.map(trace=>[trace.callId,trace]));
 const groups=new Map();
 for(const exchange of replay.exchanges){const id=exchange.request.invocation_id;const group=groups.get(id)??[];group.push(exchange);groups.set(id,group);}
 const rows=[],invocations=[];
 for(const [callId,exchanges] of groups){
  const trace=byCall.get(callId);
  if(!trace||trace.outcome!=='done')continue;
  const manifest=trace.events.find(event=>event.kind==='manifest');
  const initial=trace.events.find(event=>event.kind==='state'&&event.phase==='initial').value.$lambda;
  const final=trace.events.findLast(event=>event.kind==='state'&&event.phase==='final').value.$lambda;
  const root=manifest.definition_source;
  let before,after;
  if(root==='improveStep.nl'){
   before=journal.read('source:'+initial.args.state.incumbent)?.value;
   after=journal.read('source:'+final.return.incumbent)?.value;
   if(!before||!after)throw Error('Missing actual committed source for '+callId);
  }else if(root==='improveStep/rewriteProgram.nl'){
   before=Object.fromEntries(initial.args.request.sourceFiles.map(row=>[row.path,row.text]));
   const folder=Folder.fromFiles(before),driver=recordedDriver(exchanges,{compareSeed:false});
   const runtime=createNatlangRuntime({model:{driver,maxTurns:16,maxTokens:24000,turnTokens:2048,maxFailureRepairs:4},seed:{mode:'derived',root:0},codeEdits:'deny',network:false,services:{evaluator:{}},serviceDeclarations:{evaluator:EVALUATOR_DECLARATION},serviceScopes:{evaluator:['improveStep.nl']},signal:AbortSignal.timeout(60000)});
   let value;try{value=await runtime.run(()=>folder.apply(loadVirtualNatlang(original.authored.files,root),initial.args.request));}catch(error){await writeFile('/tmp/natlang-export-request-mismatch.json',JSON.stringify({actual:error.request,expected:error.expected},null,2));throw Error('Standalone child replay failed; request mismatch saved in /tmp/natlang-export-request-mismatch.json');}
   if(!isDeepStrictEqual(value,final.return)||driver.audit().unconsumedRequests)throw Error('Child invocation failed independent replay: '+callId);
   after=filesOf(folder.snapshot());
  }else throw Error('Unexpected optimizer invocation: '+root);
  const invocationId=replay.id+':'+callId.split('/').at(-1);
  // Test cases are sealed research confirmation material and must never enter training fixtures.
  const caseDefinition={...protocol,cases:protocol.cases.filter(row=>row.split!=='test')};
  for(const key of ['referenceFiles','reference','optimizer','executor','separatedSource','baselineEvidenceFrom'])delete caseDefinition[key];
  const populationSources={};
  for(const member of initial.args.state?.population??[]){const source=journal.read('source:'+member.source)?.value;if(!source)throw Error('Missing population source');populationSources[member.source]=source;}
  const semantics={root,files:original.authored.files,inputs:initial.args,folder_files:before,expected:final.return,expected_files:after,
   evaluation_fixture:{kind:'flat-program-evaluator',scope:'invocation',caseDefinition,populationSources,executorId:protocol.executor.model,executorTimeoutMs:protocol.executorTimeoutMs}};
  const program={version:'natlang.program/2',id:invocationId,kind:'lambda_source',family:'native-program-improvement',split:'train',source_groups:[...new Set(['improvement-target:'+replay.id,...(protocol.sourceGroups??[])])],source_ids:(protocol.incidents??[]).map(incident=>incident.id),source:'generated-failure-corpus',license:'project-generated',semantics};
  const verification=await verifyInvocation(program,replay,callId);
  let admitted=0;
  for(const [index,exchange] of exchanges.entries()){
   if(!successfulTurn(trace,exchange,index+1))continue;
   const target={role:'assistant',content:exchange.turn.text??'',...(exchange.turn.raw_calls?.length?{tool_calls:exchange.turn.raw_calls}:exchange.turn.calls?.length?{tool_calls:exchange.turn.calls.map(([name,args],i)=>({id:'recorded-'+i,type:'function',function:{name,arguments:JSON.stringify(args)}}))}:{})};
   rows.push({version:'natlang.teacher_training_turn.native/1',id:invocationId+':'+index,teacher_trajectory_id:replay.id,teacher_trajectory_digest:hash(replay),owner:'improver',program_id:invocationId,source_groups:program.source_groups,split:'train',family:program.family,task_family:program.family,task_kind:'directory-reducer',task_modality:'program-editing',
    task:{kind:'whole_program',program_ir:program},messages:exchange.request.messages,tools:exchange.request.tools,target,teacher_reasoning:exchange.turn.reasoning??null,teacher_reasoning_trained:exchange.turn.reasoning!==undefined,sourceIncidents:program.source_ids,source_ref:{trajectory_id:replay.id,source_row_sha256:replay.source.originalHash},source:Folder.fromFiles(before).snapshot().digest,
    provenance:{collection_role:'teacher',owner:'improver',runtime_api:'native-ordinary-runtime',runtime_compiler:replay.migration.currentCompiler,source_replay:replayPath,source_invocation:callId},
    training_admission:{kind:'exact-native-runtime-oracle',approved:true,reason:'Full loop reexecuted; invocation inputs, result and folder effects verified; actual action completed'},trace_admission:{admitted:true},outcome:{accepted:true},evidence:{actions:observedActions(trace,index+1),replayAudit:replay.audit}});
   admitted++;
  }
  invocations.push({id:invocationId,root,turns:exchanges.length,admitted,folderChanged:!isDeepStrictEqual(before,after),verification});
 }
 return {rows,invocations,id:replay.id};
}
if(process.argv[1]===new URL(import.meta.url).pathname){
 const input=resolve(process.argv[2]),output=resolve(process.argv[3]);await mkdir(output,{recursive:true});
 const rows=[],cases=[];
 for(const entry of await readdir(input,{withFileTypes:true})){
  if(!entry.isDirectory())continue;
  const path=join(input,entry.name,'replay.json');
  try{await readFile(path);}catch{continue;}
  const result=await exportCase(path);rows.push(...result.rows);const {rows:_,...summary}=result;cases.push(summary);
 }
 const text=rows.map(JSON.stringify).join('\n')+'\n';await writeFile(join(output,'training-turns.jsonl'),text);
 await writeFile(join(output,'manifest.json'),JSON.stringify({schema:'natlang.native-improvement-training/1',rows:rows.length,sha256:createHash('sha256').update(text).digest('hex'),cases,providerCalls:0,source:input,contextMigration:'Provider replies preserved; each exported current-context invocation independently reexecuted against exact value and file oracles. No newly sampled teacher responses.'},null,2)+'\n');
 console.log(JSON.stringify({rows:rows.length,cases:cases.length}));
}
