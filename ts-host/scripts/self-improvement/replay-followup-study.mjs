/** Reexecute real optimizer AND student actions with exact requests and independent original oracles. */
import {mkdir,mkdtemp,readFile,writeFile,readdir} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {createHash} from 'node:crypto';
import {isDeepStrictEqual} from 'node:util';
import {pinReplayRuntime,runtimeModule,migrateSystemPrompts,migrateServiceOpenings,migrateCaseContexts} from './replay-runtime.mjs';
import {pathToFileURL} from 'node:url';
export const canonicalRequest=request=>JSON.stringify({executionContext:request.invocation_id?.startsWith('case:')?request.invocation_id.split('/')[0]:undefined,messages:request.messages.map(message=>({...message,...(typeof message.content==='string'?{content:message.content.replace(/evidence: "[a-f0-9]{1,63}" (?=<<cut off:)/g,'evidence: "<clipped-evidence-ref>" ')}:{})})),tools:request.tools,seed:request.seed,max_tokens:request.max_tokens}).replace(/task-\d+-[a-z0-9]+/g,'task-ID');
export function recordedDriver(exchanges,{translate=value=>value,compareSeed=true,compareExecutionContext=true}={}){
 const key=request=>canonicalRequest({...translate(request),...(compareSeed?{}:{seed:undefined}),...(compareExecutionContext?{}:{invocation_id:undefined})});
 const order=new Map(exchanges.map((exchange,index)=>[exchange,index])),finished=new Set(),waiters=[];
 const groups=new Map();for(const exchange of exchanges){const id=exchange.request.invocation_id;const group=groups.get(id)??[];group.push(exchange);groups.set(id,group);}
 const remaining=[...groups.values()],bound=new Map();let actions=0;const mismatches=[];const replayed=[];let seedChanges=0;
 const driver=async request=>{
  let group=bound.get(request.invocation_id);
  if(!group){const index=remaining.findIndex(group=>key(group[0].request)===key(request));if(index<0){const error=new Error('No recorded invocation matches the actual request');error.request=request;error.expected=remaining.map(group=>group[0].request);mismatches.push({actual:request,expected:error.expected});throw error;}group=remaining.splice(index,1)[0].slice();bound.set(request.invocation_id,group);}
  const exchange=group.shift();if(!exchange||key(exchange.request)!==key(request)){const error=new Error('Recorded continuation differs from actual runtime request');error.request=request;error.expected=exchange?.request;mismatches.push({actual:request,expected:error.expected});throw error;}
  const rank=order.get(exchange),task=exchange.request.invocation_id.split('/')[0];
  const ready=()=>exchanges.slice(0,rank).every((earlier,index)=>earlier.request.invocation_id.split('/')[0]!==task||finished.has(index));
  if(!ready())await new Promise(resolve=>waiters.push({ready,resolve}));
  setTimeout(()=>{finished.add(rank);for(let index=waiters.length-1;index>=0;index--)if(waiters[index].ready()){waiters[index].resolve();waiters.splice(index,1);}},0);
  actions++;if(exchange.request.seed!==request.seed)seedChanges++;const turn=structuredClone(translate(exchange.turn));replayed.push({request:structuredClone(request),turn});return turn;
 };
 driver.mismatches=mismatches;driver.exchanges=replayed;
 driver.audit=()=>({providerCalls:0,requestsReplayed:actions,seedChanges,unconsumedRequests:remaining.reduce((n,g)=>n+g.length,0)+[...bound.values()].reduce((n,g)=>n+g.length,0)});return driver;
}
export async function replayCase(directory,output,pinned){
 const runtime=pinned??await pinReplayRuntime(output);
 const {Folder,improveProgram,SourceEvaluator,OperationJournal}=await runtimeModule(runtime,'index.js');
 const {UsageGateway}=await runtimeModule(runtime,'evaluation/usage.js');
 const {fingerprint}=await runtimeModule(runtime,'adaptation/identity.js');
 const {TOOLS_PROMPT,DIRECTORY_REDUCER_PROMPT}=await runtimeModule(runtime,'native/prompt.js');
 const previous=await import(pathToFileURL(resolve(directory,'runtime/native/prompt.js')).href);
 const migrate=records=>migrateSystemPrompts(records,previous.TOOLS_PROMPT,TOOLS_PROMPT,previous.DIRECTORY_REDUCER_PROMPT,DIRECTORY_REDUCER_PROMPT);
 const protocol=JSON.parse(await readFile(join(directory,'protocol.json'),'utf8'));
 const original=JSON.parse(await readFile(join(directory,'native.json'),'utf8'));
 const cases=protocol.cases.filter(row=>row.split!=='test'); // Sealed confirmation cases never enter training replay.
 const exchanges=(await readFile(join(directory,'exchanges.ndjson'),'utf8')).trim().split('\n').map(JSON.parse).filter(row=>row.command==='native');
 const migrated=migrate(exchanges);
 const {declarationNamespace}=await runtimeModule(runtime,'native/external.js');
 const ts=(await import('typescript')).default;
 const serviceReplacements=new Map();
 for(const row of cases)for(const [name,source]of Object.entries(row.services??{}))serviceReplacements.set(declarationNamespace(name,source),declarationNamespace(name,ts.transpileDeclaration(source,{fileName:name+'.ts',compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText));
 const publicServices=migrateServiceOpenings(migrated.records,serviceReplacements);
 let optimizerExchanges=publicServices.records.filter(row=>row.role==='optimizer');
 let studentExchanges=publicServices.records.filter(row=>row.role==='student');
 if(!original.baseline||!original.authored){
  await mkdir(output,{recursive:true});
  const artifact={schema:'natlang.current-improvement-replay/1',id:protocol.id,verified:false,quarantine:'Interrupted run has no complete baseline or authored manifest; preserve as failure evidence.',runtime,source:{directory}};
  await writeFile(join(output,'replay.json'),JSON.stringify(artifact,null,2));return artifact;
 }
 const traces=[];await mkdir(output,{recursive:true});
 const journalPath=await mkdtemp(join(output,'journal-'));
 const journal=new OperationJournal(journalPath);
 const evaluator=new SourceEvaluator(protocol.contract,cases,()=>{throw Error('No inference in contract check');},new UsageGateway(protocol.budget),{executorId:protocol.executor.model,timeoutMs:protocol.executorTimeoutMs,sourcePolicy:{baseline:protocol.files,mode:protocol.policy.mode,allowedFiles:protocol.policy.allowedFiles}});
 const originalTraces=(await readFile(join(directory,'native-traces.ndjson'),'utf8')).trim().split('\n').map(JSON.parse);
 const sources=new Map([protocol.files,original.sourceManifest.files].map(files=>[Folder.fromFiles(files).snapshot().digest,files]));
 const digests=new Set(sources.keys());
 function snapshots(value){if(!value||typeof value!=='object')return;if(Array.isArray(value.sourceFiles)){const files=Object.fromEntries(value.sourceFiles.map(row=>[row.path,row.text])),digest=Folder.fromFiles(files).snapshot().digest;digests.add(digest);sources.set(digest,files);}for(const v of Object.values(value))if(v&&typeof v==='object')snapshots(v);}
 snapshots(original.state);for(const trace of originalTraces)snapshots(trace.events.find(event=>event.kind==='state'&&event.phase==='initial')?.value?.$lambda?.args);
 const replacements=new Map();for(const source of digests)for(const split of ['train','validation']){const ids=protocol.cases.filter(row=>row.split===split).map(row=>row.id);const before=fingerprint({source,suite:original.baseline.suiteVersion,split,ids,seed:0}),after=fingerprint({source,suite:evaluator.suiteVersion,split,ids,seed:0});replacements.set(before,after);}
 replacements.set(original.baseline.suiteVersion,evaluator.suiteVersion);
 const translate=value=>{
  if(typeof value==='string'){
   let text=[...replacements].reduce((text,[before,after])=>text.replaceAll(before,after),value);
   text=text.replace(/evidence: "([a-f0-9]{1,63})" (?=<<cut off:)/g,(match,prefix)=>{const entries=[...replacements].filter(([before])=>before.startsWith(prefix));if(entries.length!==1)return match;return match.replace(prefix,entries[0][1].slice(0,prefix.length));});
   return text;
  }
  if(Array.isArray(value))return value.map(translate);
  if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).map(([key,item])=>[key,translate(item)]));
  return value;
 };
 const oldCompiler=await readFile(join(directory,'runtime/compiler/intrinsics.js'),'utf8');
 const compileVersion=Number(oldCompiler.match(/NATLANG_COMPILE_VERSION\s*=\s*(\d+)/)?.[1]);
 if(![4,5,6].includes(compileVersion))throw Error('Unrecognized recording compiler; recollect');
 // Fresh runtime recomputes baseline observations; only the actual recorded actions are reused.
 const contextOptions={cases,sources,entry:protocol.contract.entry,fingerprint,declarationNamespace,seed:protocol.policy.seed??0};
 const context=migrateCaseContexts(exchanges.filter(row=>row.role==='student'),contextOptions);
 studentExchanges=migrateServiceOpenings(migrate(context.records).records,serviceReplacements).records;
 const rawProbe=(await readFile(join(directory,'exchanges.ndjson'),'utf8')).trim().split('\n').map(JSON.parse).filter(row=>row.command==='probe'&&row.role==='student');
 const probe=migrateServiceOpenings(migrate(migrateCaseContexts(rawProbe,contextOptions).records).records,serviceReplacements).records;
 const body=protocol.files[protocol.contract.entry].split('---').slice(2).join('---').trim();
 const baselineTasks=new Set(probe.filter(row=>String(row.request.messages[1].content).includes(body)).map(row=>row.request.invocation_id.split('/')[0]));
 // Probe actions belong to this loop only when its baseline explicitly reused that journal.
 // Otherwise native recorded its own executions, which can differ despite identical inputs.
 const baselineExchanges=protocol.baselineEvidenceFrom?probe.filter(row=>baselineTasks.has(row.request.invocation_id.split('/')[0])):[];
 const hasExecutionContext=[...baselineExchanges,...studentExchanges].every(exchange=>exchange.request.invocation_id?.startsWith('case:'));
 const optimizer=recordedDriver(optimizerExchanges,{translate}),student=recordedDriver([...baselineExchanges,...studentExchanges],{translate,compareSeed:compileVersion>=6,compareExecutionContext:hasExecutionContext});
 const migration={prompt:migrated.migration,services:publicServices.migration,caseIdentity:context.migration,originalCompiler:compileVersion,currentCompiler:6,originalSuite:original.baseline.suiteVersion,currentSuite:evaluator.suiteVersion,evidenceReferencesTranslated:original.baseline.suiteVersion!==evaluator.suiteVersion,executionContextMigration:hasExecutionContext?'exact':'current observations gain exact case/source/seed identity; old provider observations remain unchanged; full request continuations must match',confirmationCasesExcluded:true,clippedEvidenceReferencesNormalized:true,studentSeedPolicy:compileVersion<6?'reexecute original actions under stable parent-qualified seeds; provider responses are recorded, not newly sampled':'exact'};

 try{
  const result=await improveProgram({folder:Folder.fromFiles(protocol.files),contract:protocol.contract,cases,policy:protocol.policy,improverSource:Folder.fromFiles(original.authored.files).snapshot(),improver:optimizer,executor:student,executorId:protocol.executor.model,executorTimeoutMs:protocol.executorTimeoutMs,budget:protocol.budget,directory:journalPath,signal:AbortSignal.timeout(60000),trace:trace=>traces.push(trace)});
  const audit={optimizer:optimizer.audit(),student:student.audit()};
  const equal={source:isDeepStrictEqual(result.sourceManifest.files,original.sourceManifest.files),state:isDeepStrictEqual(result.state,translate(original.state)),quality:result.validation?.quality===original.validation?.quality,modelCalls:result.validation?.modelCalls===original.validation?.modelCalls,disposition:result.disposition===original.disposition};
  const verified=Object.values(equal).every(Boolean)&&Object.values(audit).every(row=>row.unconsumedRequests===0)&&!result.error;
  if(optimizer.mismatches.length||student.mismatches.length)await writeFile(join(output,'request-mismatch.json'),JSON.stringify({optimizer:optimizer.mismatches,student:student.mismatches},null,2));
  const artifact={schema:'natlang.current-improvement-replay/1',id:protocol.id,runtime,verified,equal,audit,sourceManifest:result.sourceManifest,state:result.state,validation:result.validation,error:result.error,traces,exchanges:optimizer.exchanges,targetExchanges:student.exchanges,migration,journalPath,source:{directory,originalHash:createHash('sha256').update(JSON.stringify(original)).digest('hex')}};
  await writeFile(join(output,'replay.json'),JSON.stringify(artifact,null,2));return artifact;
 }catch(error){await writeFile(join(output,'error.json'),JSON.stringify({error:String(error),actual:error.request,expected:error.expected},null,2));throw error;}
}
if(process.argv[1]===new URL(import.meta.url).pathname){
 const input=resolve(process.argv[2]),output=resolve(process.argv[3]??join(input,'replay-current'));
 const results=[];await mkdir(output,{recursive:true});
 const runtime=await pinReplayRuntime(process.argv[4]?resolve(process.argv[4]):output);
 for(const entry of await readdir(input,{withFileTypes:true})){
  if(!entry.isDirectory())continue;const directory=join(input,entry.name);
  try{await readFile(join(directory,'native.json'));}catch{continue;}
  try{const r=await replayCase(directory,join(output,entry.name),runtime);results.push({id:r.id,verified:r.verified,equal:r.equal,audit:r.audit});}catch(error){results.push({id:entry.name,verified:false,error:String(error)});}
  console.log(JSON.stringify(results.at(-1)));await writeFile(join(output,'verification.json'),JSON.stringify({results,verified:results.filter(row=>row.verified).map(row=>row.id)},null,2));
 }
}
