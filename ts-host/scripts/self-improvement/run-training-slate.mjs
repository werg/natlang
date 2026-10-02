/** General generation lane: real native loops, independent replay and automatic shared publication. */
import {mkdir,readFile,writeFile,cp,symlink,open} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {spawn} from 'node:child_process';
import {pinReplayRuntime,runtimeModule,digest} from './replay-runtime.mjs';
import {replayCase} from './replay-followup-study.mjs';
import {exportCase} from './export-followup-training.mjs';
import {publish} from './publish-optimizer-training.mjs';
const flags=new Map();for(let i=2;i<process.argv.length;i+=2)flags.set(process.argv[i],process.argv[i+1]);
const repo=resolve(flags.get('--repo')??new URL('../../../',import.meta.url).pathname);
const root=resolve(flags.get('--output')??'../runs/optimizer-training-slate-v1');
const input=resolve(flags.get('--cases')??'../data/teacher/self-improvement/task-slate-v1/cases.jsonl');
await mkdir(root,{recursive:true});
const runtime=await pinReplayRuntime(join(root,'sdk'));
const {AUTHORED_IMPROVER}=await runtimeModule(runtime,'improvement/authored-source.js');
const {OperationJournal}=await runtimeModule(runtime,'index.js');
const rows=(await readFile(input,'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse);
if(!rows.length||rows.some(row=>row.cohort!=='development'||row.cases.some(example=>example.split==='test')))throw Error('Generation accepts nonempty development-only tasks');
const configuration={schema:'natlang.optimizer-generation/1',tasks:digest(JSON.stringify(rows)),runtime:runtime.identity,optimizer:{provider:flags.get('--optimizer-provider')??'openai-codex',model:flags.get('--optimizer-model')??'gpt-6-luna'},executor:{endpoint:flags.get('--executor-endpoint')??'http://127.0.0.1:8081',model:flags.get('--executor-model')??'/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'},source:digest(JSON.stringify(AUTHORED_IMPROVER)),limits:{tasks:rows.length,modelCalls:rows.length*180,caseExecutions:rows.length*32,elapsedMs:rows.length*3600000},waitFor:flags.get('--wait-for')?resolve(flags.get('--wait-for')):null};
let progress;try{progress=JSON.parse(await readFile(join(root,'progress.json'),'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
progress??={configuration,startedAt:null,outcomes:[]};if(JSON.stringify(progress.configuration)!==JSON.stringify(configuration))throw Error('Generation inputs changed; retain previous assignment');
const save=()=>writeFile(join(root,'progress.json'),JSON.stringify(progress,null,2));await save();
let interrupted=false,active;process.once('SIGTERM',()=>{interrupted=true;active?.kill('SIGTERM');});
if(configuration.waitFor){console.log(JSON.stringify({status:'queued',waitFor:configuration.waitFor}));while(!interrupted){try{await readFile(join(configuration.waitFor,'result.json'));break;}catch(error){if(error.code!=='ENOENT')throw error;}await new Promise(resolve=>setTimeout(resolve,30000));}}
if(interrupted)process.exit(1);
progress.startedAt??=Date.now();await save();
const collected=[],admissions=[]; // Published data is carried forward even after resuming this controller.
async function command(directory){
 const log=await open(join(directory,'native.log'),'a');
 try{return await new Promise(resolve=>{const child=spawn(process.execPath,['scripts/self-improvement/structural-study.mjs',directory,'native'],{cwd:new URL('../../',import.meta.url).pathname,stdio:['ignore',log.fd,log.fd]});active=child;let forced;const timer=setTimeout(()=>{child.kill('SIGTERM');forced=setTimeout(()=>child.kill('SIGKILL'),10000);},3600000);child.once('error',error=>{clearTimeout(timer);resolve({error:String(error)});});child.once('exit',(code,signal)=>{clearTimeout(timer);clearTimeout(forced);active=undefined;resolve({code,signal});});});}finally{await log.close();}
}
for(const row of rows){
 if(interrupted)break;
 if(progress.outcomes.some(outcome=>outcome.id===row.id))continue;
 if(Date.now()-progress.startedAt>=configuration.limits.elapsedMs){progress.outcomes.push({id:row.id,disposition:'assignment-exhausted'});await save();continue;}
 const directory=join(root,row.id);await mkdir(directory,{recursive:true});await symlink(join(root,'sdk','node_modules'),join(directory,'node_modules')).catch(error=>{if(error.code!=='EEXIST')throw error;});
 const {reference,...definition}=row;
 // Keep repeated collections distinct while retaining the same source groups.
 const protocol={...definition,id:row.id+':collection-'+digest(root).slice(0,12),optimizer:configuration.optimizer,executor:configuration.executor,executorTimeoutMs:900000,improverSource:AUTHORED_IMPROVER,design:'General native optimizer curriculum; three finite experiments; no held-out confirmation or weight-gain claim.'};
 const path=join(directory,'protocol.json');try{if(JSON.stringify(JSON.parse(await readFile(path,'utf8')))!==JSON.stringify(protocol))throw Error('Frozen task changed');}catch(error){if(error.code!=='ENOENT')throw error;await writeFile(path,JSON.stringify(protocol,null,2));}
 try{await readFile(join(directory,'runtime','replay-runtime.json'));}catch(error){if(error.code!=='ENOENT')throw error;await cp(runtime.directory,join(directory,'runtime'),{recursive:true});await symlink(new URL('../../node_modules/',import.meta.url).pathname,join(directory,'runtime','node_modules')).catch(error=>{if(error.code!=='EEXIST')throw error;});await cp(join(root,'sdk','prelude.js'),join(directory,'prelude.js'));await writeFile(join(directory,'runtime','runtime-identity.json'),JSON.stringify(runtime.hashes));}
 console.log(JSON.stringify({id:row.id,family:row.family,status:'generating',completed:progress.outcomes.length,total:rows.length}));
 const status=await command(directory);let admission;
 try{
  await replayCase(directory,join(root,'replay',row.id),runtime);
  const result=await exportCase(join(root,'replay',row.id,'replay.json'));collected.push(...result.rows);const {rows:admittedRows,...evidence}=result;admissions.push(evidence);await writeFile(join(directory,'admission.json'),JSON.stringify(evidence,null,2));
  const output=join(root,'training');await mkdir(output,{recursive:true});const text=collected.map(JSON.stringify).join('\n')+'\n';
  await writeFile(join(output,'training-turns.jsonl'),text);await writeFile(join(output,'manifest.json'),JSON.stringify({schema:'natlang.native-improvement-training/1',rows:collected.length,sha256:digest(text),cases:admissions,source:root,providerCalls:0},null,2));
  const publication=await publish(repo,[join(output,'manifest.json')]);
  admission={rows:result.rows.length,invocations:result.invocations?.length??0,quarantinedLoop:result.quarantinedLoop??!result.rows.length,publication};
 }catch(error){admission={rows:0,error:String(error),disposition:'failure-evidence-retained'};}
 const ledger=new OperationJournal(join(directory,'native-allocation')).read('ledger')?.value;
 progress.outcomes.push({id:row.id,family:row.family,status,admission,usage:ledger?.usage,caseExecutions:ledger?.rollouts});await save();
 console.log(JSON.stringify(progress.outcomes.at(-1)));
}
// Expose the full accumulated published lane to preparation, including decisions from earlier resumptions.
let registry={artifacts:[]};try{registry=JSON.parse(await readFile(join(repo,'data/teacher/self-improvement/current-manifest.json'),'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
const artifact=registry.artifacts.find(artifact=>artifact.lane==='native-program-improvement');
const destination=join(root,'verified-turns.jsonl');await writeFile(destination,artifact?await readFile(join(repo,artifact.path)):'');
await writeFile(join(root,'manifest.json'),JSON.stringify({...progress,disposition:interrupted?'interrupted':'completed',training:{path:destination,rows:artifact?.rows??0,sha256:artifact?.sha256??digest('')},failurePolicy:'Every failed task remains recorded; no failed action or rejected source becomes a positive target.'},null,2));
if(interrupted)process.exitCode=1;
