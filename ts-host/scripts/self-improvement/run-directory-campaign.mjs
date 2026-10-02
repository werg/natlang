/** One finite complete campaign; failures do not gate later families or training export. */
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {spawn} from 'node:child_process';
import {directoryCampaignCases} from './build-directory-campaign.mjs';
import {pinReplayRuntime,runtimeModule,digest} from './replay-runtime.mjs';
const root=resolve(process.argv[2]??'../runs/native-directory-campaign-20261002');
await mkdir(root,{recursive:true});
const runtime=await pinReplayRuntime(join(root,'sdk'));
const {AUTHORED_IMPROVER}=await runtimeModule(runtime,'improvement/authored-source.js');
const {OperationJournal}=await runtimeModule(runtime,'index.js');
const optimizerSourcePath=join(root,'optimizer-source.json');
let source;try{source=JSON.parse(await readFile(optimizerSourcePath,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;source=AUTHORED_IMPROVER;await writeFile(optimizerSourcePath,JSON.stringify(source,null,2));}
const priorPath=process.argv[3]?resolve(process.argv[3]):undefined;
const limits={modelCalls:2400,caseExecutions:520,commands:priorPath?25:24,elapsedMs:21600000};
let progress;try{progress=JSON.parse(await readFile(join(root,'progress.json'),'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
if(!progress&&priorPath){const prior=JSON.parse(await readFile(join(priorPath,'result.json'),'utf8'));progress={startedAt:prior.startedAt,outcomes:[],priorAttempt:{path:priorPath,charged:prior.charged,commands:prior.outcomes.length,reason:'Helper placed outside the callable companion folder. One fixture correction; all original calls and elapsed time remain charged.'}};}
progress??={startedAt:Date.now(),outcomes:[]};
const identity={runtime:runtime.identity,optimizer:digest(JSON.stringify(source)),limits};
if(progress.identity&&JSON.stringify(progress.identity)!==JSON.stringify(identity))throw Error('Campaign identity changed; retain original assignment.');
progress.identity=identity;
let interrupted=false,active;process.once('SIGTERM',()=>{interrupted=true;active?.kill('SIGTERM');});
const definitions=directoryCampaignCases();
async function save(){await writeFile(join(root,'progress.json'),JSON.stringify(progress,null,2));}
function usage(){let modelCalls=progress.priorAttempt?.charged.modelCalls??0,caseExecutions=progress.priorAttempt?.charged.caseExecutions??0;for(const row of definitions)for(const arm of ['probe','native','direct','confirm']){const ledger=new OperationJournal(join(root,row.id,arm+'-allocation')).read('ledger')?.value;if(ledger){modelCalls+=ledger.usage.modelCalls;caseExecutions+=ledger.rollouts;}}return {modelCalls,caseExecutions};}
async function command(args,logPath,timeout){
 const {open}=await import('node:fs/promises');const log=await open(logPath,'a');
 try{return await new Promise(resolve=>{const child=spawn(process.execPath,args,{cwd:new URL('../../',import.meta.url).pathname,stdio:['ignore',log.fd,log.fd]});active=child;let killTimer;const timer=setTimeout(()=>{child.kill('SIGTERM');killTimer=setTimeout(()=>child.kill('SIGKILL'),10000);},timeout);child.once('error',error=>resolve({error:String(error)}));child.once('exit',(code,signal)=>{active=undefined;clearTimeout(timer);clearTimeout(killTimer);resolve({code,signal});});});}finally{await log.close();}
}
await save();
// The optimizer and SDK are frozen before any transfer family executes. All families use that source.
for(const row of definitions){
 if(interrupted)break;
 const directory=join(root,row.id);await mkdir(directory,{recursive:true});
 const protocol={...row,improverSource:source,optimizer:{provider:'openai-codex',model:'gpt-6-luna'},executor:{endpoint:'http://127.0.0.1:8081',model:'/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'},executorTimeoutMs:300000,design:'Integrated directory campaign: native three-experiment search versus one direct rewrite. Four development families and two transfer families. Optimizer source frozen before transfer; sealed test cases never feed edits or training. One confirmation block; results descriptive, every attempt reported.'};
 const path=join(directory,'protocol.json');try{if(JSON.stringify(JSON.parse(await readFile(path,'utf8')))!==JSON.stringify(protocol))throw Error('Frozen case changed');}catch(error){if(error.code!=='ENOENT')throw error;await writeFile(path,JSON.stringify(protocol,null,2));}
 // Use the same pinned SDK for every arm, without affecting any other running process.
 const {cp,symlink}=await import('node:fs/promises');
 try{await readFile(join(directory,'runtime','runtime-identity.json'));}catch(error){if(error.code!=='ENOENT')throw error;await cp(runtime.directory,join(directory,'runtime'),{recursive:true,dereference:false});await symlink(new URL('../../node_modules/',import.meta.url).pathname,join(directory,'runtime','node_modules')).catch(error=>{if(error.code!=='EEXIST')throw error;});await writeFile(join(directory,'runtime','runtime-identity.json'),JSON.stringify(runtime.hashes));await cp(join(root,'sdk','prelude.js'),join(directory,'prelude.js'));}
 for(const arm of ['probe','native','direct','confirm']){
  if(interrupted)break;
  const prior=progress.outcomes.find(outcome=>outcome.id===row.id&&outcome.arm===arm);if(prior)continue;
  const charged=usage(),elapsed=Date.now()-progress.startedAt;
  if(charged.modelCalls+row.budget.maxModelCalls>limits.modelCalls||charged.caseExecutions+row.budget.maxRollouts>limits.caseExecutions||progress.outcomes.length+(progress.priorAttempt?.commands??0)>=limits.commands||elapsed>=limits.elapsedMs){progress.outcomes.push({id:row.id,arm,disposition:'assignment-exhausted'});await save();continue;}
  console.log(JSON.stringify({id:row.id,cohort:row.cohort,arm,charged}));
  const status=await command(['scripts/self-improvement/structural-study.mjs',directory,arm],join(directory,arm+'.log'),Math.min(row.budget.maxElapsedMs,limits.elapsedMs-elapsed));
  progress.outcomes.push({id:row.id,cohort:row.cohort,arm,status,charged:usage()});await save();
 }
}
// Offline migration and admission continue even if some paid arms failed or exhausted their allocation.
if(!interrupted){for(const [name,args]of [['replay',['scripts/self-improvement/replay-followup-study.mjs',root,join(root,'replay-current')]],['export',['scripts/self-improvement/export-followup-training.mjs',join(root,'replay-current'),join(root,'training-current')]]]){const status=await command(args,join(root,name+'.log'),600000);progress[name]=status;await save();}}
await writeFile(join(root,'result.json'),JSON.stringify({...progress,limits,charged:usage(),disposition:interrupted?'interrupted':'completed'},null,2));
