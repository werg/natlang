/** Frozen, finite comparison across real semantic tasks; every failed attempt remains a result. */
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {spawn,execFileSync} from 'node:child_process';
import {followupCases} from './build-followup-cases.mjs';
import {OperationJournal} from '../../dist/index.js';
const output=resolve(process.argv[2]??'../runs/native-next-six-20261001/clean-study');
await mkdir(output,{recursive:true});
const repository=resolve('..'),oldCommit='df1f679';
const paths=execFileSync('git',['ls-tree','-r','--name-only',oldCommit,'applications/program-improver'],{cwd:repository,encoding:'utf8'}).trim().split('\n');
const separatedSource=Object.fromEntries(paths.map(path=>[path.slice('applications/program-improver/'.length),execFileSync('git',['show',oldCommit+':'+path],{cwd:repository,encoding:'utf8'})]));
const rows=followupCases(),allocation={maxModelCalls:3600,maxElapsedMs:14400000,maxAttempts:rows.length*4,perAttemptTimeoutMs:1800000};
const design={schema:'natlang.followup-study/1',allocation,programs:rows.map(row=>row.id),arms:['native','direct','separated'],reference:'Training only; independent reference source is never passed to optimizer.',confirmation:'One fresh case per program after source selection; descriptive pilot, no significant learning-gain claim.',separated:'Frozen authored application from '+oldCommit+' on the same ordinary runtime; no compatibility mode.'};
const designPath=join(output,'design.json');
try{if(JSON.stringify(JSON.parse(await readFile(designPath,'utf8')))!==JSON.stringify(design))throw Error('Study design changed');}catch(error){if(error.code!=='ENOENT')throw error;await writeFile(designPath,JSON.stringify(design,null,2)+'\n');}
let prior;try{prior=JSON.parse(await readFile(join(output,'progress.json'),'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
const started=prior?.startedAt??Date.now()-(prior?.elapsedMs??0);let calls=prior?.calls??0,attempts=prior?.attempts??0;const outcomes=prior?.outcomes??[];
const completed=new Set(outcomes.map(row=>row.program+':'+row.arm));
for(const row of rows){
 const directory=join(output,row.id);await mkdir(directory,{recursive:true});
 const protocol={...row,optimizer:{provider:'openai-codex',model:'gpt-6-luna'},executor:{endpoint:'http://127.0.0.1:8081',model:'/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'},executorTimeoutMs:300000,separatedSource,baselineEvidenceFrom:join(directory,'probe-journal'),budget:{...row.budget,maxModelCalls:100,maxElapsedMs:1800000}};
 const protocolPath=join(directory,'protocol.json');
 try{if(JSON.stringify(JSON.parse(await readFile(protocolPath,'utf8')))!==JSON.stringify(protocol))throw Error('Case protocol changed');}catch(error){if(error.code!=='ENOENT')throw error;await writeFile(protocolPath,JSON.stringify(protocol,null,2)+'\n');}
 for(const arm of ['probe','native','direct','separated']){
  if(completed.has(row.id+':'+arm))continue;
  if(calls+protocol.budget.maxModelCalls>allocation.maxModelCalls||Date.now()-started>=allocation.maxElapsedMs||attempts>=allocation.maxAttempts){outcomes.push({program:row.id,arm,disposition:'assignment-allocation-exhausted'});continue;}
  attempts++;console.log(JSON.stringify({program:row.id,arm,event:'start',calls}));
  const status=await new Promise(resolve=>{const child=spawn(process.execPath,['scripts/self-improvement/structural-study.mjs',directory,arm],{stdio:'inherit'});const timer=setTimeout(()=>child.kill('SIGTERM'),Math.min(allocation.perAttemptTimeoutMs,allocation.maxElapsedMs-(Date.now()-started)));child.on('exit',(code,signal)=>{clearTimeout(timer);resolve({code,signal});});});
  let result;try{result=JSON.parse(await readFile(join(directory,arm+'.json'),'utf8'));}catch{}
  // Result ledgers retain actual newly incurred calls; reused baseline costs remain in report metrics.
  const ledger=result?.ledger??new OperationJournal(join(directory,arm+'-allocation')).read('ledger')?.value;
  const count=Object.values(ledger?.roles??{}).reduce((sum,role)=>sum+(role.modelCalls??0),0);calls+=count;
  outcomes.push({program:row.id,arm,status,modelCalls:count,headroom:result?.headroom,disposition:result?.disposition??(result?'completed':'execution-failed'),accepted:result?.accepted});
  await writeFile(join(output,'progress.json'),JSON.stringify({design,outcomes,calls,attempts,startedAt:started,elapsedMs:Date.now()-started},null,2)+'\n');
 }
}
await writeFile(join(output,'result.json'),JSON.stringify({design,outcomes,calls,attempts,startedAt:started,elapsedMs:Date.now()-started},null,2)+'\n');
