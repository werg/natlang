/** One fixed continuation finishes unopened program families; prior failures and freezes stay intact. */
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {existsSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {spawn} from 'node:child_process';
import {OperationJournal} from '../../dist/index.js';
import {followupAllocation} from './followup-allocation.mjs';
const root=resolve(process.argv[2]),study=join(root,'clean-study'),output=join(root,'completion-corrected');await mkdir(output,{recursive:true});
const programs=['followup-refund-files','followup-urgent-files','followup-keep-files'];
const limits={modelCalls:400,caseExecutions:100,commands:12,elapsedMs:10800000};
const allocation=followupAllocation(root);allocation.allocateOnce('finish-three-file-programs',{collection:12,modelCalls:limits.modelCalls,caseExecutions:limits.caseExecutions});
const design={programs,limits,prior:join(study,'result.json'),launcherCorrection:'The prior completion launcher exited before any new provider calls. Its zero-call failures remain in completion/result.json; this uses the same reserved allocation, not a replenished model budget.',policy:'Continue previously unconfirmed families only; retain original study and its exhausted/failed outcomes. Use existing frozen protocols and existing command journals; no closed confirmation is rerun.'};
const designPath=join(output,'design.json');try{if(JSON.stringify(JSON.parse(await readFile(designPath,'utf8')))!==JSON.stringify(design))throw Error('completion design changed');}catch(error){if(error.code!=='ENOENT')throw error;await writeFile(designPath,JSON.stringify(design,null,2));}
let progress;try{progress=JSON.parse(await readFile(join(output,'progress.json'),'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
progress??={startedAt:Date.now(),modelCalls:0,caseExecutions:0,commands:0,outcomes:[]};
const ledger=(directory,arm)=>existsSync(join(directory,arm+'-allocation'))?new OperationJournal(join(directory,arm+'-allocation')).read('ledger')?.value:undefined;
for(const program of programs)for(const arm of ['probe','native','direct','separated']){
 if(progress.outcomes.some(row=>row.program===program&&row.arm===arm))continue;
 const directory=join(study,program),resultPath=join(directory,arm+'.json');
 try{await readFile(resultPath);progress.outcomes.push({program,arm,disposition:'existing-result-retained',newModelCalls:0});continue;}catch(error){if(error.code!=='ENOENT')throw error;}
 const protocol=JSON.parse(await readFile(join(directory,'protocol.json'),'utf8')),before=ledger(directory,arm);
 const maxCalls=Math.max(0,protocol.budget.maxModelCalls-(before?.usage.modelCalls??0));
 const maxCases=Math.max(0,protocol.budget.maxRollouts-(before?.rollouts??0));
 if(progress.modelCalls+maxCalls>limits.modelCalls||progress.caseExecutions+maxCases>limits.caseExecutions||progress.commands>=limits.commands||Date.now()-progress.startedAt>=limits.elapsedMs){progress.outcomes.push({program,arm,disposition:'completion-allocation-exhausted'});continue;}
 progress.commands++;
 const status=await new Promise(resolve=>{
  const child=spawn(process.execPath,['scripts/self-improvement/structural-study.mjs',directory,arm],{stdio:'inherit',cwd:new URL('../../',import.meta.url).pathname});let forced;
  const timer=setTimeout(()=>{child.kill('SIGTERM');forced=setTimeout(()=>child.kill('SIGKILL'),10000);},Math.min(protocol.budget.maxElapsedMs,limits.elapsedMs-(Date.now()-progress.startedAt)));
  child.once('exit',(code,signal)=>{clearTimeout(timer);clearTimeout(forced);resolve({code,signal});});
 });
 const after=ledger(directory,arm),calls=(after?.usage.modelCalls??0)-(before?.usage.modelCalls??0),executions=(after?.rollouts??0)-(before?.rollouts??0);
 progress.modelCalls+=calls;progress.caseExecutions+=executions;
 let result;try{result=JSON.parse(await readFile(resultPath,'utf8'));}catch{}
 progress.outcomes.push({program,arm,status,newModelCalls:calls,newCaseExecutions:executions,unknownRequests:after?.unknownRequests??0,disposition:result?.disposition??(result?'completed':'execution-failed')});
 await writeFile(join(output,'progress.json'),JSON.stringify(progress,null,2));console.log(JSON.stringify(progress.outcomes.at(-1)));
}
await writeFile(join(output,'result.json'),JSON.stringify({design,...progress,assignment:allocation.snapshot()},null,2));
