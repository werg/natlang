/** Fresh development curriculum from observed training failures; no old held-out inputs. */
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {spawn} from 'node:child_process';
import {followupCases} from './build-followup-cases.mjs';
import {AssignmentBudget,OperationJournal} from '../../dist/index.js';
const root=resolve(process.argv[2]),prior=resolve(process.argv[3]),priorAttempt=process.argv[4]?resolve(process.argv[4]):undefined;
await mkdir(root,{recursive:true});
const limits={modelCalls:400,caseExecutions:96,commands:priorAttempt?8:6,elapsedMs:14400000};
const allocation=new AssignmentBudget({collection:6,modelCalls:400,caseExecutions:96,trainingJobs:0,trainingUpdates:0,confirmation:0,cost:1000},{},1,new OperationJournal(join(priorAttempt??root,'assignment')));
allocation.allocateOnce('file-repair-development',{collection:6,modelCalls:400,caseExecutions:96});
const specs=[
 ['refund-files','refunds',[
  [['Please repair this; I decline the refund.','Keep the item and reverse my payment.'],[1]],
  [['Return what I paid; I do not want a repair.','The refund offer is declined. Fix the hinge.','Please mend the latch.'],[0]],
  [['Please restore the faulty switch.','I want my money back, not a replacement.'],[1]]]],
 ['urgent-files','urgent',[
  [['The urgent flag is obsolete; next month is fine.','Production is down. Act immediately.'],[1]],
  [['Customers cannot sign in. Please intervene now.','The old escalation is closed; there is no rush.','Handle this next week.'],[0]],
  [['Please act now: the service is down.','No immediate attention is needed.'],[0]]]],
 ['keep-files','keep',[
  [['Lovely packaging, terrible product. Avoid it.','Despite the setup trouble, I recommend this.'],[1]],
  [['I would gladly buy this again.','I recommend the old model. This one disappoints.','I returned this expensive paperweight.'],[0]],
  [['The screen looks lovely, but I cannot recommend buying it.','Not perfect; I am keeping it and would choose it again.'],[1]]]]
];
let progress;try{progress=JSON.parse(await readFile(join(root,'progress.json'),'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
if(!progress&&priorAttempt){
 const old=JSON.parse(await readFile(join(priorAttempt,'progress.json'),'utf8'));
 let modelCalls=0,caseExecutions=0,commands=0;
 for(const id of ['refund-files','urgent-files','keep-files'])for(const arm of ['probe','native']){const ledger=new OperationJournal(join(priorAttempt,id,arm+'-allocation')).read('ledger')?.value;if(ledger){commands++;modelCalls+=ledger.usage.modelCalls;caseExecutions+=ledger.rollouts;}}
 progress={startedAt:old.startedAt,modelCalls,caseExecutions,commands,outcomes:[],priorAttempt:{directory:priorAttempt,modelCalls,caseExecutions,commands,reason:'Fixture schema mismatch: goal refunds versus erroneous oracle refund. Original reports and provider observations preserved; one fixture-correction retry uses the original assignment and elapsed allowance.'}};
}
progress??={startedAt:Date.now(),modelCalls:0,caseExecutions:0,commands:0,outcomes:[]};
await writeFile(join(root,'progress.json'),JSON.stringify(progress,null,2));
let interrupted=false,active;process.once('SIGTERM',()=>{interrupted=true;active?.kill('SIGTERM');});
for(const [id,key,examples] of specs){
 if(interrupted)break;
 const base=followupCases().find(row=>row.id==='followup-'+id),directory=join(root,id);
 if(!base.policy.goal.includes('"'+key+'"'))throw Error('Fixture output key differs from the declared goal: '+id);
 await mkdir(directory,{recursive:true});
 const cases=examples.map(([texts,matches],index)=>{
  const names=index===1?['notes/earlier.txt','notes/later.txt','notes/extra.txt']:['notes/first.txt','notes/second.txt'];
  const folder={...Object.fromEntries(texts.map((text,i)=>[names[i],text])),'README.md':'Preserve this unrelated file.'};
  return {id:id+'-repair-'+index,group:'file-repair-new-'+id+'-'+index,split:index<2?'train':'validation',args:[],folder,expected:'done',expectedFiles:{...folder,'report.json':JSON.stringify({[key]:matches.map(i=>names[i]).sort()})}};
 });
 const protocol={...base,id:'file-repair-'+id,cases,sourceGroups:['file-repair-'+id],incidents:[{id:'observed-'+id,split:'train',source:join(prior,'followup-'+id,'native.json'),mechanisms:['missing output effects','source/runtime folder confusion','unavailable filesystem API','copying training paths']}],optimizer:{provider:'openai-codex',model:'gpt-6-luna'},executor:{endpoint:'http://127.0.0.1:8081',model:'/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'},executorTimeoutMs:300000,policy:{...base.policy,maxExperiments:2},budget:{maxModelCalls:120,maxRollouts:28,maxProposals:2,maxElapsedMs:3600000},design:'Fresh development examples only. Two varied training folders, one validation folder; no confirmation cohort and no learning-gain claim. One ordinary native optimizer, existing exact flat evaluator and file effects. Previous held-out cases are neither loaded nor reused.'};
 const path=join(directory,'protocol.json');try{if(JSON.stringify(JSON.parse(await readFile(path,'utf8')))!==JSON.stringify(protocol))throw Error('Frozen file-repair protocol changed');}catch(error){if(error.code!=='ENOENT')throw error;await writeFile(path,JSON.stringify(protocol,null,2));}
 for(const arm of ['probe','native']){
  if(interrupted)break;
  if(progress.outcomes.some(row=>row.program===id&&row.arm===arm))continue;
  if(progress.modelCalls+120>limits.modelCalls||progress.caseExecutions+28>limits.caseExecutions||progress.commands>=limits.commands||Date.now()-progress.startedAt>=limits.elapsedMs){progress.outcomes.push({program:id,arm,disposition:'assignment-exhausted'});continue;}
  progress.commands++;
  await writeFile(join(root,'progress.json'),JSON.stringify(progress,null,2));
  const status=await new Promise(resolve=>{
   const child=spawn(process.execPath,['scripts/self-improvement/structural-study.mjs',directory,arm],{cwd:new URL('../../',import.meta.url).pathname,stdio:'inherit'});active=child;let forced;
   const timer=setTimeout(()=>{child.kill('SIGTERM');forced=setTimeout(()=>child.kill('SIGKILL'),10000);},Math.min(3600000,limits.elapsedMs-(Date.now()-progress.startedAt)));
   child.once('exit',(code,signal)=>{active=undefined;clearTimeout(timer);clearTimeout(forced);resolve({code,signal});});
  });
  const ledger=new OperationJournal(join(directory,arm+'-allocation')).read('ledger')?.value;
  const calls=ledger?.usage.modelCalls??0,executions=ledger?.rollouts??0;
  progress.modelCalls+=calls;progress.caseExecutions+=executions;
  progress.outcomes.push({program:id,arm,status,modelCalls:calls,caseExecutions:executions});
  await writeFile(join(root,'progress.json'),JSON.stringify(progress,null,2));
 }
}
await writeFile(join(root,'result.json'),JSON.stringify({limits,...progress,disposition:interrupted?'interrupted':'completed',allocation:allocation.snapshot()},null,2));
