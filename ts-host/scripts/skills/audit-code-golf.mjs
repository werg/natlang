#!/usr/bin/env node
/** Execute every reference against host-held gold in the pinned CPU sandbox. */
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {Worker,isMainThread,parentPort,workerData} from 'node:worker_threads';
import {canonical} from '../../dist/adaptation/identity.js';
import {runPythonSolveBatch} from '../../dist/skills/code-objective.js';
import {validateEpisode} from '../../dist/skills/episode.js';
import {PYTHON_SANDBOX_IMAGE} from '../../dist/skills/graded.js';
if(!isMainThread){
 const reports=workerData.map(({episode,row})=>{
  const task=row.expected,source=task.sizeObjective.referenceSource;
  const visible=JSON.parse(row.args[0]);
  if(visible.reference_source!==source)throw Error('public/host reference mismatch');
  const result=runPythonSolveBatch({source,inputs:task.cases.map(c=>c.input)});
  const correct=result.kind==='results'&&result.results.length===task.cases.length&&result.results.every((r,i)=>r.kind==='ok'&&canonical(r.value)===canonical(task.cases[i].expected));
  return {episode,task:row.id,checks:task.cases.length,correct,task_sha256:createHash('sha256').update(canonical(task)).digest('hex'),...(correct?{}:{result})};
 });
 parentPort.postMessage(reports);
}else{
 const [input,out]=process.argv.slice(2);if(!input||!out)throw Error('usage: audit-code-golf.mjs INPUT OUTPUT');
 const body=await readFile(input,'utf8'),episodes=body.trim().split('\n').map(JSON.parse);
 const jobs=episodes.flatMap(e=>{const errors=validateEpisode(e);if(errors.length)throw Error(JSON.stringify(errors));return [...e.support.cases,...e.query.cases].map(row=>({episode:e.id,row}));});
 const reports=(await Promise.all([0,1].map(lane=>new Promise((resolve,reject)=>{
  const worker=new Worker(new URL(import.meta.url),{workerData:jobs.filter((_,i)=>i%2===lane)});worker.once('message',resolve);worker.once('error',reject);worker.once('exit',code=>{if(code)reject(Error('audit worker exit '+code));});
 })))).flat().sort((a,b)=>a.task.localeCompare(b.task));
 const evaluator_sha256=createHash('sha256').update(await readFile(new URL('../../dist/skills/code-objective.js',import.meta.url))).digest('hex');
 const audit={schema:'natlang.code-golf-reference-audit/1',input_sha256:createHash('sha256').update(body).digest('hex'),evaluator_sha256,sandbox_image:PYTHON_SANDBOX_IMAGE,episodes:episodes.length,tasks:reports.length,checks:reports.reduce((n,r)=>n+r.checks,0),errors:reports.filter(r=>!r.correct),reports,provider_calls:0,publication:'Reference validation only; not model trajectories or training admission'};
 await writeFile(out,JSON.stringify(audit,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify({...audit,reports:undefined}));if(audit.errors.length)process.exitCode=1;
}
