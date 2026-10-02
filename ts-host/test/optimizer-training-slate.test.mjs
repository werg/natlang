import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {spawnSync} from 'node:child_process';
import {trainingSlate,SLATE_FAMILIES} from '../scripts/self-improvement/build-training-slate.mjs';
import {Folder,SourceEvaluator} from '../dist/index.js';
import {UsageGateway} from '../dist/evaluation/usage.js';

test('large optimizer slate has independent varied cases, no sealed inputs, and exact preservation',()=>{
 const rows=trainingSlate();assert.equal(rows.length,192);assert.equal(SLATE_FAMILIES.length,16);
 assert.equal(new Set(rows.map(row=>row.id)).size,192);
 assert.equal(new Set(rows.flatMap(row=>row.cases.map(example=>JSON.stringify(example.folder)))).size,960);
 for(const row of rows){
  assert.equal(row.cohort,'development');assert.equal(row.cases.filter(example=>example.split==='train').length,3);
  assert.equal(row.cases.filter(example=>example.split==='validation').length,2);
  assert.equal(row.policy.maxExperiments,3);assert.ok(row.policy.allowedFiles.includes('solve/workflow.ts'));
  for(const example of row.cases){
   assert.equal(example.expectedFiles['README.md'],example.folder['README.md']);
   assert.equal(example.expectedFiles['archive/irrelevant.txt'],example.folder['archive/irrelevant.txt']);
   assert.ok(!JSON.stringify(example.folder).includes('"yes":'));assert.ok(!JSON.stringify(example.folder).includes('"category":'));
  }
 }
 assert.deepEqual(rows.map(row=>row.id),trainingSlate().map(row=>row.id));
 assert.notDeepEqual(rows.map(row=>row.id),trainingSlate({seed:1}).map(row=>row.id));
});

test('each family executes its easy partial implementation against independent effects',async()=>{
 const rows=trainingSlate({variants:1});
 for(const row of rows){
  const driver=async request=>{
   if(String(request.messages[1].content).includes('Run workflow.run(folder)'))return {calls:[['eval',{code:'return await workflow.run(folder);',finish:true}]]};
   const text=request.messages.map(message=>String(message.content)).join('\n');
   if(row.family==='corrected-extraction'){const match=text.match(/Confirmed assignment: (\w+) at (\w+)\./);return {text:match?JSON.stringify({owner:match[1],place:match[2]}):'null'};}
   if(row.family==='semantic-routing')return {text:JSON.stringify(text.includes('invoice charge')?'billing':text.includes('malfunctioning device')?'technical':'account')};
   return {text:String(!text.includes('No affirmative decision exists'))};
  };
  const evaluator=new SourceEvaluator(row.contract,row.cases,driver,new UsageGateway({maxModelCalls:30,maxRollouts:2,maxProposals:0}),{executorId:'independent-easy-labels',timeoutMs:10000});
  let result;try{result=await evaluator.evaluate(Folder.fromFiles(row.files).snapshot(),{split:'train',caseIds:[row.cases[0].id]});}catch(error){throw Error(row.family+': '+error);}
  assert.equal(result.quality,1,row.family+': '+JSON.stringify(result.outcomes));
 }
});

test('generation records failed tasks, continues the slate, and emits a consumable empty corpus',async()=>{
 const root=await mkdtemp(join(tmpdir(),'natlang-optimizer-slate-'));
 try{
  const cases=join(root,'cases.jsonl');await writeFile(cases,trainingSlate({variants:1}).slice(0,2).map(JSON.stringify).join('\n')+'\n');
  const output=join(root,'generation');
  const result=spawnSync(process.execPath,[new URL('../scripts/self-improvement/run-training-slate.mjs',import.meta.url).pathname,'--repo',root,'--cases',cases,'--output',output,'--optimizer-provider','deliberately-missing-provider'],{encoding:'utf8',timeout:60000});
  assert.equal(result.status,0,result.stderr+result.stdout);
  const report=JSON.parse(await readFile(join(output,'manifest.json'),'utf8'));
  assert.equal(report.disposition,'completed');assert.equal(report.outcomes.length,2);
  assert.ok(report.outcomes.every(row=>row.admission.rows===0&&row.admission.disposition==='failure-evidence-retained'));
  assert.equal(await readFile(join(output,'verified-turns.jsonl'),'utf8'),'');
 }finally{await rm(root,{recursive:true,force:true});}
});
