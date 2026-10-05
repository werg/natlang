#!/usr/bin/env node
// Static controller programs, runtime scope reads and audited decision-model answers; no fabricated teacher prose.
import { readFile, mkdir, writeFile, appendFile, rename } from 'node:fs/promises';
import { resolve,join } from 'node:path';
import { parseArgs } from 'node:util';
import { createHash } from 'node:crypto';
import { cloudflareLogin,clefDecision } from './clef-client.mjs';
import { referenceRow } from './references.mjs';
import { admitRow } from '../../dist/teacher/curriculum.js';
import { materializeNativeRows } from '../../dist/teacher/native-materializer.js';
import { defaultToolSurfaceHash } from '../../dist/teacher/collector.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';
const {values}=parseArgs({options:{cases:{type:'string'},out:{type:'string'},receipts:{type:'string'},'max-tokens':{type:'string',default:'60000'},'min-probability':{type:'string',default:'0.8'}}});
if(!values.cases||!values.out)throw new Error('--cases FILE --out DIR required');
const maxTokens=Number(values['max-tokens']),minimum=Number(values['min-probability']);
if(!Number.isSafeInteger(maxTokens)||maxTokens<=0||!Number.isFinite(minimum)||minimum<0||minimum>1)throw new Error('Invalid budget or probability gate');
const output=resolve(values.out),staged=output+`.building-${process.pid}`;
await mkdir(staged,{recursive:false});
const rows=(await readFile(values.cases,'utf8')).trim().split('\n').map(JSON.parse);
let login;
const cache=new Map();
if(values.receipts)for(const line of (await readFile(values.receipts,'utf8')).trim().split('\n')){
 const receipt=JSON.parse(line);
 const digest=createHash('sha256').update(JSON.stringify({model:receipt.model,...receipt.request})).digest('hex');
 if(digest!==receipt.request_sha256)throw new Error('Cached teacher request hash mismatch');
 cache.set(digest,receipt);
}
const stats={cases:rows.length,accepted_cases:0,held_cases:0,requests:0,cached_requests:0,input_tokens:0,estimated_neurons:0,approved_decisions:0,held_decisions:0,by_family:{}};
const options={modelId:'clef-decision-distillation',rootSeed:0,systemPrompt:TOOLS_PROMPT,contextTokens:65536,maxTurns:60,toolSurfaceSha256:await defaultToolSurfaceHash(),collectionRole:'reference',authoredActionPlans:true,followCutoffPages:true,followEvalCutoffPages:true};
const failures=[];
for(const file of ['cases.jsonl','teacher.requests.jsonl','reference.results.jsonl','native.jsonl','rejections.jsonl'])await writeFile(join(staged,file),'',{flag:'wx'});
try {
 for(let index=0;index<rows.length;index++) {
  const record=rows[index],receipts=[];let held=false;
  for(const question of record.generation.decision_distillation.requests) {
   let chosen;
   for(const model of ['clef-flash','clef']) {
    // Byte bound plus overhead prevents this pilot approaching the account's free daily allocation.
    const reserve=Buffer.byteLength(JSON.stringify({model,state:question.state,questions:question.questions}))+2048;
    const requestHash=createHash('sha256').update(JSON.stringify({model,state:question.state,questions:question.questions})).digest('hex');
    const cached=cache.get(requestHash);
    if(!cached && stats.input_tokens+reserve>maxTokens)throw new Error('Pilot input-token budget exhausted; partial evidence retained unsealed');
    const result=cached?.response ?? await clefDecision(login??=cloudflareLogin(),model,question),answer=result.answers.decision;
    const allowed=Object.keys(question.questions.decision.criteria);
    if(answer?.type!=='choice'||!allowed.includes(answer.choice)||!allowed.every(key=>Number.isFinite(answer.probabilities?.[key])&&answer.probabilities[key]>=0&&answer.probabilities[key]<=1)||Math.abs(allowed.reduce((sum,key)=>sum+answer.probabilities[key],0)-1)>.01)throw new Error('Invalid cached or live typed answer');
    if(cached)stats.cached_requests++;else{stats.requests++;stats.input_tokens+=result.usage.input_tokens;
    stats.estimated_neurons+=result.usage.input_tokens*(model==='clef'?21818:8182)/1e6;}
    const probabilities=Object.values(answer.probabilities).sort((a,b)=>b-a);
    const accepted=answer.choice===question.expected && answer.probabilities[answer.choice]>=minimum && probabilities[0]-probabilities[1]>=0.2;
    const receipt={case_id:record.id,decision_id:question.id,model,request:{state:question.state,questions:question.questions},request_sha256:createHash('sha256').update(JSON.stringify({model,state:question.state,questions:question.questions})).digest('hex'),response:result,...(cached?{cached_from:resolve(values.receipts)}:{}),expected:question.expected,accepted,quality_gate:'authored-world agreement; chosen probability>=minimum; margin>=0.2'};
    await appendFile(join(staged,'teacher.requests.jsonl'),JSON.stringify(receipt)+'\n');receipts.push(receipt);
    if(accepted){chosen=answer.choice;break;}
   }
   if(chosen===undefined){held=true;break;}
  }
  if(held){stats.held_cases++;const failure={id:record.id,reason:'teacher disagreement or insufficient probability/margin',receipts};failures.push(failure);await appendFile(join(staged,'rejections.jsonl'),JSON.stringify(failure)+'\n');continue;}
  // Every decision label has an actual teacher receipt, independently checked against the world oracle.
  const traceSeed=Number(/^s(\d+)-/.exec(record.curriculum.shape)?.[1] ?? 0);
  const row=await referenceRow(record,index,{...options,rootSeed:traceSeed});
  row.provenance.decision_teacher={version:1,providers:[...new Set(receipts.filter(x=>x.accepted).map(x=>x.model))],decision_ids:receipts.filter(x=>x.accepted).map(x=>x.decision_id),receipt_file:'teacher.requests.jsonl',controller:'authored-static-program',tools:'runtime-executed',labels:'live-Workers-AI; exact-authored-world-agreement',reasoning:'authored-action-plan; not model reasoning'};
  if(!admitRow(row).admitted)throw new Error(`Runtime replay rejected ${record.id}`);
  const native=materializeNativeRows([row],{directAnswers:true});
  if(native.unlinked.length)throw new Error(`Unlinked runtime decisions ${record.id}`);
  await appendFile(join(staged,'cases.jsonl'),JSON.stringify(record)+'\n');
  await appendFile(join(staged,'reference.results.jsonl'),JSON.stringify(row)+'\n');
  for(const turn of native.turns){await appendFile(join(staged,'native.jsonl'),JSON.stringify(turn)+'\n');stats[turn.training_admission.approved?'approved_decisions':'held_decisions']++;}
  stats.accepted_cases++;stats.by_family[record.curriculum.family]=(stats.by_family[record.curriculum.family]??0)+1;
  console.log(JSON.stringify(stats));
 }
 await writeFile(join(staged,'summary.json'),JSON.stringify({schema:'natlang.clef-distillation/1',...stats,minimum_probability:minimum,source_cases:resolve(values.cases),source_cases_sha256:createHash('sha256').update(await readFile(values.cases)).digest('hex'),admission:'independently-oracle-verified native decisions; still requires source/split/graph closure',free_quota:'shared account allocation, not one allowance per model; pilot budget only, not daily quota tracker'},null,2)+'\n');
 await rename(staged,output);
} catch(error){await writeFile(join(staged,'failure.json'),JSON.stringify({error:String(error),stats},null,2)+'\n');throw error;}
