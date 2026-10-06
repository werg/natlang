#!/usr/bin/env node
// Autonomous whole-task execution on held-out IR; no reference actions or teacher hints supplied to the student.
import {readFile,mkdir,writeFile,appendFile} from 'node:fs/promises';
import {resolve,join,dirname} from 'node:path';
import {parseArgs} from 'node:util';
import {createHash} from 'node:crypto';
import {nativeJobRunner,expectedProvenance} from '../dist/teacher/collector.js';
import {generationHoldReason,quarantineReason,retiredFamily} from '../dist/teacher/curriculum-policy.js';
import {TOOLS_PROMPT} from '../dist/native/prompt.js';
const {values}=parseArgs({options:{cases:{type:'string'},endpoint:{type:'string'},model:{type:'string',default:'natlang-neuralese'},out:{type:'string'},limit:{type:'string',default:'4'},'max-output':{type:'string',default:'1024'},'max-requests':{type:'string',default:'24'},'context-tokens':{type:'string',default:'65536'}}});
if(!values.cases||!values.endpoint||!values.out)throw Error('--cases --endpoint --out required');
const bytes=await readFile(values.cases),records=bytes.toString().trim().split('\n').map(JSON.parse);
if(records.some(r=>r.split!=='test'||generationHoldReason(r)||quarantineReason(r)||retiredFamily(r)))throw Error('requires reviewed held-out cases');
for(const flag of ['limit','max-output','max-requests','context-tokens'])if(!Number.isSafeInteger(Number(values[flag]))||Number(values[flag])<1)throw Error('positive integer bounds required');
const out=resolve(values.out);await mkdir(dirname(out),{recursive:true});await mkdir(out,{recursive:false});
const selected=records.slice(0,Number(values.limit)),abort=new AbortController();
process.on('SIGINT',()=>abort.abort());process.on('SIGTERM',()=>abort.abort());
const options={endpoint:values.endpoint,modelId:values.model,systemPrompt:TOOLS_PROMPT,temperature:0,contextTokens:Number(values['context-tokens']),maxTurns:12,maxModelRequests:Number(values['max-requests']),modelConcurrency:1,collectionRole:'heldout_task_execution',rootSeed:7203,request:{max_tokens:Number(values['max-output'])},jobs:join(out,'jobs'),workers:1,transportRetries:0};
await mkdir(options.jobs);
const pins={cases_sha256:createHash('sha256').update(bytes).digest('hex'),endpoint:values.endpoint,model:values.model,options};
await writeFile(join(out,'plan.json'),JSON.stringify(pins,null,2)+'\n',{flag:'wx'});
const results=[];const runner=nativeJobRunner(options);
for(const [index,record] of selected.entries()){
 abort.signal.throwIfAborted();let row,error;
 try{row=await runner({index,record},expectedProvenance(record,options),abort.signal);}catch(e){abort.signal.throwIfAborted();error={name:e.name,code:e.code,message:String(e.message).slice(0,800)};}
 const result={id:record.id,source_groups:record.source_groups,family:record.curriculum?.family,passed:row?.outcome?.accepted===true,turns:row?.trajectory?.length??0,row,error,training_publication:false};
 await appendFile(join(out,'results.jsonl'),JSON.stringify(result)+'\n');results.push(result);
 console.log(JSON.stringify({completed:results.length,passed:results.filter(r=>r.passed).length,error}));
}
const summary={schema:'natlang.recurrence-task-execution/1',cases:results.length,passed:results.filter(r=>r.passed).length,resource_limited:results.filter(r=>r.error?.code==='NATLANG_MODEL_REQUEST_BUDGET').length,by_family:Object.fromEntries([...new Set(results.map(r=>r.family))].map(f=>[f,{cases:results.filter(r=>r.family===f).length,passed:results.filter(r=>r.family===f&&r.passed).length}])),role:'autonomous held-out full-task execution; no teacher replay',prompt_mode:'crisp runtime prompt; port payload behavior measured separately by conditional-return ablations',training_publication:false,pins};
await writeFile(join(out,'summary.json'),JSON.stringify(summary,null,2)+'\n');console.log(JSON.stringify(summary));
