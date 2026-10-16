#!/usr/bin/env node
/** Static exact-verifier task inputs only; no generated teacher/student traces. */
import {createHash} from 'node:crypto';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const repo=resolve(dirname(fileURLToPath(import.meta.url)),'../..');
const sha=value=>createHash('sha256').update(value).digest('hex');

export const specs=[
 {name:'integer-return',level:0,returns:'number',instructions:'Return the integer 17.',expected:17},
 {name:'string-return',level:0,returns:'string',instructions:'Return exactly the text copper meadow.',expected:'copper meadow'},
 {name:'boolean-return',level:0,returns:'boolean',instructions:'Return true.',expected:true},
 {name:'addition',level:1,returns:'number',instructions:'Add 14 and 9. Return the total.',expected:14+9},
 {name:'sort',level:1,returns:'number[]',instructions:'Sort [9, 2, 5] in increasing order. Return the sorted numbers.',expected:[9,2,5].sort((a,b)=>a-b)},
 {name:'field-extract',level:1,returns:'string',instructions:'The record is {"label":"cedar","count":8}. Return its label.',expected:'cedar'},
 {name:'invoice-total',level:2,returns:'number',instructions:'There are 3 items costing 7 each and 4 items costing 6 each. Calculate each line total and return the grand total. You may use eval.',expected:3*7+4*6},
 {name:'filter-sum',level:2,returns:'number',instructions:'For [3, 8, 11, 4], select the even numbers, square each selected number and return their sum. You may use eval.',expected:[3,8,11,4].filter(x=>x%2===0).reduce((sum,x)=>sum+x*x,0)},
 {name:'normalize-join',level:2,returns:'string',instructions:'Trim each string in ["  lake", "stone  ", " reed "] and join the results with one space. Return the text. You may use eval.',expected:['  lake','stone  ',' reed '].map(x=>x.trim()).join(' ')},
 {name:'private-writer-return',level:3,returns:'string',instructions:'Ask a private typed writer for the text violet river, then return its result.',expected:'violet river',root:'const writer: Neuralese<() => Promise<Neuralese<string>>> = nl.with<Neuralese<string>>({})`Return exactly the text violet river.`; return await writer();'},
 {name:'private-writer-reader',level:3,returns:'string',instructions:'Ask a private typed writer for the text silver maple. Give its result to a reader and return the reader\'s exact copy.',expected:'silver maple',root:'const writer: Neuralese<() => Promise<Neuralese<string>>> = nl.with<Neuralese<string>>({})`Return exactly the text silver maple.`; const reader: Neuralese<(note: Neuralese<string>) => Promise<string>> = nl.with<string>({})`Read the input note and return an exact copy of its text.`; return await reader(await writer());'},
 {name:'private-two-inputs',level:3,returns:'string',instructions:'Ask separate private typed writers for north and lamp. Give both results to a reader and return their text joined by one space.',expected:'north lamp',root:'const first: Neuralese<() => Promise<Neuralese<string>>> = nl.with<Neuralese<string>>({})`Return exactly the text north.`; const second: Neuralese<() => Promise<Neuralese<string>>> = nl.with<Neuralese<string>>({})`Return exactly the text lamp.`; const reader: Neuralese<(first: Neuralese<string>, second: Neuralese<string>) => Promise<string>> = nl.with<string>({})`Return the first input text followed by one space and the second input text.`; return await reader(await first(), await second());'},
];

export async function buildScaffolds(runtime=resolve(repo,'ts-host')) {
 const {definitionProject}=await import(pathToFileURL(resolve(runtime,'dist/teacher/program.js')).href);
 return specs.map(spec=>{
  const id='rl-execution-curriculum-20261016:'+spec.name;
  const project=definitionProject('task',{returns:spec.returns,instructions:spec.instructions});
  const root=spec.root??'return '+JSON.stringify(spec.expected)+';';
  return {version:'natlang.program/2',id,kind:'lambda_source',family:'rl-execution-'+spec.level,
   source:'natlang-authored-rl-execution-curriculum',split:'train',source_ids:[id],source_groups:[id],
   license:'project-generated',gold_sources:['independent-constructed-input-oracle'],
   semantics:{root:project.root,files:project.files,inputs:{},expected:spec.expected,operation:'exact'},
   curriculum:{version:'natlang.inline_curriculum/1',family:'rl-execution-'+spec.level,
    reference:{root:[['eval',{code:root}]]},
    rl:{level:spec.level,family:spec.name,qualification:'pending-live-48'},
    ...(spec.root?{trajectory_contract:{version:'natlang.trajectory_contract/1',authored_root_eval:true,
     authored_root_sha256:sha(root),min_child_invocations:0,min_state_edges_from_children:0,min_iteration_steps:0}}:{})},
   generation:{method:'static task construction; no sampled trajectory',model_facing_qualification:'pending-live-48',
    reward:'existing exact task outcome; no communication-edge or node-count condition'}};
 });
}

async function main(){
 const args=process.argv.slice(2);const option=name=>{const at=args.indexOf(name);return at<0?undefined:args[at+1];};
 const out=option('--out');if(!out)throw Error('--out is required');
 const basePath=option('--base-cases')??resolve(repo,'runs/decision-rich-train-20261005-v2/cases.jsonl');
 const baseRaw=await readFile(basePath,'utf8');const base=baseRaw.trim().split('\n').map(JSON.parse);
 if(base.some(row=>row.split!=='train'))throw Error('base task packet must contain train rows only');
 const scaffolds=await buildScaffolds(option('--runtime')??resolve(repo,'ts-host'));
 if(new Set([...base,...scaffolds].map(row=>row.id)).size!==base.length+scaffolds.length)throw Error('duplicate task IDs');
 await mkdir(out,{recursive:true});const outputs={};
 for(const [name,rows] of [['qualification-cases.jsonl',scaffolds],['pending-full-cases.jsonl',[...base,...scaffolds]]]){
  const bytes=rows.map(row=>JSON.stringify(row)+'\n').join('');await writeFile(resolve(out,name),bytes);
  outputs[name]={sha256:sha(bytes),bytes:Buffer.byteLength(bytes),rows:rows.length};
 }
 const manifest={schema:'natlang.rl-curriculum-prepared-tasks/1',status:'pending-model-facing-qualification',
  training_admitted:false,base:{path:resolve(basePath),sha256:sha(baseRaw),rows:base.length},outputs,
  measurement:{cases:12,samples_per_case:4,executor_samples:48,private_call_cases:3,
   private_mode:'authored root for diagnostic transport coverage; expected outcome alone determines reward',
   production_mode:'whole student root; using Neuralese is optional'},
  admission:'After live executor measurement, bind exact runtime/checkpoint/sampling and publish a qualified full packet; preserve every base train row.'};
 await writeFile(resolve(out,'prepared-manifest.json'),JSON.stringify(manifest,null,2)+'\n');console.log(JSON.stringify(manifest));
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))await main();
