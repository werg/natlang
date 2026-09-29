#!/usr/bin/env node
// Discover completed native teacher jobs, retaining immutable inputs and an exclusion ledger.
// Admission here selects candidates; native materialization and final split/token/dedup audits still apply.
import {readdir, readFile, mkdir, stat, open, rename, writeFile} from 'node:fs/promises';
import {join, resolve} from 'node:path';
import {createHash, randomUUID} from 'node:crypto';
import {parseArgs} from 'node:util';
import {admitRow} from '../dist/teacher/curriculum.js';
import {classifyAdmissionReasons} from './admission-dispositions.mjs';
const {values} = parseArgs({options:{repo:{type:'string'},out:{type:'string'}}});
const repo = resolve(values.repo ?? process.cwd());
const out = resolve(values.out ?? join(repo,'data/teacher/generated-snapshots'));
await mkdir(out,{recursive:true});
const policyIdentity=async()=>Object.fromEntries(await Promise.all(['../dist/teacher/curriculum.js',
  '../dist/teacher/curriculum-policy.js','../dist/teacher/source-conversion.js','../dist/teacher/source-review.js',
  './admission-dispositions.mjs'].map(async name=>[name,createHash('sha256').update(await readFile(new URL(name,import.meta.url))).digest('hex')])));
const admissionPolicy=await policyIdentity();
async function* files(dir) {
  for (const e of (await readdir(dir,{withFileTypes:true})).sort((a,b)=>a.name.localeCompare(b.name))) {
    if (e.isDirectory() && !/^(runtime|node_modules|\.git|generated-snapshots)/.test(e.name)) yield* files(join(dir,e.name));
    else if (e.isFile() && e.name.endsWith('.result.json')) yield join(dir,e.name);
  }
}
const ledger=[], selected=new Map(), summary={completed_files:0,eligible_files:0,selected_trajectories:0,unique_programs:0,by_model:{},by_family:{},excluded:{}};
const bump=(obj,key)=>obj[key]=(obj[key]??0)+1;
for (const base of [join(repo,'runs'),join(repo,'data/teacher')]) for await (const file of files(base)) {
  summary.completed_files++;
  let row,bytes,reasons=[];
  try {bytes=await readFile(file);row=JSON.parse(bytes);}
  catch(e){reasons=['unreadable_completed_result'];}
  const p=row?.task?.program_ir;
  if(row){
    if(!p?.curriculum) reasons.push('requires_non_curriculum_adapter');
    if(p?.version!=='natlang.program/2') reasons.push('obsolete_program_ir');
    if(p?.split!=='train') reasons.push('not_training_split');
    if(row.provenance?.collection_role!=='teacher') reasons.push('not_explicit_teacher_collection');
    if(!row.id) reasons.push('missing_trajectory_id');
    if(!reasons.length){try{const a=admitRow(row);if(!a.admitted)reasons.push(...a.reasons);}catch(e){reasons.push('admission_error:'+String(e));}}
  }
  const item={file,id:row?.id,program_id:p?.id,sha256:bytes&&createHash('sha256').update(bytes).digest('hex'),reasons};
  ledger.push(item);
  if(reasons.length){for(const reason of reasons)bump(summary.excluded,reason.split(':')[0]);continue;}
  summary.eligible_files++;
  const mtime=(await stat(file)).mtimeMs, old=selected.get(row.id);
  // Saved range exports and repair directories may copy one trajectory ID. Keep the latest eligible artifact.
  if(!old || mtime>old.mtime || (mtime===old.mtime&&file>old.item.file)){
    if(old)old.item.reasons=['superseded_eligible_trajectory_id'];
    selected.set(row.id,{item,mtime,model:row.provenance.model,family:p.curriculum.family});
  }else item.reasons=['superseded_eligible_trajectory_id'];
}
const nonce=randomUUID(), tmp=join(out,nonce+'.results.tmp'), stream=await open(tmp,'wx'), hash=createHash('sha256'), programs=new Set();
for(const item of ledger)if(item.reasons.includes('superseded_eligible_trajectory_id')){
  const kept=selected.get(item.id).item;
  item.replacement={file:kept.file,sha256:kept.sha256};
  item.duplicate_bytes=item.sha256===kept.sha256;
}
summary.exclusion_categories={};
for(const item of ledger){
  item.dispositions=classifyAdmissionReasons(item.reasons);
  for(const category of new Set(item.dispositions.map(d=>d.category)))bump(summary.exclusion_categories,category);
  item.dpo_negative_eligible=false;
  if(item.dispositions.some(d=>d.category==='candidate_failure'))
    item.dpo_next_action='Requires an approved replacement at the same prompt and current-runtime in-place failure proof; this hold is not itself a negative label.';
}
try{
  for(const [id,c] of [...selected].sort(([a],[b])=>a.localeCompare(b))){
    const bytes=await readFile(c.item.file);
    if(createHash('sha256').update(bytes).digest('hex')!==c.item.sha256)throw Error('Completed artifact changed during snapshot: '+c.item.file);
    const line=JSON.stringify(JSON.parse(bytes))+'\n';await stream.writeFile(line);hash.update(line);programs.add(c.item.program_id);
    bump(summary.by_model,c.model??'unknown');bump(summary.by_family,c.family??'unknown');
  }
  await stream.sync();
}finally{await stream.close();}
const digest=hash.digest('hex'), path=join(out,digest+'.results.jsonl');
if(JSON.stringify(admissionPolicy)!==JSON.stringify(await policyIdentity()))throw Error('Admission policy changed during inventory; rerun snapshot.');
// Same digest means same immutable bytes. A repeated inventory can safely reuse them.
await rename(tmp,path);
summary.selected_trajectories=selected.size;summary.unique_programs=programs.size;
const ledgerPath=join(out,nonce+'.ledger.jsonl');
summary.repeated_eligible_trajectory_ids=summary.eligible_files-selected.size;
const manifest={version:'natlang.generated_training_snapshot/1',time:new Date().toISOString(),selection:'Explicit teacher role, train split, current IR and admission; latest eligible artifact per trajectory ID',admission_policy_identity:admissionPolicy,final_training_audited:false,results:{path,sha256:digest},ledger:ledgerPath,summary};
await writeFile(ledgerPath,ledger.map(x=>JSON.stringify(x)+'\n').join(''),{flag:'wx'});
await writeFile(join(out,nonce+'.manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});
console.log(path);
