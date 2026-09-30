#!/usr/bin/env node
// Discover completed native teacher jobs, retaining immutable inputs and an exclusion ledger.
// Admission here selects candidates; native materialization and final split/token/dedup audits still apply.
import {readdir, readFile, mkdir, stat, open, link, unlink} from 'node:fs/promises';
import {createReadStream, createWriteStream} from 'node:fs';
import {join, resolve} from 'node:path';
import {createHash, randomUUID} from 'node:crypto';
import {Readable, Transform} from 'node:stream';
import {pipeline} from 'node:stream/promises';
import {createGzip} from 'node:zlib';
import {parseArgs} from 'node:util';
import {admitRow} from '../dist/teacher/curriculum.js';
import {classifyAdmissionReasons} from './admission-dispositions.mjs';
const {values} = parseArgs({options:{repo:{type:'string'},out:{type:'string'},json:{type:'boolean',default:false}}});
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
const nonce=randomUUID(), tmp=join(out,nonce+'.results.jsonl.gz.tmp'), contentHash=createHash('sha256'), compressedHash=createHash('sha256'), programs=new Set();
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
async function* selectedLines(){
  for(const [id,c] of [...selected].sort(([a],[b])=>a.localeCompare(b))){
    const bytes=await readFile(c.item.file);
    if(createHash('sha256').update(bytes).digest('hex')!==c.item.sha256)throw Error('Completed artifact changed during snapshot: '+c.item.file);
    const line=JSON.stringify(JSON.parse(bytes))+'\n';programs.add(c.item.program_id);
    bump(summary.by_model,c.model??'unknown');bump(summary.by_family,c.family??'unknown');
    yield line;
  }
}
const contentDigest=new Transform({transform(chunk,encoding,callback){contentHash.update(chunk);callback(null,chunk);}});
const compressedDigest=new Transform({transform(chunk,encoding,callback){compressedHash.update(chunk);callback(null,chunk);}});
await pipeline(Readable.from(selectedLines()),contentDigest,createGzip(),compressedDigest,createWriteStream(tmp,{flags:'wx'}));
const completed=await open(tmp,'r+');
try { await completed.sync(); } finally { await completed.close(); }
const contentSha256=contentHash.digest('hex'), digest=compressedHash.digest('hex'), path=join(out,digest+'.results.jsonl.gz');
if(JSON.stringify(admissionPolicy)!==JSON.stringify(await policyIdentity()))throw Error('Admission policy changed during inventory; rerun snapshot.');
// Publish without replacing any existing immutable snapshot.
try { await link(tmp,path); }
catch(error) {
  if(error.code!=='EEXIST') throw error;
  const existingHash=createHash('sha256');
  for await(const chunk of createReadStream(path)) existingHash.update(chunk);
  if(existingHash.digest('hex')!==digest) throw Error('Existing snapshot path has unexpected bytes: '+path);
}
await unlink(tmp);
summary.selected_trajectories=selected.size;summary.unique_programs=programs.size;
const ledgerPath=join(out,nonce+'.ledger.jsonl');
summary.repeated_eligible_trajectory_ids=summary.eligible_files-selected.size;
const manifest={version:'natlang.generated_training_snapshot/2',time:new Date().toISOString(),selection:'Explicit teacher role, train split, current IR and admission; latest eligible artifact per trajectory ID',admission_policy_identity:admissionPolicy,final_training_audited:false,results:{path,sha256:digest,content_sha256:contentSha256,encoding:'gzip'},ledger:ledgerPath,summary};
async function writeDurableExclusive(path,content){
  const staging=`${path}.${randomUUID()}.tmp`,handle=await open(staging,'wx');
  try { await handle.writeFile(content); await handle.sync(); } finally { await handle.close(); }
  try { await link(staging,path); } finally { await unlink(staging); }
}
await writeDurableExclusive(ledgerPath,ledger.map(x=>JSON.stringify(x)+'\n').join(''));
if(JSON.stringify(admissionPolicy)!==JSON.stringify(await policyIdentity()))throw Error('Admission policy changed while failure inventory was assembled; rerun snapshot.');
const failureCandidates=ledger.filter(item=>item.dispositions.some(d=>d.category==='candidate_failure')&&item.sha256&&item.file)
  .map(item=>({item,candidate_id:createHash('sha256').update(`${item.file}\0${item.sha256}`).digest('hex')}))
  .sort((a,b)=>a.candidate_id.localeCompare(b.candidate_id));
async function* failureLines(){
  for(const {item,candidate_id} of failureCandidates){
    const bytes=await readFile(item.file);
    if(createHash('sha256').update(bytes).digest('hex')!==item.sha256)throw Error('Failure candidate changed during inventory: '+item.file);
    let result;try{result=JSON.parse(bytes);}catch{throw Error('Failure candidate became unreadable: '+item.file);}
    const program=result.task?.program_ir;
    yield JSON.stringify({version:'natlang.teacher_failure_candidate/1',candidate_id,
      source_artifact:{path:item.file,sha256:item.sha256},trajectory_id:result.id??null,
      program_id:program?.id??item.program_id??null,task:result.task??null,
      source_groups:program?.source_groups??null,provenance:result.provenance??null,
      outcome:result.outcome??null,admission_reasons:item.reasons,dispositions:item.dispositions,
      dpo_negative_eligible:false,raw_result:result})+'\n';
  }
}
const failureContentHash=createHash('sha256');
const failureContentDigest=new Transform({transform(chunk,encoding,callback){failureContentHash.update(chunk);callback(null,chunk);}});
const failureTmp=join(out,nonce+'.failure-candidates.jsonl.gz.tmp'),failureCompressed=createHash('sha256');
const failureCompressedDigest=new Transform({transform(chunk,encoding,callback){failureCompressed.update(chunk);callback(null,chunk);}});
await pipeline(Readable.from(failureLines()),failureContentDigest,createGzip(),failureCompressedDigest,createWriteStream(failureTmp,{flags:'wx'}));
if(JSON.stringify(admissionPolicy)!==JSON.stringify(await policyIdentity()))throw Error('Admission policy changed during failure inventory; rerun snapshot.');
const failureContentSha256=failureContentHash.digest('hex');
const failureDigest=failureCompressed.digest('hex'),failurePath=join(out,failureDigest+'.failure-candidates.jsonl.gz');
const failureCompleted=await open(failureTmp,'r+');try{await failureCompleted.sync();}finally{await failureCompleted.close();}
try{await link(failureTmp,failurePath);}catch(error){
  if(error.code!=='EEXIST')throw error;
  const check=createHash('sha256');for await(const chunk of createReadStream(failurePath))check.update(chunk);
  if(check.digest('hex')!==failureDigest)throw Error('Existing failure candidate path has unexpected bytes: '+failurePath);
}
await unlink(failureTmp);
const ledgerSha256=createHash('sha256').update(ledger.map(x=>JSON.stringify(x)+'\n').join('')).digest('hex');
const failureManifestPath=join(out,nonce+'.failure-candidates.manifest.json');
const failureManifest={version:'natlang.teacher_failure_inventory/1',created_at:new Date().toISOString(),
  candidates:failureCandidates.length,artifact:{path:failurePath,sha256:failureDigest,content_sha256:failureContentSha256,encoding:'gzip'},
  positive_snapshot:{path,sha256:digest,content_sha256:contentSha256},source_ledger:{path:ledgerPath,sha256:ledgerSha256},
  policy_identity:admissionPolicy,negative_labels_assigned:false,automatic_pair_generation:false};
await writeDurableExclusive(failureManifestPath,JSON.stringify(failureManifest,null,2)+'\n');
manifest.failure_candidates={path:failurePath,sha256:failureDigest,manifest:failureManifestPath,candidates:failureCandidates.length};
await writeDurableExclusive(join(out,nonce+'.manifest.json'),JSON.stringify(manifest,null,2)+'\n');
console.log(values.json?JSON.stringify({results:path,manifest:join(out,nonce+'.manifest.json'),failure_candidates:failurePath,failure_manifest:failureManifestPath}):path);
