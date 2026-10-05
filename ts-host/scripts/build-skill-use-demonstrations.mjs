#!/usr/bin/env node
/** Fresh, checked teacher replays with skill retrieval or inventory-only controls.
 * These are teacher SFT candidates, not student on-policy samples. No provider calls.
 * Usage: PLAN [--execute EXACT_SHA]. Nothing is automatically published to training.
 */
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
const hash=b=>createHash('sha256').update(b).digest('hex');
const [path,mode,digest]=process.argv.slice(2),bytes=await readFile(path),plan=JSON.parse(bytes);
if(plan.schema!=='natlang.skill_use_demonstration_plan/1')throw Error('unsupported plan');
if(mode==='--execute'&&(!plan.root_approved||digest!==hash(bytes)))throw Error('exact reviewed plan required');
for(const [file,sha] of Object.entries(plan.pins))if(hash(await readFile(file))!==sha)throw Error('input changed: '+file);
const frozen=JSON.parse(await readFile(join(plan.runtime,'frozen-runtime.json')));
if(!plan.pins[join(plan.runtime,'frozen-runtime.json')])throw Error('runtime manifest must be pinned');
for(const [file,sha] of Object.entries(frozen.files))if(hash(await readFile(join(plan.runtime,file)))!==sha)throw Error('runtime changed: '+file);
const load=p=>import(pathToFileURL(join(plan.runtime,'dist',p)));
const [collector,policy,prompts,materializer,curriculum]=await Promise.all(['teacher/collector.js','teacher/curriculum-policy.js','native/prompt.js','teacher/native-materializer.js','teacher/curriculum.js'].map(load));
const originals=new Map((await readFile(plan.ir,'utf8')).split('\n').filter(Boolean).map(JSON.parse).map(r=>[r.id,r]));
const closure=JSON.parse(await readFile(plan.source_closure));
if(closure.status!=='passed'||closure.selected_ir_sha256!==plan.pins[plan.ir]||closure.combined_identity_overlap_count!==0||closure.selected_groups_not_train?.length||closure.errors?.length)throw Error('source closure is not valid');
const library=Object.fromEntries(await Promise.all(Object.entries(plan.library).map(async([file,key])=>[key,await readFile(file,'utf8')])));
const seen=new Set(),jobs=[];
for(const item of plan.demonstrations){
 const teacher=JSON.parse(await readFile(item.teacher));
 const record=teacher.task?.program_ir,original=originals.get(record?.id);
 if(!plan.pins[item.teacher]||!original||collector.recordDigest(record)!==collector.recordDigest(original)||record.split!=='train'||policy.generationHoldReason(record)||policy.quarantineReason(record)||policy.retiredFamily(record)||seen.has(record.id))throw Error('unapproved or duplicate teacher');
 seen.add(record.id);
 const mode=item.mode??'retrieve';
 if(!['retrieve','inventory-control'].includes(mode)||!item.selection_rationale?.trim())throw Error('demonstration mode and visible-task rationale required');
 if(mode==='retrieve'&&!library['skills/'+item.skill+'/SKILL.md'])throw Error('bound skill required');
 if(mode==='inventory-control'&&item.skill!=null)throw Error('inventory control must not prescribe a skill');
 const native=materializer.materializeNativeRows([teacher],{directAnswers:true});
 if(native.acceptedRows!==1||!native.turns.length)throw Error('teacher is not admitted');
 jobs.push({item:{...item,mode},teacher,record});
}
if(mode!=='--execute'){console.log(JSON.stringify({status:'preflight_passed',cases:jobs.length,provider_calls:0,plan_sha256:hash(bytes)}));process.exit(0);}
await mkdir(plan.output,{recursive:true});
const options={systemPrompt:prompts.TOOLS_PROMPT,contextTokens:plan.context_tokens,maxTurns:plan.max_turns,rootSeed:plan.root_seed,modelId:'static-teacher-skill-demonstration',collectionRole:'skill_use_demonstration',toolSurfaceSha256:await collector.defaultToolSurfaceHash(plan.runtime)};
const rows=[],reviews=[];
for(const [index,{item,teacher,record:original}] of jobs.entries()){
 const record=structuredClone(original),root=record.semantics.root.replace(/\.nl$/,'');
 record.id=original.id+':skill-demo:'+(item.mode==='retrieve'?item.skill:'inventory-control');
 record.source_ids=[...new Set([...(record.source_ids??[]),original.id])];
 for(const [key,value] of Object.entries(library)){const target=root+'/'+key;if(target in record.semantics.files)throw Error('overlay would overwrite source');record.semantics.files[target]=value;}
 const actions=[...(item.mode==='retrieve'?[{calls:[['read_code',{name:'skills.'+item.skill}]]}]:[]),...teacher.trajectory.map(t=>({calls:(t.assistant?.calls??[]).map(c=>[c.tool,structuredClone(c.arguments)]),text:t.assistant?.content??''}))];
 const trajectory=[];let cursor=0;
 const driver=async request=>{
  if(cursor>=actions.length)throw Error('teacher action sequence exhausted');
  const response=actions[cursor++];
  for(const [name] of response.calls)if(!request.tools.some(t=>t.name===name||t.function?.name===name))throw Error('teacher calls an unavailable tool');
  trajectory.push(collector.trajectoryTurn(request,response));return response;
 };
 let row,error,run;
 try{
  const provenance={...collector.expectedProvenance(record,options),transport:'static-replay',skill_demonstration:{version:1,original_program_ir_sha256:collector.recordDigest(original),teacher_sha256:plan.pins[item.teacher],plan_sha256:hash(bytes),mode:item.mode,skill:item.skill??null,selection_rationale:item.selection_rationale}};
  run=await collector.executeProgram(record,driver,{...options,runId:hash(record.id+':'+plan.root_seed).slice(0,32)});
  row=collector.programRow(record,options.modelId,hash(record.id+':'+plan.root_seed).slice(0,32),provenance,run,trajectory);
  const admission=materializer.materializeNativeRows([row],{directAnswers:true});
  if(cursor!==actions.length||!run.outcome.accepted||admission.acceptedRows!==1||!admission.turns.length||(record.curriculum&&!curriculum.admitRow(row).admitted))throw Error('fresh demonstration failed complete native/task admission');
  rows.push(row);
 }catch(e){error={name:e.name,message:e.message};}
 reviews.push({program_id:original.id,mode:item.mode,skill:item.skill??null,accepted:!error,error,turns:trajectory.length,training_publication:false});
 await writeFile(join(plan.output,index+'.json'),JSON.stringify({plan_sha256:hash(bytes),review:reviews.at(-1),row,trace:run?.trace})+'\n',{flag:'wx'});
}
await writeFile(join(plan.output,'candidates.jsonl'),rows.map(r=>JSON.stringify(r)+'\n').join(''),{flag:'wx'});
await writeFile(join(plan.output,'summary.json'),JSON.stringify({schema:'natlang.skill_use_demonstration_summary/1',plan_sha256:hash(bytes),attempted:jobs.length,accepted:rows.length,turns:rows.reduce((n,r)=>n+r.trajectory.length,0),reviews,method:'static-teacher-replay-with-skill-inventory',on_policy:false,skill_helpfulness_proven:false,automatic_training_publication:false},null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({attempted:jobs.length,accepted:rows.length,provider_calls:0}));
