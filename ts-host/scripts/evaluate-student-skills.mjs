#!/usr/bin/env node
/** Ordinary execution ablation; no teacher hints, training publication, or reliability certificate. */
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';import {join,resolve} from 'node:path';import {pathToFileURL} from 'node:url';
const hash=b=>createHash('sha256').update(b).digest('hex');
const [input,mode,digest]=process.argv.slice(2);const planBytes=await readFile(input),plan=JSON.parse(planBytes);
if(plan.schema!=='natlang.student_skill_evaluation_plan/1')throw new Error('unsupported plan');
if(mode==='--execute'&&(plan.root_approved!==true||digest!==hash(planBytes)))throw new Error('requires exact approved plan');
for(const [path,sha] of Object.entries(plan.pins))if(hash(await readFile(path))!==sha)throw new Error('pinned input changed: '+path);
const manifest=JSON.parse(await readFile(join(plan.runtime,'frozen-runtime.json')));
for(const [path,sha] of Object.entries(manifest.files))if(hash(await readFile(join(plan.runtime,path)))!==sha)throw new Error('runtime changed: '+path);
const module=p=>import(pathToFileURL(join(plan.runtime,'dist',p)));
const [collector,policy,prompts]=await Promise.all(['teacher/collector.js','teacher/curriculum-policy.js','native/prompt.js'].map(module));
const records=(await readFile(plan.ir,'utf8')).split('\n').filter(Boolean).map(JSON.parse);
if(records.length!==plan.cases||records.some(r=>r.split!=='train'||policy.generationHoldReason(r)||policy.quarantineReason(r)||policy.retiredFamily(r)))throw new Error('development packet is not reviewed train-only');
const closure=JSON.parse(await readFile(plan.source_closure));
if(closure.status!=='passed'||closure.selected_ir_sha256!==plan.pins[plan.ir]||closure.combined_identity_overlap_count!==0||closure.selected_groups_not_train?.length!==0||closure.errors?.length!==0)throw new Error('source closure does not approve this IR');
const library={};for(const [path,key] of Object.entries(plan.library))library[key]=await readFile(path,'utf8');
if(mode!=='--execute'){console.log(JSON.stringify({status:'preflight_passed',cases:records.length,arms:3,provider_calls:0,plan_sha256:hash(planBytes)}));process.exit(0);}
const identity=await (await fetch(plan.endpoint+'/natlang/student-identity')).json();
if(identity.adapter!==plan.model||identity.checkpoint_map||identity.revision!==plan.student.revision||identity.base_model!==plan.student.base_model||Object.keys(identity.weight_pins).length!==Object.keys(plan.student.weight_pins).length||Object.entries(plan.student.weight_pins).some(([path,sha])=>identity.weight_pins[path]!==sha))throw new Error('student identity changed');
const abort=new AbortController();process.on('SIGTERM',()=>abort.abort());process.on('SIGINT',()=>abort.abort());
const out=resolve(plan.output);await mkdir(out,{recursive:true});const results=[];
for(const arm of ['baseline','discovery','instructed']){
 const system=prompts.TOOLS_PROMPT+(arm==='instructed'?'\nBefore solving, read an applicable bound skill with read_code("skills.<name>"). Use its procedure when helpful; do not read unrelated skills.':'');
 const jobs=join(out,arm,'jobs');await mkdir(jobs,{recursive:true});
 const options={endpoint:plan.endpoint,modelId:plan.model,systemPrompt:system,temperature:0,contextTokens:plan.context_tokens,maxTurns:plan.max_turns,maxModelRequests:plan.max_requests,modelConcurrency:1,collectionRole:'student_skill_development_evaluation',rootSeed:plan.root_seed,request:{max_tokens:plan.max_output_tokens},jobs,output:join(out,arm,'unused.jsonl'),workers:1,transportRetries:0};
 const runner=collector.nativeJobRunner(options);
 for(const [index,base] of records.entries()){
  abort.signal.throwIfAborted();const resultPath=join(out,arm,index+'.json');try{results.push(JSON.parse(await readFile(resultPath,'utf8')));continue;}catch(e){if(e.code!=='ENOENT')throw e;}
  const record=structuredClone(base);
  if(arm!=='baseline'){
   const root=record.semantics.root.replace(/\.nl$/,'');
   for(const [path,text] of Object.entries(library)){
    const target=root+'/'+path;if(target in record.semantics.files)throw new Error('overlay would overwrite a source file');record.semantics.files[target]=text;
   }
  }
  const provenance=collector.expectedProvenance(record,options);let row,error;
  try{row=await runner({index,record},provenance,abort.signal);}catch(e){abort.signal.throwIfAborted();error={name:e.name,message:String(e.message).slice(0,500)};}
  const calls=(row?.trajectory??[]).flatMap(t=>t.assistant?.calls??[]);
  const reads=calls.filter(c=>c.tool==='read_code'&&String(c.arguments?.name??'').startsWith('skills.')).map(c=>c.arguments.name);
  const result={schema:'natlang.student_skill_evaluation_result/1',plan_sha256:hash(planBytes),arm,program_id:base.id,source_groups:base.source_groups??[],family:base.family,original_ir_sha256:collector.recordDigest(base),effective_ir_sha256:collector.recordDigest(record),accepted:row?.outcome?.accepted===true,skill_reads:reads,model_turns:row?.trajectory?.length??null,row,error,training_publication:false};
  await writeFile(resultPath,JSON.stringify(result)+'\n',{flag:'wx'});results.push(result);
 }
}
const arms={};for(const arm of ['baseline','discovery','instructed']){const rows=results.filter(r=>r.arm===arm);arms[arm]={cases:rows.length,successes:rows.filter(r=>r.accepted).length,cases_reading_skills:rows.filter(r=>r.skill_reads.length).length,infrastructure_errors:rows.filter(r=>r.error).length};}
await writeFile(join(out,'summary.json'),JSON.stringify({schema:'natlang.student_skill_evaluation_summary/1',plan_sha256:hash(planBytes),arms,teacher_guidance:false,role:'development_train_cases_not_qualification',qualification_certified:false,automatic_training_publication:false},null,2)+'\n',{flag:'wx'});console.log(JSON.stringify(arms));
