#!/usr/bin/env node
/** Threshold-triggered local trajectory repair with fresh replay and complete-action prefix SFT. */
import {readFile,writeFile,mkdir,open,unlink} from 'node:fs/promises';
import {join,resolve,basename} from 'node:path';import {pathToFileURL,fileURLToPath} from 'node:url';import {createHash} from 'node:crypto';
import {trajectoryNll,firstDifficultChunk,replaceChunk,candidateDecision,correctivePrefix} from './chunk-search.mjs';
import {canonical} from './projection-search.mjs';
import {postInferenceJson} from './repair-transport.mjs';
const hash=b=>createHash('sha256').update(b).digest('hex');
const [input,mode,pin]=process.argv.slice(2),bytes=await readFile(input),plan=JSON.parse(bytes),planHash=hash(bytes);
if(plan.schema!=='natlang.student_chunk_rewrite_plan/1')throw Error('unsupported plan');
if(mode==='--execute'&&(!plan.root_approved||pin!==planHash))throw Error('requires exact approved plan');
for(const [p,h] of Object.entries(plan.pins))if(hash(await readFile(p))!==h)throw Error('input changed: '+p);
for(const name of ['rewrite-student-chunks.mjs','chunk-search.mjs','projection-search.mjs','repair-transport.mjs']){
 const actual=hash(await readFile(fileURLToPath(new URL(name,import.meta.url))));
 if(!Object.entries(plan.pins).some(([p,h])=>basename(p)===name&&h===actual))throw Error('executing collector source not approved: '+name);
}
const controls=plan.controls;
for(const key of ['trajectory_nll','chunk_nll','token_nll','regression_tolerance','continuation_nll'])if(!Number.isFinite(controls[key])||controls[key]<0)throw Error('invalid threshold: '+key);
for(const key of ['max_edits','candidates','max_failures','max_requests','max_turns'])if(!Number.isSafeInteger(controls[key])||controls[key]<1||controls[key]>128)throw Error('invalid budget: '+key);
if(!Number.isSafeInteger(controls.max_output_tokens)||controls.max_output_tokens<1||controls.max_output_tokens>4096)throw Error('invalid output budget');
if(!Number.isSafeInteger(plan.root_seed))throw Error('seed required');
const frozen=JSON.parse(await readFile(join(plan.runtime,'frozen-runtime.json')));
if(!plan.pins[join(plan.runtime,'frozen-runtime.json')])throw Error('runtime seal must be pinned');
for(const [p,h] of Object.entries(frozen.files))if(hash(await readFile(join(plan.runtime,p)))!==h)throw Error('runtime changed: '+p);
const load=p=>import(pathToFileURL(join(plan.runtime,'dist',p)));
const [collector,materializer,curriculum,policy,prompts,model]=await Promise.all(['teacher/collector.js','teacher/native-materializer.js','teacher/curriculum.js','teacher/curriculum-policy.js','native/prompt.js','model/chat-completion.js'].map(load));
const ir=(await readFile(plan.ir,'utf8')).trim().split('\n').map(JSON.parse),byId=new Map(ir.map(r=>[r.id,r]));
const closure=JSON.parse(await readFile(plan.source_closure));
if(closure.status!=='passed'||closure.selected_ir_sha256!==plan.pins[plan.ir]||closure.combined_identity_overlap_count!==0||closure.selected_groups_not_train?.length||closure.errors?.length)throw Error('source closure rejected');
const references=[];
for(const artifact of plan.teacher_artifacts){
 if(plan.pins[artifact.path]!==artifact.sha256)throw Error('reference pin missing');
 const teacher=JSON.parse(await readFile(artifact.path));const record=teacher.task?.program_ir;
 if(!record||!byId.has(record.id)||collector.recordDigest(record)!==collector.recordDigest(byId.get(record.id))||record.split!=='train')throw Error('reference differs from train IR');
 if(record.semantics.world||record.semantics.services)throw Error('external state unsupported');
 if(policy.generationHoldReason(record)||policy.quarantineReason(record)||policy.retiredFamily(record))throw Error('held source');
 const admission=materializer.materializeNativeRows([teacher],{directAnswers:true});
 if(admission.acceptedRows!==1||admission.unlinked.length||(record.curriculum&&!curriculum.admitRow(teacher).admitted))throw Error('reference not admitted');
 references.push({teacher,record,artifact});
}
if(new Set(references.map(r=>r.record.id)).size!==references.length||canonical(references.map(r=>r.record.id))!==canonical(plan.selected_program_ids))throw Error('selected programs differ');
console.log(JSON.stringify({status:'preflight_passed',cases:references.length,provider_calls:0,plan_sha256:planHash}));
if(mode!=='--execute')process.exit(0);
const abort=new AbortController();for(const sig of ['SIGTERM','SIGINT'])process.once(sig,()=>abort.abort());
const post=async(endpoint,path,body)=>{
 return postInferenceJson(endpoint+path,body,{signal:abort.signal,onRetry:async event=>{
  const receipt={...event,path,time:new Date().toISOString()};
  process.stderr.write('inference_retry: '+JSON.stringify(receipt)+'\n');
  await writeFile(join(output,'transport-retries.jsonl'),JSON.stringify(receipt)+'\n',{flag:'a'});
 }});
};
const identity=await (await fetch(plan.endpoint+'/natlang/student-identity')).json();
if(identity.adapter!==plan.student.adapter||identity.base_model!==plan.student.base_model||identity.revision!==plan.student.revision||identity.checkpoint_map||canonical(identity.weight_pins)!==canonical(plan.student.weight_pins))throw Error('student identity differs');
const output=resolve(plan.output);await mkdir(output,{recursive:true});const lock=await open(join(output,'active.lock'),'wx');
const save=(path,row)=>writeFile(path,JSON.stringify(row)+'\n',{flag:'wx'});
const wire=request=>({messages:request.messages,tools:model.modelTools(request.tools)});
const fingerprint=request=>canonical(wire(request));
const observations=request=>canonical({tools:model.modelTools(request.tools),messages:request.messages.filter(m=>m.role!=='assistant').map(({role,content})=>({role,content}))});
const assistant=response=>({role:'assistant',content:response.text||null,...(response.reasoning?{reasoning_content:response.reasoning}:{}),...(response.calls?.length?{tool_calls:response.calls.map(([name,args],i)=>({id:'chunk_'+i,type:'function',function:{name,arguments:args}}))}:{})});
const score=async(request,response)=>post(plan.endpoint,'/natlang/score',{...wire(request),assistant:assistant(response),chunk_scores:true,score_temperature:1});
const provider=plan.teacher??{endpoint:plan.endpoint,model:plan.student.adapter,temperature:.9};
const proposals=async(turn,chunk,seed,dir,round)=>{
 const messages=[{role:'system',content:'Rewrite only the specified chunk of a correct expert action. Preserve task correctness and compatibility with its surrounding action. Return ONLY a JSON object with one key "replacement", whose value has the same type as the chunk. Do not execute tools, explain, or include Markdown.'},{role:'user',content:JSON.stringify({context:wire(turn.request),correct_action:assistant(turn.response),chunk:{kind:chunk.kind,value:chunk.value,argument_name:chunk.argument_name},goal:'An equivalent, clear action a smaller model can learn.'})}];
 const rows=[];
 for(let i=0;i<controls.candidates;i++){
  abort.signal.throwIfAborted();
  const generated=await post(provider.endpoint,'/v1/chat/completions',{model:provider.model,messages,temperature:provider.temperature,top_p:1,max_tokens:controls.max_output_tokens,seed:seed+i,chat_template_kwargs:{enable_thinking:false}});
  await save(join(dir,`proposal-${round}-${i}-wire.json`),generated);
  const choice=generated.choices?.[0],text=choice?.message?.content;
  if(choice?.finish_reason!=='stop'||typeof text!=='string'){rows.push({rejected:'incomplete_teacher_response',generated});continue;}
  let parsed;try{parsed=JSON.parse(text);if(Object.keys(parsed).length!==1||!Object.hasOwn(parsed,'replacement'))throw Error('invalid shape');}
  catch{rows.push({rejected:'invalid_replacement_json',generated});continue;}
  if(plan.teacher){
   // Score actual teacher JSON replacement tokens at temperature1, without hidden proposal temperature.
   const prefix=await post(plan.teacher.endpoint,'/tokenize',{model:plan.teacher.model,messages,add_generation_prompt:true,chat_template_kwargs:{enable_thinking:false}});
   const suffix=await post(plan.teacher.endpoint,'/tokenize',{model:plan.teacher.model,prompt:text,add_special_tokens:false});
   const scored=await post(plan.teacher.endpoint,'/v1/completions',{model:plan.teacher.model,prompt:[...prefix.tokens,...suffix.tokens],temperature:0,max_tokens:1,prompt_logprobs:1});
   const logs=scored.choices?.[0]?.prompt_logprobs;
   const values=suffix.tokens.map((id,j)=>logs?.[prefix.tokens.length+j]?.[String(id)]?.logprob);
   if(!values.length||values.some(x=>!Number.isFinite(x)))throw Error('teacher prompt-logprob support missing');
   const nll=-values.reduce((a,b)=>a+b,0)/values.length;
   rows.push({replacement:parsed.replacement,teacher_token_ids:suffix.tokens,teacher_token_logprobs:values,teacher_nll:nll,ranking_nll:nll,proposal_source:'teacher',proposal_score_scope:'complete JSON replacement response conditioned on rewrite request; temperature1',generated});
  }else{
   const scored=await post(plan.endpoint,'/natlang/score',{messages,tools:[],assistant:{role:'assistant',content:text},score_temperature:1});
   rows.push({replacement:parsed.replacement,ranking_nll:-scored.mean_logprob,proposal_source:'student',proposal_score_scope:'student JSON replacement response conditioned on rewrite request; temperature1',score:scored,generated});
  }
  await save(join(dir,`proposal-${round}-${i}-score.json`),rows.at(-1));
 }
 return rows;
};
const summaries=[],finalRows=[],finalTurns=[];
try{
 await save(join(output,'plan.json'),plan);
 for(const {teacher,record,artifact} of references){
  abort.signal.throwIfAborted();const dir=join(output,hash(record.id).slice(0,20));await mkdir(dir,{recursive:true});
  const modelId='verified-chunk-rewrite/'+provider.model;
  const expected=collector.expectedProvenance(record,{modelId,rootSeed:plan.root_seed,systemPrompt:prompts.TOOLS_PROMPT,temperature:0,contextTokens:plan.context_tokens,maxTurns:controls.max_turns,toolSurfaceHash:await collector.defaultToolSurfaceHash(plan.runtime)});
  expected.collection_role='student_chunk_rewrite';expected.student_chunk_rewrite={method:'verified-threshold-chunk-rewrite/1',plan_sha256:planHash,parent_trajectory_sha256:artifact.sha256,student_scoring_context:'ordinary task, no teacher hints',teacher_proposal_only:!!plan.teacher,proposal_model:provider.model,student_likelihood_model:plan.student.adapter,reference_model:teacher.model??teacher.provenance?.model??null};
  const runId=collector.programRunId(0,expected);
  const execute=async(reference,edit=null,continuation=null)=>{
   const turns=[],trajectory=[];let busy=false,error,row,admitted=false,regenerating=continuation!==null,consecutiveHard=0,fatalStop;
   const driver=async request=>{
    if(fatalStop)throw fatalStop;
    if(busy)throw Error('parallel replay unsupported');busy=true;
    try{
     const index=turns.length;if(index>=controls.max_requests){fatalStop=Error('resource_request_budget');throw fatalStop;}
     let response;
     const old=reference[index];
     if(edit&&index>edit.turnIndex&&(!old||observations(request)!==observations(old.request)))regenerating=true;
     if(regenerating&&!plan.teacher)throw Error('changed observations require a teacher or prefix fallback');
     if(regenerating&&(continuation===null||index>=continuation)){
      const teacherRequest=wire(request);
      teacherRequest.messages=structuredClone(teacherRequest.messages);
      teacherRequest.messages[0].content+='\nTraining-only expert reference; actual current inputs and observations take precedence. Complete the actual task, using equivalent actions when necessary.\n'+JSON.stringify(teacher.trajectory.map(t=>t.assistant));
      const data=await post(plan.teacher.endpoint,'/v1/chat/completions',{...teacherRequest,model:plan.teacher.model,temperature:.2,max_tokens:controls.max_output_tokens,seed:plan.root_seed+index,chat_template_kwargs:{enable_thinking:false}});
      response=await model.chatCompletionModelTurn(async()=>data)(request,abort.signal);
     }else{
      if(!old)throw Error('unchanged suffix exhausted; continuation needed');
      if(edit&&index<edit.turnIndex&&fingerprint(request)!==old.fingerprint)throw Error('prefix context mismatch');
      response=structuredClone(edit&&index===edit.turnIndex?edit.response:old.response);
     }
     if(response.truncated)throw Error('truncated action');
     const statistics=await score(request,response);
     turns.push({request:structuredClone(request),response:structuredClone(response),fingerprint:fingerprint(request),score:statistics,teacher_regenerated:regenerating});
     if(regenerating){
      consecutiveHard=(-statistics.mean_logprob>controls.continuation_nll)?consecutiveHard+1:0;
      if(consecutiveHard>=controls.max_failures){fatalStop=Error('teacher_continuation_student_difficulty_stop');throw fatalStop;}
     }
     trajectory.push(collector.trajectoryTurn(request,response));return response;
    }finally{busy=false;}
   };
   try{
    const run=await collector.executeProgram(record,driver,{systemPrompt:prompts.TOOLS_PROMPT,contextTokens:plan.context_tokens,maxTurns:controls.max_turns,temperature:0,rootSeed:plan.root_seed,runId,signal:abort.signal});
    if(!edit&&continuation===null&&turns.length!==reference.length)throw Error('reference replay not exact');
    if(fatalStop)throw fatalStop;
    row=collector.programRow(record,modelId,runId,expected,run,trajectory);
    const native=materializer.materializeNativeRows([row],{directAnswers:true});
    const approved=new Set(native.turns.filter(t=>t.training_admission.approved).map(t=>t.decision.index));
    for(const [i,turn] of turns.entries())turn.training_target=approved.has(i);
    admitted=run.outcome.accepted&&native.acceptedRows===1&&!native.unlinked.length&&approved.size>0&&(!record.curriculum||curriculum.admitRow(row).admitted);
   }catch(e){if(abort.signal.aborted)throw e;error={name:e.name,message:String(e.message).slice(0,1000)};}
   return {turns,row,admitted,error};
  };
  const initial=await execute(teacher.trajectory.map(t=>({response:{...t.model_response,...(t.assistant.reasoning?{reasoning:t.assistant.reasoning}:{})}})));
  await save(join(dir,'initial.json'),initial);
  if(!initial.admitted){summaries.push({program_id:record.id,disposition:'reference_replay_rejected',error:initial.error});continue;}
  let current=initial,bestValid=initial,failures=0,edits=0,chunkEdits=0,continuations=0,rounds=0,breakage=null,disposition='vanilla_sft';const attempts=[];
  while(edits<controls.max_edits&&rounds<controls.max_edits*controls.max_failures){
   rounds++;
   let location=firstDifficultChunk(current.turns,controls);
   if(!location&&trajectoryNll(current.turns)>controls.trajectory_nll){
    const turnIndex=current.turns.findIndex(t=>t.training_target!==false&&-t.score.mean_logprob>controls.trajectory_nll);
    if(turnIndex>=0){const t=current.turns[turnIndex];location={turnIndex,chunk:{kind:'action',value:assistant(t.response),mean_nll:-t.score.mean_logprob,max_nll:Math.max(...t.score.token_logprobs.map(x=>-x))}};}
   }
   if(trajectoryNll(current.turns)<=controls.trajectory_nll&&!location)break;
   if(!location){disposition='trajectory_threshold_unmet_without_local_chunk';break;}
   breakage=location;const turn=current.turns[location.turnIndex];
   const candidates=await proposals(turn,location.chunk,plan.root_seed+edits*100+failures*10,dir,rounds);
   const evaluated=[];
   for(const [i,candidate] of candidates.entries()){
    if(candidate.rejected){evaluated.push(candidate);continue;}
    try{
     const response=replaceChunk(turn.response,location.chunk,candidate.replacement);
     if(canonical(response.calls)===canonical(turn.response.calls)&&response.text===turn.response.text){evaluated.push({...candidate,rejected:'unchanged_chunk'});continue;}
     const localScore=await score(turn.request,response);
     const matching=localScore.chunks.filter(c=>c.kind===location.chunk.kind&&c.call_index===location.chunk.call_index&&c.argument_name===location.chunk.argument_name&&(c.value_start??0)===(location.chunk.value_start??0));
     const chunkNll=location.chunk.kind==='action'?-localScore.mean_logprob:matching.length?Math.max(...matching.map(c=>c.mean_nll)):Infinity;
     const chunkMax=location.chunk.kind==='action'?Math.max(...localScore.token_logprobs.map(x=>-x)):matching.length?Math.max(...matching.map(c=>c.max_nll)):Infinity;
     if(chunkNll>controls.chunk_nll||chunkMax>controls.token_nll){evaluated.push({...candidate,rejected:'student_chunk_too_difficult',localScore});continue;}
     const result=await execute(current.turns,{...location,response});
     result.chunk_nll=chunkNll;result.chunk_max_nll=chunkMax;result.ranking_nll=candidate.ranking_nll;
     const decision=candidateDecision(result,current,controls);evaluated.push({...candidate,decision,result});
    }catch(e){if(abort.signal.aborted)throw e;evaluated.push({...candidate,rejected:'candidate_error',error:String(e)});}
   }
   await save(join(dir,`edit-${edits}-attempt-${failures}.json`),{location,current_nll:trajectoryNll(current.turns),candidates:evaluated});
   attempts.push({turn:location.turnIndex,kind:location.chunk.kind,qualified:evaluated.filter(c=>c.decision?.accepted).length});
   const winner=evaluated.filter(c=>c.decision?.accepted).sort((a,b)=>a.ranking_nll-b.ranking_nll)[0];
   if(winner){current=winner.result;if(trajectoryNll(current.turns)<trajectoryNll(bestValid.turns))bestValid=current;chunkEdits++;edits++;failures=0;disposition='rewritten_sft';continue;}
   failures++;
   if(failures<controls.max_failures)continue;
   if(!plan.teacher){disposition='corrective_prefix_sft';break;}
   const continued=await execute(current.turns,null,location.turnIndex);
   await save(join(dir,`teacher-continuation-${rounds}.json`),continued);
   if(continued.admitted){current=continued;if(trajectoryNll(current.turns)<trajectoryNll(bestValid.turns))bestValid=current;continuations++;disposition='teacher_continuation_sft';breakage=null;failures=0;edits++;continue;}
   disposition='corrective_prefix_sft';break;
  }
  if((edits===controls.max_edits||rounds===controls.max_edits*controls.max_failures)&&firstDifficultChunk(current.turns,controls)){breakage=firstDifficultChunk(current.turns,controls);disposition='corrective_prefix_sft';}
  if(trajectoryNll(current.turns)>trajectoryNll(bestValid.turns)){
   current=bestValid;breakage=firstDifficultChunk(current.turns,controls);
   disposition=breakage?'corrective_prefix_sft':'rewritten_sft';
  }
  current.row.provenance.student_chunk_rewrite.disposition=disposition;
  current.row.provenance.student_chunk_rewrite.supervision_cutoff_decision=disposition==='corrective_prefix_sft'?breakage.turnIndex:null;
  let turns=materializer.materializeNativeRows([current.row],{directAnswers:true}).turns;
  if(disposition==='corrective_prefix_sft')turns=correctivePrefix(turns,breakage.turnIndex);
  for(const t of turns){t.student_chunk_rewrite={disposition,plan_sha256:planHash,complete_corrective_action:true,cutoff_decision:disposition==='corrective_prefix_sft'?breakage.turnIndex:null};}
  finalRows.push(current.row);finalTurns.push(...turns);
  const summary={program_id:record.id,disposition,edits,accepted_chunk_edits:chunkEdits,accepted_teacher_continuations:continuations,initial_nll:trajectoryNll(initial.turns),final_nll:trajectoryNll(current.turns),initial_target_tokens:initial.turns.filter(t=>t.training_target!==false).reduce((n,t)=>n+t.score.token_count,0),final_target_tokens:current.turns.filter(t=>t.training_target!==false).reduce((n,t)=>n+t.score.token_count,0),initial_requests:initial.turns.length,final_requests:current.turns.length,attempts,exported_decisions:turns.length,positive_decisions:turns.filter(t=>t.training_admission.approved).length,automatic_training_publication:false};
  await save(join(dir,'final.json'),{summary,row:current.row,turns});summaries.push(summary);
  console.log(JSON.stringify(summary));
 }
 await writeFile(join(output,'admitted.jsonl'),finalRows.map(r=>JSON.stringify(r)+'\n').join(''),{flag:'wx'});
 await writeFile(join(output,'turns.jsonl'),finalTurns.map(r=>JSON.stringify(r)+'\n').join(''),{flag:'wx'});
 await save(join(output,'summary.json'),{method:'verified-threshold-chunk-rewrite/1',cases:summaries,automatic_training_publication:false,not_mcmc:true,not_accuracy_evaluation:true});
}finally{await lock.close();await unlink(join(output,'active.lock'));}
