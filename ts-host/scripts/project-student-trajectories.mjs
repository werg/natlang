#!/usr/bin/env node
/** Fixed-student, proposal-corrected MH over freshly executed interactive episode suffixes.
 * Usage: node rewrite-student-trajectories.mjs PLAN [--execute --sha256 PLAN_SHA]
 */
import { readFile, mkdir, open } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';
import {mhDecision, canonical, randomStream} from './projection-search.mjs';

const hash = data => createHash('sha256').update(data).digest('hex');
const json = value => JSON.stringify(value);
const load = async path => JSON.parse(await readFile(path, 'utf8'));
async function pinned(path, expected) {
  const bytes = await readFile(path);
  if (hash(bytes) !== expected) throw new Error(`pinned input changed: ${path}`);
  return bytes;
}
async function append(path, value) {
  const file = await open(path, 'a');
  try { await file.writeFile(json(value) + '\n'); await file.sync(); }
  finally { await file.close(); }
}

const { values, positionals } = parseArgs({ allowPositionals: true, options: {
  execute: { type: 'boolean', default: false }, sha256: { type: 'string' },
} });
if (positionals.length !== 1) throw new Error('expected one immutable rewrite plan');
const planPath = resolve(positionals[0]), planBytes = await readFile(planPath);
const planHash = hash(planBytes), plan = JSON.parse(planBytes);
if (plan.schema !== 'natlang.student_projection_plan/1') throw new Error('unsupported rewrite plan');
if (values.execute && (plan.root_approved !== true || values.sha256 !== planHash))
  throw new Error('execution requires the exact reviewed plan');
for (const [path, expected] of Object.entries(plan.pins)) await pinned(path, expected);
if (!(plan.attempts >= 1 && plan.attempts <= 2 && Number.isInteger(plan.attempts)) ||
    !(plan.max_turns >= 1 && plan.max_turns <= 16) || !Number.isInteger(plan.max_turns) ||
    !(plan.max_requests >= plan.max_turns && plan.max_requests <= 64) || !Number.isInteger(plan.max_requests) ||
    !(plan.max_hint_chars > 0) || !Number.isInteger(plan.max_hint_chars) ||
    !(plan.context_tokens > 0) || !Number.isInteger(plan.context_tokens) ||
    !(plan.max_output_tokens > 0) || !Number.isInteger(plan.max_output_tokens) ||
    !(plan.temperature >= 0 && plan.temperature <= 2) || !Number.isFinite(plan.temperature) ||
    !Number.isSafeInteger(plan.root_seed)) throw new Error('invalid collection resource controls');
const runtime = resolve(plan.runtime), frozen = await load(join(runtime, 'frozen-runtime.json'));
if (plan.pins[join(runtime, 'frozen-runtime.json')] !== hash(await readFile(join(runtime, 'frozen-runtime.json'))))
  throw new Error('runtime manifest must be pinned by the plan');
for (const [path, expected] of Object.entries(frozen.files)) await pinned(join(runtime, path), expected);
const module = relative => import(pathToFileURL(join(runtime, 'dist', relative)).href);
const [collector, materializer, policy, curriculum, prompts, transport] = await Promise.all([
  module('teacher/collector.js'), module('teacher/native-materializer.js'),
  module('teacher/curriculum-policy.js'), module('teacher/curriculum.js'),
  module('native/prompt.js'), module('model/openai-compatible.js'),
]);
if (!Number.isInteger(plan.search_steps) || plan.search_steps<1 || !Number.isInteger(plan.cases_limit) || plan.cases_limit<1) throw new Error('invalid search budget');
const {modelTools}=await module('model/chat-completion.js');
const toolSurfaceHash = await collector.defaultToolSurfaceHash(runtime);
const selectedIr = (await pinned(plan.ir, plan.pins[plan.ir])).toString().split('\n').filter(Boolean).map(JSON.parse);
const byId = new Map(selectedIr.map(record => [record.id, record]));
if (byId.size !== selectedIr.length || selectedIr.some(record => record.split !== 'train'))
  throw new Error('reviewed IR must contain unique, explicitly train cases');
const closure = await load(plan.source_closure);
// The root approval binds the exact IR and its existing source-split review.
if (!plan.pins[plan.source_closure] || !plan.teacher_artifacts.length)
  throw new Error('requires a pinned source closure and teacher artifacts');
if (closure.status !== 'passed' || closure.selected_ir_sha256 !== plan.pins[plan.ir] ||
    closure.combined_identity_overlap_count !== 0 || closure.selected_groups_not_train?.length !== 0 ||
    closure.errors?.length !== 0) throw new Error('source closure does not approve this exact train IR');
const teachers = [];
for (const artifact of plan.teacher_artifacts) {
  const teacher = JSON.parse(await pinned(artifact.path, artifact.sha256));
  const record = teacher.task?.program_ir;
  if (!record || !byId.has(record.id) || collector.recordDigest(record) !== collector.recordDigest(byId.get(record.id)))
    throw new Error('teacher does not match exact selected IR');
  if (policy.generationHoldReason(record) || policy.quarantineReason(record) || policy.retiredFamily(record))
    throw new Error(`source held: ${record.id}`);
  const native = materializer.materializeNativeRows([teacher], { directAnswers: true });
  if (native.acceptedRows !== 1 || native.unlinked.length || !native.turns.length ||
      (record.curriculum && !curriculum.admitRow(teacher).admitted))
    throw new Error(`teacher is not currently admitted: ${record.id}`);
  // Preserve complete teacher actions; never silently truncate the reference.
  const hint = json(teacher.trajectory.map(turn => turn.assistant));
  if (hint.length > plan.max_hint_chars) throw new Error(`teacher exceeds hint allowance: ${record.id}`);
  if (record.semantics.world || record.semantics.services) throw new Error('external state is unsupported by projection replay');
  teachers.push({ teacher, record, hint, artifact });
}
if (new Set(teachers.map(item => item.record.id)).size !== teachers.length)
  throw new Error('duplicate teacher programs');
if (!Array.isArray(plan.selected_program_ids) ||
    new Set(plan.selected_program_ids).size !== plan.selected_program_ids.length ||
    teachers.length !== plan.selected_program_ids.length ||
    teachers.some(item => !plan.selected_program_ids.includes(item.record.id)))
  throw new Error('teacher artifacts differ from the exact selected subset');
console.log(json({ status: 'preflight_passed', cases: teachers.length, provider_calls: 0,
  plan_sha256: planHash, source_closure_sha256: plan.pins[plan.source_closure], method: 'request-boundary-proposal-corrected-mh/1' }));
if(values.execute) {
  const endpoint=plan.endpoint.replace(/\/$/,'');
  const identity=await (await fetch(endpoint+'/natlang/student-identity')).json();
  for(const [p,h] of Object.entries(plan.student.weight_pins)) await pinned(p,h);
  if(identity.prompt_tokenization!=='chat-template-single-bos/1' || identity.projection_scoring!=='closed-assistant-temperature-logprob/1') throw new Error('student lacks exact proposal/scoring format');
  if(identity.checkpoint_map || identity.adapter!==plan.model || identity.revision!==plan.student.revision || identity.base_model!==plan.student.base_model ||
    canonical(identity.weight_pins)!==canonical(plan.student.weight_pins)) throw new Error('immutable student identity mismatch');
  const output=resolve(plan.output);await mkdir(output,{recursive:true});
  const lock=await open(join(output,'collector.lock'),'wx');
  let stop=false;const abort=new AbortController();
  const stopNow=()=>{stop=true;abort.abort();};process.on('SIGTERM',stopNow);process.on('SIGINT',stopNow);
  const post=async (body) => {
    const r=await fetch(endpoint+'/natlang/score',{method:'POST',headers:{'content-type':'application/json'},body:json(body),signal:abort.signal});
    const d=await r.json();if(!r.ok)throw new Error('student_score: '+json(d));return d;
  };
  try {
    const savedPlan=join(output,'plan.json');
    try {if(hash(await readFile(savedPlan))!==planHash)throw new Error('output belongs to another plan');}
    catch(e){if(e.code!=='ENOENT')throw e;await collector.writeAtomic(savedPlan,planBytes.toString());}
    const finalRows=[],finalTurns=[],results=[];
    for(const {teacher,record,hint,artifact} of teachers.slice(0,plan.cases_limit)) {
      if(stop)break;
      const key=hash(record.id).slice(0,20),dir=join(output,key);await mkdir(dir,{recursive:true});
      const finalPath=join(dir,'final.json');
      try {
        const previous=await load(finalPath);if(previous.plan_sha256!==planHash)throw new Error('stale chain');
        results.push(previous.summary);if(previous.row){finalRows.push(previous.row);finalTurns.push(...previous.turns);}continue;
      } catch(e){if(e.code!=='ENOENT')throw e;}
      const privileged='\n\nTraining-only expert trajectory. Execute the original task using its visible inputs and actual tool observations. Different correct actions are allowed. Do not mention this reference.\n'+hint;
      const rng=randomStream(plan.root_seed+Number.parseInt(key.slice(0,8),16));
      const guided=request=>{const r=structuredClone(request);if(typeof r.messages[0]?.content!=='string')throw new Error('nontextual system context');r.messages[0].content+=privileged;return r;};
      const fingerprint=r=>canonical({messages:r.messages,tools:r.tools,tool_choice:r.tool_choice??'auto'});
      const wire=r=>({messages:r.messages,tools:modelTools(r.tools)});
      const score=async(request,ids,assistant)=>{
        const payload={...wire(request),...(ids?{completion_token_ids:ids}:{assistant})};
        const p=await post({...payload,score_temperature:1});
        if (p.token_count>Math.min(plan.max_output_tokens,request.max_tokens??plan.max_output_tokens)) throw new Error('reverse proposal outside output-budget support');
        const q=await post({...wire(guided(request)),completion_token_ids:p.completion_token_ids,score_temperature:plan.temperature});
        return {target:p.sum_logprob,proposal:q.sum_logprob,token_ids:p.completion_token_ids,
          target_token_logprobs:p.token_logprobs,proposal_token_logprobs:q.token_logprobs};
      };
      const expected=collector.expectedProvenance(record,{modelId:plan.model,rootSeed:plan.root_seed,systemPrompt:prompts.TOOLS_PROMPT,
        temperature:plan.temperature,contextTokens:plan.context_tokens,maxTurns:plan.max_turns,toolSurfaceHash});
      expected.collection_role='student_projection';
      expected.student_projection={method:'request-boundary-proposal-corrected-mh/1',plan_sha256:planHash,parent_trajectory_sha256:artifact.sha256,
        parent_trajectory_id:teacher.id,privileged_proposal_only:true,target_temperature:1,proposal_temperature:plan.temperature};
      const runId=collector.programRunId(0,expected);
      const execute=async(prefix,cut,iteration,initial=false)=>{
        const turns=[],trajectory=[],exchanges=[];let busy=false;
        const driver=async request=>{
          if(busy)throw new Error('parallel model requests unsupported by sequential suffix kernel');busy=true;
          try {
            const i=turns.length;if(i>=plan.max_requests)throw new Error('collection_request_budget');
            let response,statistics;
            if(i<cut || initial) {
              const old=prefix[i];if(!old || fingerprint(request)!==old.fingerprint)throw new Error('fresh replay changed prefix observation');
              response=structuredClone(old.response);statistics=old.token_ids?old:await score(request,null,old.assistant);
            } else {
              let exchange;
              const send=transport.openAICompatibleModelTurn({endpoint,model:plan.model,stream:false,
                request:{max_tokens:plan.max_output_tokens,natlang_projection:true},
                onExchange:e=>{if(exchange)throw new Error('transport retry cannot be assigned a suffix probability');exchange=e;}});
              const g=guided(request);g.temperature=plan.temperature;g.seed=plan.root_seed+iteration*plan.max_requests+i;
              response=await send(g,abort.signal);exchanges.push(exchange);
              const meta=response.raw_response?.natlang_projection;
              if(response.truncated || !meta?.terminated || !Array.isArray(meta.completion_token_ids))throw new Error('incomplete_projection_response');
              statistics=await score(request,meta.completion_token_ids);
            }
            turns.push({request:structuredClone(request),fingerprint:fingerprint(request),response:structuredClone(response),...statistics});
            trajectory.push(collector.trajectoryTurn(request,response));return response;
          }finally{busy=false;}
        };
        let row,admitted=false,error;
        try {
          const run=await collector.executeProgram(record,driver,{systemPrompt:prompts.TOOLS_PROMPT,contextTokens:plan.context_tokens,
            maxTurns:plan.max_turns,temperature:plan.temperature,rootSeed:plan.root_seed,runId,signal:abort.signal});
          if(turns.length<cut)throw new Error('fresh replay terminated before cut');
          if(initial && turns.length!==prefix.length)throw new Error('teacher replay did not consume exact trajectory');
          row=collector.programRow(record,plan.model,runId,expected,run,trajectory);
          const native=materializer.materializeNativeRows([row],{directAnswers:true});
          admitted=run.outcome.accepted && native.acceptedRows===1 && !native.unlinked.length &&
            native.turns.some(t=>t.training_admission?.approved===true) && (!record.curriculum||curriculum.admitRow(row).admitted);
        } catch(e){error={name:e.name,message:String(e.message).slice(0,1000),classification:stop?'operator_interrupted':'proposal_or_replay_error'};}
        return {turns,row,admitted,error,exchanges};
      };
      // Initial teacher trace is a known correct state, replayed fresh. Canonical
      // tokenization initializes raw-token state; subsequent proposals retain
      // the actual sampled tokens including their EOS.
      const baseline=teacher.trajectory.map(t=>({fingerprint:fingerprint({messages:t.context,tools:t.tools_offered}),
        assistant:{role:'assistant',content:t.assistant.content||null,
          ...(t.assistant.reasoning?{reasoning_content:t.assistant.reasoning}:{}),
          ...(t.assistant.calls?.length?{tool_calls:t.assistant.calls.map((c,i)=>({id:'projection_'+i,type:'function',function:{name:c.tool,arguments:json(c.arguments)}}))}: {})},
        response:{...t.model_response,...(t.assistant.reasoning?{reasoning:t.assistant.reasoning}:{})}}));
      let current;
      try {current=await load(join(dir,'initial.json'));}
      catch(e){if(e.code!=='ENOENT')throw e;current=await execute(baseline,baseline.length,0,true);await collector.writeAtomic(join(dir,'initial.json'),json(current)+'\n');}
      if(current.error?.classification==='operator_interrupted')throw new Error('interrupted initialization requires a fresh reviewed continuation');
      const start=current;
      // Initialization may fail because the reference used a different prompt or
      // turn allowance. Bootstrap attempts are recorded, never called MH moves.
      for(let b=1;!current.admitted && b<=plan.attempts && !stop;b++) {
        const path=join(dir,`bootstrap-${b}.json`);
        try {current=await load(path);}catch(e){if(e.code!=='ENOENT')throw e;current=await execute([],0,b);await collector.writeAtomic(path,json(current)+'\n');}
      }
      const cuts=Math.max(1,current.turns.length);let accepted=0,completed=0;
      if(current.admitted) for(let step=0;step<plan.search_steps&&!stop;step++) {
        const receiptPath=join(dir,`search-${step}.json`);
        const cut=Math.floor(rng()*cuts),draw=rng();
        try {
          const receipt=await load(receiptPath);if(receipt.plan_sha256!==planHash)throw new Error('stale search receipt');
          if(receipt.candidate?.error?.classification==='operator_interrupted')throw new Error('interrupted proposal requires a fresh reviewed continuation');
          if(receipt.accepted) {current=receipt.candidate;accepted++;}completed++;continue;
        }catch(e){if(e.code!=='ENOENT')throw e;}
        let candidate,decision={accepted:false,reason:'cut_outside_current_trace'};
        if(cut<current.turns.length) {
          candidate=await execute(current.turns,cut,plan.attempts+step+1);
          if(candidate.admitted)decision=mhDecision(current.turns,candidate.turns,cut,draw);
          else decision={accepted:false,reason:candidate.error?'proposal_error':'task_constraint_rejected',draw};
        }
        await collector.writeAtomic(receiptPath,json({plan_sha256:planHash,step,cut,fixed_cut_slots:cuts,...decision,candidate})+'\n');
        if(decision.accepted){current=candidate;accepted++;}completed++;
        await append(join(output,'events.jsonl'),{program_id:record.id,step,cut,accepted:decision.accepted,reason:decision.reason??null});
      }
      if(stop)break;
      // Never pass an unchanged teacher or unsearched bootstrap off as a new
      // on-policy sample. Rejected/partial chains remain available for review.
      const changed=canonical(current.turns.map(t=>t.token_ids))!==canonical(start.turns.map(t=>t.token_ids));
      let turns=[],row;
      if(current.admitted && accepted>0 && changed && completed===plan.search_steps) {
        row=current.row;const n=materializer.materializeNativeRows([row],{directAnswers:true});turns=n.turns;
        finalRows.push(row);finalTurns.push(...turns);
      }
      const summary={program_id:record.id,initialized:current.admitted,reference_replay_admitted:start.admitted,steps:completed,accepted_moves:accepted,published_candidate:!!row,
        training_publication:false,initial_error:start.error??null};
      await collector.writeAtomic(finalPath,json({plan_sha256:planHash,summary,row,turns})+'\n');results.push(summary);
    }
    if(!stop){
      for(const [file,rows] of [['admitted.jsonl',finalRows],['turns.jsonl',finalTurns]])
        await collector.writeAtomic(join(output,file),rows.map(json).join('\n')+(rows.length?'\n':''));
      await collector.writeAtomic(join(output,'summary.json'),json({schema:'natlang.student_projection_summary/1',plan_sha256:planHash,
        method:'request-boundary-proposal-corrected-mh/1',cases:results.length,admitted:finalRows.length,turns:finalTurns.length,
        results,automatic_training_publication:false,convergence_claimed:false})+'\n');
    }else process.exitCode=130;
  } finally {process.off('SIGINT',stopNow);process.off('SIGTERM',stopNow);await lock.close();const {unlink}=await import('node:fs/promises');await unlink(join(output,'collector.lock'));}
}
