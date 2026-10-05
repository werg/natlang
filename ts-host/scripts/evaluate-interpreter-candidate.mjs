#!/usr/bin/env node
/** Review-only runner: complete native program execution against one pinned interpreter candidate. */
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, open, readFile, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { classifyCaughtError, classifyReturnedOutcome } from './evaluation-classification.mjs';

const [planPath,modeArg,planPin]=process.argv.slice(2);
const planBytes=await readFile(planPath),plan=JSON.parse(planBytes);
const digestBytes=b=>createHash('sha256').update(b).digest('hex');
if(plan.schema!=='natlang.interpreter_candidate_evaluation/1')throw Error('unsupported plan');
const preflightOnly=modeArg==='--preflight';
if(!preflightOnly&&(modeArg!=='--execute'||plan.root_approved!==true||planPin!==digestBytes(planBytes)))throw Error('requires exact approved plan');
for(const [path,sha] of Object.entries(plan.pins)){
 const digest=createHash('sha256');
 for await(const block of createReadStream(path))digest.update(block);
 if(digest.digest('hex')!==sha)throw Error('pinned input changed: '+path);
}
const runtime=resolve(plan.runtime),packet=resolve(plan.packet),out=resolve(plan.output);
const endpoint=plan.endpoint,modelId=plan.model,serverSelection='fixed-external-candidate',mode='candidate',step=0;
const snapshotSha256=plan.model_sha256;
if(!/^[a-f0-9]{64}$/.test(snapshotSha256)||!/^http:\/\/127\.0\.0\.1:[0-9]+$/.test(endpoint))throw Error('invalid candidate identity/endpoint');
if(plan.max_output_tokens!==1024)throw Error('primary comparison retains the original1024token budget');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const load = relative => import(pathToFileURL(join(runtime, relative)).href);
const manifestPath = join(runtime, 'frozen-runtime.json');
const runtimeManifestBytes=await readFile(manifestPath);
const runtimeManifestSha256 = hash(runtimeManifestBytes);
for(const [path,sha] of Object.entries(JSON.parse(runtimeManifestBytes).files))if(hash(await readFile(join(runtime,path)))!==sha)throw Error('runtime changed: '+path);
const packetManifest = JSON.parse(await readFile(join(packet, 'packet-manifest.json'), 'utf8'));
const casesBytes = await readFile(join(packet, 'cases.ir.jsonl'));
const goldBytes = await readFile(join(packet, 'gold-reference.jsonl'));
const requestBytes = await readFile(join(packet, 'first-requests.jsonl'));
const promptFit = JSON.parse(await readFile(join(packet, 'prompt-fit.json'), 'utf8'));
const reviewExclusion = JSON.parse(await readFile(join(packet, 'source-review-exclusion.json'), 'utf8'));
if (hash(casesBytes) !== packetManifest.artifact_pins.cases_ir.sha256 ||
    hash(goldBytes) !== packetManifest.artifact_pins.gold_reference.sha256)
  throw new Error('packet inputs differ from their prepared pins');
if (packetManifest.protocol?.max_turns !== 8 ||
    packetManifest.protocol?.max_model_requests_per_case !== 16 ||
    packetManifest.protocol?.context_tokens !== 16384 ||
    packetManifest.protocol?.client_max_tokens !== 1024)
  throw new Error('packet resource controls differ from the reviewed limits');
const collector = await load('dist/teacher/collector.js');
const curriculumPolicy = await load('dist/teacher/curriculum-policy.js');
const cases = casesBytes.toString('utf8').split(/\r?\n/).filter(Boolean).map(JSON.parse);
const goldRows = goldBytes.toString('utf8').split(/\r?\n/).filter(Boolean).map(JSON.parse);
const requestRows = requestBytes.toString('utf8').split(/\r?\n/).filter(Boolean).map(JSON.parse);
if (cases.length !== 24 || goldRows.length !== cases.length || requestRows.length !== cases.length ||
    promptFit.fit !== true || !Array.isArray(promptFit.cases) || promptFit.cases.length !== cases.length || promptFit.context_tokens !== 16384 ||
    promptFit.reserved_output_tokens !== 1024)
  throw new Error('fixed evaluation case count or prompt-fit proof changed');
const goldByKey = new Map(goldRows.map(row => [`${row.id}\0${row.program_ir_sha256}`, row]));
const pendingReviewValues = new Set(reviewExclusion.pending_ids_and_aliases);
if (reviewExclusion.source_review_sha256 !== packetManifest.runtime.current_source_review_ts_sha256 ||
    hash(await readFile(reviewExclusion.source_review_path)) !== reviewExclusion.source_review_sha256)
  throw new Error('source-review exclusion and runtime policy pins differ');
for (const [index, record] of cases.entries()) {
  const key = `${record.id}\0${collector.recordDigest(record)}`;
  if (!goldByKey.has(key)) throw new Error(`missing gold-separation join for ${record.id}`);
  const identities = [record.id, ...(record.source_ids ?? []), ...(record.dataset_records ?? []), ...(record.source_groups ?? [])];
  if (identities.some(identity => pendingReviewValues.has(identity)))
    throw new Error(`current source-review hold intersects selected case ${record.id}`);
  const request = requestRows[index];
  if (request.id !== record.id || request.program_ir_sha256 !== collector.recordDigest(record) ||
      !Array.isArray(request.messages) || !Array.isArray(request.tools) ||
      ['expected', 'oracle', 'gold', 'gold_reference', 'semantics', 'program_ir'].some(name => name in request))
    throw new Error(`model-visible request capture or gold separation changed for ${record.id}`);
  if (record.semantics.world || record.semantics.services?.world)
    throw new Error(`world task must remain excluded until its completion guard is deployed: ${record.id}`);
}
if (preflightOnly) {
  const policyHeld = cases.filter(record => curriculumPolicy.generationHoldReason(record)).length;
  console.log(JSON.stringify({ status: 'preflight_passed', model_requests: 0, output_created: false,
    packet_manifest_sha256: hash(await readFile(join(packet, 'packet-manifest.json'))),
    runtime_manifest_sha256: runtimeManifestSha256, cases: cases.length, gold_rows: goldRows.length,
    visible_request_rows_checked: requestRows.length, source_review_hold_matches: 0,
    generation_policy_held_cases: policyHeld, provider_eligible_cases: cases.length - policyHeld,
    limits: { context_tokens: 16384, max_tokens: 1024, max_turns: 8, max_model_requests_per_case: 16 } }));
  process.exit(0);
}
const serverCheck = await fetch(endpoint+'/v1/models');
if (!serverCheck.ok) throw new Error(`candidate server health failed: ${serverCheck.status}`);
const serverModels = await serverCheck.json();
if (!serverModels.data?.some(item => item.id === modelId))
  throw new Error(`candidate model identity mismatch: expected ${modelId}`);
const metadataProtocol=plan.metadata_protocol??'llama.cpp';
if(!['llama.cpp','vllm'].includes(metadataProtocol))throw Error('unsupported metadata protocol');
const properties=metadataProtocol==='llama.cpp'?await (await fetch(endpoint+'/props')).json():serverModels.data.find(item=>item.id===modelId);
if(metadataProtocol==='vllm'){
 if(properties.root!==plan.model_path||properties.max_model_len!==16384)throw Error('candidate model path/context differs');
 if(!plan.chat_template_path||hash(await readFile(plan.chat_template_path))!==plan.chat_template_sha256)throw Error('native template pin differs');
}
if(metadataProtocol==='llama.cpp'){
if(properties.model_path!==plan.model_path)throw Error('candidate server loaded a different model path');
if(typeof properties.chat_template!=='string'||hash(Buffer.from(properties.chat_template))!==plan.chat_template_sha256)
  throw Error('candidate server template differs from pinned native template');
if(properties.default_generation_settings?.n_ctx!==16384)throw Error('candidate server context differs');
}
await writeFile(resolve(plan.server_properties_receipt),JSON.stringify(properties,null,2)+'\n',{flag:'wx'});

if (await import('node:fs/promises').then(fs => fs.stat(out).then(() => true, () => false)))
  throw new Error(`refusing to overwrite prior evaluation attempt: ${out}`);
await mkdir(out, { recursive: true });
const postMetadata=async(path,body)=>{
 const response=await fetch(endpoint+path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
 if(!response.ok)throw Error(`candidate metadata ${path}: HTTP${response.status}`);
 return response.json();
};
// Count with the deployed candidate's native renderer/tokenizer, never another student's tokenizer.
const candidateFit=[];
for(const row of requestRows){
 if(metadataProtocol==='vllm'){
  const tokenized=await postMetadata('/tokenize',{model:modelId,messages:row.messages,tools:row.tools,add_generation_prompt:true,chat_template_kwargs:plan.request_options?.chat_template_kwargs??{}});
  if(!Array.isArray(tokenized.tokens)||tokenized.tokens.length+1024>16384)throw Error('candidate prompt exceeds context: '+row.id);
  candidateFit.push({id:row.id,token_ids_sha256:hash(Buffer.from(JSON.stringify(tokenized.tokens))),input_tokens:tokenized.tokens.length,reserved_output_tokens:1024});
  continue;
 }
 const rendered=await postMetadata('/apply-template',{model:modelId,messages:row.messages,tools:row.tools});
 if(typeof rendered.prompt!=='string')throw Error('candidate template returned no prompt');
 const tokenized=await postMetadata('/tokenize',{content:rendered.prompt,add_special:true});
 if(!Array.isArray(tokenized.tokens)||tokenized.tokens.length+1024>16384)throw Error('candidate prompt exceeds context: '+row.id);
 candidateFit.push({id:row.id,prompt_sha256:hash(Buffer.from(rendered.prompt)),input_tokens:tokenized.tokens.length,reserved_output_tokens:1024});
}
await writeFile(join(out,'candidate-prompt-fit.json'),JSON.stringify({model:modelId,chat_template_sha256:plan.chat_template_sha256,context_tokens:16384,fit:true,cases:candidateFit},null,2)+'\n',{flag:'wx'});
const jobs = join(out, 'jobs'); await mkdir(jobs, { recursive: true });
const trajectories=join(out,'trajectories'); await mkdir(trajectories);
const resultsPath = join(out, 'case-results.jsonl');
const resultHandle = await open(resultsPath, 'wx');
const options = {
  endpoint, modelId, systemPrompt: collector.defaultSystemPrompt, contextTokens: 16384,
  maxTurns: 8, maxModelRequests: 16, modelConcurrency: 1, collectionRole: 'student',
  rootSeed: 909, request: { ...plan.request_options, max_tokens: 1024 }, jobs, output: join(out, 'aggregate-unused.jsonl'), workers: 1,
  transportRetries: 0, temperature:0,
};
const runner = collector.nativeJobRunner(options);
const surface = await collector.defaultToolSurfaceHash(runtime);
const counts = { attempted: 0, policy_held: 0, complete_success: 0, semantic_failure: 0,
  contract_failure: 0, review_required: 0, resource_failure: 0, infrastructure_failure: 0, incomplete_task: 0 };
const caseSummaries = [];
let completedTrajectoryTurns = 0;
try {
  for (let index = 0; index < cases.length; index++) {
    const record = cases[index], irSha256 = collector.recordDigest(record);
    const provenance = collector.expectedProvenance(record, {
      ...options, toolSurfaceSha256: surface, collectionRole: 'student',
    });
    const startedAt = new Date().toISOString();
    let summary;
    const generationHold = curriculumPolicy.generationHoldReason(record);
    if (generationHold) {
      counts.policy_held++;
      summary = {
        schema: 'natlang.interpreter_candidate_case_result/1', id: record.id, program_ir_sha256: irSha256,
        source_groups: record.source_groups ?? [], family: record.family ?? null,
        source: record.source ?? null, mode, step, snapshot_sha256: snapshotSha256,
        server_selection: serverSelection, started_at: startedAt, finished_at: new Date().toISOString(),
        disposition: 'policy_held', policy_hold_reason: generationHold, provider_requests: 0,
        accepted: null, preference_label: null,
      };
      const line = `${JSON.stringify(summary)}\n`;
      await resultHandle.write(line); await resultHandle.sync();
      caseSummaries.push(summary);
      continue;
    }
    counts.attempted++;
    const caseSignal = new AbortController().signal;
    try {
      const row = await runner({ index, record }, provenance, caseSignal);
      const trajectoryBytes=Buffer.from(JSON.stringify(row)+'\n');
      const trajectoryPath=join(trajectories,`${index.toString().padStart(2,'0')}-${hash(Buffer.from(record.id)).slice(0,16)}.json`);
      await writeFile(trajectoryPath,trajectoryBytes,{flag:'wx'});
      const outcome = row.outcome ?? {};
      const requestTurns = Array.isArray(row.trajectory) ? row.trajectory.length : null;
      if (requestTurns !== null) completedTrajectoryTurns += requestTurns;
      const disposition = classifyReturnedOutcome(outcome);
      counts[disposition]++;
      summary = {
        schema: 'natlang.interpreter_candidate_case_result/1', id: record.id, program_ir_sha256: irSha256,
        source_groups: record.source_groups ?? [], family: record.family ?? null,
        source: record.source ?? null, mode, step, snapshot_sha256: snapshotSha256,
        server_selection: serverSelection, started_at: startedAt, finished_at: new Date().toISOString(),
        trajectory_path:trajectoryPath, trajectory_sha256:hash(trajectoryBytes), training_publication:false,
        elapsed_ms:Date.now()-Date.parse(startedAt),
        disposition, status: outcome.status ?? null, detail: outcome.detail ?? null, accepted: outcome.accepted ?? null,
        value: outcome.value ?? null, checks: outcome.checks ?? null, request_turns: requestTurns,
        rejection_reasons: outcome.rejection_reasons ?? [], oracle: outcome.oracle ?? null,
        trace_sha256: row.provenance?.trace_sha256 ?? null,
      };
    } catch (error) {
      const text = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
      const caseDeadlineExceeded = caseSignal.aborted && caseSignal.reason?.name === 'TimeoutError';
      const disposition = classifyCaughtError(error, { caseDeadlineExceeded });
      counts[disposition]++;
      summary = {
        schema: 'natlang.interpreter_candidate_case_result/1', id: record.id, program_ir_sha256: irSha256,
        source_groups: record.source_groups ?? [], family: record.family ?? null,
        source: record.source ?? null, mode, step, snapshot_sha256: snapshotSha256,
        server_selection: serverSelection, started_at: startedAt, finished_at: new Date().toISOString(),
        disposition, error: text, classification_basis: caseDeadlineExceeded ? 'runner_case_deadline_signal' : 'typed_error_code_or_incomplete_default', preference_label: null,
      };
    }
    const line = `${JSON.stringify(summary)}\n`;
    await resultHandle.write(line); await resultHandle.sync();
    caseSummaries.push(summary);
  }
  await resultHandle.close();
  const report = {
    plan_sha256:hash(planBytes),candidate_identity:plan.candidate_identity,training_publication:false,
    schema: 'natlang.interpreter_candidate_evaluation_report/1', status: 'completed', mode, step,
    selected_cases: cases.length, provider_attempted_cases: counts.attempted, policy_held_cases: counts.policy_held,
    endpoint, server_selection: serverSelection, model_id: modelId, snapshot_sha256: snapshotSha256,
    packet_manifest_sha256: hash(await readFile(join(packet, 'packet-manifest.json'))),
    cases_ir_sha256: hash(casesBytes), gold_reference_sha256: hash(goldBytes),
    runtime_manifest_sha256: runtimeManifestSha256, system_prompt_sha256: hash(Buffer.from(collector.defaultSystemPrompt)),
    tool_surface_sha256: surface, completed_trajectory_turns: completedTrajectoryTurns,
    request_count_note: 'trajectory turn counts cover completed rows; failed rows may have partial request journals and are not added to this total',
    limits: { context_tokens: 16384, reserved_output_tokens: 1024, max_turns: 8,
      max_model_requests_per_case: 16, case_timeout_ms: null, concurrency: 1 },
    counts, completed_cases: counts.complete_success + counts.semantic_failure + counts.contract_failure,
    semantic_denominator: counts.complete_success + counts.semantic_failure,
    semantic_accuracy: (counts.complete_success + counts.semantic_failure) ?
      counts.complete_success / (counts.complete_success + counts.semantic_failure) : null,
    review_required_cases: counts.review_required,
    resource_or_infrastructure_or_incomplete: counts.resource_failure + counts.infrastructure_failure + counts.incomplete_task,
    preference_labels_created: 0, case_results: resultsPath,
  };
  await writeFile(join(out, 'report.json'), `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' });
  console.log(JSON.stringify(report));
} catch (error) {
  await resultHandle.close();
  throw error;
}
