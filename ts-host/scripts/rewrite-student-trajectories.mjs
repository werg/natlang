#!/usr/bin/env node
/** Student-conditioned rewriting with privileged proposal context and ordinary SFT contexts.
 * No MCMC, likelihood ranking, reward optimizer, or automatic training publication.
 * Usage: node rewrite-student-trajectories.mjs PLAN [--execute --sha256 PLAN_SHA]
 */
import { readFile, mkdir, open } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';

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
if (plan.schema !== 'natlang.student_rewrite_plan/1') throw new Error('unsupported rewrite plan');
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
  plan_sha256: planHash, source_closure_sha256: plan.pins[plan.source_closure], method: 'verified-student-rewrite' }));
if (values.execute) {
  if (!plan.student || !Object.keys(plan.student.weight_pins ?? {}).length)
    throw new Error('student needs immutable checkpoint weight pins');
  for (const [path, expected] of Object.entries(plan.student.weight_pins)) await pinned(path, expected);
  const identityResponse = await fetch(`${plan.endpoint.replace(/\/$/, '')}/natlang/student-identity`);
  if (!identityResponse.ok) throw new Error('student endpoint has no loaded-checkpoint identity');
  const loaded = await identityResponse.json();
  if (loaded.checkpoint_map || loaded.base_model !== plan.student.base_model ||
      loaded.revision !== plan.student.revision || loaded.adapter !== plan.model ||
      json(Object.entries(loaded.weight_pins ?? {}).sort()) !== json(Object.entries(plan.student.weight_pins).sort()))
    throw new Error('student endpoint did not load the pinned checkpoint');
  const models = await (await fetch(`${plan.endpoint.replace(/\/$/, '')}/v1/models`)).json();
  if (!models.data?.some(model => model.id === plan.model)) throw new Error('unexpected student endpoint identity');
  const output = resolve(plan.output); await mkdir(output, { recursive: true });
  const lock = await open(join(output, 'collector.lock'), 'wx');
  let stop = false; const abort = new AbortController();
  const requestStop = () => { stop = true; abort.abort(); };
  process.on('SIGINT', requestStop); process.on('SIGTERM', requestStop);
  try {
    const identity = join(output, 'plan.json');
    try { if (hash(await readFile(identity)) !== planHash) throw new Error('output belongs to another plan'); }
    catch (error) { if (error.code !== 'ENOENT') throw error; await collector.writeAtomic(identity, planBytes.toString()); }
    for (const { teacher, record, hint, artifact } of teachers) {
      if (stop) break;
      const key = hash(record.id).slice(0, 20);
      for (let attempt = 0; attempt < plan.attempts; attempt++) {
        if (stop) break;
        const path = join(output, `${key}-${attempt}.json`);
        let previous;
        try { previous = await load(path); } catch (error) { if (error.code !== 'ENOENT') throw error; }
        if (previous) { if (previous.plan_sha256 !== planHash) throw new Error('stale attempt');
          if (previous.admitted) break;
          if (previous.classification === 'operator_interrupted')
            throw new Error('interrupted attempt requires a reviewed continuation; preserve its proposal evidence');
          continue; }
        try {
          await readFile(join(output, `${key}-${attempt}.proposal.json`));
          throw new Error('orphan proposal evidence requires reviewed continuation; refusing overwrite');
        } catch (error) { if (error.code !== 'ENOENT') throw error; }
        const trajectory = [], proposals = [];
        let sent = 0;
        const send = transport.openAICompatibleModelTurn({ endpoint: plan.endpoint, model: plan.model,
          stream: false, request: { max_tokens: plan.max_output_tokens } });
        const privileged = '\n\nTraining-only reference, available to help you execute this task. ' +
          'Use the original visible inputs and actual tool observations. Follow the task contract; ' +
          'you may choose different valid actions. Do not mention this reference in your answer.\n' + hint;
        const driver = async request => {
          if (++sent > plan.max_requests) throw new Error('collection_request_budget');
          const proposal = structuredClone(request);
          if (typeof proposal.messages[0]?.content !== 'string') throw new Error('expected textual system prompt');
          proposal.messages[0].content += privileged;
          const response = await send(proposal, abort.signal);
          proposals.push({ request: proposal, response });
          if (response.truncated) throw new Error('incomplete_model_response');
          // Store actual privileged requests separately; SFT uses ORIGINAL request contexts.
          trajectory.push(collector.trajectoryTurn(request, response)); return response;
        };
        const seed = plan.root_seed + attempt;
        const expected = collector.expectedProvenance(record, { modelId: plan.model,
          rootSeed: seed, systemPrompt: prompts.TOOLS_PROMPT, temperature: plan.temperature,
          contextTokens: plan.context_tokens, maxTurns: plan.max_turns, toolSurfaceHash });
        expected.collection_role = 'student_rewrite';
        expected.student_rewrite = { method: 'verified-student-rewrite/1', plan_sha256: planHash,
          parent_trajectory_sha256: artifact.sha256, parent_trajectory_id: teacher.id, privileged_proposal_only: true };
        const runId = collector.programRunId(0, expected);
        let receipt;
        try {
          const run = await collector.executeProgram(record, driver, { systemPrompt: prompts.TOOLS_PROMPT,
            contextTokens: plan.context_tokens, maxTurns: plan.max_turns, temperature: plan.temperature,
            rootSeed: seed, runId, signal: abort.signal });
          const row = collector.programRow(record, plan.model, runId, expected, run, trajectory);
          const native = materializer.materializeNativeRows([row], { directAnswers: true });
          const admitted = run.outcome.accepted && native.acceptedRows === 1 && !native.unlinked.length &&
            native.turns.some(turn => turn.training_admission?.approved === true) &&
            (!record.curriculum || curriculum.admitRow(row).admitted);
          receipt = { plan_sha256: planHash, program_id: record.id, attempt, admitted, row, requests: sent };
          // Standard native turns preserve source lineage; candidates are still separate from the training corpus.
          if (admitted) await collector.writeAtomic(join(output, `${key}.turns.jsonl`),
            native.turns.map(turn => json(turn)).join('\n') + '\n');
        } catch (error) {
          receipt = { plan_sha256: planHash, program_id: record.id, attempt, admitted: false,
            classification: stop ? 'operator_interrupted' : 'collection_error', error_type: error.name,
            error_message: String(error.message).slice(0, 1000),
            program_ir_sha256: collector.recordDigest(record), partial_trajectory: trajectory, requests: sent };
        }
        await collector.writeAtomic(join(output, `${key}-${attempt}.proposal.json`), json(proposals) + '\n');
        await collector.writeAtomic(path, json(receipt) + '\n');
        await append(join(output, 'events.jsonl'), { program_id: record.id, attempt, admitted: receipt.admitted,
          requests: sent, receipt_sha256: hash(json(receipt) + '\n'), checked_at: new Date().toISOString() });
        if (receipt.admitted) break;
      }
    }
    if (!stop) {
      const admittedRows = [], allTurns = [], results = [];
      for (const { record } of teachers) {
        const key = hash(record.id).slice(0, 20);
        let admitted;
        for (let attempt = 0; attempt < plan.attempts; attempt++) {
          const result = await load(join(output, `${key}-${attempt}.json`));
          if (result.admitted) { admitted = result; break; }
        }
        results.push({ program_id: record.id, admitted: !!admitted });
        if (admitted) {
          const native = materializer.materializeNativeRows([admitted.row], { directAnswers: true });
          if (native.acceptedRows !== 1 || native.unlinked.length) throw new Error('saved admission changed');
          admittedRows.push(admitted.row); allTurns.push(...native.turns);
        }
      }
      await collector.writeAtomic(join(output, 'admitted.jsonl'), admittedRows.map(json).join('\n') + (admittedRows.length ? '\n' : ''));
      await collector.writeAtomic(join(output, 'turns.jsonl'), allTurns.map(json).join('\n') + (allTurns.length ? '\n' : ''));
      await collector.writeAtomic(join(output, 'summary.json'), json({ schema: 'natlang.student_rewrite_summary/1',
        plan_sha256: planHash, cases: teachers.length, admitted: admittedRows.length, turns: allTurns.length,
        results, automatic_training_publication: false }) + '\n');
    }
    if (stop) process.exitCode = 130;
  } finally {
    process.off('SIGINT', requestStop); process.off('SIGTERM', requestStop);
    await lock.close(); const { unlink } = await import('node:fs/promises'); await unlink(join(output, 'collector.lock'));
  }
}
