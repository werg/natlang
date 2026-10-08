#!/usr/bin/env node
/** Project-authored diagnostic skill evaluation; does not emit training corpus rows. */
import {readFile, mkdir, writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {join, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const [planPath, mode, approvedPlanSha] = process.argv.slice(2);
if (!planPath || !['--preflight', '--execute'].includes(mode))
  throw new Error('usage: node diagnostic-evaluator.mjs PLAN.json --preflight | --execute PLAN_SHA256');
const planBytes = await readFile(planPath);
const plan = JSON.parse(planBytes);
if (plan.schema !== 'natlang.criterion_grounded_skill_diagnostic_plan/1') throw new Error('unsupported plan schema');
if (mode === '--execute' && (plan.root_approved !== true || approvedPlanSha !== hash(planBytes)))
  throw new Error('execution requires root-approved plan bytes and exact SHA-256 argument');
if (mode === '--preflight' && plan.root_approved !== false) throw new Error('preflight plan must remain unapproved');
for (const [path, sha] of Object.entries(plan.pins)) {
  if (hash(await readFile(path)) !== sha) throw new Error(`pinned input changed: ${path}`);
}
const runtimeManifestPath = join(plan.runtime, 'frozen-runtime.json');
const runtimeManifest = JSON.parse(await readFile(runtimeManifestPath));
if (hash(await readFile(runtimeManifestPath)) !== plan.runtime_manifest_sha256) throw new Error('runtime manifest pin changed');
for (const [relative, sha] of Object.entries(runtimeManifest.files))
  if (hash(await readFile(join(plan.runtime, relative))) !== sha) throw new Error(`frozen runtime file changed: ${relative}`);
const module = relative => import(pathToFileURL(join(plan.runtime, 'dist', relative)));
const [collector, prompts, programs] = await Promise.all([module('teacher/collector.js'), module('native/prompt.js'), module('teacher/program.js')]);
const contexts = JSON.parse(await readFile(plan.contexts, 'utf8'));
const oracle = JSON.parse(await readFile(plan.oracle, 'utf8'));
if (contexts.schema !== 'natlang.criterion-grounded-fixtures/1' || oracle.schema !== 'natlang.criterion-grounded-fixture-oracles/1' ||
    contexts.cases.length !== 4 || oracle.decisions.length !== 4) throw new Error('unexpected fixture schema or count');
const decisions = new Map(oracle.decisions.map(row => [row.id, row.expected]));
if (decisions.size !== 4 || contexts.cases.some(row => !decisions.has(row.id)) ||
    JSON.stringify(plan.fixture_ids) !== JSON.stringify(contexts.cases.map(row => row.id))) throw new Error('oracle/plan does not exactly cover fixture IDs');
const arms = ['discovery', 'instructed'];
if (JSON.stringify(plan.arms) !== JSON.stringify(arms) || plan.workers !== 1 || plan.transport_retries !== 0 ||
    plan.max_requests_per_execution !== 4 || plan.max_total_requests !== 32 || contexts.cases.length * arms.length !== 8)
  throw new Error('execution bounds/design mismatch');
const candidate = await readFile(plan.skill, 'utf8');
const candidateInstructions = candidate.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/, '').trim();
const records = contexts.cases.map((fixture, index) => ({
  version: 'natlang.program/2', id: `criterion-diagnostic-${index + 1}`,
  kind: 'lambda_source', source: 'project-authored-diagnostic-fixture', split: 'diagnostic',
  source_groups: ['criterion-grounded-skill-diagnostic-v1'], license: 'project-authored',
  semantics: {
    root: 'judge.nl',
    files: {
      'judge.nl': `---\nargs:\n  contract: string\n  entityId: string\n  evidenceScope: string\n  criterion: string\n  evidence: string\nreturns: string\n---\nApply the supplied contract to the criterion and evidence for entityId. Return only the requested result label.\n`,
      'judge/skills/judge-against-criteria/SKILL.md': candidate,
    },
    inputs: {
      contract: fixture.contract.rule,
      entityId: fixture.entityId,
      evidenceScope: fixture.evidenceScope,
      criterion: fixture.criterion,
      evidence: fixture.evidence,
    },
    expected: decisions.get(fixture.id),
  },
  diagnostic: {fixture_id: fixture.id, source_artifact: plan.contexts, source_sha256: plan.pins[plan.contexts],
    sampled: false, training_admission: false, split_reason: 'project-authored paired skill diagnostic'},
}));
for (const record of records) {
  collector.validateFocusedRecord(record);
  const node = await programs.prepareProgramNode(record);
  if (!node.skills?.listing?.includes('judge-against-criteria') || !node.skills.listing.includes('Boolean eligibility') ||
      node.skills.listing.includes(candidateInstructions) || !node.skills.documents?.['skills.judge-against-criteria']?.startsWith(candidateInstructions))
    throw new Error(`native runtime skill listing/body binding mismatch for ${record.id}`);
  const visibleTask = JSON.stringify({root:record.semantics.files['judge.nl'], inputs:record.semantics.inputs});
  for (const row of oracle.decisions) if (visibleTask.includes(row.reason)) throw new Error('host oracle rationale leaked into task input');
}
if (mode === '--preflight') {
  console.log(JSON.stringify({status:'preflight_passed', provider_calls:0, split:'diagnostic', corpus_publication:false,
    cases:records.length, arms:arms.length, root_executions:records.length*arms.length, max_requests:plan.max_total_requests,
    runtime_manifest_sha256:plan.runtime_manifest_sha256, plan_sha256:hash(planBytes)}));
  process.exit(0);
}
if (!plan.output || !plan.endpoint && !plan.provider) throw new Error('approved plan must provide output and provider route');
if (plan.output.includes('training') || plan.output.includes('corpora')) throw new Error('diagnostic output cannot target corpus paths');
const output = resolve(plan.output); await mkdir(output, {recursive:true});
const systemBase = prompts.TOOLS_PROMPT;
const results = [];
for (const arm of arms) for (let index = 0; index < records.length; index++) {
  const caseDir = join(output, arm, String(index).padStart(2,'0'));
  await mkdir(caseDir, {recursive:true});
  const resultPath = join(caseDir, 'result.json');
  try {
    const old = JSON.parse(await readFile(resultPath, 'utf8'));
    if (old.plan_sha256 !== hash(planBytes) || old.arm !== arm || old.program_id !== records[index].id)
      throw new Error('existing result belongs to a different plan');
    results.push(old); continue;
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  const systemPrompt = systemBase + (arm === 'instructed'
    ? '\nBefore solving, read an applicable bound skill with read_code("skills.<name>"). Use its procedure when helpful; do not read unrelated skills.' : '');
  const options = {
    endpoint: plan.endpoint, ...(plan.provider ? {provider:plan.provider, piOptions:plan.pi_options??{}} : {}),
    modelId: plan.model, systemPrompt, contextTokens: plan.context_tokens, rootSeed: plan.root_seed + index,
    temperature: 0, maxTurns: plan.max_turns, maxModelRequests: plan.max_requests_per_execution,
    modelConcurrency: 1, collectionRole: 'student', collectionRoleLabel: 'student_skill_diagnostic',
    textNeuraleseEmulation: plan.text_neuralese_emulation === true,
    executionPlans: plan.execution_plans === true, executionPlanTokens: plan.execution_plan_tokens,
    request: {max_tokens: plan.max_output_tokens}, jobs: join(caseDir,'jobs'), output: join(caseDir,'unused.jsonl'),
    workers: 1, transportRetries: 0,
  };
  await mkdir(options.jobs, {recursive:true});
  const runner = collector.nativeJobRunner(options);
  const expected = collector.expectedProvenance(records[index], options);
  let row, error;
  try { row = await runner({index, record: structuredClone(records[index])}, expected); }
  catch (cause) { error = {name:cause?.name, code:cause?.code, message:String(cause?.message??cause).slice(0,1000)}; }
  let traceEvents = [];
  try { traceEvents = (await readFile(join(options.jobs, `${String(index).padStart(6,'0')}-${collector.recordDigest(records[index]).slice(0,16)}.trace.jsonl`),'utf8'))
    .split('\n').filter(Boolean).map(JSON.parse); } catch {}
  const skillEvents = traceEvents.filter(event => event.kind === 'skill_use' && event.skill_name === 'judge-against-criteria');
  const reads = (row?.trajectory ?? []).flatMap(turn => (turn.assistant?.calls ?? [])
    .filter(call => call.tool === 'read_code' && String(call.arguments?.name??'').startsWith('skills.'))
    .map(call => ({target:call.arguments.name, arguments:call.arguments})));
  const result = {schema:'natlang.criterion-grounded-diagnostic-result/1', plan_sha256:hash(planBytes), arm,
    program_id:records[index].id, fixture_index:index, split:'diagnostic', training_admission:false,
    expected:decisions.get(contexts.cases[index].id), accepted:row?.outcome?.accepted===true,
    actual:row?.outcome?.value, skill_events:skillEvents, skill_reads:reads, row, error,
    provider_request_telemetry:row?.request_telemetry??null, corpus_publication:false};
  await writeFile(resultPath, JSON.stringify(result)+'\n', {flag:'wx'}); results.push(result);
}
const summary = {schema:'natlang.criterion-grounded-diagnostic-summary/1', plan_sha256:hash(planBytes),
  arms:Object.fromEntries(arms.map(arm=>{const rows=results.filter(row=>row.arm===arm);return [arm,{cases:rows.length,
    accepted:rows.filter(row=>row.accepted).length, body_reads:rows.filter(row=>row.skill_events.some(event=>event.phase==='body_read')).length,
    offered:rows.filter(row=>row.skill_events.some(event=>event.phase==='offered')).length,
    failures:rows.filter(row=>row.error).length}]})), all_cases_preserved:true, training_admission:false, corpus_publication:false};
await writeFile(join(output,'summary.json'), JSON.stringify(summary,null,2)+'\n', {flag:'wx'});
console.log(JSON.stringify(summary));
