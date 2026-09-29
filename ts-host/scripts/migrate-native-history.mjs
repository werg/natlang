#!/usr/bin/env node
/**
 * Offline, fail-closed pilot for replaying historically accepted native teacher rows whose only current
 * admission problem is an obsolete runtime outcome. This emits review candidates; it never promotes them.
 */
import { createHash } from 'node:crypto';
import { readdir, readFile, mkdir, open } from 'node:fs/promises';
import { basename, join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { admitRow } from '../dist/teacher/curriculum.js';
import { defaultToolSurfaceHash, executeProgram, programRunId, trajectoryTurn } from '../dist/teacher/collector.js';
import { recorded, callNumbers, observed, REPLAY_END } from '../dist/teacher/replay.js';
import { openingText } from '../dist/teacher/opening.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';
import { sourceConversionDigest } from '../dist/teacher/source-conversion.js';

const { values, positionals } = parseArgs({ allowPositionals: true, options: {
  repo: { type: 'string', default: process.cwd() }, out: { type: 'string' }, max: { type: 'string', default: '10' },
  'timeout-ms': { type: 'string', default: '30000' },
} });
const repo = resolve(values.repo), out = resolve(values.out ?? join(repo, 'runs/native-history-migration-20260930'));
const max = Math.max(0, Math.min(10, Number(values.max))), timeoutMs = Math.max(1, Math.min(30000, Number(values['timeout-ms'])));
if (!Number.isInteger(max) || !Number.isFinite(timeoutMs)) throw new Error('--max must be an integer 0..10 and --timeout-ms a number 1..30000');
const canonical = value => Array.isArray(value) ? `[${value.map(canonical).join(',')}]` :
  value && typeof value === 'object' ? `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(',')}}` : JSON.stringify(value);
const sha = value => createHash('sha256').update(value).digest('hex');
const digest = value => sourceConversionDigest(value);
const stableEqual = (a, b) => canonical(a) === canonical(b);
const rows = [], seenFiles = [];
const hostRoot = join(repo, 'ts-host');
async function compiledRuntimeIdentity() {
  const nativeDir = join(hostRoot, 'dist/native'), names = (await readdir(nativeDir)).filter(name => name.endsWith('.js'))
    .map(name => `dist/native/${name}`);
  names.push('dist/teacher/collector.js', 'dist/teacher/curriculum.js', 'dist/teacher/curriculum-policy.js',
    'dist/teacher/source-conversion.js', 'dist/teacher/replay.js', 'dist/teacher/opening.js');
  const files = {};
  for (const name of [...new Set(names)].sort()) files[name] = sha(await readFile(join(hostRoot, name)));
  return { files, aggregate_sha256: sha(JSON.stringify(files)) };
}
const compiledIdentityAtStart = await compiledRuntimeIdentity();
const toolSurfaceAtStart = await defaultToolSurfaceHash(hostRoot);

async function* resultFiles(dir) {
  let entries;
  try { entries = await readdir(dir, { withFileTypes: true }); } catch { return; }
  for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (!['node_modules', '.git', 'runtime', 'native-history-migration-20260930'].includes(entry.name))
        yield* resultFiles(path);
    } else if (entry.isFile() && entry.name.endsWith('.result.json')) yield path;
  }
}

function obsoleteOnly(row) {
  try {
    if (row?.version !== 'natlang.teacher_trajectory.native/1' || !row.id || row.outcome?.accepted !== true ||
        row.provenance?.collection_role !== 'teacher' || !Array.isArray(row.trajectory)) return null;
    const record = row.task?.program_ir, generation = record?.generation?.generator;
    if (record?.version !== 'natlang.program/2' || record.split !== 'train' || !record.curriculum ||
        record.external_source || record.handoff || row.handoff || row.provenance?.variant ||
        row.provenance?.source_conversion || /adapter|source|world/i.test(String(generation ?? '')) ||
        /^(?:world|textworld|scienceworld|alfworld)_/i.test(String(record.curriculum.family ?? ''))) return null;
    const admission = admitRow(row);
    if (!admission.reasons.length || !admission.reasons.every(reason => reason.startsWith('obsolete_outcome:'))) return null;
    if (row.provenance?.program_ir_sha256 !== digest(record)) return null;
    return admission.reasons;
  } catch { return null; }
}

for (const root of [join(repo, 'runs'), join(repo, 'data/teacher')]) {
  for await (const path of resultFiles(root)) {
    let bytes, row;
    try { bytes = await readFile(path); row = JSON.parse(bytes); } catch { continue; }
    seenFiles.push(path);
    const reasons = obsoleteOnly(row);
    if (!reasons) continue;
    rows.push({ path, bytes, row, reasons, sha256: sha(bytes) });
  }
}
// One source artifact per trajectory id. Prefer the shortest trace and then a stable path ordering.
const unique = new Map();
for (const candidate of rows) {
  const old = unique.get(candidate.row.id);
  if (!old || candidate.row.trajectory.length < old.row.trajectory.length ||
      (candidate.row.trajectory.length === old.row.trajectory.length && candidate.path < old.path)) unique.set(candidate.row.id, candidate);
}
const simple = [...unique.values()].filter(item => String(item.row.task.program_ir.curriculum.family ?? '').startsWith('simple_') &&
  item.row.trajectory.length <= 20).sort((a, b) => a.row.trajectory.length - b.row.trajectory.length || a.row.id.localeCompare(b.row.id));

function strictReplayDriver(trajectory) {
  const calls = callNumbers(trajectory), oldOpenings = [], openingCalls = new Map();
  trajectory.forEach((turn, index) => {
    const opening = openingText(turn.context);
    if (!openingCalls.has(opening)) { openingCalls.set(opening, calls[index]); oldOpenings.push(opening); }
    else if (openingCalls.get(opening) !== calls[index]) throw Error('ambiguous_recorded_opening_ownership');
  });
  const invocationOwners = new Map();
  trajectory.forEach((turn, index) => {
    const invocation = turn.invocation_id;
    if (!invocation) return;
    const opening = openingText(turn.context), owners = invocationOwners.get(opening) ?? new Set();
    owners.add(String(invocation)); invocationOwners.set(opening, owners);
  });
  if ([...invocationOwners.values()].some(owners => owners.size > 1)) throw Error('ambiguous_identical_opening_invocations');
  const responseByPlace = new Map(), originalPlace = new Map();
  trajectory.forEach((turn, index) => {
    const call = calls[index], nth = calls.slice(0, index).filter(other => other === call).length;
    responseByPlace.set(`${call}:${nth}`, recorded(turn)); originalPlace.set(`${call}:${nth}`, index);
  });
  const assigned = new Map(), requestCalls = new Map(), claimed = new Set(), perCall = new Map();
  const allTurns = [], places = [], driverErrors = [];
  const match = opening => {
    if (requestCalls.has(opening)) return requestCalls.get(opening);
    const exact = oldOpenings.indexOf(opening);
    let call;
    if (exact >= 0) call = openingCalls.get(oldOpenings[exact]);
    else {
      const words = value => new Set(value.split(/\W+/u).filter(Boolean));
      const actual = words(opening), scores = oldOpenings.map((candidate, at) => {
        const index = openingCalls.get(candidate);
        if (claimed.has(index)) return { index, score: -1 };
        const other = words(candidate), shared = [...actual].filter(word => other.has(word)).length;
        return { index, score: shared / Math.max(1, new Set([...actual, ...other]).size) };
      }).sort((a, b) => b.score - a.score);
      const [best, second] = scores;
      if (!best || best.score < 0.8 || best.score - (second?.score ?? 0) < 0.15)
        throw Error('fresh_opening_has_no_unique_high_confidence_owner');
      call = best.index;
    }
    if (claimed.has(call)) throw Error('multiple_fresh_openings_claim_same_recorded_call');
    claimed.add(call); requestCalls.set(opening, call); return call;
  };
  const driver = async request => {
    let call;
    try { call = match(openingText(request.messages)); }
    catch (error) {
      driverErrors.push(String(error?.message ?? error));
      const response = structuredClone(REPLAY_END);
      allTurns.push(trajectoryTurn(request, response)); places.push(null); return response;
    }
    const nth = perCall.get(call) ?? 0; perCall.set(call, nth + 1);
    const key = `${call}:${nth}`, response = responseByPlace.get(key);
    if (!response || assigned.has(key)) {
      driverErrors.push('fresh_runtime_requested_an_unrecorded_or_duplicate_turn');
      const fallback = structuredClone(REPLAY_END);
      allTurns.push(trajectoryTurn(request, fallback)); places.push(null); return fallback;
    }
    assigned.set(key, true);
    allTurns.push(trajectoryTurn(request, response)); places.push({ call, nth, original_index: originalPlace.get(key) });
    return structuredClone(response);
  };
  return { driver, allTurns, places, assigned, responseByPlace, driverErrors, expectedTurns: trajectory.length };
}

const report = { version: 'natlang.native_history_migration_pilot/1', scan: { result_files: seenFiles.length,
  eligible_artifacts: rows.length, unique_rows: unique.size, simple_unique_candidates: simple.length },
  policy: { criteria: 'current native IR/train/explicit teacher/original accepted/admission reasons only obsolete_outcome:*; no external source, handoff or variant; simple_* only for replay',
    max_cases: max, per_case_timeout_ms: timeoutMs, replay_sequentially: true, promotion: false },
  candidates: [], counts: {} };
report.replay_identity = { runtime: 'typescript-native', runtime_contract_version: 17,
  tool_surface_sha256: toolSurfaceAtStart,
  compiled_runtime_policy: compiledIdentityAtStart,
  system_prompt_sha256: sha(TOOLS_PROMPT), context_tokens_source: 'each original row provenance',
  max_turns_source: 'each original row provenance', seed_source: 'each original row provenance',
  replay_mode: 'offline recorded model actions; no provider calls' };
const migrated = [];
const count = key => report.counts[key] = (report.counts[key] ?? 0) + 1;

async function processCandidate(candidate) {
  const { row, path, bytes, reasons, sha256: sourceSha } = candidate, record = row.task.program_ir;
  const audit = { id: row.id, source_path: path, source_sha256: sourceSha, program_id: record.id,
    family: record.curriculum.family, original_reasons: reasons, old_policy: { admitted: false, reasons },
    checks: [], status: 'rejected', reasons: [] };
  const reject = reason => { audit.reasons.push(reason); audit.status = 'rejected'; return audit; };
  const fileSequence = basename(path).match(/^(\d+)-.*\.result\.json$/)?.[1];
  if (!fileSequence) return reject('original_run_index_unavailable');
  const runIndex = Number(fileSequence), runId = programRunId(runIndex, row.provenance);
  const recoveredId = `teacher-program:${digest([record.id, row.provenance.model, runId]).slice(0, 20)}`;
  if (recoveredId !== row.id) return reject('original_run_identity_did_not_reconstruct_from_saved_index');
  audit.checks.push('source_row_and_original_run_identity_verified');
  const recordDigest = digest(record), expectedDigest = digest(record.semantics.expected);
  const trajectoryDigest = digest(row.trajectory), outcomeDigest = digest(row.outcome);
  let replay;
  try { replay = strictReplayDriver(row.trajectory); }
  catch (error) { return reject(String(error?.message ?? error)); }
  const controller = new AbortController(); let timedOut = false;
  const timeout = setTimeout(() => { timedOut = true; controller.abort(new Error('per_case_timeout')); }, timeoutMs);
  let run;
  try {
    run = await executeProgram(record, replay.driver, { systemPrompt: TOOLS_PROMPT,
      contextTokens: row.provenance.context_tokens ?? 16384, maxTurns: row.provenance.max_turns,
      rootSeed: row.provenance.seed_policy?.root ?? 909, runId, signal: controller.signal });
  } catch (error) {
    if (timedOut) return reject('per_case_timeout');
    return reject(`replay_error:${String(error?.message ?? error).split('\n')[0].slice(0, 120)}`);
  } finally { clearTimeout(timeout); }
  if (timedOut) return reject('per_case_timeout');
  if (replay.driverErrors.length) return reject(`strict_replay_driver:${replay.driverErrors[0]}`);
  if (replay.assigned.size !== replay.expectedTurns || replay.allTurns.length !== replay.expectedTurns)
    return reject('not_all_recorded_model_actions_replayed_exactly_once');
  audit.checks.push('all_recorded_actions_replayed_once_with_unique_call_ownership');
  const oldCalls = callNumbers(row.trajectory);
  const newByOriginal = new Map();
  replay.places.forEach((place, freshIndex) => { if (place) newByOriginal.set(place.original_index, replay.allTurns[freshIndex]); });
  if (newByOriginal.size !== row.trajectory.length) return reject('call_ownership_mapping_incomplete');
  const toolOutputChanges = [];
  for (let index = 0; index < row.trajectory.length; index++) {
    const fresh = newByOriginal.get(index), old = row.trajectory[index];
    if (!fresh) return reject(`call_ownership_missing_recorded_turn:${index}`);
    const oldObserved = observed(old.context), newObserved = observed(fresh.context);
    if (!stableEqual(oldObserved, newObserved)) toolOutputChanges.push({ index, call: oldCalls[index],
      old_sha256: digest(oldObserved), new_sha256: digest(newObserved) });
  }
  audit.tool_output_changes = toolOutputChanges;
  audit.checks.push('source_program_ir_and_gold_unchanged');
  if (toolOutputChanges.length) return reject(`observed_tool_outputs_changed:${toolOutputChanges.length}`);
  audit.checks.push('all_normalized_tool_outputs_unchanged');
  if (!run.outcome.accepted || run.outcome.status !== 'done')
    return reject(`fresh_runtime_result_not_accepted:${run.outcome.rejection_reasons?.join(',') ?? run.outcome.status}`);
  if (!stableEqual(row.outcome.value, run.outcome.value)) return reject('fresh_final_value_differs_from_original');
  if (!stableEqual(row.outcome.files ?? null, run.outcome.files ?? null)) return reject('fresh_output_files_differ_from_original');
  if (!stableEqual(row.outcome.effects ?? null, run.outcome.effects ?? null)) return reject('fresh_effects_differ_from_original');
  if (!stableEqual(record.semantics.expected, run.outcome.derived_expected ?? record.semantics.expected) ||
      expectedDigest !== digest(record.semantics.expected)) return reject('gold_contract_changed_or_fresh_expected_differs');
  audit.checks.push('accepted_result_value_files_effects_and_gold_match_original');
  const freshTrajectory = replay.allTurns;
  const traceSha = digest(run.trace);
  const migrationLineage = { version: 'natlang.history_migration/1', source_path: path, source_sha256: sourceSha,
        source_row_sha256: sha(bytes), original_program_ir_sha256: recordDigest,
        original_outcome_sha256: outcomeDigest, original_trajectory_sha256: trajectoryDigest,
        replay_outcome_sha256: digest(run.outcome), replay_trajectory_sha256: digest(freshTrajectory),
        replay_trace_sha256: traceSha, run_index: runIndex, run_id: runId, tool_output_changes: toolOutputChanges.length,
        original_provenance: row.provenance, current_replay_identity: report.replay_identity };
  const preMigrationRow = { ...row, outcome: run.outcome, trajectory: freshTrajectory,
    provenance: { ...row.provenance, trace_sha256: traceSha } };
  const beforeHold = admitRow(preMigrationRow);
  audit.pre_migration_policy = { admitted: beforeHold.admitted, reasons: beforeHold.reasons };
  if (!beforeHold.admitted) return reject(`fresh_replay_admission_rejected:${beforeHold.reasons.join(',')}`);
  const migratedRow = { ...preMigrationRow,
    provenance: { collection_role: 'migration_candidate', history_migration: migrationLineage } };
  const after = admitRow(migratedRow);
  audit.new_policy = { admitted: after.admitted, reasons: after.reasons };
  if (after.admitted || after.reasons.length !== 1 || after.reasons[0] !== 'history_migration_review_pending')
    return reject(`migration_candidate_not_held_exactly:${after.reasons.join(',')}`);
  audit.checks.push('migration_candidate_blocked_pending_review');
  if (record.id !== migratedRow.task.program_ir.id || !stableEqual(row.task, migratedRow.task) ||
      row.id !== migratedRow.id || !stableEqual(row.outcome.value, migratedRow.outcome.value))
    return reject('original_identity_task_or_gold_not_preserved');
  audit.checks.push('pre_hold_admission_accepts_and_candidate_remains_held');
  audit.status = 'review_candidate_held';
  migrated.push(migratedRow);
  return audit;
}

for (const candidate of simple.slice(0, max)) {
  // Sequential by design: fresh runtime state is isolated and each case has its own 30-second ceiling.
  const result = await processCandidate(candidate);
  report.candidates.push(result); count(result.status);
  if (result.reasons?.some(reason => reason.startsWith('new_admission_rejected:'))) count('new_admission_rejections');
  if (result.reasons?.some(reason => reason.includes('tool_output'))) count('tool_output_related_rejections');
}
const candidatePath = join(out, 'review-candidates.jsonl'), auditPath = join(out, 'audit.jsonl');
const compiledIdentityAtEnd = await compiledRuntimeIdentity();
const toolSurfaceAtEnd = await defaultToolSurfaceHash(hostRoot);
if (!stableEqual(compiledIdentityAtStart, compiledIdentityAtEnd) || toolSurfaceAtStart !== toolSurfaceAtEnd)
  throw new Error('compiled runtime, policy, or source tool files changed during the migration pilot; refusing to write results');
const priorPath = join(repo, 'runs/native-history-migration-20260930/review-candidates.jsonl');
let priorBytes = Buffer.alloc(0), priorRows = [];
try {
  priorBytes = await readFile(priorPath);
  for (const line of priorBytes.toString('utf8').split(/\r?\n/).filter(Boolean)) {
    const prior = JSON.parse(line), policy = admitRow(prior);
    priorRows.push({ id: prior.id, row_sha256: digest(prior), admitted: policy.admitted, reasons: policy.reasons });
  }
} catch { /* No earlier pilot output is a valid first-run state. */ }
report.prior_pilot_reconciliation = { source_path: priorPath, source_sha256: priorBytes.length ? sha(priorBytes) : null,
  rows: priorRows, rows_held_pending_review: priorRows.filter(row => !row.admitted &&
    row.reasons.length === 1 && row.reasons[0] === 'history_migration_review_pending').length };
await mkdir(out, { recursive: true });
const candidateBytes = migrated.map(row => JSON.stringify(row)).join('\n') + (migrated.length ? '\n' : '');
const auditBytes = report.candidates.map(row => JSON.stringify(row)).join('\n') + (report.candidates.length ? '\n' : '');
const writeNew = async (path, data) => { const file = await open(path, 'wx'); try { await file.writeFile(data); await file.sync(); } finally { await file.close(); } };
await writeNew(candidatePath, candidateBytes);
await writeNew(auditPath, auditBytes);
report.output = { candidates: candidatePath, audit: auditPath, candidate_sha256: sha(candidateBytes), audit_sha256: sha(auditBytes),
  candidate_rows: migrated.length, audit_rows: report.candidates.length };
await writeNew(join(out, 'manifest.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ ...report, candidates: report.candidates.map(({ id, program_id, family, status, reasons, new_policy, checks, tool_output_changes }) =>
  ({ id, family, status, reasons, new_policy, checks, tool_output_changes: tool_output_changes?.length })) }, null, 2));
