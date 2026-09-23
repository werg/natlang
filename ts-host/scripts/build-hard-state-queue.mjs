#!/usr/bin/env node
/** Select exact student-visited prefixes for replay-and-handoff collection. */
import { createHash } from 'node:crypto';
import { readFile, writeFile, rename } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

const canonical = value => Array.isArray(value) ? `[${value.map(canonical).join(',')}]` :
  value && typeof value === 'object' ? `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(',')}}` : JSON.stringify(value);
const digest = value => createHash('sha256').update(typeof value === 'string' || Buffer.isBuffer(value) ? value : canonical(value)).digest('hex');
async function writeImmutable(path, content) {
  try {
    if (await readFile(path, 'utf8') === content) return;
    throw new Error(`refusing to overwrite changed hard-state artifact: ${path}`);
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
  }
  const temporary = `${path}.pending-${process.pid}`;
  await writeFile(temporary, content);
  await rename(temporary, path);
}

function failedEvalIndex(row) {
  const ledger = row.outcome?.action_ledger ?? [];
  let cursor = 0;
  for (let index = 0; index < row.trajectory.length; index++) {
    for (const call of row.trajectory[index].assistant?.calls ?? []) {
      let found = -1;
      for (let at = cursor; at < ledger.length; at++) {
        if (ledger[at].name === (call.source_tool ?? call.tool) &&
            canonical(ledger[at].arguments) === canonical(call.arguments)) { found = at; break; }
      }
      if (found < 0) continue;
      cursor = found + 1;
      if (call.tool === 'eval' && ['rejected', 'refused', 'error'].includes(ledger[found].outcome))
        return index;
    }
  }
  return -1;
}

export function selectHardState(row) {
  if (row.version !== 'natlang.teacher_trajectory.native/1' || row.outcome?.accepted !== false ||
      !Array.isArray(row.trajectory) || !row.trajectory.length) return { reason: 'not_failed_student_trajectory' };
  if (row.provenance?.collection_role !== 'student') return { reason: 'not_student' };
  if ((row.outcome.host_events ?? []).some(event => event.event?.operation !== 'typescript.eval'))
    return { reason: 'nonreplayable_host_effect' };
  const failedAt = failedEvalIndex(row);
  const handoffAt = failedAt >= 0 && failedAt + 1 < row.trajectory.length ? failedAt + 1 :
    row.trajectory.length - 1;
  const target = row.trajectory[handoffAt];
  if (!target?.request_sha256 || !row.trajectory.slice(0, handoffAt)
      .every(turn => turn.request_sha256 && turn.model_response))
    return { reason: 'missing_exact_replay' };
  const program = row.task?.program_ir;
  if (!program?.id) return { reason: 'missing_program' };
  return { handoff: { version: 'natlang.hard_state/1', id: program.id,
    program_ir_sha256: row.provenance.program_ir_sha256,
    student_trajectory_id: row.id, student_trajectory_sha256: digest(row),
    student_provenance: row.provenance, handoff_at: handoffAt,
    target_request_sha256: target.request_sha256,
    prefix: row.trajectory.slice(0, handoffAt).map(turn => ({ request_sha256: turn.request_sha256,
      response: turn.model_response })),
    failure: { kind: failedAt >= 0 ? 'scope_eval' : 'unsolved',
      failed_decision: failedAt >= 0 ? failedAt : null,
      student_outcome: row.outcome.status, student_detail: row.outcome.detail,
      scope_failures: row.outcome.scope_failures ?? [] } }, program };
}

export async function buildHardStateQueue(input, output) {
  const rows = (await readFile(input, 'utf8')).split(/\r?\n/).filter(Boolean).map(JSON.parse);
  const selected = [], rejected = {}, ids = new Set();
  for (const row of rows) {
    const result = selectHardState(row);
    if (!result.handoff) { rejected[result.reason] = (rejected[result.reason] ?? 0) + 1; continue; }
    if (ids.has(result.handoff.id)) { rejected.duplicate_program = (rejected.duplicate_program ?? 0) + 1; continue; }
    ids.add(result.handoff.id);
    selected.push(result);
  }
  await writeImmutable(output, selected.map(item => JSON.stringify(item.handoff)).join('\n') + (selected.length ? '\n' : ''));
  await writeImmutable(`${output}.programs.jsonl`, selected.map(item => JSON.stringify(item.program)).join('\n') + (selected.length ? '\n' : ''));
  const manifest = { version: 'natlang.hard_state_queue/1', input_sha256: digest(await readFile(input)),
    count: selected.length, rejected,
    queue_sha256: digest(await readFile(output)), programs_sha256: digest(await readFile(`${output}.programs.jsonl`)) };
  await writeImmutable(`${output}.manifest.json`, JSON.stringify(manifest, null, 2) + '\n');
  return manifest;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [input, output] = process.argv.slice(2);
  if (!input || !output) throw new Error('usage: build-hard-state-queue.mjs STUDENT-TRAJECTORIES QUEUE.jsonl');
  console.log(JSON.stringify(await buildHardStateQueue(input, output)));
}
