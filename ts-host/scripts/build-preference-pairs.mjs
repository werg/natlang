#!/usr/bin/env node
/** Preserve only same-request, oracle-supported preference candidates. */
import { createHash } from 'node:crypto';
import { readFile, writeFile, rename } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const rows = async path => (await readFile(path, 'utf8')).split(/\r?\n/).filter(Boolean).map(JSON.parse);
async function immutable(path, content) {
  try {
    if (await readFile(path, 'utf8') === content) return;
    throw new Error(`refusing to overwrite changed preference artifact: ${path}`);
  } catch (error) { if (error?.code !== 'ENOENT') throw error; }
  const pending = `${path}.pending-${process.pid}`;
  await writeFile(pending, content);
  await rename(pending, path);
}
function target(assistant, index) {
  const response = { role: 'assistant', content: assistant.content ?? '' };
  if (assistant.calls?.length) response.tool_calls = assistant.calls.map((call, offset) => ({
    id: `pair_${index}_${offset}`, type: 'function', function: { name: call.source_tool ?? call.tool,
      arguments: JSON.stringify(call.arguments ?? {}) } }));
  return response;
}
function actionSignature(message) {
  return JSON.stringify({ content: message.content ?? '',
    calls: (message.tool_calls ?? []).map(call => call.function) });
}
function linkedFailure(student, index) {
  const decision = student.trajectory[index];
  if (!decision.assistant?.calls?.length) return 'premature_reply';
  const ledger = student.outcome?.action_ledger ?? [];
  return decision.assistant.calls.some(call => ledger.some(event =>
    event.name === (call.source_tool ?? call.tool) && JSON.stringify(event.arguments) === JSON.stringify(call.arguments) &&
    ['rejected', 'refused', 'error'].includes(event.outcome))) ? 'failed_action' : null;
}
export function preferencePairs(studentRows, teacherRows, teacherTurns) {
  const students = new Map(studentRows.map(row => [row.id, row]));
  const turns = new Map(teacherTurns.map(row => [`${row.source_ref?.trajectory_id}:${row.decision?.index}`, row]));
  const pairs = [], excluded = {};
  for (const teacher of teacherRows) {
    const handoff = teacher.handoff;
    if (!handoff || teacher.outcome?.accepted !== true) continue;
    const student = students.get(handoff.student_trajectory_id);
    const index = handoff.handoff_at;
    const chosen = turns.get(`${teacher.id}:${index}`);
    const bad = student?.trajectory?.[index], good = teacher.trajectory?.[index];
    let reason = null;
    if (!student || student.outcome?.accepted !== false) reason = 'missing_failed_student';
    else if (!bad || !good || bad.request_sha256 !== good.request_sha256) reason = 'different_request';
    else if (bad.phase !== 'action' || good.phase !== 'action') reason = 'not_action_decisions';
    else if (!chosen?.training_admission?.approved) reason = 'unapproved_teacher_decision';
    else if (!(reason = linkedFailure(student, index))) reason = 'rejected_action_not_locally_failed';
    if (reason && !['premature_reply', 'failed_action'].includes(reason)) {
      excluded[reason] = (excluded[reason] ?? 0) + 1; continue;
    }
    const rejected = target(bad.assistant, index);
    if (actionSignature(rejected) === actionSignature(chosen.target)) {
      excluded.identical_decisions = (excluded.identical_decisions ?? 0) + 1; continue;
    }
    pairs.push({ version: 'natlang.preference_pair.native/1',
      id: `${teacher.id}:preference:${index}`, program_id: chosen.program_id,
      source_groups: chosen.source_groups, split: chosen.split ?? null,
      request_sha256: bad.request_sha256, messages: chosen.messages, tools: chosen.tools,
      chosen: chosen.target, rejected,
      evidence: { kind: reason, student_trajectory_id: student.id,
        teacher_trajectory_id: teacher.id, student_final_accepted: false,
        teacher_final_accepted: true, teacher_decision_training_approved: true } });
  }
  return { pairs, excluded };
}
export async function buildPreferencePairs(studentPath, teacherPath, turnsPath, output) {
  const result = preferencePairs(await rows(studentPath), await rows(teacherPath), await rows(turnsPath));
  const data = result.pairs.map(row => JSON.stringify(row)).join('\n') + (result.pairs.length ? '\n' : '');
  await immutable(output, data);
  const manifest = { version: 'natlang.preference_pairs/1', count: result.pairs.length,
    excluded: result.excluded, student_sha256: digest(await readFile(studentPath)),
    teacher_sha256: digest(await readFile(teacherPath)), turns_sha256: digest(await readFile(turnsPath)),
    output_sha256: digest(await readFile(output)) };
  await immutable(`${output}.manifest.json`, JSON.stringify(manifest, null, 2) + '\n');
  return manifest;
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [student, teacher, turns, output] = process.argv.slice(2);
  if (!output) throw new Error('usage: build-preference-pairs.mjs STUDENT-ROWS TEACHER-ROWS TEACHER-TURNS OUT');
  console.log(JSON.stringify(await buildPreferencePairs(student, teacher, turns, output)));
}
