import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { collectBatch, defaultSystemPrompt, defaultToolSurfaceHash, nativeJobRunner } from '../dist/teacher/collector.js';
import { failsInPlace, handoffAt, handoffRecord, handoffSites, preferencePair } from '../dist/teacher/handoff.js';
import { materializeNativeRows } from '../dist/teacher/native-materializer.js';
import { handoffTurns } from '../dist/teacher/replay.js';

const record = { version: 'natlang.program/2', id: 'student-hard-case', kind: 'lambda_source',
  source_groups: ['student-hard-case'], split: 'train', semantics: {
    root: 'one.nl', files: { 'one.nl': '---\nargs: {}\nreturns: number\n---\nReturn one.\n' },
    inputs: {}, expected: 1, operation: 'exact' } };
function modelServer(responses) {
  let count = 0;
  const requests = [];
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8');
    request.on('data', chunk => { body += chunk; });
    request.on('end', () => {
      requests.push(JSON.parse(body));
      const calls = responses[Math.min(count++, responses.length - 1)];
      const message = { role: 'assistant', content: '', ...(calls ? { tool_calls: calls.map(([name, args], index) => ({
        id: `call-${count}-${index}`, type: 'function', function: { name, arguments: JSON.stringify(args) } })) } : {}) };
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ choices: [{ message }], usage: { prompt_tokens: 10, completion_tokens: 4 } }));
    });
  });
  return { server, count: () => count, requests };
}
const config = (dir, endpoint, role, surface) => ({ jobs: join(dir, `${role}-jobs`),
  output: join(dir, `${role}.jsonl`), workers: 1, modelId: role, rootSeed: 7,
  systemPrompt: defaultSystemPrompt, contextTokens: 16384,
  toolSurfaceSha256: surface, endpoint, collectionRole: role, maxTurns: 3 });
const options = { systemPrompt: defaultSystemPrompt, contextTokens: 16384, maxTurns: 3, rootSeed: 7 };

test('a teacher takes over a failed run at its failure, and its decision is preferred to the failed one', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'student-handoff-'));
  // The student reads, then fails, then gives up; the teacher, handed the failed turn, returns the answer.
  const student = modelServer([[['eval', { code: '1 + 1' }]], [['eval', { code: 'throw new Error("wrong branch")' }]],
    [['return_result', { status: 'failed', reason: 'stuck' }]]]);
  const teacher = modelServer([[['return_result', { status: 'success', value: 1 }]]]);
  await Promise.all([new Promise(resolve => student.server.listen(0, '127.0.0.1', resolve)),
    new Promise(resolve => teacher.server.listen(0, '127.0.0.1', resolve))]);
  try {
    const surface = await defaultToolSurfaceHash();
    const studentConfig = config(dir, `http://127.0.0.1:${student.server.address().port}`, 'student', surface);
    await collectBatch([{ index: 0, record }], studentConfig, nativeJobRunner(studentConfig));
    const studentRow = JSON.parse((await readFile(studentConfig.output, 'utf8')).trim());
    assert.equal(studentRow.outcome.accepted, false);
    assert.deepEqual(handoffSites(studentRow), [{ index: 1, kind: 'failed_action' }]);
    const handoff = handoffAt(studentRow, { index: 1, kind: 'failed_action' });
    assert.equal(await failsInPlace(studentRow, 1, handoff.rejected, 'failed_action', options, 'check'), null);
    // A response that does not fail in that place is no rejected side.
    assert.equal(await failsInPlace(studentRow, 1, { calls: [['eval', { code: '2' }]] }, 'failed_action', options, 'check'),
      'the action no longer fails');

    const task = handoffRecord(studentRow, handoff);
    assert.equal(task.handoff.source.program_id, record.id);
    const teacherConfig = config(dir, `http://127.0.0.1:${teacher.server.address().port}`, 'teacher', surface);
    await collectBatch([{ index: 0, record: task }], teacherConfig, nativeJobRunner(teacherConfig));
    const repaired = JSON.parse((await readFile(teacherConfig.output, 'utf8')).trim());
    assert.equal(repaired.outcome.accepted, true);
    assert.deepEqual(handoffTurns(repaired.trajectory, task.handoff), { prefix: [0], at: 1 });
    assert.equal(student.count(), 3);
    assert.equal(teacher.count(), 1, 'the student prefix replays without another decode');
    const { turns } = materializeNativeRows([repaired]);
    assert.deepEqual(turns.map(turn => turn.training_admission.approved), [false, true]);
    assert.match(turns[0].training_admission.reason, /student replay prefix/);

    const pair = await preferencePair(repaired, 1, task.handoff.rejected, 'failed_action', { kind: 'handoff' }, options,
      repaired.handoff.run_id);
    assert.equal(pair.program_id, record.id);
    assert.deepEqual(pair.messages, turns[1].messages);
    assert.equal(pair.chosen.target.tool_calls[0].function.name, 'return_result');
    assert.match(pair.rejected.target.tool_calls[0].function.arguments, /wrong branch/);
  } finally {
    await Promise.all([new Promise(resolve => student.server.close(resolve)),
      new Promise(resolve => teacher.server.close(resolve))]);
  }
});

test('replacing the planted failing action is a valid handoff without reinjecting the failure', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'seeded-handoff-'));
  const seeded = { ...record, semantics: { ...record.semantics,
    failure_seed: { kind: 'compile', code: 'while (true) {}' } } };
  const student = modelServer([[['return_result', { status: 'failed', reason: 'stuck' }]]]);
  const teacher = modelServer([[['return_result', { status: 'success', value: 1 }]]]);
  await Promise.all([new Promise(resolve => student.server.listen(0, '127.0.0.1', resolve)),
    new Promise(resolve => teacher.server.listen(0, '127.0.0.1', resolve))]);
  try {
    const surface = await defaultToolSurfaceHash();
    const s = config(dir, `http://127.0.0.1:${student.server.address().port}`, 'student', surface);
    await collectBatch([{ index: 0, record: seeded }], s, nativeJobRunner(s));
    const failed = JSON.parse((await readFile(s.output, 'utf8')).trim());
    assert.equal(failed.outcome.seeded_failure.observed, true);
    const site = handoffSites(failed).find(site => site.index === 0 && site.kind === 'failed_action');
    assert.ok(site);
    const task = handoffRecord(failed, handoffAt(failed, site));
    const t = config(dir, `http://127.0.0.1:${teacher.server.address().port}`, 'teacher', surface);
    await collectBatch([{ index: 0, record: task }], t, nativeJobRunner(t));
    const repaired = JSON.parse((await readFile(t.output, 'utf8')).trim());
    assert.equal(repaired.outcome.accepted, true);
    assert.deepEqual(repaired.outcome.seeded_failure, { observed: false, replaced_by_handoff: true });
    assert.equal(teacher.count(), 1);
    assert.equal(repaired.trajectory[0].assistant.calls[0].tool, 'return_result');
    assert.equal(await failsInPlace(repaired, 0, task.handoff.rejected, 'failed_action', options, 'seed-check'), null);
    const invalid = structuredClone(failed);
    invalid.outcome.checks.seeded_failure_requirement = false;
    assert.deepEqual(handoffSites(invalid), [], 'an unmet exercise prerequisite is not a wrong model result');
    const stale = structuredClone(failed);
    stale.task.program_ir.curriculum = { family: 'inline_type_repair' };
    assert.deepEqual(handoffSites(stale), []);
    await assert.rejects(nativeJobRunner(t)({ index: 0, record: stale.task.program_ir }, {}), /retired curriculum family/);
    assert.equal(await failsInPlace(stale, 0, task.handoff.rejected, 'failed_action', options, 'stale'),
      'the exercise belongs to a retired curriculum family');
  } finally { await Promise.all([new Promise(resolve => student.server.close(resolve)),
    new Promise(resolve => teacher.server.close(resolve))]); }
});

test('recorded turns retain usage that calibrates replay compaction and token budgets', async () => {
  const { recorded } = await import('../dist/teacher/replay.js');
  const response = recorded({ context: [], model_response: { prompt_tokens: 12000, completion_tokens: 900 },
    assistant: { content: '', reasoning: 'Check the current batch.', calls: [
      { tool: 'eval', arguments: { code: 'batch' } },
    ] } });
  assert.equal(response.prompt_tokens, 12000);
  assert.equal(response.completion_tokens, 900);
  assert.deepEqual(response.calls, [['eval', { code: 'batch' }]]);
});
