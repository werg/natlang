import assert from 'node:assert/strict';
import { test } from 'node:test';
import ts from 'typescript';
import { checkSequentialNlLoops } from '../dist/compiler/sequential-loops.js';
import { occupancyOf } from '../dist/cli/calls.js';

const check = body => checkSequentialNlLoops(ts.createSourceFile('a.ts', `import classify from './classify.nl';\nimport { nl } from '@natlang/node';\nimport other from './x';\n${body}`, ts.ScriptTarget.ES2022, true));

test('a loop that only awaits one independent nl call is flagged with the Promise.all rewrite', () => {
  for (const body of [
    'async function f(ts: string[]) { const out: string[] = []; for (const t of ts) out.push(await classify(t, "r")); return out; }',
    'async function f(ts: string[]) { const out: string[] = []; for (const t of ts) { out[out.length] = await classify(t); } return out; }',
    'async function f(ts: string[]) { for (const t of ts) { const label = await classify(t.trim); } }',
    'async function f(ts: string[]) { const out: string[] = []; for (const t of ts) out.push(await nl<string>`Summarize t.`(t)); return out; }',
  ]) {
    const found = check(body);
    assert.equal(found.length, 1, body);
    assert.equal(found[0].severity, 'warning');
    assert.match(found[0].message, /Promise\.all/);
  }
});

test('loops with effects, carried state, other work or non-nl calls are left alone', () => {
  for (const body of [
    'async function f(ts: string[]) { let prev = ""; for (const t of ts) { prev = await classify(prev + t); } }',
    'async function f(ts: string[]) { const out: string[] = []; for (const t of ts) out.push(await classify(out.join(t))); }',
    'async function f(ts: string[]) { for (const t of ts) { await classify(t); console.log(t); } }',
    'async function f(ts: string[]) { for (const t of ts) { const x = await classify(save(t)); } }',
    'async function f(ts: string[]) { for (const t of ts) { const x = await other(t); } }',
    'async function f(ts: string[]) { const labels = await Promise.all(ts.map(t => classify(t))); }',
    'async function f(ts: string[]) { for (const t of ts) { if (t) break; const x = await classify(t); } }',
  ]) assert.equal(check(body).length, 0, body);
});

test('occupancy summarises batch fields of model request events', () => {
  const events = [
    { kind: 'model_request', phase: 'end', batch_id: 'b1', batch_size: 3, in_flight: 3, queue_wait_ms: 0, schedule_priority: 'new' },
    { kind: 'model_request', phase: 'end', batch_id: 'b1', batch_size: 3, in_flight: 3, queue_wait_ms: 4, schedule_priority: 'new' },
    { kind: 'model_request', phase: 'end', batch_id: 'b2', batch_size: 1, in_flight: 1, queue_wait_ms: 10, schedule_priority: 'running' },
    { kind: 'model_request', phase: 'end' },
    { kind: 'model_request', phase: 'start', batch_id: 'b9' },
  ];
  const summary = occupancyOf(events);
  assert.equal(summary.requests, 4);
  assert.equal(summary.scheduled, 3);
  assert.equal(summary.batches, 2);
  assert.equal(summary.max_in_flight, 3);
  assert.equal(summary.running_turns, 1);
  assert.equal(summary.queue_wait_ms_p95, 10);
});
