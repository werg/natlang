// `natlang traces good|bad|fix`: tester verdicts become annotations and tune.examples.jsonl root examples.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, existsSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { openCallStore } from '../dist/calls/store.js';
import { hexDigest } from '../dist/native/hash.js';

const bin = new URL('../bin/natlang.mjs', import.meta.url).pathname;
const ref = text => ({ complete: true, hash: hexDigest(text), bytes: Buffer.byteLength(text) });

/** A recorded call of `triage/label-ticket.nl` in program `root`. */
function record(store, id, root, options = {}) {
  const input = JSON.stringify(options.input ?? 'refund please'), output = JSON.stringify(options.output ?? 'billing');
  const instructions = JSON.stringify('Label the ticket.');
  const blobs = new Map([[hexDigest(input), input], [hexDigest(output), output], [hexDigest(instructions), instructions]]);
  store.record({
    version: 'natlang.calls/1', call_id: id, parent_call_id: options.parent ?? null, parent_action_index: options.parent ? 0 : null,
    task_id: `task-${id}`, program_id: null, build_hash: null, program_root: root,
    definition: { id: 'label-id', name: 'label-ticket', source: options.source ?? 'triage/label-ticket.nl', key: 'label-key', interface: 'iface', site: 'named',
      subtype: 'function', params: [{ name: 'ticket', type: 'string' }], returns: 'string', instructions: ref(instructions), types: {} },
    executor: { kind: 'agent', model_id: 'model-a', model_revision: null }, inputs: { ticket: ref(input) }, captures: {}, capture_writes: [],
    output: ref(output), outcome: 'done', detail: '', started_at: '2026-03-04T10:00:00.000Z', ended_at: '2026-03-04T10:00:01.000Z',
    effects: [], folder: null, approach: { evals: [], hash: null }, cost: { model_requests: 1, tokens_in: 1, tokens_out: 1, wall_ms: 1, turns: 1, evals: 1 },
    features: {}, audit_of: null, events: null }, blobs);
}

function setup(...calls) {
  const dir = mkdtempSync(join(tmpdir(), 'natlang-feedback-'));
  const root = join(dir, 'program'), storeRoot = join(dir, 'calls');
  mkdirSync(join(root, 'triage'), { recursive: true });
  writeFileSync(join(root, 'triage', 'label-ticket.nl'), 'Label the ticket.\n');
  const store = openCallStore(storeRoot);
  for (const [id, options] of calls) record(store, id, root, options);
  const run = (...argv) => {
    const result = spawnSync(process.execPath, [bin, 'traces', ...argv, '--store', storeRoot],
      { encoding: 'utf8', env: { ...process.env, NATLANG_FEEDBACK_GROUP: '' } });
    return { code: result.status, out: result.stdout, err: result.stderr };
  };
  const examples = () => { const path = join(root, 'tune.examples.jsonl'); return existsSync(path) ? readFileSync(path, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line)) : []; };
  return { dir, root, store, run, examples, annotations: id => store.annotations(id), cleanup() { try { store.close(); } catch { /* closed */ } rmSync(dir, { recursive: true, force: true }); } };
}

test('good appends the recorded input and output with a per-day group and annotates the call', () => {
  const t = setup(['c/1']);
  try {
    const result = t.run('good', 'c/1');
    assert.equal(result.code, 0, result.err);
    const [example, ...rest] = t.examples();
    assert.equal(rest.length, 0);
    assert.deepEqual({ ...example, group: undefined }, { function: 'triage/label-ticket.nl', input: { ticket: 'refund please' }, expected: 'billing', group: undefined });
    assert.match(example.group, /^feedback-20260304-[0-9a-f]{8}$/);
    assert.deepEqual(t.annotations('c/1').map(item => [item.kind, item.value, item.source]), [['feedback', { verdict: 'good' }, 'cli']]);
    assert.match(result.out, /appended to/);
  } finally { t.cleanup(); }
});

test('fix replaces the expected result of the same function and input, and dedupes identical lines', () => {
  const t = setup(['c/1'], ['c/2']);
  try {
    assert.equal(t.run('good', 'c/1').code, 0);
    assert.equal(t.run('good', 'c/2').code, 0);
    assert.equal(t.examples().length, 1, 'the same function and input is not added twice');
    assert.match(t.run('good', 'c/1').out, /already in/);
    const fixed = t.run('fix', 'c/1', '--expected', '"refunds"');
    assert.equal(fixed.code, 0, fixed.err);
    assert.match(fixed.out, /updated the expected result/);
    const examples = t.examples();
    assert.equal(examples.length, 1);
    assert.equal(examples[0].expected, 'refunds');
    assert.equal(examples[0].input.ticket, 'refund please');
    assert.deepEqual(readdirSync(t.root).filter(name => name.endsWith('.tmp')), [], 'temp file renamed away');
    assert.equal(t.annotations('c/1').length, 3);
  } finally { t.cleanup(); }
});

test('group comes from --group, else NATLANG_FEEDBACK_GROUP, else the per-day default shared by the session', () => {
  const t = setup(['c/1'], ['c/2', { input: 'where is my parcel', output: 'shipping' }], ['c/3', { input: 'login fails', output: 'account' }]);
  try {
    assert.equal(t.run('good', 'c/1', '--group', 'customer-17').code, 0);
    const env = spawnSync(process.execPath, [bin, 'traces', 'good', 'c/2', '--store', join(t.dir, 'calls')], { encoding: 'utf8', env: { ...process.env, NATLANG_FEEDBACK_GROUP: 'session-9' } });
    assert.equal(env.status, 0, env.stderr);
    assert.equal(t.run('good', 'c/3').code, 0);
    const [a, b, c] = t.examples();
    assert.deepEqual([a.group, b.group], ['customer-17', 'session-9']);
    assert.match(c.group, /^feedback-20260304-[0-9a-f]{8}$/);
    const again = setup(['d/1', { input: 'x' }], ['d/2', { input: 'y' }]);
    try { again.run('good', 'd/1'); again.run('good', 'd/2'); const [p, q] = again.examples(); assert.equal(p.group, q.group); assert.equal(p.group, c.group); } finally { again.cleanup(); }
  } finally { t.cleanup(); }
});

test('plain bad annotates only; bad with --expected appends; --dry-run changes nothing', () => {
  const t = setup(['c/1']);
  try {
    const plain = t.run('bad', 'c/1');
    assert.equal(plain.code, 0, plain.err);
    assert.match(plain.out, /no example appended/);
    assert.deepEqual(t.examples(), []);
    assert.equal(t.annotations('c/1')[0].value.verdict, 'bad');
    const dry = t.run('bad', 'c/1', '--expected', '"refunds"', '--dry-run');
    assert.equal(dry.code, 0, dry.err);
    assert.match(dry.out, /"function":"triage\/label-ticket.nl"/);
    assert.deepEqual(t.examples(), []);
    assert.equal(t.annotations('c/1').length, 1, 'dry run annotates nothing');
    assert.equal(t.run('bad', 'c/1', '--expected', '"refunds"').code, 0);
    assert.deepEqual(t.examples().map(item => item.expected), ['refunds']);
    assert.notEqual(t.run('fix', 'c/1').code, 0, 'fix needs --expected');
  } finally { t.cleanup(); }
});

test('a nested call or a call without a program root fails with a reason and still keeps the annotation', () => {
  const t = setup(['p/1'], ['p/2', { parent: 'p/1' }]);
  try {
    const nested = t.run('good', 'p/2');
    assert.equal(nested.code, 1);
    assert.match(nested.err, /annotated p\/2 as good; not added as a tune example: .*ran inside p\/1/);
    assert.deepEqual(t.annotations('p/2').map(item => item.kind), ['feedback']);
    assert.deepEqual(t.examples(), []);
    const rootless = t.run('fix', 'p/1', '--expected', '1', '--program', join(t.dir, 'missing'));
    assert.equal(rootless.code, 1);
    assert.match(rootless.err, /does not exist under/);
    assert.equal(t.annotations('p/1').length, 1);
    const override = t.run('good', 'p/1', '--program', t.root);
    assert.equal(override.code, 0, override.err);
    assert.equal(t.examples().length, 1);
    assert.notEqual(t.run('good', 'nope').code, 0);
  } finally { t.cleanup(); }
});
