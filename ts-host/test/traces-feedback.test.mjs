// `natlang traces`: tester verdicts become annotations and examples.json examples (the file natlang test checks and
// natlang tune learns from); call IDs resolve from prefixes; list defaults to the current program; options are checked.
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
    task_id: `task-${id}`, program_id: null, build_hash: null, program_root: options.root === undefined ? root : options.root,
    definition: { id: 'label-id', name: 'label-ticket', source: options.source ?? 'triage/label-ticket.nl', key: 'label-key', interface: 'iface', site: 'named',
      subtype: 'function', params: [{ name: 'ticket', type: 'string' }], returns: 'string', instructions: ref(instructions), types: {} },
    executor: { kind: 'agent', model_id: 'model-a', model_revision: null }, inputs: { ticket: ref(input) }, captures: {}, capture_writes: [],
    output: ref(output), outcome: options.outcome ?? 'done', detail: options.detail ?? '', started_at: options.started ?? '2026-03-04T10:00:00.000Z', ended_at: '2026-03-04T10:00:01.000Z',
    effects: [], folder: null, approach: { evals: [], hash: null }, cost: { model_requests: 1, tokens_in: 1, tokens_out: 1, wall_ms: 1, turns: 1, evals: 1 },
    features: {}, audit_of: null, events: null }, blobs, options.events?.map(event => JSON.stringify(event)).join('\n'));
}

function setup(...calls) {
  const dir = mkdtempSync(join(tmpdir(), 'natlang-feedback-'));
  const root = join(dir, 'program'), storeRoot = join(dir, 'calls');
  mkdirSync(join(root, 'triage'), { recursive: true });
  writeFileSync(join(root, 'triage', 'label-ticket.nl'), 'Label the ticket.\n');
  writeFileSync(join(root, 'natlang.json'), '{}\n');
  const store = openCallStore(storeRoot);
  for (const [id, options] of calls) record(store, id, root, options);
  const runIn = (cwd, ...argv) => {
    const result = spawnSync(process.execPath, [bin, 'traces', ...argv, '--store', storeRoot],
      { cwd, encoding: 'utf8', env: { ...process.env, NATLANG_FEEDBACK_GROUP: '', NATLANG_CONFIG_HOME: dir } });
    return { code: result.status, out: result.stdout, err: result.stderr };
  };
  const run = (...argv) => runIn(dir, ...argv);
  const examples = () => { const path = join(root, 'examples.json'); return existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) : []; };
  return { dir, root, store, run, runIn, examples, annotations: id => store.annotations(id), cleanup() { try { store.close(); } catch { /* closed */ } rmSync(dir, { recursive: true, force: true }); } };
}

test('good appends the recorded input and output with a per-day group and annotates the call', () => {
  const t = setup(['c/1']);
  try {
    const result = t.run('good', 'c/1');
    assert.equal(result.code, 0, result.err);
    const [example, ...rest] = t.examples();
    assert.equal(rest.length, 0);
    assert.deepEqual({ ...example, group: undefined }, { name: 'label-ticket: refund please', function: 'triage/label-ticket.nl', inputs: { ticket: 'refund please' },
      expected: 'billing', group: undefined, producer: 'model-a' });
    assert.match(example.group, /^feedback-20260304-[0-9a-f]{8}$/);
    assert.deepEqual(t.annotations('c/1').map(item => [item.kind, item.value, item.source]), [['feedback', { verdict: 'good' }, 'cli']]);
    assert.match(result.out, /added to .*examples\.json.*natlang test checks it/);
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
    assert.equal(examples[0].inputs.ticket, 'refund please');
    assert.deepEqual(readdirSync(t.root).filter(name => name.endsWith('.tmp')), [], 'temp file renamed away');
    assert.equal(t.annotations('c/1').length, 3);
    const path = join(t.root, 'examples.json');
    writeFileSync(path, JSON.stringify([{ ...t.examples()[0], protected: true }]));
    const held = t.run('fix', 'c/1', '--expected', '"other"');
    assert.notEqual(held.code, 0);
    assert.match(held.err, /protected example/);
    assert.equal(t.examples()[0].expected, 'refunds', 'a protected example is never rewritten');
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
    assert.match(plain.out, /no example added/);
    assert.deepEqual(t.examples(), []);
    assert.equal(t.annotations('c/1')[0].value.verdict, 'bad');
    const dry = t.run('bad', 'c/1', '--expected', '"refunds"', '--dry-run');
    assert.equal(dry.code, 0, dry.err);
    assert.match(dry.out, /"function":"triage\/label-ticket.nl"/);
    assert.deepEqual(t.examples(), []);
    assert.equal(t.annotations('c/1').length, 1, 'dry run annotates nothing');
    assert.equal(t.run('bad', 'c/1', '--expected', '"refunds"').code, 0);
    assert.deepEqual(t.examples().map(item => item.expected), ['refunds']);
    assert.equal(t.run('fix', 'c/1').code, 2, 'fix needs --expected');
  } finally { t.cleanup(); }
});

test('a nested call or a call without a program root fails with a reason and still keeps the annotation', () => {
  const t = setup(['p/1'], ['p/2', { parent: 'p/1' }]);
  try {
    const nested = t.run('good', 'p/2');
    assert.equal(nested.code, 1);
    assert.match(nested.err, /annotated p\/2 as good; not added as an example: .*ran inside p\/1/);
    assert.deepEqual(t.annotations('p/2').map(item => item.kind), ['feedback']);
    assert.deepEqual(t.examples(), []);
    const rootless = t.run('fix', 'p/1', '--expected', '1', '--program', join(t.dir, 'missing'));
    assert.equal(rootless.code, 1);
    assert.match(rootless.err, /does not exist under/);
    assert.equal(t.annotations('p/1').length, 1);
    const override = t.run('good', 'p/1', '--program', t.root);
    assert.equal(override.code, 0, override.err);
    assert.equal(t.examples().length, 1);
    assert.equal(t.run('good', 'nope').code, 2);
  } finally { t.cleanup(); }
});

test('good refuses outputs of models that are neither a tuning teacher nor our own; fix records the tester', () => {
  const t = setup(['c/1'], ['c/2', { input: 'invoice missing', output: 'billing' }]);
  try {
    const sha = 'ab'.repeat(32), models = join(t.dir, 'models');
    mkdirSync(models);
    writeFileSync(join(models, 'pins.json'), JSON.stringify({ [sha]: { identity: { architecture: 'lfm2', checkpoint_sha256: sha,
      embedding_weight_sha256: 'cd'.repeat(32), tokenizer_fingerprint: 'ef'.repeat(32) } } }));
    // Escalation alone does not authorize external tuning labels: tuning defaults
    // to self-teaching. External teachers must be selected explicitly.
    const config = (escalate, teacher) => writeFileSync(join(t.dir, 'config.json'), JSON.stringify({ defaultProfile: 'ours', profiles: {
      ours: { native: { checkpoint: join(models, `${sha}.pt`), sha256: sha }, ...(escalate ? { escalate: 'hosted' } : {}) },
      hosted: { endpoint: 'http://127.0.0.1:9', model: 'model-a' }, other: { endpoint: 'http://127.0.0.1:9', model: 'model-b' } },
      tune: teacher ? { teacher } : {} }));
    const run = (...argv) => spawnSync(process.execPath, [bin, 'traces', ...argv, '--store', join(t.dir, 'calls')],
      { encoding: 'utf8', env: { ...process.env, NATLANG_FEEDBACK_GROUP: '', NATLANG_CONFIG_HOME: t.dir, NATLANG_NATIVE_MODELS: models } });
    config(true);
    const implicit = run('good', 'c/1');
    assert.notEqual(implicit.status, 0);
    assert.match(implicit.stderr, /not a tuning teacher or one of our models/);
    assert.equal(t.examples().length, 0);
    config(true, 'hosted');
    const authorized = run('good', 'c/1');
    assert.equal(authorized.status, 0, authorized.stderr);
    config(false, 'other');
    const refused = run('good', 'c/2');
    assert.notEqual(refused.status, 0);
    assert.match(refused.stderr, /came from model-a, which is not a tuning teacher or one of our models/);
    assert.match(refused.stderr, /natlang traces fix c\/2 --expected JSON/);
    assert.equal(t.examples().length, 1);
    assert.equal(run('fix', 'c/2', '--expected', '"billing"').status, 0);
    assert.deepEqual(t.examples().map(item => item.producer), ['model-a', 'tester']);
  } finally { t.cleanup(); }
});

test('a call recorded without a program root becomes an example of the program the command runs in', () => {
  const t = setup(['n/1', { root: null }]);
  try {
    const outside = t.run('good', 'n/1');
    assert.equal(outside.code, 1);
    assert.match(outside.err, /records no program root and the current directory is not in a program holding triage\/label-ticket.nl; run this in the program's directory or pass --program DIR/);
    const inside = t.runIn(join(t.root, 'triage'), 'fix', 'n/1', '--expected', '"technical"');
    assert.equal(inside.code, 0, inside.err);
    assert.deepEqual(t.examples().map(item => item.expected), ['technical']);
  } finally { t.cleanup(); }
});

test('escalated inputs join examples.json without an expected result, once', async () => {
  const { recordEscalation } = await import('../dist/calls/feedback.js');
  const t = setup(['e/1']);
  try {
    assert.match(recordEscalation(t.store, 'e/1'), /inputs join .*examples\.json as an example without an expected result/);
    assert.equal(recordEscalation(t.store, 'e/1'), undefined, 'an input already listed is left as it is');
    const [example, ...rest] = t.examples();
    assert.equal(rest.length, 0);
    assert.equal('expected' in example, false);
    assert.equal(example.escalated, 'e/1');
    assert.match(example.group, /^escalations-/);
  } finally { t.cleanup(); }
});

test('call IDs resolve from a unique prefix or the short form without task-N-', () => {
  const ids = ['task-3-aaaa1111-0000/1', 'task-3-aaaa1111-0000/2', 'task-4-bbbb2222-0000/1', 'task-5-bbbb3333-0000/1'];
  const t = setup([ids[0]], [ids[1], { parent: ids[0] }], [ids[2]], [ids[3]]);
  try {
    const shown = t.run('show', 'aaaa1111');
    assert.equal(shown.code, 0, shown.err);
    assert.match(shown.out, /call task-3-aaaa1111-0000\/1;/, 'a task prefix means its root call');
    assert.match(t.run('show', 'task-4').out, /call task-4-bbbb2222-0000\/1;/);
    assert.match(t.run('show', 'aaaa1111-0000/2').out, /call task-3-aaaa1111-0000\/2 \(inside task-3-aaaa1111-0000\/1\)/);
    const ambiguous = t.run('show', 'bbbb');
    assert.equal(ambiguous.code, 2);
    assert.match(ambiguous.err, /bbbb matches 2 calls; give more of the ID:\n  task-/);
    const missing = t.run('pin', 'zzzz');
    assert.equal(missing.code, 2);
    assert.match(missing.err, /no call matches zzzz; natlang traces list shows recent calls/);
    assert.equal(t.run('fix', 'bbbb2222', '--expected', '"x"').code, 0);
    assert.equal(t.annotations(ids[2])[0].value.expected, 'x');
    assert.match(t.run('show').err, /usage: natlang traces show CALL/);
  } finally { t.cleanup(); }
});

test('show renders each turn: reasoning, text, tool calls with results and rejections, child calls and the outcome', () => {
  const long = 'Because the ticket mentions a refund. '.repeat(40);
  const events = [
    { kind: 'model_request', phase: 'end', turn: 1, completion_tokens: 42, duration_ms: 1500 },
    { kind: 'model_turn', turn: 1, completion_tokens: 42 },
    { kind: 'proposal', phase: 'generated', turn: 1, reasoning: long, text: 'Let me check.', calls: [['eval', { code: 'return lookup(ticket)' }]] },
    { kind: 'action', name: 'eval', arguments: { code: 'return lookup(ticket)' }, outcome: 'error', result_text: 'lookup is not defined' },
    { kind: 'model_turn', turn: 2 },
    { kind: 'proposal', phase: 'generated', turn: 2, text: '', calls: [['eval', { code: 'return label(ticket)' }]] },
    { kind: 'action', name: 'eval', arguments: { code: 'return label(ticket)' }, outcome: 'ok', result_text: 'Staged "billing".' },
    { kind: 'model_turn', turn: 3 },
    { kind: 'action', name: 'return_result', arguments: { status: 'success', value: 'billing' }, outcome: 'completed', result_text: 'Returned "billing".' },
  ];
  const t = setup(['s/1', { events }], ['s/2', { parent: 's/1', output: 'inner' }]);
  try {
    t.store.db.prepare('UPDATE calls SET parent_action_index = 1 WHERE call_id = ?').run('s/2');
    const shown = t.run('show', 's/1');
    assert.equal(shown.code, 0, shown.err);
    const out = shown.out;
    assert.match(out, /^label-ticket \(triage\/label-ticket.nl\): done in/);
    assert.match(out, /input ticket: "refund please"/);
    assert.match(out, /turn 1 \(42 tokens, 1\.5 s\)\n  thinking: Because the ticket/);
    assert.match(out, /more characters; --full shows all\)/);
    assert.match(out, /  says: Let me check\.\n  → eval\n      \| return lookup\(ticket\)\n    ✗ error: lookup is not defined/);
    assert.match(out, /= ok: Staged "billing"\.\n    ↳ child label-ticket → "inner" \(agent, done; s\/2\)/);
    assert.match(out, /→ return_result \{"status":"success","value":"billing"\}/);
    assert.match(out, /\nresult: "billing"$/m);
    assert.doesNotMatch(t.run('show', 's/1', '--full').out, /--full shows all/);
    const raw = t.run('show', 's/1', '--events').out.trim().split('\n').map(line => JSON.parse(line));
    assert.equal(raw.length, events.length);
    assert.equal(JSON.parse(t.run('show', 's/1', '--json').out).call_id, 's/1');
  } finally { t.cleanup(); }
});

test('list defaults to the program of the current directory; --all is the machine, --program another program', () => {
  const t = setup(['l/1'], ['l/2', { root: '/elsewhere' }], ['l/3', { root: null }]);
  try {
    const ids = out => [...out.matchAll(/^(l\/\d)\s/gm)].map(match => match[1]).sort();
    const inside = t.runIn(t.root, 'list');
    assert.equal(inside.code, 0, inside.err);
    assert.deepEqual(ids(inside.out), ['l/1']);
    assert.match(inside.out, /calls of .*program \(--all for every program's\)/);
    assert.deepEqual(ids(t.runIn(t.root, 'list', '--all').out), ['l/1', 'l/2', 'l/3']);
    assert.deepEqual(ids(t.run('list').out), ['l/1', 'l/2', 'l/3'], 'outside a program: every call');
    assert.deepEqual(JSON.parse(t.run('list', '--program', t.root, '--json').out).map(row => row.call_id), ['l/1']);
    assert.equal(t.runIn(t.root, 'list', '--all', '--program', t.root).code, 2);
    assert.match(t.run('list', '--since', '2026-03-05').out, /no calls on this machine/);
    assert.deepEqual(ids(t.run('list', '--since', '2026-03-04T09:00Z').out), ['l/1', 'l/2', 'l/3']);
  } finally { t.cleanup(); }
});

test('options are checked per action; usage errors exit 2; help never opens the store', () => {
  const t = setup(['o/1']);
  try {
    const bogus = t.run('list', '--bogus', '1');
    assert.equal(bogus.code, 2);
    assert.match(bogus.err, /unknown option --bogus for natlang traces list/);
    assert.match(t.run('list', '--limt', '3').err, /did you mean --limit\?/);
    assert.match(t.run('show', 'o/1', '--expected', '"x"').err, /unknown option --expected for natlang traces show/);
    const date = t.run('list', '--since', 'yesterday');
    assert.equal(date.code, 2);
    assert.match(date.err, /--since takes an ISO date or time .* or an age \(30m, 2h, 7d\), not "yesterday"/);
    assert.match(t.run('list', '--limit', 'ten').err, /--limit takes a positive whole number/);
    assert.match(t.run('lst').err, /unknown action natlang traces lst \(did you mean list\?\)/);
    for (const argv of [['--help'], ['help'], ['show', '--help']]) {
      const result = spawnSync(process.execPath, [bin, 'traces', ...argv], { encoding: 'utf8', env: { ...process.env, NATLANG_CALL_STORE: join(t.dir, 'never') } });
      assert.equal(result.status, 0, result.stderr);
      assert.match(result.stdout, /natlang traces show CALL \[--full\]/);
      assert.equal(existsSync(join(t.dir, 'never')), false, 'help opens no store');
    }
  } finally { t.cleanup(); }
});
