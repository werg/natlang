// Malleable code through a transaction (plans/semantic-fp/01 L3): what judges a call is protected from its edits, a
// successful call records the revisions it committed, and `natlang improve adopt-call` checks, tests and installs them
// (or reports why not), with a byte-identical rollback.
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { CallStore, adoptCall, createNatlangRuntime, loadNatlang, loadVirtualNatlang, rollbackSource } from '../dist/index.js';

const REVIEW = '---\nargs: { n: number }\nreturns: string\nprotect: [checks/verdict.ts]\n---\nCall label with n and return its result.\n';
const LABEL = 'export default function label(n: number): string { return "old"; }\n';
const VERDICT = 'export default function verdict(text: string): boolean { return text === "ok"; }\n';

function program() {
  const root = mkdtempSync(join(tmpdir(), 'natlang-code-edits-'));
  mkdirSync(join(root, 'review', 'checks'), { recursive: true });
  writeFileSync(join(root, 'natlang.json'), '{}\n');
  writeFileSync(join(root, 'review.nl'), REVIEW);
  writeFileSync(join(root, 'review', 'label.ts'), LABEL);
  writeFileSync(join(root, 'review', 'checks', 'verdict.ts'), VERDICT);
  return root;
}

/** A model that plays `calls` in order and keeps every request. */
function scripted(calls) {
  const requests = [];
  let step = 0;
  return { requests, model: async request => { requests.push(request); const call = calls[step++];
    return call ? { calls: [call] } : { calls: [['return_result', { status: 'failed', reason: 'script ended' }]] }; } };
}
const lastToolText = request => JSON.stringify(request.messages?.at(-1) ?? request).replace(/\\"/g, '"');

async function editingRun(root, store, calls) {
  const { model, requests } = scripted(calls);
  const runtime = createNatlangRuntime({ model, calls: store, programRoot: root });
  const review = loadNatlang(join(root, 'review.nl'), root);
  const value = await runtime.run(() => review(3)).catch(error => error);
  return { value, requests, record: store.call(store.calls().find(row => !row.parent_call_id).call_id) };
}

const FIX = ['edit_code', { name: 'label', find: 'return "old";', replace_with: 'return "new";' }];

test('protected items, types.ts and protect: lists cannot be edited; the error says what to fix instead', async () => {
  const root = program(), store = CallStore.open(mkdtempSync(join(tmpdir(), 'natlang-calls-')));
  try {
    const { requests, record } = await editingRun(root, store, [
      ['edit_code', { name: 'checks.verdict', find: 'text === "ok"', replace_with: 'true' }],
      ['edit_code', { name: 'types.ts', find: 'a', replace_with: 'b' }],
      FIX, ['eval', { code: 'return label(n)' }], ['return_result', { status: 'success' }]]);
    assert.match(lastToolText(requests[1]), /review\/checks\/verdict\.ts decides whether this call succeeded, so it cannot be edited here; fix the code that produces the value it rejects\./);
    assert.match(lastToolText(requests[2]), /types\.ts declares the types this call's result is checked against, so it cannot be edited here/);
    assert.equal(readFileSync(join(root, 'review', 'checks', 'verdict.ts'), 'utf8'), VERDICT, 'edits never touch disk');
    assert.equal(record.outcome, 'done');
    assert.equal(store.value(record.output), 'new');
    // Only the committed revision is recorded: the refused edits are not proposals.
    assert.equal(record.code_revisions.length, 1);
    const [revision] = record.code_revisions;
    assert.equal(revision.source, 'review/label.ts');
    assert.equal(revision.function, 'label');
    assert.deepEqual(revision.patch, { find: 'return "old";', replace_with: 'return "new";' });
    assert.equal(store.value(revision.after), LABEL.replace('"old"', '"new"'));
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test('a protect: entry that names nothing is a load error, and a nested function cannot drop its own protect: list', async () => {
  assert.throws(() => loadVirtualNatlang({ 'f.nl': '---\nreturns: string\nprotect: [checks/missing.ts]\n---\nReturn ok.\n', 'f/x.ts': LABEL }, 'f.nl'),
    /protect names "checks\/missing.ts", which is not an item of f\//);
  const files = { 'outer.nl': '---\nreturns: string\n---\nCall inner.\n',
    'outer/inner.nl': '---\nreturns: string\nprotect: [judge.ts]\n---\nReturn ok.\n', 'outer/inner/judge.ts': VERDICT };
  const { model, requests } = scripted([['edit_code', { name: 'inner', find: 'protect: [judge.ts]\n', replace_with: '' }],
    ['edit_code', { name: 'inner.judge', find: 'text === "ok"', replace_with: 'true' }],
    ['return_result', { status: 'success', value: 'ok' }]]);
  const outer = loadVirtualNatlang(files, 'outer.nl');
  assert.equal(await createNatlangRuntime({ model }).run(() => outer()), 'ok');
  assert.match(lastToolText(requests[1]), /The protect: list of outer\/inner\.nl names what judges its calls, so an edit cannot change it/);
  assert.match(lastToolText(requests[2]), /outer\/inner\/judge\.ts decides whether this call succeeded/);
});

test('adopt-call installs a checked and tested fix, refuses a failing one, and rollback restores the exact bytes', async () => {
  const root = program(), store = CallStore.open(mkdtempSync(join(tmpdir(), 'natlang-calls-')));
  try {
    const { record } = await editingRun(root, store, [FIX, ['eval', { code: 'return label(n)' }], ['return_result', { status: 'success' }]]);
    const passing = async directory => {
      // The gates see the proposal, not the checkout.
      assert.notEqual(directory, root);
      assert.match(readFileSync(join(directory, 'review', 'label.ts'), 'utf8'), /"new"/);
      return { check: { ok: true, diagnostics: [] }, test: { ok: true, passed: 1, failed: 0 } };
    };
    const failing = async () => ({ check: { ok: true, diagnostics: [] }, test: { ok: false, passed: 0, failed: 1, detail: '1 of 1 saved examples fail with the fix' } });

    const refused = await adoptCall(store, record.call_id, { verify: failing });
    assert.equal(refused.status, 'rejected');
    assert.match(refused.reasons.join('\n'), /natlang test failed: 1 of 1 saved examples fail/);
    assert.equal(readFileSync(join(root, 'review', 'label.ts'), 'utf8'), LABEL, 'a rejected proposal changes nothing');

    const dry = await adoptCall(store, record.call_id, { verify: passing, dryRun: true });
    assert.equal(dry.status, 'checked');
    assert.equal(readFileSync(join(root, 'review', 'label.ts'), 'utf8'), LABEL);

    const accepted = await adoptCall(store, record.call_id, { verify: passing });
    assert.equal(accepted.status, 'accepted', accepted.reasons.join('\n'));
    assert.deepEqual(accepted.changed, ['review/label.ts']);
    assert.equal(readFileSync(join(root, 'review', 'label.ts'), 'utf8'), LABEL.replace('"old"', '"new"'));
    const saved = JSON.parse(readFileSync(accepted.recordPath, 'utf8'));
    assert.equal(saved.status, 'installed');

    // The source moved on: the same call's proposal is stale now.
    const again = await adoptCall(store, record.call_id, { verify: passing });
    assert.equal(again.status, 'rejected');
    assert.match(again.reasons.join('\n'), /changed since the call edited it/);

    rollbackSource(saved);
    assert.equal(readFileSync(join(root, 'review', 'label.ts'), 'utf8'), LABEL, 'rollback is byte-identical');
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test('adopt-call refuses calls that failed or changed no code, and a proposal that touches a protected item', async () => {
  const root = program(), store = CallStore.open(mkdtempSync(join(tmpdir(), 'natlang-calls-')));
  const verify = async () => ({ check: { ok: true, diagnostics: [] }, test: { ok: true } });
  try {
    const { record: failed } = await editingRun(root, store, [FIX, ['return_result', { status: 'failed', reason: 'no' }]]);
    assert.equal(failed.code_revisions, undefined, 'a failed call proposes nothing');
    assert.match((await adoptCall(store, failed.call_id, { verify })).reasons[0], /only a call that succeeded proposes/);
    const store2 = CallStore.open(mkdtempSync(join(tmpdir(), 'natlang-calls-')));
    try {
      const { record: plain } = await editingRun(root, store2, [['eval', { code: 'return label(n)' }], ['return_result', { status: 'success' }]]);
      assert.match((await adoptCall(store2, plain.call_id, { verify })).reasons[0], /changed no code/);
    } finally { store2.close(); }
    // A forged record whose revision touches the protected verifier is refused before any check runs.
    const forged = { ...failed, outcome: 'done', code_revisions: [{ function: 'checks.verdict', source: 'review/checks/verdict.ts', revision: 1,
      originalHash: '', effectiveHash: (await import('../dist/native/hash.js')).hexDigest(VERDICT), patchedHash: '', patch: { find: '', replace_with: '' },
      after: { complete: true, hash: 'x' } }] };
    const { hexDigest } = await import('../dist/native/hash.js');
    const text = 'export default function verdict(): boolean { return true; }\n';
    forged.code_revisions[0].patchedHash = hexDigest(text);
    const fake = { call: () => forged, value: () => text };
    let verified = false;
    const result = await adoptCall(fake, 'forged', { verify: async () => { verified = true; return verify(); } });
    assert.equal(result.status, 'rejected');
    assert.match(result.reasons.join('\n'), /review\/checks\/verdict\.ts decides whether calls succeed \(protect:\)/);
    assert.equal(verified, false);
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test('natlang improve adopt-call runs natlang check and natlang test, and --untested adopts on check alone', async () => {
  const { spawnSync } = await import('node:child_process');
  const { fileURLToPath } = await import('node:url');
  const cli = fileURLToPath(new URL('../dist/cli/main.js', import.meta.url));
  const root = program(), storeRoot = mkdtempSync(join(tmpdir(), 'natlang-calls-')), store = CallStore.open(storeRoot);
  try {
    const { record } = await editingRun(root, store, [FIX, ['eval', { code: 'return label(n)' }], ['return_result', { status: 'success' }]]);
    store.close();
    const run = (...args) => spawnSync(process.execPath, [cli, 'improve', 'adopt-call', record.call_id, '--store', storeRoot, '--json', ...args], { encoding: 'utf8' });
    const untested = run();
    assert.equal(untested.status, 1, untested.stderr);
    assert.match(JSON.parse(untested.stdout).reasons.join('\n'), /no saved examples to test the fix against; save one with natlang traces fix CALL --expected JSON, or pass --untested/);
    assert.equal(readFileSync(join(root, 'review', 'label.ts'), 'utf8'), LABEL);
    const adopted = run('--untested');
    assert.equal(adopted.status, 0, adopted.stdout + adopted.stderr);
    const report = JSON.parse(adopted.stdout);
    assert.equal(report.status, 'accepted');
    assert.equal(report.verdict.check.ok, true);
    assert.equal(readFileSync(join(root, 'review', 'label.ts'), 'utf8'), LABEL.replace('"old"', '"new"'));
    const rollback = spawnSync(process.execPath, [cli, 'improve', 'rollback', report.record], { encoding: 'utf8' });
    assert.equal(rollback.status, 0, rollback.stderr);
    assert.equal(readFileSync(join(root, 'review', 'label.ts'), 'utf8'), LABEL);
  } finally { rmSync(root, { recursive: true, force: true }); rmSync(storeRoot, { recursive: true, force: true }); }
});
