// One contract suite for every call-store variant that can run under Node (plans/ARCHITECTURE_IMPROVEMENT.md B8):
//
//   node-file     calls/store.ts      node:sqlite database in WAL mode + blobs in files (FileMedium), the machine store
//   wasm-memory   calls/store-wasm.ts sqlite-wasm in-memory database + blobs in the same database (DatabaseMedium)
//
// Both share the logic in calls/store-core.ts, so this suite is what proves the two media and SQLite bindings agree.
//
// Not runnable under Node: browser/call-store-worker.ts. It needs `self`, Worker messaging and the OPFS SAH-pool VFS
// (navigator.storage.getDirectory), none of which exist in Node. It only wraps wasmCallStore (covered below as
// wasm-memory) behind a message protocol; its OPFS behaviour is covered by `npm run test:browser-call-store`
// (scripts/browser-call-store-smoke.mjs), which needs a browser. The worker's QUERY_METHODS list is checked below against
// the store so a renamed method cannot silently break the page-to-worker protocol.
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { mkdtempSync, rmSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { pathToFileURL } from 'node:url';
import sqlite3InitModule from '@sqlite.org/sqlite-wasm';
import { CallStore, FileMedium, openCallStore } from '../dist/calls/store.js';
import { wasmCallStore } from '../dist/calls/store-wasm.js';
import { hexDigest } from '../dist/native/hash.js';

const require = createRequire(import.meta.url);
const { DatabaseSync } = require('node:sqlite');
const sqlite3 = await sqlite3InitModule();
const storeModule = pathToFileURL(new URL('../dist/calls/store.js', import.meta.url).pathname).href;

const BASE = Date.parse('2026-01-01T00:00:00.000Z');
const at = index => new Date(BASE + index * 1000).toISOString();
const ref = text => ({ complete: true, hash: hexDigest(text), bytes: Buffer.byteLength(text) });

/** A complete call record with its value blobs. `text` is the output value. */
function makeCall(id, options = {}) {
  const definition = options.definition ?? 'price';
  const output = JSON.stringify(options.output ?? { id, answer: id.length });
  const input = JSON.stringify(options.input ?? `in-${id}`);
  const instructions = JSON.stringify('Look up the price.');
  const blobs = new Map([[hexDigest(output), output], [hexDigest(input), input], [hexDigest(instructions), instructions]]);
  if (options.padding) { const pad = JSON.stringify('x'.repeat(options.padding)); blobs.set(hexDigest(pad), pad); options = { ...options, padRef: ref(pad) }; }
  const record = {
    version: 'natlang.calls/1', call_id: id, parent_call_id: options.parent ?? null, parent_action_index: options.parent ? 0 : null,
    task_id: options.task ?? `task-${id}`, program_id: null, build_hash: null, program_root: options.programRoot ?? null,
    definition: { id: `${definition}-id`, name: definition, source: options.source ?? `${definition}.nl`, key: options.key ?? `${definition}-key`,
      interface: 'iface', site: options.site ?? 'named', subtype: 'function', params: [{ name: 'id', type: 'string' }], returns: 'number',
      instructions: ref(instructions), types: {} },
    executor: { kind: options.executor ?? 'agent', model_id: 'model-a', model_revision: null, ...(options.caseHash ? { case_hash: options.caseHash } : {}) },
    inputs: { id: ref(input), ...(options.padRef ? { pad: options.padRef } : {}) }, captures: {}, capture_writes: [], output: ref(output),
    outcome: options.outcome ?? 'done', detail: '', started_at: options.startedAt ?? at(options.n ?? 0), ended_at: at((options.n ?? 0) + 1),
    effects: [], folder: null, approach: { evals: ['return 1'], hash: options.approach ?? 'approach-1' },
    cost: { model_requests: 1, tokens_in: options.tokensIn ?? 10, tokens_out: options.tokensOut ?? 5, wall_ms: options.wall ?? 100, turns: 1, evals: 1 },
    features: {}, audit_of: options.auditOf ?? null, events: null,
  };
  return { record, blobs, output, input };
}
const events = (id, size = 0) => [JSON.stringify({ type: 'start', id }), JSON.stringify({ type: 'pad', text: 'e'.repeat(size) })].join('\n');
const put = (store, id, options = {}, withEvents = options.events) => {
  const call = makeCall(id, options);
  store.record(call.record, call.blobs, withEvents === undefined ? undefined : events(id, withEvents));
  return call;
};
const compile = (store, files, caseHashes, extra = {}) => store.saveCompilation({ definitionKey: 'price-key', definitionId: 'price-id',
  definitionName: 'price', definitionSource: 'price.nl', interfaceHash: 'iface', programRoot: null, files, caseHashes, ...extra });

const variants = [
  { name: 'node-file',
    open() {
      const root = mkdtempSync(join(tmpdir(), 'natlang-store-contract-'));
      const store = openCallStore(root);
      return { store, root, cleanup() { try { store.close(); } catch { /* closed */ } rmSync(root, { recursive: true, force: true }); },
        /** Make an open row belong to a writer that no longer exists. */
        orphan(callId) {
          const dead = spawnSync(process.execPath, ['-e', '0']).pid;
          store.db.prepare('UPDATE calls SET pid = ?, process_scope = ? WHERE call_id = ?').run(dead, store.medium.process.scope, callId);
        },
        /** Another store over the same files, like another process. */
        sibling() { return new CallStore(root, new DatabaseSync(join(root, 'calls.sqlite')), new FileMedium(root)); },
        /** Close and reopen the same root. */
        reopen() { store.close(); return openCallStore(root); } };
    } },
  { name: 'wasm-memory',
    open() {
      const db = new sqlite3.oo1.DB(':memory:');
      const { store } = wasmCallStore('memory', db, { scope: 'session:first' });
      return { store, cleanup() { try { store.close(); } catch { /* closed */ } },
        orphan(callId) { store.db.prepare('UPDATE calls SET process_scope = ? WHERE call_id = ?').run('session:ended', callId); },
        sibling: undefined,
        reopen() { return wasmCallStore('memory', db, { scope: 'session:second' }).store; } };
    } },
];

for (const variant of variants) {
  const scenario = (name, body) => test(`call store contract [${variant.name}]: ${name}`, async () => {
    const handle = variant.open();
    try { await body(handle); } finally { handle.cleanup(); }
  });

  scenario('records a call and reads it back with its values and events', ({ store }) => {
    const call = put(store, 't/1', { events: 10 });
    const loaded = store.call('t/1');
    assert.equal(loaded.call_id, 't/1');
    assert.equal(loaded.definition.name, 'price');
    assert.deepEqual(store.value(loaded.output), JSON.parse(call.output));
    assert.equal(store.value(loaded.inputs.id), 'in-t/1');
    assert.deepEqual(store.events('t/1').map(event => event.type), ['start', 'pad']);
    assert.equal(store.call('missing'), undefined);
    assert.equal(store.events('missing'), undefined);
    assert.equal(store.value(null), undefined);
    assert.equal(store.value({ complete: false, reason: 'oversize' }), undefined);
    assert.equal(store.value({ complete: true, hash: 'f'.repeat(64), bytes: 1 }), undefined, 'a hash that was never stored');
    assert.ok(store.diskBytes() >= store.totalBytes() && store.totalBytes() > 0);
  });

  scenario('re-recording a call id replaces it and identical values are stored once', ({ store }) => {
    put(store, 't/1', { outcome: 'failed', input: 'shared' });
    const bytes = store.totalBytes();
    put(store, 't/2', { input: 'shared', output: 'same', n: 1 });
    put(store, 't/3', { input: 'shared', output: 'same', n: 2 });
    const added = store.totalBytes() - bytes;
    put(store, 't/1', { outcome: 'done', input: 'shared' });
    assert.equal(store.calls().filter(row => row.call_id === 't/1').length, 1);
    assert.equal(store.call('t/1').outcome, 'done');
    // t/2 and t/3 differ only in ids, so the shared input/output/instructions blobs were not duplicated.
    assert.ok(added < 2 * (bytes), `shared blobs were stored once (${added} vs ${bytes})`);
    assert.equal(store.putBlob('hello'), hexDigest('hello'));
    const total = store.totalBytes();
    store.putBlob('hello');
    assert.equal(store.totalBytes(), total, 'putBlob of an existing blob adds nothing');
    assert.equal(store.blob(hexDigest('hello')), 'hello');
  });

  scenario('queries filter, order newest first and paginate', ({ store }) => {
    put(store, 'a/1', { definition: 'price', n: 1 });
    put(store, 'a/2', { definition: 'price', n: 2, outcome: 'failed' });
    put(store, 'a/3', { definition: 'ship', n: 3, executor: 'crisp', caseHash: 'case-1' });
    put(store, 'a/4', { definition: 'price', n: 4, auditOf: 'a/1' });
    put(store, 'a/5', { definition: 'ship', n: 5, programRoot: '/work/app' });
    assert.deepEqual(store.calls().map(row => row.call_id), ['a/5', 'a/3', 'a/2', 'a/1'], 'audits are hidden, newest first');
    assert.deepEqual(store.calls({ audits: true }).map(row => row.call_id), ['a/5', 'a/4', 'a/3', 'a/2', 'a/1']);
    assert.deepEqual(store.calls({ definition: 'price' }).map(row => row.call_id), ['a/2', 'a/1']);
    assert.deepEqual(store.calls({ source: 'ship.nl' }).map(row => row.call_id), ['a/5', 'a/3']);
    assert.deepEqual(store.calls({ outcome: 'failed' }).map(row => row.call_id), ['a/2']);
    assert.deepEqual(store.calls({ executor: 'crisp' }).map(row => row.call_id), ['a/3']);
    assert.deepEqual(store.calls({ caseHash: 'case-1' }).map(row => row.call_id), ['a/3']);
    assert.deepEqual(store.calls({ program: '/work/app' }).map(row => row.call_id), ['a/5']);
    assert.deepEqual(store.calls({ since: at(3) }).map(row => row.call_id), ['a/5', 'a/3']);
    assert.deepEqual(store.calls({ limit: 2 }).map(row => row.call_id), ['a/5', 'a/3']);
    assert.deepEqual(store.calls({ limit: 2, after: 'a/3' }).map(row => row.call_id), ['a/2', 'a/1']);
    assert.deepEqual(store.calls({ after: 'a/1' }), []);
    assert.deepEqual(store.calls({ definition: 'nothing' }), []);
  });

  scenario('groups calls into hot definitions, children, and per-definition counts', ({ store }) => {
    put(store, 'h/1', { definition: 'price', n: 1, tokensIn: 100 });
    put(store, 'h/2', { definition: 'price', n: 2, tokensIn: 100 });
    put(store, 'h/3', { definition: 'price', n: 3, executor: 'crisp', tokensIn: 0, tokensOut: 0 });
    put(store, 'h/4', { definition: 'ship', n: 4, tokensIn: 1000, tokensOut: 1000 });
    put(store, 'h/5', { definition: 'inner', n: 5, site: 'internal', parent: 'h/1' });
    put(store, 'h/6', { definition: 'child', n: 6, parent: 'h/1', key: 'child-key' });
    put(store, 'h/7', { definition: 'ship', n: 7, tokensIn: 1000, tokensOut: 1000 });
    const hot = store.hot();
    assert.deepEqual(hot.map(row => [row.definition_name, row.calls, row.agent_calls, row.crisp_calls]),
      [['price', 3, 2, 1], ['ship', 2, 2, 0], ['child', 1, 1, 0]], 'internal sites are not hot; ordered by calls');
    assert.deepEqual(store.hot({ by: 'tokens' }).map(row => row.definition_name), ['ship', 'price', 'child']);
    assert.equal(store.hot({ limit: 1 }).length, 1);
    assert.deepEqual(store.hot({ since: at(4) }).map(row => row.definition_name), ['ship', 'child']);
    assert.deepEqual(store.children('h/1').map(row => row.call_id), ['h/5', 'h/6']);
    assert.deepEqual(store.children('h/4'), []);
    assert.equal(store.countCalls('price-key'), 2, 'agent calls only');
    assert.deepEqual(store.agentCalls('price-key').map(row => row.call_id), ['h/1', 'h/2'], 'oldest first');
  });

  scenario('annotations pin calls, and settings round-trip', ({ store }) => {
    put(store, 'n/1');
    store.annotate('n/1', 'score', { value: 0.5 }, 'judge', false);
    store.annotate('n/1', 'note', 'looked fine');
    assert.deepEqual(store.annotations('n/1').map(row => [row.kind, row.value, row.source]),
      [['score', { value: 0.5 }, 'judge'], ['note', 'looked fine', null]]);
    assert.deepEqual(store.annotations('nothing'), []);
    assert.equal(store.settings().auditRate, 0.05);
    assert.equal(store.writeSettings({ auditRate: 0.5 }).auditRate, 0.5);
    store.writeSettings({ minCalls: 7 });
    assert.deepEqual([store.settings().auditRate, store.settings().minCalls], [0.5, 7], 'a later write keeps earlier settings');
  });

  scenario('eviction drops events of unpinned calls first, then unpinned calls, and never pinned or case-linked ones', ({ store }) => {
    store.writeSettings({ minFreeBytes: 0 });
    for (let index = 0; index < 10; index++) put(store, `e/${index}`, { n: index, events: 1000, padding: 2000 });
    store.annotate('e/0', 'keep', true);                   // pinned by annotation
    store.linkCase('case-x', 'e/1', 'training');           // protected by its case link
    const before = store.totalBytes();
    assert.equal(store.evict(before), 0, 'nothing to do under the bound');
    const freed = store.evict(before - 3000);
    assert.ok(freed >= 3000, `freed ${freed}`);
    assert.equal(store.calls({ limit: 100 }).length, 10, 'events went first; every call row is still there');
    assert.equal(store.events('e/2'), undefined, 'the oldest unpinned call lost its event stream first');
    assert.ok(store.events('e/9')?.length, 'eviction stopped once the bound was met, so the newest keeps its events');
    assert.ok(store.events('e/0')?.length, 'the pinned call keeps its events');
    assert.ok(store.events('e/1')?.length, 'a case-linked call keeps its events');
    assert.equal(store.totalBytes(), before - freed);

    store.evict(0);
    assert.deepEqual(store.calls({ limit: 100 }).map(row => row.call_id).sort(), ['e/0', 'e/1']);
    assert.ok(store.value(store.call('e/0').output), 'a pinned call keeps its values');
    assert.ok(store.value(store.call('e/1').inputs.pad), 'so does a case-linked one');
  });

  scenario('the configured bound limits the store as calls arrive', ({ store }) => {
    store.writeSettings({ maxStoreBytes: 20_000, minFreeBytes: 0 });
    assert.equal(store.bound(), 20_000);
    for (let index = 0; index < 400; index++) put(store, `b/${String(index).padStart(3, '0')}`, { n: index, output: `value-${index}-${'v'.repeat(200)}` });
    assert.ok(store.totalBytes() <= 20_000, `store is ${store.totalBytes()} bytes`);
    const remaining = store.calls({ limit: 1000 }).map(row => row.call_id);
    assert.ok(remaining.length > 0 && remaining.length < 400, `${remaining.length} calls remain`);
    assert.equal(remaining[0], 'b/399', 'the newest call survives');
    assert.ok(!remaining.includes('b/000'), 'the oldest call was evicted');
  });

  scenario('free space lowers the bound', ({ store }) => {
    store.writeSettings({ maxStoreBytes: 1_000_000, minFreeBytes: 1000 });
    put(store, 'f/1', { padding: 5000 });
    const total = store.totalBytes();
    store.medium.freeBytes = () => 400;
    assert.equal(store.bound(), total - 600, 'drops by the shortfall in free bytes');
    store.medium.freeBytes = () => { throw new Error('unknown'); };
    assert.equal(store.bound(), 1_000_000, 'an unknown free size does not shrink the bound');
  });

  scenario('running calls show up, and those of a writer that is gone become interrupted', ({ store, orphan, reopen }) => {
    const definition = { id: 'x', name: 'x', source: null, key: 'k', interface: 'i', site: 'named' };
    const begin = id => store.begin({ callId: id, parentCallId: null, parentActionIndex: null, taskId: id, programId: null, programRoot: null,
      definition, modelId: null, startedAt: at(1), auditOf: null });
    begin('r/1'); begin('r/2');
    store.progress('r/1', events('r/1'));
    assert.deepEqual(store.calls({ outcome: 'running' }).map(row => row.call_id).sort(), ['r/1', 'r/2']);
    assert.equal(store.events('r/1')?.[0].type, 'start');
    assert.equal(store.markInterrupted(), 0, 'this writer is still alive');
    orphan('r/2');
    assert.equal(store.markInterrupted(), 1);
    assert.deepEqual(store.calls({ outcome: 'interrupted' }).map(row => row.call_id), ['r/2']);
    assert.deepEqual(store.calls({ outcome: 'running' }).map(row => row.call_id), ['r/1']);
    put(store, 'r/1', { outcome: 'done' });
    assert.equal(store.calls({ outcome: 'running' }).length, 0, 'recording replaces the running row');
    // Reopening recovers rows left behind by an earlier session or process.
    begin('r/3'); orphan('r/3');
    const next = reopen();
    assert.equal(next.calls({ outcome: 'interrupted' }).some(row => row.call_id === 'r/3'), true);
  });

  scenario('compilations keep case tiers across revisions and bump the version', ({ store }) => {
    put(store, 'c/1', { caseHash: undefined });
    const v0 = store.version();
    const first = compile(store, { 'cases.ts': 'v1' }, ['h1', 'h2'], { links: [{ caseHash: 'h1', callId: 'c/1', role: 'group' }] });
    assert.ok(store.version() > v0);
    assert.equal(store.currentCompilation('price-key').id, first);
    assert.deepEqual(store.currentCompilation('price-key').files, { 'cases.ts': 'v1' });
    assert.deepEqual(store.cases(first).map(item => [item.hash, item.tier, item.position]), [['h1', 'shadow', 0], ['h2', 'shadow', 1]]);
    store.setTier('h1', 'active', 'looks right');
    assert.equal(store.caseStats('h1').tier, 'active');
    assert.equal(store.caseStats('h1').note, 'looks right');
    assert.equal(store.caseStats('nope'), undefined);

    const second = compile(store, { 'cases.ts': 'v2' }, ['h1', 'h3']);
    assert.notEqual(second, first);
    assert.equal(store.currentCompilation('price-key').id, second);
    assert.equal(store.compilation(first).status, 'superseded');
    assert.equal(store.compilation(second).parent_id, first);
    assert.deepEqual(store.cases(second).map(item => [item.hash, item.tier]), [['h1', 'active'], ['h3', 'shadow']], 'h1 kept its tier');
    assert.deepEqual(store.compilations({ status: 'current' }).map(row => row.id), [second]);
    assert.deepEqual(store.compilations({ definition: 'price' }).map(row => row.id).sort(), [first, second].sort());
    assert.deepEqual(store.compilations({ definition: 'ship' }), []);
    store.setCompilationStatus(second, 'disabled');
    assert.equal(store.currentCompilation('price-key'), undefined);
    assert.equal(compile(store, { 'cases.ts': 'v2' }, ['h1', 'h3']), second, 'identical files get the same compilation id');
  });

  scenario('cases link to calls by role and record verdicts, serving and jobs', ({ store }) => {
    put(store, 'k/1'); put(store, 'k/2', { n: 1 }); put(store, 'k/3', { n: 2 });
    compile(store, { 'cases.ts': 'x' }, ['hc']);
    store.linkCase('hc', 'k/1', 'group');
    store.linkCase('hc', 'k/2', 'group');
    store.linkCase('hc', 'k/3', 'held-out', 'equal');
    store.linkCase('hc', 'k/3', 'held-out', 'worse');          // same link again updates the verdict
    assert.deepEqual(store.caseCalls('hc', 'group').map(row => row.call_id).sort(), ['k/1', 'k/2']);
    assert.deepEqual(store.caseCalls('hc', 'held-out').map(row => [row.call_id, row.verdict]), [['k/3', 'worse']]);
    assert.equal(store.caseCalls('hc').length, 3);
    assert.deepEqual(store.callCases('k/3').map(row => ({ ...row })), [{ case_hash: 'hc', role: 'held-out', verdict: 'worse' }]);

    store.caseServed('hc', 'k/1', false);
    store.caseServed('hc', 'k/2', true);
    const served = store.caseStats('hc');
    assert.deepEqual([served.served, served.handed_off], [1, 1], 'a hand-off is counted apart from serving');
    store.caseVerdict('hc', 'k/2', 'shadow', 'better');
    store.caseVerdict('hc', 'k/3', 'audit', 'worse');
    const judged = store.caseStats('hc');
    assert.deepEqual([judged.compared, judged.better, judged.audited, judged.audit_worse], [2, 1, 1, 1]);

    store.enqueue('audit', 'hc', 'k/1');
    store.enqueue('audit', 'hc', 'k/1');                        // idempotent
    store.enqueue('shadow', 'hc', 'k/2');
    assert.deepEqual(store.pendingJobs().map(job => [job.kind, job.call_id]), [['audit', 'k/1'], ['shadow', 'k/2']]);
    store.finishJob(store.pendingJobs()[0].id, 'done', 'equal', 'ok');
    assert.equal(store.pendingJobs().length, 1);
    assert.deepEqual(store.jobs('hc').map(job => [job.status, job.verdict]), [['pending', null], ['done', 'equal']]);
    assert.ok(store.calls({ limit: 10 }).length === 3);
    store.evict(0);
    assert.equal(store.calls({ limit: 10 }).length >= 2, true, 'calls queued for jobs or linked to cases are pinned');
  });

  scenario('declines, findings and spend accumulate', ({ store }) => {
    store.decline({ definitionKey: 'price-key', definitionId: 'price-id', reason: 'not-worth-it', why: 'cheap', calls: 30 });
    store.decline({ definitionKey: 'price-key', definitionId: 'price-id', reason: 'effects', why: 'writes', calls: 40 });
    assert.equal(store.declineFor('price-key').reason, 'effects', 'one decline per definition');
    assert.equal(store.declines().length, 1);
    compile(store, { 'cases.ts': 'x' }, ['d1']);
    assert.equal(store.declineFor('price-key'), undefined, 'compiling clears the decline');

    const finding = { definitionKey: 'price-key', definitionId: 'price-id', definitionName: 'price', definitionSource: 'price.nl', kind: 'drift', summary: 'output drifts' };
    store.finding({ ...finding, detail: { n: 1 } });
    store.finding({ ...finding, detail: { n: 2 } });
    store.finding({ ...finding, summary: 'another' });
    const open = store.findings();
    assert.equal(open.length, 2);
    const drift = open.find(row => row.summary === 'output drifts');
    assert.deepEqual([drift.seen, drift.detail], [2, { n: 2 }]);
    assert.equal(store.acknowledgeFinding(drift.id), true);
    assert.equal(store.acknowledgeFinding(99999), false);
    assert.equal(store.findings().length, 1);
    assert.equal(store.findings({ all: true }).length, 2);
    assert.equal(store.findings({ definition: 'price' }).length, 1);

    put(store, 's/1', { n: 1, tokensIn: 200, tokensOut: 0, wall: 400 });
    put(store, 's/2', { n: 2, executor: 'crisp', tokensIn: 0, tokensOut: 0, wall: 10 });
    put(store, 's/3', { n: 3, executor: 'crisp', tokensIn: 0, tokensOut: 0, wall: 10 });
    store.spent('price-key', 'write', 50, 100);
    store.spent('price-key', 'noop', 0, 0);
    const [row] = store.savings();
    assert.deepEqual([row.agent_calls, row.served, row.agent_tokens_per_call, row.saved_tokens, row.spent_tokens, row.spent_ms],
      [1, 2, 200, 400, 50, 100]);
    assert.deepEqual(store.savings({ definition: 'ship' }), []);
  });

  scenario('iteration statistics read, write, export and reset by prefix', ({ store }) => {
    const statistics = store.iterationStatistics();
    assert.equal(statistics.read('a:1'), undefined);
    statistics.write('a:1', { n: 1 }); statistics.write('a:2', { n: 2 }); statistics.write('b_1', { n: 3 }); statistics.write('a:1', { n: 4 });
    assert.deepEqual(statistics.read('a:1'), { n: 4 });
    assert.deepEqual(Object.keys(statistics.export()).sort(), ['a:1', 'a:2', 'b_1']);
    statistics.reset('a:');
    assert.deepEqual(Object.keys(statistics.export()), ['b_1']);
    statistics.reset('b%');                                    // LIKE wildcards are escaped
    assert.deepEqual(Object.keys(statistics.export()), ['b_1']);
    statistics.reset();
    assert.deepEqual(statistics.export(), {});
  });

  scenario('refinement verdicts are keyed, replaceable, counted, cleared, and survive reopening', ({ store, reopen }) => {
    const verdicts = store.refinementVerdicts();
    assert.equal(verdicts.get('v|p|j'), undefined);
    verdicts.set('v|p|j', { probability: 0.25, judge: 'j' });
    verdicts.set('v2|p|j', { probability: 0.9, judge: 'j' });
    verdicts.set('v|p|j', { probability: 0.75, judge: 'j' });
    assert.deepEqual(verdicts.get('v|p|j'), { probability: 0.75, judge: 'j' });
    assert.equal(verdicts.count(), 2);
    const again = reopen().refinementVerdicts();
    assert.deepEqual(again.get('v2|p|j'), { probability: 0.9, judge: 'j' });
    again.clear();
    assert.equal(again.count(), 0);
  });

  scenario('a store written before the verdict table existed gains it when opened', ({ store, reopen }) => {
    store.db.exec('DROP TABLE refinement_verdicts');
    const migrated = reopen().refinementVerdicts();
    assert.equal(migrated.get('v|p|j'), undefined);
    migrated.set('k', { probability: 0.5, judge: 'j' });
    assert.equal(migrated.count(), 1);
  });

  scenario('data and settings survive reopening the store', ({ store, reopen }) => {
    put(store, 'p/1', { events: 5 });
    store.writeSettings({ minCalls: 3 });
    compile(store, { 'cases.ts': 'x' }, ['p1']);
    const next = reopen();
    assert.equal(next.call('p/1').call_id, 'p/1');
    assert.ok(next.events('p/1')?.length);
    assert.equal(next.settings().minCalls, 3);
    assert.equal(next.currentCompilation('price-key').files['cases.ts'], 'x');
  });
}

test('call store contract [node-file]: concurrent writers in one process share a database', () => {
  const handle = variants[0].open();
  try {
    const second = handle.sibling();
    for (let index = 0; index < 30; index++) {
      put(handle.store, `w1/${index}`, { n: index * 2 });
      put(second, `w2/${index}`, { n: index * 2 + 1 });
    }
    assert.equal(handle.store.calls({ limit: 1000 }).length, 60);
    assert.equal(second.calls({ limit: 1000 }).length, 60);
    assert.equal(second.value(second.call('w1/7').output).id, 'w1/7', 'blobs written by one are readable by the other');
    // Both write the same content-addressed blob: no error, one blob.
    handle.store.putBlob('same'); second.putBlob('same');
    assert.equal(handle.store.blob(hexDigest('same')), 'same');
    second.close();
  } finally { handle.cleanup(); }
});

test('call store contract [node-file]: concurrent writer processes lose no calls', async () => {
  const handle = variants[0].open();
  try {
    const script = `
      import { CallStore, FileMedium } from ${JSON.stringify(storeModule)};
      import { createRequire } from 'node:module';
      const { DatabaseSync } = createRequire(import.meta.url)('node:sqlite');
      const [root, tag] = process.argv.slice(1);
      const db = new DatabaseSync(root + '/calls.sqlite');
      db.exec('PRAGMA busy_timeout = 10000; PRAGMA journal_mode = WAL;');
      const store = new CallStore(root, db, new FileMedium(root));
      for (let i = 0; i < 40; i++) {
        const out = JSON.stringify({ tag, i });
        const hash = (await import('node:crypto')).createHash('sha256').update(out).digest('hex');
        const r = { version: 'natlang.calls/1', call_id: tag + '/' + i, parent_call_id: null, parent_action_index: null, task_id: tag, program_id: null,
          build_hash: null, program_root: null, definition: { id: 'd', name: 'd', source: null, key: 'k', interface: 'i', site: 'named', subtype: 'function',
          params: [], returns: 'number', instructions: { complete: false, reason: 'excluded' }, types: {} },
          executor: { kind: 'agent', model_id: 'm', model_revision: null }, inputs: {}, captures: {}, capture_writes: [],
          output: { complete: true, hash, bytes: out.length }, outcome: 'done', detail: '', started_at: new Date(Date.now() + i).toISOString(),
          ended_at: new Date().toISOString(), effects: [], folder: null, approach: { evals: [], hash: null },
          cost: { model_requests: 0, tokens_in: 1, tokens_out: 1, wall_ms: 1, turns: 1, evals: 0 }, features: {}, audit_of: null, events: null };
        store.record(r, new Map([[hash, out]]));
      }
      store.close();`;
    const run = tag => new Promise((resolve, reject) => {
      const child = spawn(process.execPath, ['--no-warnings', '--input-type=module', '-e', script, handle.root, tag], { stdio: ['ignore', 'inherit', 'pipe'] });
      let stderr = '';
      child.stderr.on('data', chunk => { stderr += chunk; });
      child.on('exit', code => code === 0 ? resolve() : reject(new Error(`writer ${tag} exited ${code}: ${stderr}`)));
    });
    await Promise.all([run('p1'), run('p2'), run('p3')]);
    const reader = handle.sibling();
    assert.equal(reader.calls({ limit: 1000 }).length, 120);
    assert.deepEqual(reader.value(reader.call('p2/39').output), { tag: 'p2', i: 39 });
    reader.close();
  } finally { handle.cleanup(); }
});

test('call store contract: the browser worker protocol only names methods the store has', () => {
  const source = readFileSync(new URL('../src/browser/call-store-protocol.ts', import.meta.url), 'utf8');
  const list = /QUERY_METHODS = \[([^\]]+)\]/.exec(source)[1];
  const methods = [...list.matchAll(/'([A-Za-z]+)'/g)].map(match => match[1]);
  const writes = /method: ('[A-Za-z]+'(?: \| '[A-Za-z]+')*);/.exec(source)[1].match(/[A-Za-z]+/g);
  const handle = variants[1].open();
  try {
    for (const method of [...methods, ...writes]) assert.equal(typeof handle.store[method], 'function', `store.${method}`);
  } finally { handle.cleanup(); }
});
