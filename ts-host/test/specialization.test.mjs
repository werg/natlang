import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang, CallStore, ReplayServices, approaches, induceRules, measureGuard, inputFeatures,
  splitOf, study, crispDecline, approachHash, normalizeProgram, renderEvidence, verifyCases, saveAccepted, keepCases, runJobs, caseHashes, Folder,
  groupsOf, measure, assembleCases, renderGroup, renderFunction, detectFindings } from '../dist/index.js';

// Call IDs embed a random UUID, and a call is held out by a hash of its ID (a fifth of calls). With random IDs a recording of 30
// calls held out fewer than the two the assertions need about once in a hundred runs. Fix the IDs so the split is the same every run.
let uuidSequence = 0;
Object.defineProperty(globalThis.crypto, 'randomUUID', { configurable: true, value: () => `00000000-0000-4000-8000-${(++uuidSequence).toString(16).padStart(12, '0')}` });

const freshStore = () => CallStore.open(mkdtempSync(join(tmpdir(), 'natlang-spec-')));
const done = store => { const root = store.root; store.close(); rmSync(root, { recursive: true, force: true }); };

test('approaches group calls that ran the same code, with differing literals as holes', () => {
  const example = (callId, request, code) => ({ callId, args: { request }, features: inputFeatures({ request }), evals: [code], approach: '', split: 'training' });
  const found = approaches([
    example('a', 'refund the last one', "await orders.refund('A-7')"), example('b', 'refund the last one', "await orders.refund('B-9')"),
    example('c', 'status 3', "return orders.lookup('3').status"), example('d', 'status 3 now', "return orders.lookup('3').status")]);
  assert.equal(found.length, 2);
  const refunds = found.find(item => item.calls.includes('a'));
  assert.deepEqual(refunds.calls.sort(), ['a', 'b']);
  assert.match(refunds.template[0], /\$h0/);
  assert.deepEqual(refunds.holes.a, ['"A-7"']);
});

test('normalization abstracts input parts, logging and shown values', () => {
  const a = approachHash(["const o = orders.lookup('144');\nconsole.log(o);\no"], { request: 'What is the status of order 144?' });
  const b = approachHash(['const x = await orders.lookup("136")\nconsole.log("result:", x)'], { request: 'What is the status of order 136?' });
  assert.equal(a, b);
  const refund = (id, request) => normalizeProgram(`const r = orders.refund("${id}");\nreturn \`refunded ${id}\``, { request });
  assert.equal(refund('140', 'refund 140'), refund('132', 'refund 132'));
  assert.match(refund('140', 'refund 140'), /\$part\.request/);
  assert.equal(approachHash([], { request: 'thanks' }), 'answer-only');
  assert.notEqual(approachHash(['orders.refund("1")'], { request: 'refund 1' }), approachHash(['orders.lookup("1")'], { request: 'refund 1' }));
});

test('rule induction finds a guard with no counterexample and leaves the rest unclassified', () => {
  const examples = [];
  for (let index = 0; index < 12; index++) examples.push({ callId: `r${index}`, args: { request: `refund ${index}` },
    features: inputFeatures({ request: `refund ${index}` }), evals: [], approach: 'refund', split: 'training' });
  for (let index = 0; index < 12; index++) examples.push({ callId: `s${index}`, args: { request: `where is order ${index}` },
    features: inputFeatures({ request: `where is order ${index}` }), evals: [], approach: 'status', split: 'training' });
  examples.push({ callId: 'odd', args: { request: 'hmm' }, features: inputFeatures({ request: 'hmm' }), evals: [], approach: 'other', split: 'training' });
  const { rules, unclassified } = induceRules(examples);
  assert.ok(rules.length >= 2);
  for (const rule of rules) {
    const measured = measureGuard(rule.guard, rule.label, examples);
    assert.equal(measured.counterexamples.length, 0, rule.guard);
  }
  assert.deepEqual(unclassified, ['odd']);
  assert.match(rules.map(rule => rule.guard).join('\n'), /refund|where|order/);
});

test('replay services answer from recorded effects and report anything new as a divergence', async () => {
  const services = new ReplayServices([{ service: 'orders', method: 'lookup', args: ['7'], result: { total: 3 }, async: false, complete: true }]);
  assert.deepEqual(services.services.orders.lookup('7'), { total: 3 });
  assert.throws(() => services.services.orders.lookup('7'), /no recorded result for orders\.lookup\("7"\)/);
  assert.throws(() => services.services.orders.refund('7'), /a replay cannot perform new effects/);
  assert.equal(services.divergences.length, 2);
});

test('replay answers a call whose scalars are written differently, and the comparison still sees the difference', async () => {
  const services = new ReplayServices([{ service: 'orders', method: 'refund', args: [999], result: { ok: true }, async: false, complete: true }]);
  assert.deepEqual(services.services.orders.refund('999'), { ok: true });
  assert.deepEqual(services.observed, [{ service: 'orders', method: 'refund', args: ['999'] }]);
  assert.equal(services.divergences.length, 0);
  assert.throws(() => services.services.orders.refund('998'), /no recorded result/);
});

test('keepCases keeps only the chosen cases and the rest of the file', () => {
  const text = "import { a } from 'natlang:services';\nexport const cases = [\n  { when: () => true, run: async () => 1 },\n  { when: () => false, run: async () => 2 },\n];\n";
  const kept = keepCases(text, [1]);
  assert.match(kept, /^import \{ a \}/);
  assert.equal(caseHashes(kept).length, 1);
  assert.match(kept, /run: async \(\) => 2/);
});

const HANDLE = '---\nargs: { request: string }\nreturns: string\n---\nHandle a support request: refund an order, or say its status.\n';
const orderService = () => {
  const log = [];
  return { log, orders: { async refund(id) { log.push(`refund ${id}`); return { ok: true }; }, lookup(id) { log.push(`lookup ${id}`); return { status: `shipped ${id}` }; } } };
};
/**
 * A scripted executor standing in for a model: it reads the request, then runs the refund or the status approach. It
 * declares a model and reports token use, as a model driver does; only such calls are evidence (isModelEvidence).
 */
function executor() {
  return Object.assign(async request => {
    const tools = request.messages.filter(message => message.role === 'tool').map(message => String(message.content));
    const opening = String(request.messages.find(message => message.role === 'user')?.content ?? '');
    if (opening.includes('Two executions')) return { prompt_tokens: 10, calls: [['return_result', { status: 'success', value: 'equal' }]] };
    const seen = tools.map(text => /(refund|status) (\d+)/.exec(text)).find(Boolean);
    if (tools.length <= 1) return { prompt_tokens: 10, calls: [['eval', { code: 'return request' }]] };
    if (tools.length === 2) return { prompt_tokens: 10, calls: [['eval', { code: seen[1] === 'refund' ? "const id = request.split(' ')[1];\nawait orders.refund(id);\nreturn 'refunded ' + id" :
      "return orders.lookup(request.split(' ')[1]).status" }]] };
    return { prompt_tokens: 10, calls: [['return_result', { status: 'success', value: seen[1] === 'refund' ? `refunded ${seen[2]}` : `shipped ${seen[2]}` }]] };
  }, { model: 'test-model' });
}
const CASES = `import { orders } from 'natlang:services';
export const cases = [
  { when: (args: { request: string }) => /^refund \\d+$/.test(args.request),
    run: async (args: { request: string }) => { const id = args.request.split(' ')[1]; await orders.refund(id); return 'refunded ' + id; } },
  { when: (args: { request: string }) => /^status \\d+$/.test(args.request),
    run: async (args: { request: string }) => orders.lookup(args.request.split(' ')[1]).status },
];
`;

async function recordCalls(store, count, model = executor()) {
  const handle = loadVirtualNatlang({ 'handle.nl': HANDLE }, 'handle.nl');
  for (let index = 0; index < count; index++) {
    const service = orderService();
    const runtime = createNatlangRuntime({ model, calls: store, services: { orders: service.orders } });
    await runtime.run(() => handle(`${index % 2 ? 'refund' : 'status'} ${100 + index}`));
  }
  return handle;
}

test('calls of a scripted executor that declares no model are recorded but are never evidence', async () => {
  const store = freshStore();
  try {
    const scripted = executor();
    delete scripted.model;
    await recordCalls(store, 12, async request => ({ ...(await scripted(request)), prompt_tokens: undefined }));
    const key = store.hot()[0].definition_key;
    const record = store.call(store.calls({ key, limit: 1 })[0].call_id);
    assert.match(record.executor.model_id, /^undeclared:/);
    assert.equal(study(store, key), undefined);
  } finally { done(store); }
});

test('study, verify and save: recorded calls become a compilation whose cases are promoted by held-out evidence', async () => {
  const store = freshStore();
  try {
    store.writeSettings({ promotionComparisons: 1, promotionLiveComparisons: 0 });
    const handle = await recordCalls(store, 30);
    const key = store.hot()[0].definition_key;
    const subject = study(store, key);
    assert.equal(subject.examples.length, 30);
    assert.equal(subject.approaches.filter(item => item.calls.length >= 10).length, 2);
    assert.ok(subject.rules.length >= 2);
    assert.equal(crispDecline(subject, 20), undefined);
    const evidence = renderEvidence(store, subject);
    assert.match(evidence['function.md'], /Handle a support request/);
    assert.match(evidence['conditions.md'], /## 1\./);
    assert.ok(Object.keys(evidence).some(path => /^approaches\/a1\/examples\//.test(path)));
    const verifier = createNatlangRuntime({ model: executor(), calls: false });
    const { checks } = await verifyCases(verifier, store, subject, CASES);
    assert.equal(checks.length, 2);
    for (const check of checks) {
      assert.equal(check.accepted, true, check.reason);
      assert.ok(check.results.every(result => result.verdict === 'equal'), JSON.stringify(check.results));
    }
    const saved = saveAccepted(store, subject, CASES, checks, 'report');
    assert.equal(saved.cases, 2);
    const compilation = store.currentCompilation(key);
    const heldOut = subject.examples.filter(example => example.split === 'held-out').length;
    assert.ok(heldOut >= 2, 'the split holds out some calls');
    assert.ok(compilation.cases.every(item => item.compared >= 1 && item.tier === 'active'), JSON.stringify(compilation.cases));
    // The runtime now serves matching calls without the model.
    const service = orderService();
    const live = createNatlangRuntime({ model: async () => { throw new Error('the model must not be asked'); }, calls: store, services: { orders: service.orders } });
    assert.equal(await live.run(() => handle('refund 999')), 'refunded 999');
    assert.deepEqual(service.log, ['refund 999']);
  } finally { done(store); }
});

test('a case that answers wrongly is not accepted, and the judge decides differing behavior', async () => {
  const store = freshStore();
  try {
    await recordCalls(store, 24);
    const subject = study(store, store.hot()[0].definition_key);
    const wrong = CASES.replace("return 'refunded ' + id;", "return 'refused ' + id;");
    let judged = 0;
    const judgeModel = async request => {
      const opening = String(request.messages.find(message => message.role === 'user')?.content ?? '');
      if (!opening.includes('Two executions')) throw new Error('only the judge runs here');
      judged++;
      const reads = request.messages.filter(message => message.role === 'tool').map(message => String(message.content)).join('\n');
      if (!reads) return { calls: [['eval', { code: 'return [first, second]' }]] };
      return { calls: [['return_result', { status: 'success', value: /^\[\s*"Result: \\"refunded/.test(reads) ? 'first' : 'second' }]] };
    };
    const { checks } = await verifyCases(createNatlangRuntime({ model: judgeModel, calls: false }), store, subject, wrong);
    const refund = checks.find(check => /refund/.test(check.source));
    assert.equal(refund.accepted, false);
    assert.ok(refund.results.some(result => result.verdict === 'worse'), JSON.stringify(refund.results));
    assert.ok(judged > 0);
    assert.equal(checks.find(check => /status/.test(check.source)).accepted, true);
  } finally { done(store); }
});

test('shadow and audit jobs compare a case with the agent and count toward its tier', async () => {
  const store = freshStore();
  try {
    store.writeSettings({ promotionComparisons: 1000, auditRate: 1 });
    const handle = await recordCalls(store, 4);
    const key = store.hot()[0].definition_key;
    const subject = study(store, key);
    const checks = [0, 1].map(position => ({ position, accepted: true, results: [] }));
    saveAccepted(store, subject, CASES, checks, 'report');
    const [refundHash] = caseHashes(CASES);
    // Shadow: the agent serves, and the call is queued for a shadow replay.
    await createNatlangRuntime({ model: executor(), calls: store, services: orderService() }).run(() => handle('refund 5'));
    assert.equal(store.pendingJobs().filter(job => job.kind === 'shadow').length, 1);
    const worker = createNatlangRuntime({ model: executor(), calls: false });
    assert.equal(await runJobs(worker, store), 1);
    assert.equal(store.caseStats(refundHash).compared, 1);
    assert.equal(store.jobs(refundHash)[0].verdict, 'equal');
    // Audit: an active case serves, and the call is re-run through the agent offline.
    store.setTier(refundHash, 'active');
    await createNatlangRuntime({ model: async () => { throw new Error('no model'); }, calls: store, services: orderService() }).run(() => handle('refund 6'));
    const audits = store.pendingJobs().filter(job => job.kind === 'audit');
    assert.equal(audits.length, 1);
    const auditor = createNatlangRuntime({ model: executor(), calls: store, specialization: 'off' });
    assert.equal(await runJobs(auditor, store), 1);
    const stats = store.caseStats(refundHash);
    assert.equal(stats.audited, 1);
    assert.equal(stats.audit_worse, 0);
    const rerun = store.calls({ audits: true, limit: 100 }).find(call => call.audit_of === audits[0].call_id);
    assert.ok(rerun, 'the audit run is recorded and names the call it audited');
    assert.equal(rerun.executor, 'agent');
  } finally { done(store); }
});

test('the evidence folder renders as files a reducer can read', () => {
  const folder = Folder.fromFiles({ 'evidence/function.md': '# f' });
  assert.deepEqual(folder.filePaths(), ['evidence/function.md']);
  assert.equal(splitOf('x'), splitOf('x'));
});

test('groups split calls by what they did, and a proposed guard is measured exactly', async () => {
  const store = freshStore();
  try {
    await recordCalls(store, 30);
    const subject = study(store, store.hot()[0].definition_key);
    const groups = groupsOf(subject);
    assert.deepEqual(groups.map(group => group.label).sort(), ['orders.lookup', 'orders.refund']);
    const refunds = groups.find(group => group.label === 'orders.refund');
    const exact = measure(subject, refunds, '/^refund \\d+$/.test(args.request)');
    assert.deepEqual([exact.valid, exact.ofGroup, exact.groupSize, exact.others], [true, refunds.training.length, refunds.training.length, 0]);
    const loose = measure(subject, refunds, '/\\d/.test(args.request)');
    assert.ok(loose.others > 0 && loose.counterexamples.every(item => item.did === 'orders.lookup'));
    assert.equal(measure(subject, refunds, 'args.request ===').valid, false);
    const files = renderGroup(store, subject, refunds, renderFunction(subject));
    assert.match(files['group.md'], /orders\.refund/);
    assert.match(files['group.md'], /0 of other groups/);
    assert.match(files['others.md'], /did orders\.lookup/);
    assert.ok(Object.keys(files).some(path => path.startsWith('examples/')));
  } finally { done(store); }
});

test('per-group case files assemble into one cases file with merged imports and private helpers', () => {
  const refund = "import { orders } from 'natlang:services';\nconst numberOf = (text: string) => text.split(' ')[1];\n" +
    "export const when = (args: { request: string }) => /^refund \\d+$/.test(args.request);\n" +
    "export const run = async (args: { request: string }) => { await orders.refund(numberOf(args.request)); return 'refunded ' + numberOf(args.request); };\n";
  const status = "import { orders } from 'natlang:services';\nconst numberOf = (text: string) => text.split(' ')[1];\n" +
    "export function when(args: { request: string }) { return /^status \\d+$/.test(args.request); }\n" +
    "export async function run(args: { request: string }) { return orders.lookup(numberOf(args.request)).status; }\n";
  const { text, included, errors } = assembleCases([{ id: 'g1', text: refund }, { id: 'g2', text: status }, { id: 'g3', text: 'export const when = () => true;' }]);
  assert.deepEqual(included, ['g1', 'g2']);
  assert.deepEqual(errors.map(error => error.id), ['g3']);
  assert.equal((text.match(/import \{ orders \} from "natlang:services"/g) ?? []).length, 1);
  assert.equal(caseHashes(text).length, 2, 'two cases, each with its own numberOf');
});

test('findings: identical inputs handled differently, and service arguments whose type varies', async () => {
  const store = freshStore();
  try {
    const typed = executor();
    let flip = 0;
    const mixed = Object.assign(async request => {
      const turn = await typed(request);
      // Every other refund passes its number as a number instead of a string.
      const code = turn.calls?.[0]?.[1]?.code;
      if (typeof code === 'string' && code.includes('await orders.refund(id)') && flip++ % 2)
        return { ...turn, calls: [['eval', { code: code.replace('await orders.refund(id)', 'await orders.refund(Number(id))') }]] };
      return turn;
    }, { model: typed.model });
    await recordCalls(store, 30, mixed);
    const subject = study(store, store.hot()[0].definition_key);
    const kinds = detectFindings(store, subject).map(finding => `${finding.kind}: ${finding.summary}`);
    assert.ok(kinds.some(kind => /argument-types: orders\.refund argument 1 is passed as (string or number|number or string)/.test(kind)), kinds.join('\n'));
    store.finding(detectFindings(store, subject)[0]);
    store.finding(detectFindings(store, subject)[0]);
    const [row] = store.findings();
    assert.equal(row.seen, 2);
    assert.ok(store.acknowledgeFinding(row.id));
    assert.equal(store.findings().length, 0);
  } finally { done(store); }
});

test('promotion needs comparisons on live calls, not only replays of the recorded ones', async () => {
  const store = freshStore();
  try {
    store.writeSettings({ promotionComparisons: 1 });
    await recordCalls(store, 30);
    const key = store.hot()[0].definition_key;
    const subject = study(store, key);
    const verifier = createNatlangRuntime({ model: executor(), calls: false });
    const { checks } = await verifyCases(verifier, store, subject, CASES);
    saveAccepted(store, subject, CASES, checks, 'report');
    const [first] = store.currentCompilation(key).cases;
    assert.equal(first.tier, 'shadow', 'held-out replays alone do not promote');
    for (const id of ['live-1', 'live-2', 'live-3']) store.caseVerdict(first.hash, id, 'shadow', 'equal');
    const after = store.caseStats(first.hash);
    assert.deepEqual([after.tier, after.live_compared], ['active', 3]);
  } finally { done(store); }
});

test('spend and savings: what specializing cost against what serving saved', async () => {
  const store = freshStore();
  try {
    store.writeSettings({ promotionComparisons: 1, promotionLiveComparisons: 0 });
    const handle = await recordCalls(store, 30);
    const key = store.hot()[0].definition_key;
    const subject = study(store, key);
    const { checks } = await verifyCases(createNatlangRuntime({ model: executor(), calls: false }), store, subject, CASES);
    saveAccepted(store, subject, CASES, checks, 'report');
    store.spent(key, 'specialize', 500, 1000);
    const service = orderService();
    await createNatlangRuntime({ model: async () => { throw new Error('not asked'); }, calls: store, services: { orders: service.orders } }).run(() => handle('refund 5'));
    const [row] = store.savings();
    assert.equal(row.served, 1);
    assert.equal(row.spent_tokens, 500);
    assert.equal(row.saved_tokens, row.agent_tokens_per_call);
  } finally { done(store); }
});
