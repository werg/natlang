import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';
import { createNatlangRuntime, loadNatlang, nodeSourceFiles } from '../dist/index.js';
import { fusionFacts, crispPlan, verifyPlan, repairedPlan, nlPlan, nlPlannerFrom, planFusion, fusedEdges, fusionStatus, fusionReport,
  parseFusionSettings } from '../dist/fusion/index.js';
import { createTextNeuraleseEmulation, TEXT_NEURALESE_DIALECT } from '../dist/model/text-neuralese-emulation.js';
import { MemoryNeuraleseStore, StandInNeuralesePort, hashingEmbedder } from '../dist/native/neuralese-store.js';

const sha = text => createHash('sha256').update(text).digest('hex');
function folder(files) {
  const root = mkdtempSync(join(tmpdir(), 'natlang-fusion-'));
  for (const [path, content] of Object.entries(files)) {
    mkdirSync(join(root, path, '..'), { recursive: true });
    writeFileSync(join(root, path), content);
  }
  return root;
}
const stage = (args, returns, body, extra = '') => `---\n${extra}args:\n${args}\nreturns: ${returns}\n---\n${body}\n`;

/** An orchestrator with a fusible chain, a chain a crisp service reads, a chain the model reads, and a chain across models. */
const PROJECT = {
  'natlang.json': '{}',
  'orch.nl': stage('  source: string', 'string', `ORCH. Compile source.

1. tree = parse(source). checked = analyze(tree).
2. draft = write(source). Call audit.check(draft). Then render(draft).
3. lexed = lex(source). When lexed.length is above 3, return lexed.
4. big = outline(source). small = shorten(big).
5. Return checked.`),
  'orch/parse.nl': stage('  source: string', 'string', 'STAGE-PARSE: parse source.'),
  'orch/analyze.nl': stage('  syntax: string', 'string', 'STAGE-ANALYZE: analyze syntax.'),
  'orch/write.nl': stage('  source: string', 'string', 'STAGE-WRITE: write.'),
  'orch/render.nl': stage('  text: string', 'string', 'STAGE-RENDER: render.'),
  'orch/lex.nl': stage('  source: string', 'string', 'STAGE-LEX: lex.'),
  'orch/outline.nl': stage('  source: string', 'string', 'STAGE-OUTLINE: outline.'),
  'orch/shorten.nl': stage('  text: string', 'string', 'STAGE-SHORTEN: shorten.', 'model: small\n'),
  'host.ts': `import first from './first.nl';\nimport second from './second.nl';
export async function fused(x: string) { return second(await first(x)); }
export async function read(x: string) { const a = await first(x); console.log(a.length); return second(a); }
`,
  'first.nl': stage('  x: string', 'string', 'FIRST'),
  'second.nl': stage('  a: string', 'string', 'SECOND'),
};

test('the fact service lists candidate edges with their readers and the crisp planner applies the exact conditions', () => {
  const root = folder(PROJECT);
  const facts = fusionFacts(root, nodeSourceFiles(root));
  const byChain = chain => facts.edges.filter(edge => edge.chain === chain);
  const plan = crispPlan(facts);
  const decision = edge => plan.edges.find(entry => entry.edge === edge.id);
  const [parseAnalyze] = byChain('parse -> analyze');
  assert.equal(parseAnalyze.flow, 'bound');
  assert.equal(parseAnalyze.variable, 'tree');
  assert.deepEqual(parseAnalyze.readers.map(reader => reader.kind), ['consumer']);
  assert.equal(decision(parseAnalyze).decision, 'fuse');
  const [writeRender] = byChain('write -> render');
  assert.ok(writeRender.readers.some(reader => reader.kind === 'service'), 'audit.check(draft) is a service reader');
  assert.equal(decision(writeRender).decision, 'keep-text');
  assert.match(decision(writeRender).reason, /service reads/);
  const [outlineShorten] = byChain('outline -> shorten');
  assert.match(decision(outlineShorten).reason, /different models/);
  assert.equal(facts.edges.filter(edge => edge.chain.startsWith('lex')).length, 0, 'lexed has no stage consumer');
  // Crisp TypeScript: the exact data flow of the host file.
  const typescript = facts.edges.filter(edge => edge.scopeKind === 'typescript');
  const nested = typescript.find(edge => edge.flow === 'nested');
  const read = typescript.find(edge => edge.flow === 'bound');
  assert.equal(decision(nested).decision, 'fuse');
  assert.ok(read.readers.some(reader => reader.kind === 'crisp-code'));
  assert.equal(decision(read).decision, 'keep-text');
  // Only edges inside natural-language scopes can engage at run time.
  assert.deepEqual(fusedEdges(facts, plan).map(edge => edge.producer.source), ['orch/parse.nl']);
  const report = fusionReport(facts, plan, parseFusionSettings(undefined));
  assert.equal(report.counts.fuse, 2);
  assert.equal(report.counts.nlScopeFuse, 1);
  assert.equal(report.counts.typescriptScopeFuse, 1);
});

test('the verifier rejects a plan that fuses an edge with an outside reader or two models, and the retry then the fallback follow', async () => {
  const root = folder(PROJECT);
  const facts = fusionFacts(root, nodeSourceFiles(root));
  const everything = { schema: 'natlang.fusion-plan/1', planner: 'nl',
    edges: facts.edges.map(edge => ({ edge: edge.id, decision: 'fuse', reason: 'scripted' })) };
  const checked = verifyPlan(facts, everything);
  assert.equal(checked.ok, false);
  const problemChains = checked.problems.map(problem => facts.edges.find(edge => edge.id === problem.edge).chain);
  assert.ok(problemChains.includes('write -> render'));
  assert.ok(problemChains.includes('outline -> shorten'));
  assert.ok(!problemChains.includes('parse -> analyze'));
  assert.equal(verifyPlan(facts, { edges: [] }).ok, false, 'an undecided edge is a problem');
  assert.equal(verifyPlan(facts, { edges: [...everything.edges, { edge: 'nope', decision: 'fuse', reason: 'x' }] }).problems.at(-1).problem, 'no such edge in the facts');

  // A planner that fuses everything on the first answer and obeys the problems on the retry.
  const calls = [];
  const stubborn = async (edges, problems) => {
    calls.push(problems.length);
    const bad = new Set(problems.map(item => item.edge));
    return edges.map(edge => ({ edge: edge.id, decision: bad.has(edge.id) ? 'keep-text' : 'fuse', reason: 'scripted' }));
  };
  const plan = await nlPlan(facts, stubborn);
  assert.equal(calls.length, 2);
  assert.ok(calls[0] === 0 && calls[1] > 0, 'the retry carries the verifier problems');
  assert.equal(verifyPlan(facts, plan).ok, true);
  // A planner that ignores the problems: the rejected edges fall back to text.
  const deaf = async edges => edges.map(edge => ({ edge: edge.id, decision: 'fuse', reason: 'scripted' }));
  const fallback = await nlPlan(facts, deaf);
  assert.equal(verifyPlan(facts, fallback).ok, true);
  const writeRender = facts.edges.find(edge => edge.chain === 'write -> render');
  assert.match(fallback.edges.find(entry => entry.edge === writeRender.id).reason, /^kept as text: fused, but/);
  // A planner that fails: every edge is text.
  const broken = await nlPlan(facts, async () => { throw new Error('no model'); });
  assert.ok(broken.edges.every(entry => entry.decision === 'keep-text'));
  assert.equal(broken.planner, 'crisp-fallback');
  assert.equal(repairedPlan(facts, undefined, [], 'nl').edges.length, facts.edges.length);
});

test('the natural-language planner program runs on a scripted model and a crisp verifier checks its plan', async () => {
  const root = folder(PROJECT);
  const facts = fusionFacts(root, nodeSourceFiles(root));
  const requests = [];
  // The scripted model plays planner.nl: its eval code builds the decisions from `edges` and `problems`.
  const driver = ({ messages }) => {
    requests.push(messages.length);
    if (requests.length % 2 === 0) return { calls: [['return_result', { status: 'success' }]] };
    return { calls: [['eval', { code: `const bad = new Set((problems ?? []).map(p => p.edge));\n` +
      `return_result(edges.map(e => ({ edge: e.id, decision: bad.has(e.id) ? "keep-text" : "fuse", reason: "scripted planner" })));` }]] };
  };
  const runtime = createNatlangRuntime({ model: driver });
  const plan = await runtime.run(() => planFusion(facts, { planner: 'nl', nl: nlPlannerFrom() }));
  assert.equal(plan.planner, 'nl');
  assert.equal(verifyPlan(facts, plan).ok, true);
  assert.deepEqual(plan.edges.map(entry => entry.decision), crispPlan(facts).edges.map(entry => entry.decision),
    'after the retry the scripted NL plan equals the crisp plan');
  const shadow = await runtime.run(() => planFusion(facts, { planner: 'shadow', nl: nlPlannerFrom() }));
  assert.equal(shadow.planner, 'crisp', 'shadow serves the crisp plan');
});

// --- Runtime ------------------------------------------------------------------------------------------------------

/** Scripted stages: the driver answers by the marker each function's instructions carry. */
function scriptedModel(log, { parseValues = ['TREE-1'] } = {}) {
  const counters = new Map();
  return ({ messages }) => {
    const text = JSON.stringify(messages);
    const which = ['STAGE-ANALYZE', 'STAGE-PARSE', 'ORCH'].find(marker => text.includes(marker));
    const turn = counters.get(which) ?? 0;
    counters.set(which, turn + 1);
    log.push({ which, turn, text });
    if (process.env.FUSION_DEBUG) console.error(which, turn, text.slice(-(Number(process.env.FUSION_DEBUG_LEN) || 600)));
    if (which === 'STAGE-PARSE') return { calls: [['return_result', { status: 'success', value: parseValues[0] }]] };
    if (which === 'STAGE-ANALYZE') {
      const body = /exact JSON string body=("[^"]*")/.exec(text)?.[1];
      return { calls: [['return_result', { status: 'success', value: `checked:${body ? JSON.parse(body) : (/TREE-\d/.exec(text)?.[0] ?? '?')}` }]] };
    }
    if (turn === 0) return { calls: [['eval', { code: 'const tree = await parse(source);\nconst checked = await analyze(tree);\nreturn_result(checked);' }]] };
    return { calls: [['return_result', { status: 'success' }]] };
  };
}
const ORCH = {
  'natlang.json': '{}',
  'orch.nl': stage('  source: string', 'string', 'ORCH. tree = parse(source). checked = analyze(tree). Return checked.'),
  'orch/parse.nl': stage('  source: string', 'string', 'STAGE-PARSE: parse source.'),
  'orch/analyze.nl': stage('  syntax: string', 'string', 'STAGE-ANALYZE: analyze syntax.'),
};
const fusionEvents = traces => traces.flatMap(trace => trace.events.filter(event => event.kind.startsWith('fusion')).map(event => ({ ...event, call: trace.name })));

async function runOrch({ fusion, emulate, source = 'int main(){}' }) {
  const root = folder(ORCH);
  const facts = fusionFacts(root, nodeSourceFiles(root));
  const plan = crispPlan(facts);
  const edges = fusedEdges(facts, plan);
  const log = [], traces = [];
  const emulation = emulate ? createTextNeuraleseEmulation() : undefined;
  const script = scriptedModel(log);
  const model = emulation ? emulation.wrap(Object.assign(script, {})) : script;
  const runtime = createNatlangRuntime({ model, trace: trace => traces.push(trace), calls: false,
    ...(emulation ? { neuralese: emulation.runtime } : {}),
    ...(fusion ? { fusion: { edges, ...fusion(emulation) } } : {}) });
  const orch = loadNatlang(join(root, 'orch.nl'), root);
  const value = await runtime.run(() => orch(source));
  return { value, log, traces, edges, emulation, runtime, facts };
}

test('fusion off (the default) changes nothing and traces nothing', async () => {
  const plain = await runOrch({ fusion: undefined });
  const off = await runOrch({ fusion: () => ({ mode: 'off' }) });
  assert.equal(plain.value, 'checked:TREE-1');
  assert.equal(off.value, plain.value);
  assert.equal(fusionEvents(plain.traces).length, 0);
  assert.equal(fusionEvents(off.traces).length, 0);
  assert.equal(plain.log.length, off.log.length);
  assert.equal(plain.edges.length, 1);
});

test('on, under the text emulation: the producer returns a block, the consumer reads it, traces mark the emulation', async () => {
  const run = await runOrch({ emulate: true, fusion: emulation => ({ mode: 'on', emulation: { dialect: TEXT_NEURALESE_DIALECT } }) });
  assert.equal(run.value, 'checked:TREE-1', 'the analyzer read the producer\'s block');
  const events = fusionEvents(run.traces);
  const fused = events.find(event => event.kind === 'fusion_edge' && event.status === 'fused');
  const consumed = events.find(event => event.kind === 'fusion_edge' && event.status === 'consumed');
  assert.ok(fused && consumed, JSON.stringify(events));
  assert.equal(fused.emulation, true);
  assert.equal(fused.call, 'orch');
  assert.match(consumed.block, /^nz1_/);
  assert.equal(consumed.param, 'syntax');
  const analyze = run.log.find(entry => entry.which === 'STAGE-ANALYZE');
  assert.match(analyze.text, /Neuralese text block id=nz1_/, 'the analyzer was shown a block, not the text');
});

test('on without a certificate or a Neuralese backend falls back to text and says why', async () => {
  const noCertificate = await runOrch({ emulate: false, fusion: () => ({ mode: 'on' }) });
  assert.equal(noCertificate.value, 'checked:TREE-1');
  const fallback = fusionEvents(noCertificate.traces).find(event => event.kind === 'fusion_fallback');
  assert.match(fallback.reason, /no Neuralese store and write port/);
  // A store and port but a driver that cannot carry Neuralese, and no certificate.
  const store = new MemoryNeuraleseStore();
  const port = new StandInNeuralesePort(store, hashingEmbedder(8), 8, 'nd:natlang@1');
  const status = options => fusionStatus({ mode: 'on', edges: [], ...options }, { store, port }, { id: 'm', driver: Object.assign(() => ({}), { neuralese: true }) });
  assert.match(status({}).reason, /no runtime-qualification certificate/);
  const certificate = { schema: 'natlang.fusion-certificate/1', model: { id: 'm', weights_sha256: 'abc' }, dialect: 'nd:natlang@1',
    runtime_qualification: { status: 'passed', scope: ['transport', 'gradient-replay', 'typed-function-execution'], report_sha256: 'r',
      self_feedback: { channel_equivalence: 'passed', generation_quality: 'passed' } } };
  assert.match(status({ certificate }).reason, /different weights/, 'the profile must name the weights');
  assert.match(status({ certificate, weights: 'other' }).reason, /different weights/);
  assert.match(status({ certificate: { ...certificate, dialect: 'nd:natlang@2' }, weights: 'abc' }).reason, /dialect/);
  assert.match(status({ certificate: { ...certificate, runtime_qualification: { ...certificate.runtime_qualification, self_feedback: undefined } }, weights: 'abc' }).reason, /self-feedback/);
  assert.deepEqual(status({ certificate, weights: 'abc' }), { ok: true, emulation: false });
  const plainDriver = fusionStatus({ mode: 'on', edges: [], certificate, weights: 'abc' }, { store, port }, { id: 'm', driver: () => ({}) });
  assert.match(plainDriver.reason, /does not carry Neuralese/);
});

test('shadow serves text and records the fused producer\'s agreement', async () => {
  const readback = ref => async () => undefined;
  const agreeing = await runOrch({ emulate: true, fusion: emulation => ({ mode: 'shadow', emulation: { dialect: TEXT_NEURALESE_DIALECT },
    readback: async ref => {
      const block = await emulation.store.get(ref.$neuralese.id);
      return block.meta.producer.text_body_sha256 === sha('TREE-1') ? 'TREE-1' : undefined;
    } }) });
  assert.equal(agreeing.value, 'checked:TREE-1');
  assert.ok(!agreeing.log.some(entry => entry.which === 'STAGE-ANALYZE' && /Neuralese text block/.test(entry.text)), 'text is served');
  const shadow = fusionEvents(agreeing.traces).find(event => event.kind === 'fusion_shadow');
  assert.ok(shadow, 'a fusion_shadow event');
  assert.equal(shadow.agree, true);
  assert.equal(shadow.emulation, true);
  const disagreeing = await runOrch({ emulate: true, fusion: emulation => ({ mode: 'shadow', emulation: { dialect: TEXT_NEURALESE_DIALECT },
    readback: async () => 'something else' }) });
  assert.equal(fusionEvents(disagreeing.traces).find(event => event.kind === 'fusion_shadow').agree, false);
  void readback;
});

test('the CLI helpers: fusion settings validate and default to off with the crisp planner', () => {
  assert.deepEqual(parseFusionSettings(undefined), { mode: 'off', planner: 'crisp' });
  assert.deepEqual(parseFusionSettings({ mode: 'shadow', planner: 'nl', certificate: 'cert.json' }), { mode: 'shadow', planner: 'nl', certificate: 'cert.json' });
  assert.throws(() => parseFusionSettings({ mode: 'sometimes' }), /fusion.mode/);
  assert.throws(() => parseFusionSettings({ planner: 'x' }), /fusion.planner/);
  assert.throws(() => parseFusionSettings({ other: 1 }), /unknown fusion field/);
});

test('natlang check reports the plan in text and in --json, and a natlang.json fusion block selects mode and planner', () => {
  const root = folder({ ...ORCH, 'tsconfig.json': JSON.stringify({ compilerOptions: { target: 'ES2022', module: 'NodeNext', moduleResolution: 'NodeNext', strict: true, skipLibCheck: true }, include: ['*.ts'] }),
    'natlang.json': JSON.stringify({ schema: 'natlang.package/v2', name: 'fz', version: '0.1.0', include: ['orch.nl', 'orch'], fusion: { mode: 'shadow' } }) });
  const bin = fileURLToPath(new URL('../bin/natlang.mjs', import.meta.url));
  const json = spawnSync(process.execPath, [bin, 'check', root, '--json'], { encoding: 'utf8' });
  assert.equal(json.status, 0, json.stderr + json.stdout);
  const report = JSON.parse(json.stdout).fusion;
  assert.equal(report.mode, 'shadow');
  assert.equal(report.planner, 'crisp');
  assert.equal(report.counts.fuse, 1);
  assert.equal(report.edges[0].chain, 'parse -> analyze');
  assert.equal(report.edges[0].decision, 'fuse');
  const text = spawnSync(process.execPath, [bin, 'check', root], { encoding: 'utf8' });
  assert.match(text.stdout, /fusion: 1 candidate hand-off in 1 scope, 1 fusible/);
  assert.match(text.stdout, /fuse\s+parse -> analyze/);
});

test('the benchmark harness runs a chain as text and fused and applies the gate; an emulated run never passes it', async () => {
  const { runBench, DEFAULT_GATE } = await import('../scripts/fusion-bench.mjs');
  const root = folder(ORCH);
  const emulation = createTextNeuraleseEmulation();
  const createDriver = () => {
    const script = scriptedModel([]);
    return emulation.wrap(async request => ({ ...(await script(request)), prompt_tokens: 100, completion_tokens: 10 }));
  };
  const report = await runBench({ app: root, entry: 'orch.nl', cases: [{ id: 'c1', args: ['int main(){}'], expected: 'checked:TREE-1' }],
    createDriver, neuralese: emulation.runtime, emulation: { dialect: TEXT_NEURALESE_DIALECT } });
  assert.equal(report.modes.text.accuracy, 1);
  assert.equal(report.modes.fused.accuracy, 1);
  assert.equal(report.comparison.agreement, 1);
  assert.equal(report.comparison.engaged, true);
  assert.ok(report.modes.fused.tokens.total > 0 && report.modes.text.tokens.total > 0);
  assert.equal(report.modes.fused.fusion.emulation, true);
  assert.equal(report.gate.result.emulation, false);
  assert.equal(report.gate.pass, false, 'a text emulation cannot enable fusion');
  assert.equal(DEFAULT_GATE.minAgreement, 0.98);
  const dry = await runBench({ app: root, entry: 'orch.nl', cases: [], dryRun: true });
  assert.equal(dry.dryRun, true);
  assert.equal(dry.edges.length, 1);
});
