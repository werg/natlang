import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';
import { buildProject, createNatlangRuntime, loadNatlang, nodeSourceFiles } from '../dist/index.js';
import { openCallStore } from '../dist/calls/store.js';
import { fusionFacts, observedFromStore, formatFusionReport, crispPlan, verifyPlan, repairedPlan, nlPlan, nlPlannerFrom, planFusion, fusedEdges, fusionStatus, fusionReport,
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
  // Edges of natural-language scopes engage by their calling function, edges of TypeScript scopes by the marked call sites.
  assert.deepEqual(fusedEdges(facts, plan).map(edge => edge.producer.source), ['orch/parse.nl', 'first.nl']);
  assert.equal(fusedEdges(facts, plan).filter(edge => edge.sites).length, 1);
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

// --- Observed readers ---------------------------------------------------------------------------------------------

const PIPE = {
  'natlang.json': '{}',
  'pipe.nl': stage('  x: string', 'string', 'PIPE. Call first(x), then second(the result). Return the answer of second.'),
  'pipe/first.nl': stage('  x: string', 'string', 'STAGE-FIRST.'),
  'pipe/second.nl': stage('  a: string', 'string', 'STAGE-SECOND.'),
};
const PIPE_INSTRUCTIONS = 'PIPE. Call first(x), then second(the result). Return the answer of second.';
const CHAIN_BOUND = 'const a = await first(x);\nreturn await second(a);';
const CHAIN_NESTED = 'return await second(await first(x));';
const CHAIN_READ = 'const a = await first(x);\nconsole.log(a.length);\nreturn await second(a);';
const runs = (...programs) => programs.map((evals, index) => ({ id: `run-${index}`, evals: Array.isArray(evals) ? evals : [evals] }));
const observing = (list, minRuns = 3) => ({ minRuns, runs: (scope, instructions) => ({ revision: `test:${list.length}`, runs: scope === 'pipe.nl' && instructions.trim() === PIPE_INSTRUCTIONS ? list : [] }) });
const pipeEdge = facts => facts.edges.find(edge => edge.chain === 'first -> second');

test('observed readers: recorded eval code proves a hand-off the prose only implies, with its count and store revision', () => {
  const root = folder(PIPE);
  const files = nodeSourceFiles(root);
  const plain = fusionFacts(root, files);
  assert.equal(pipeEdge(plain).flow, 'implicit');
  assert.equal(pipeEdge(plain).observed, undefined, 'without recorded runs the readers are proven or unknown, never observed');
  assert.equal(crispPlan(plain).edges[0].decision, 'keep-text');

  const supporting = runs(CHAIN_BOUND, CHAIN_NESTED, ['const a = await first(x);', 'return await second(a);']);
  const facts = fusionFacts(root, files, { observed: observing(supporting) });
  const edge = pipeEdge(facts);
  assert.deepEqual(edge.observed, { runs: 3, contradicted: 0, minRuns: 3, revision: 'test:3' });
  assert.deepEqual(edge.readers.map(reader => [reader.kind, reader.certain]), [['consumer', true]]);
  const plan = crispPlan(facts, { observedMinRuns: 3 });
  assert.equal(plan.edges[0].decision, 'fuse');
  assert.match(plan.edges[0].reason, /3 recorded runs/);
  assert.deepEqual(plan.evidence, [{ edge: edge.id, readers: 'observed', runs: 3, contradicted: 0, minRuns: 3, revision: 'test:3' }]);
  assert.equal(fusedEdges(facts, plan).length, 1);
  const report = fusionReport(facts, plan, parseFusionSettings(undefined));
  assert.equal(report.counts.observedFuse, 1);
  assert.equal(report.edges[0].readerEvidence.observed.runs, 3);
  assert.match(formatFusionReport(report, false), /readers observed: 3 runs, store test:3/);
});

test('observed readers: the verifier needs the minimum number of supporting runs, and a single contradicting run keeps text', async () => {
  const root = folder(PIPE);
  const files = nodeSourceFiles(root);
  const few = fusionFacts(root, files, { observed: observing(runs(CHAIN_BOUND, CHAIN_NESTED), 3) });
  assert.equal(crispPlan(few, { observedMinRuns: 3 }).edges[0].decision, 'keep-text');
  assert.match(crispPlan(few, { observedMinRuns: 3 }).edges[0].reason, /only observed \(2 recorded runs, at least 3 needed\)/);
  assert.equal(crispPlan(few, { observedMinRuns: 2 }).edges[0].decision, 'fuse');
  // A planner that fuses anyway is rejected by the verifier at the configured minimum, and its edge falls back to text.
  const fuseAll = async edges => edges.map(item => ({ edge: item.id, decision: 'fuse', reason: 'scripted' }));
  const rejected = await nlPlan(few, fuseAll, { observedMinRuns: 3 });
  assert.equal(rejected.edges[0].decision, 'keep-text');
  assert.match(rejected.edges[0].reason, /only observed/);
  assert.equal(rejected.evidence[0].runs, 2);
  const accepted = await nlPlan(few, fuseAll, { observedMinRuns: 2 });
  assert.equal(accepted.edges[0].decision, 'fuse');
  assert.equal(accepted.evidence[0].minRuns, 2);

  const contradicted = fusionFacts(root, files, { observed: observing(runs(CHAIN_BOUND, CHAIN_NESTED, CHAIN_BOUND, CHAIN_READ)) });
  const edge = pipeEdge(contradicted);
  assert.equal(edge.observed.runs, 3);
  assert.equal(edge.observed.contradicted, 1);
  assert.ok(edge.readers.some(reader => reader.kind === 'service' || reader.kind === 'eval'), 'the reader the run shows is listed');
  const plan = crispPlan(contradicted, { observedMinRuns: 1 });
  assert.equal(plan.edges[0].decision, 'keep-text');
  assert.equal(verifyPlan(contradicted, { edges: [{ edge: edge.id, decision: 'fuse', reason: 'x' }] }, { observedMinRuns: 1 }).ok, false);
});

test('observed readers: the model\'s own reads, results and fan-out count as readers', () => {
  const root = folder(PIPE);
  const files = nodeSourceFiles(root);
  const contradicting = {
    'a field read': 'const a = await first(x);\nconst n = a.length;\nreturn await second(a);',
    'returned': 'const a = await first(x);\nawait second(a);\nreturn a;',
    'shown': 'const a = await first(x);\nreturn { a, b: await second(a) };',
    'passed to two calls': 'const a = await first(x);\nawait second(a);\nreturn await second(a);',
    'unbound': 'await first(x);\nreturn await second(x);',
    'destructured': 'const { y } = await first(x);\nreturn await second(y);',
    'never handed on': 'const a = await first(x);\nreturn a;',
  };
  for (const [name, program] of Object.entries(contradicting)) {
    const edge = pipeEdge(fusionFacts(root, files, { observed: observing(runs(program)) }));
    assert.equal(edge.observed.contradicted, 1, name);
    assert.equal(edge.observed.runs, 0, name);
  }
  // A value that crosses evals still counts: the first eval's variable is read in the second.
  const across = pipeEdge(fusionFacts(root, files, { observed: observing(runs([['const a = await first(x);', 'return await second(a);']])) }));
  assert.equal(across.observed.runs, 1);
});

test('observed readers come from the call store: only model-driven runs of the same instructions count', () => {
  const root = folder(PIPE);
  const storeRoot = mkdtempSync(join(tmpdir(), 'natlang-fusion-store-'));
  const store = openCallStore(storeRoot);
  const ref = text => ({ complete: true, hash: sha(text), bytes: Buffer.byteLength(text) });
  let n = 0;
  const put = (evals, { instructions = PIPE_INSTRUCTIONS, model = 'model-a', tokens = 10, source = 'app/pipe.nl' } = {}) => {
    const id = `run-${++n}`;
    const text = JSON.stringify(instructions), output = JSON.stringify('done');
    store.record({ version: 'natlang.calls/1', call_id: id, parent_call_id: null, parent_action_index: null, task_id: `task-${n}`, program_id: null, build_hash: null,
      program_root: null, definition: { id: 'pipe-id', name: 'pipe', source, key: 'pipe-key', interface: 'iface', site: 'named', subtype: 'function',
        params: [{ name: 'x', type: 'string' }], returns: 'string', instructions: ref(text), types: {} },
      executor: { kind: 'agent', model_id: model, model_revision: null }, inputs: {}, captures: {}, capture_writes: [], output: ref(output),
      outcome: 'done', detail: '', started_at: new Date(Date.UTC(2026, 0, 1, 0, 0, n)).toISOString(), ended_at: new Date(Date.UTC(2026, 0, 1, 0, 0, n + 1)).toISOString(),
      effects: [], folder: null, approach: { evals, hash: `h${n}` }, cost: { model_requests: 1, tokens_in: tokens, tokens_out: 5, wall_ms: 10, turns: 1, evals: evals.length },
      features: {}, audit_of: null, events: null }, new Map([[sha(text), text], [sha(output), output]]));
  };
  try {
    put([CHAIN_BOUND]); put([CHAIN_NESTED]); put([CHAIN_BOUND]);
    put([CHAIN_READ], { model: 'undeclared:scripted' });
    put([CHAIN_READ], { instructions: 'an older version of the prose' });
    put([CHAIN_READ], { tokens: 0 });
    put([CHAIN_READ], { source: 'other/elsewhere.nl' });
    const facts = fusionFacts(root, nodeSourceFiles(root), { observed: observedFromStore(store, 3) });
    const edge = pipeEdge(facts);
    assert.equal(edge.observed.runs, 3);
    assert.equal(edge.observed.contradicted, 0);
    assert.match(edge.observed.revision, /^3:[0-9a-f]{16}$/);
    assert.equal(crispPlan(facts, { observedMinRuns: 3 }).edges[0].decision, 'fuse');
    // A fourth run that reads the value (a real model's) turns the observation into a contradiction.
    put([CHAIN_READ]);
    const after = pipeEdge(fusionFacts(root, nodeSourceFiles(root), { observed: observedFromStore(store, 3) }));
    assert.equal(after.observed.contradicted, 1);
    assert.notEqual(after.observed.revision, edge.observed.revision, 'the revision names the runs read');
  } finally { store.close(); rmSync(storeRoot, { recursive: true, force: true }); }
});

test('observed settings validate, default to off, and are never applied by a plain facts call', () => {
  assert.equal(parseFusionSettings({ mode: 'off' }).observed, undefined);
  assert.deepEqual(parseFusionSettings({ observed: true }).observed, { minRuns: 20 });
  assert.deepEqual(parseFusionSettings({ observed: { minRuns: 5, store: 'calls' } }).observed, { minRuns: 5, store: 'calls' });
  assert.throws(() => parseFusionSettings({ observed: { minRuns: 0 } }), /minRuns/);
  assert.throws(() => parseFusionSettings({ observed: { other: 1 } }), /unknown fusion.observed field/);
});

test('explicit chains: a nested call or a variable used only by the next stage is proven in prose, a field read is not', () => {
  const decide = body => {
    const root = folder({ ...PIPE, 'pipe.nl': stage('  x: string', 'string', body) });
    const facts = fusionFacts(root, nodeSourceFiles(root));
    return { edge: pipeEdge(facts), plan: crispPlan(facts) };
  };
  const nested = decide('Return second(first(x)).');
  assert.equal(nested.edge.flow, 'nested');
  assert.equal(nested.plan.edges[0].decision, 'fuse');
  const bound = decide('a = first(x). Return second(a).');
  assert.equal(bound.edge.flow, 'bound');
  assert.equal(bound.plan.edges[0].decision, 'fuse');
  const read = decide('a = first(x). When a.length is 0 return "". Return second(a).');
  assert.equal(read.plan.edges[0].decision, 'keep-text');
});

// --- Crisp TypeScript orchestrators: the compiler marks the calls, the runtime fuses the planned edges -------------

const TS_APP = {
  'package.json': JSON.stringify({ name: 'fixture-fusion', private: true, type: 'module' }),
  'natlang.json': '{}',
  'tsconfig.json': JSON.stringify({ compilerOptions: { target: 'ES2022', module: 'NodeNext', moduleResolution: 'NodeNext', strict: true, skipLibCheck: true,
    rootDir: 'src', outDir: 'dist' }, include: ['src/**/*.ts'] }),
  'src/parse.nl': stage('  source: string', 'string', 'STAGE-PARSE: parse source.'),
  'src/analyze.nl': stage('  syntax: string', 'string', 'STAGE-ANALYZE: analyze syntax.'),
  'src/host.ts': `import parse from './parse.nl';
import analyze from './analyze.nl';
export async function nested(source: string) { return analyze(await parse(source)); }
export async function bound(source: string) { const tree = await parse(source); return analyze(tree); }
export async function reads(source: string) { const tree = await parse(source); console.log(tree.length); return analyze(tree); }
export async function twice(source: string) { const tree = await parse(source); return [await analyze(tree), await analyze(tree)]; }
export async function unrelated(source: string) { return analyze(source); }
`,
};
const RUNTIME_MODULE = { url: new URL('../dist/index.js', import.meta.url).href,
  types: fileURLToPath(new URL('../dist/index.d.ts', import.meta.url)), specifiers: ['@natlang/node'] };

test('TypeScript orchestrators: the facts name each hand-off\'s call sites; a re-read, shared or exported value is not fusible', () => {
  const root = folder({ ...TS_APP, 'src/exported.ts': `import parse from './parse.nl';
import analyze from './analyze.nl';
export const tree = await parse('x');
export const checked = await analyze(tree);
` });
  const facts = fusionFacts(root, nodeSourceFiles(root));
  const plan = crispPlan(facts);
  const byScope = Object.fromEntries(['nested', 'bound', 'reads', 'twice', 'unrelated'].map(name => [name, facts.edges.filter(edge => edge.scope === `src/host.ts#${name}`)]));
  const decision = edge => plan.edges.find(entry => entry.edge === edge.id).decision;
  assert.deepEqual(byScope.nested.map(decision), ['fuse']);
  assert.deepEqual(byScope.bound.map(decision), ['fuse']);
  assert.deepEqual(byScope.reads.map(decision), ['keep-text']);
  assert.ok(byScope.twice.length && byScope.twice.every(edge => decision(edge) === 'keep-text'), 'a value two calls receive is not fused');
  assert.equal(byScope.unrelated.length, 0);
  const exported = facts.edges.filter(edge => edge.scope.startsWith('src/exported.ts'));
  assert.equal(exported.length, 1);
  assert.equal(decision(exported[0]), 'keep-text');
  assert.ok(exported[0].readers.some(reader => reader.kind === 'host-return' && /exported/.test(reader.detail)));
  for (const edge of byScope.nested.concat(byScope.bound)) assert.match(edge.sites.producer, /^src\/host\.ts@\d+$/);
  assert.equal(fusedEdges(facts, plan).filter(edge => edge.sites).length, 2);
  assert.equal(fusionReport(facts, plan, parseFusionSettings(undefined)).edges.filter(edge => edge.engaged).length, 2);
});

test('TypeScript orchestrators: with fusion on under the emulation the planned hand-offs pass a block, the others stay text; off changes nothing', async () => {
  const root = folder(TS_APP);
  const result = buildProject({ project: root, runtimeModule: RUNTIME_MODULE });
  assert.equal(result.ok, true, JSON.stringify(result.diagnostics));
  const host = join(root, 'dist/host.js');
  const code = readFileSync(host, 'utf8');
  // nested: parse, analyze; bound: parse, analyze; reads: parse, analyze; twice: parse, both analyzes.
  assert.equal((code.match(/__natlang\.fuseSite\(/g) ?? []).length, 9, code);
  assert.ok(!/unrelated[^\n]*fuseSite/.test(code), 'a call with no producer is left alone');
  const facts = fusionFacts(root, nodeSourceFiles(root));
  const edges = fusedEdges(facts, crispPlan(facts));
  const module = await import(pathToFileURL(host).href);

  const run = async (name, fusion) => {
    const log = [];
    const emulation = createTextNeuraleseEmulation();
    const runtime = createNatlangRuntime({ model: emulation.wrap(scriptedModel(log)), calls: false, neuralese: emulation.runtime,
      ...(fusion ? { fusion: { edges, ...fusion } } : {}) });
    const value = await runtime.run(() => module[name]('int main(){}'));
    return { value, log, analyze: log.filter(entry => entry.which === 'STAGE-ANALYZE') };
  };
  const on = { mode: 'on', emulation: { dialect: TEXT_NEURALESE_DIALECT } };
  for (const name of ['nested', 'bound']) {
    const fused = await run(name, on);
    assert.equal(fused.value, 'checked:TREE-1', name);
    assert.match(fused.analyze[0].text, /Neuralese text block id=nz1_/, `${name}: the analyzer was shown a block`);
  }
  for (const name of ['reads', 'twice']) {
    const kept = await run(name, on);
    assert.ok(kept.analyze.length && kept.analyze.every(entry => !/Neuralese text block/.test(entry.text)), `${name}: text`);
  }
  const off = await run('nested', undefined);
  assert.equal(off.value, 'checked:TREE-1');
  assert.ok(!/Neuralese text block/.test(off.analyze[0].text), 'fusion off: text');
  const explicitOff = await run('nested', { mode: 'off' });
  assert.ok(!/Neuralese text block/.test(explicitOff.analyze[0].text));
});
