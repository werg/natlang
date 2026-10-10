/**
 * The improver's training exporter on recorded fixture runs of both generations: the stages of the current improver
 * (diagnose, hypothesize, editSource, the natural-language policies) and the single rewriteProgram editor of runs
 * recorded earlier. Each row names its stage and generation, declares whole-trajectory supervision, and is admitted by the
 * run's own measurement of the experiment it belonged to. Every model is scripted; the "recordings" are real runs.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, mkdirSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Folder, improveProgram, createNatlangRuntime } from '../dist/index.js';
import { AUTHORED_IMPROVER } from '../dist/improvement/authored-source.js';
import { loadVirtualNatlang } from '../dist/runtime/virtual-project.js';
import { EVALUATOR_DECLARATION } from '../dist/improvement/services.js';
import { exportCase, stageCounts } from '../scripts/self-improvement/export-followup-training.mjs';
import { IMPROVER_DEFINITIONS, OMITTED_DEFINITIONS, stageOf, supervisionOf, linkExperiments, stageAdmission, measuredEditAdmission, invocationFields } from '../scripts/self-improvement/improver-stages.mjs';
import { scriptedModel } from './support/natlang.mjs';
import { stagedImprover, finishing } from './support/improver.mjs';

const dist = fileURLToPath(new URL('../dist/', import.meta.url));
const contract = { entry: 'main.ts', exportName: 'solve', programId: 'export-fixture', signature: 'solve(value: number): number' };
const cases = ['train', 'validation', 'test'].flatMap(split => [1, 2].map(value => ({ id: `${split}-${value}`, group: `${split}-${value}`, split, args: [value], expected: value + 1 })));
const baseline = { 'main.ts': 'export function solve(value: number): number { return value; }' };
const policy = (extra = {}) => ({ maxExperiments: 3, maxPopulation: 3, strategy: 'adaptive', mode: 'structural', goal: 'increment', allowedFiles: ['main.ts'], ...extra });
const budget = { maxRollouts: 60, maxModelCalls: 80, maxProposals: 8 };
const write = delta => `await folder.file("main.ts").writeText("export function solve(value: number): number { return value + ${delta}; }"); return {summary:"add ${delta}",preserves:["number contract"]};`;

/** A recorded run of the current improver: its traces and provider exchanges, with the run directory the exporter reads. */
async function recordStagesRun(script, options = {}) {
  const root = mkdtempSync(join(tmpdir(), 'natlang-export-stages-'));
  const exchanges = [], traces = [];
  const model = scriptedModel(stagedImprover(script));
  const recording = finishing(model);
  const improver = async (request, signal) => {
    const turn = await recording(request, signal);
    exchanges.push({ command: 'native', role: 'optimizer', request: structuredClone(request), turn: structuredClone(turn) });
    return turn;
  };
  const journal = join(root, 'journal');
  const result = await improveProgram({ folder: Folder.fromFiles(baseline), contract, cases, policy: policy(options.policy), improver,
    executor: () => { throw Error('no inference'); }, executorId: 'exact', budget, directory: journal, trace: trace => traces.push(structuredClone(trace)), captureExactRewriteIO: true });
  assert.equal(result.disposition, 'improved', result.error);
  return recording_({ root, exchanges, traces, journal, state: result.state, authored: AUTHORED_IMPROVER, id: 'fixture-stages' });
}

/** The files the exporter reads (protocol.json, native.json, replay.json), written beside a run. */
function recording_({ root, exchanges, traces, journal, state, authored, id }) {
  const protocol = { id, files: baseline, contract, cases, policy: policy(), budget, executor: { model: 'exact' }, executorTimeoutMs: 5000, sourceGroups: [], incidents: [] };
  writeFileSync(join(root, 'protocol.json'), JSON.stringify(protocol));
  writeFileSync(join(root, 'native.json'), JSON.stringify({ state, authored: { files: authored } }));
  const replay = { id, runtime: { directory: dist, identity: 'test-dist' }, verified: true, source: { directory: root, originalHash: 'fixture' },
    traces, exchanges, targetExchanges: [], journalPath: journal, audit: { providerCalls: 0 }, migration: { currentCompiler: 6 } };
  const path = join(root, 'replay.json');
  writeFileSync(path, JSON.stringify(replay));
  return { path, root, replay };
}
const rewrite = (path, change) => { const replay = JSON.parse(readFileSync(path, 'utf8')); change(replay); writeFileSync(path, JSON.stringify(replay)); };

test('stages generation: every stage of an accepted experiment is a row that names its definition and generation', async () => {
  const run = await recordStagesRun({ edit: write(1) });
  const result = await exportCase(run.path);
  assert.deepEqual(result.omitted, []);
  const stages = new Set(result.rows.map(row => row.improver_stage.stage));
  assert.deepEqual([...stages].sort(), ['diagnose', 'editSourceStructural', 'hypothesize']);
  for (const row of result.rows) {
    assert.equal(row.improver_stage.generation, 'stages/1');
    assert.equal(row.improver_stage.definition_source, `improveStep/${row.improver_stage.stage}.nl`);
    assert.equal(row.task.program_ir.semantics.root, row.improver_stage.definition_source, 'the row is keyed on the recorded definition');
    assert.equal(row.training_admission.approved, true);
  }
  const edit = result.rows.find(row => row.improver_stage.stage === 'editSourceStructural');
  assert.equal(edit.improver_stage.mode, 'structural');
  assert.equal(edit.task_kind, 'directory-reducer');
  assert.equal(result.rows.find(row => row.improver_stage.stage === 'diagnose').task_kind, 'function');
  assert.match(edit.training_admission.reason, /add 1/, 'the admission reason is the run\'s own record of the accepted experiment');
  assert.deepEqual(stageCounts(result.rows)['improveStep/diagnose.nl'], { generation: 'stages/1', rows: result.rows.filter(row => row.improver_stage.stage === 'diagnose').length });
});

test('every row declares whole-trajectory supervision: instructions and inputs trained, feedback at the lower weight, nothing masked', async () => {
  const run = await recordStagesRun({ edit: write(1) });
  const { rows } = await exportCase(run.path);
  assert.ok(rows.length > 0);
  for (const row of rows) {
    assert.equal(row.supervision.scope, 'whole-trajectory');
    assert.deepEqual(row.supervision.masked, []);
    assert.equal(row.supervision.classes.length, row.messages.length);
    assert.equal(row.supervision.classes[0], 'instructions-inputs');
    assert.equal(row.supervision.weights['instructions-inputs'], 'context');
    assert.equal(row.supervision.weights.feedback, 'feedback');
  }
  const later = rows.find(row => row.messages.some(message => message.role === 'tool'));
  if (later) assert.ok(later.supervision.classes.includes('feedback') || later.supervision.classes.includes('earlier-turn'));
});

test('the supervision classes follow the trainer: before the first reply trained, up to the last reply supervised by its row, after it feedback', () => {
  const messages = [{ role: 'system' }, { role: 'user' }, { role: 'assistant' }, { role: 'tool' }, { role: 'assistant' }, { role: 'tool' }, { role: 'user' }];
  assert.deepEqual(supervisionOf(messages).classes, ['instructions-inputs', 'instructions-inputs', 'earlier-turn', 'earlier-turn', 'earlier-turn', 'feedback', 'feedback']);
  assert.deepEqual(supervisionOf(messages.slice(0, 2)).classes, ['instructions-inputs', 'instructions-inputs']);
});

test('a rejected experiment stays context only: its stages are registered, not trained on', async () => {
  let edits = 0;
  const run = await recordStagesRun({ edit: () => (++edits === 1 ? write(2) : write(1)) });
  const result = await exportCase(run.path);
  const contextOnly = result.failureInvocations.map(row => row.stage).sort();
  assert.deepEqual(contextOnly, ['diagnose', 'editSourceStructural', 'hypothesize'], 'the rejected experiment');
  for (const row of result.failureInvocations) { assert.equal(row.generation, 'stages/1'); assert.match(row.disposition, /^context-only-/); assert.match(row.reason, /Measured candidate/); }
  const trained = new Set(result.rows.map(row => row.program_id));
  for (const row of result.failureInvocations) assert.equal(trained.has(row.id), false);
  assert.equal(result.rows.filter(row => row.improver_stage.stage === 'editSourceStructural').length, 1, 'only the accepted edit trains');
});

test('natural-language search policies are rows too, admitted by replay and the verifier that bounded them', async () => {
  const run = await recordStagesRun({ other: opening => opening.includes('Decide whether the improvement search ends') ? 'return {stop:true,reason:"Declared objective satisfied."};' : null, edit: write(1) },
    { policy: { policies: { shouldStop: 'nl' } } });
  const { rows } = await exportCase(run.path);
  const stop = rows.filter(row => row.improver_stage.stage === 'shouldStop');
  assert.equal(stop.length, 1);
  assert.equal(stop[0].improver_stage.shape, 'policy');
  assert.equal(stop[0].improver_stage.definition_source, 'improveStep/shouldStop.nl');
  assert.equal(stop[0].task.program_ir.semantics.expected.stop, true);
});

test('a shadow disagreement with the served crisp decision keeps the natural-language side as context', () => {
  const stage = stageOf('improveStep/shouldStop.nl');
  assert.equal(stageAdmission({ stage, invocation: {}, shadowDisagreements: new Set(['shouldStop']) }).approved, false);
  assert.equal(stageAdmission({ stage, invocation: {}, shadowDisagreements: new Set(['chooseParent']) }).approved, true);
});

test('an unknown definition is an error that lists the exported ones; a known non-exported one is a registered omission', async () => {
  assert.throws(() => stageOf('improveStep/bogus.nl'), /Unexpected optimizer invocation: improveStep\/bogus\.nl.*improveStep\/editSource\.nl/);
  const run = await recordStagesRun({ edit: write(1) });
  rewrite(run.path, replay => {
    const manifest = replay.traces.find(trace => trace.events.some(event => event.kind === 'manifest' && event.definition_source === 'improveStep/diagnose.nl')).events.find(event => event.kind === 'manifest');
    manifest.definition_source = 'counterexampleStep.nl';
  });
  const result = await exportCase(run.path);
  assert.equal(result.omitted.length, 1);
  assert.equal(result.omitted[0].root, 'counterexampleStep.nl');
  assert.equal(result.omitted[0].reason, OMITTED_DEFINITIONS.get('counterexampleStep.nl'));
  assert.equal(result.rows.some(row => row.improver_stage.stage === 'diagnose'), false);
  rewrite(run.path, replay => { replay.traces.find(trace => trace.events.some(event => event.kind === 'manifest' && event.definition_source === 'improveStep/hypothesize.nl')).events.find(event => event.kind === 'manifest').definition_source = 'improveStep/bogus.nl'; });
  await assert.rejects(() => exportCase(run.path), /Unexpected optimizer invocation: improveStep\/bogus\.nl/);
});

test('an edit with no measured experiment in the run state is context only, never a silent positive', () => {
  assert.deepEqual(measuredEditAdmission([{ candidate: 'a', accepted: true, reason: 'ok' }], 'a', false), { approved: true, reason: 'ok' });
  assert.equal(measuredEditAdmission([{ candidate: 'a', accepted: false, reason: 'worse' }], 'a', false).approved, false);
  assert.equal(measuredEditAdmission([], 'b', false).approved, false);
  assert.equal(measuredEditAdmission([], 'b', true).approved, true, 'an honest unchanged result');
});

test('stages are linked into experiments by the values they pass on', () => {
  const diagnosis = { observations: [], pattern: 'p' }, hypothesis = { kind: 'helper', statement: 's', files: ['main.ts'], predictedChange: 'c' };
  const invocations = [
    { callId: 'd1', stage: stageOf('improveStep/diagnose.nl'), args: { request: {} }, value: diagnosis },
    { callId: 'h1', stage: stageOf('improveStep/hypothesize.nl'), args: { request: { diagnosis } }, value: hypothesis },
    { callId: 'e1', stage: stageOf('improveStep/editSource.nl'), args: { request: { hypothesis, diagnosis } }, value: { summary: 'x', preserves: [] } },
    { callId: 'd2', stage: stageOf('improveStep/diagnose.nl'), args: { request: {} }, value: { observations: [], pattern: 'q' } },
  ];
  const links = linkExperiments(invocations);
  assert.deepEqual(links.get('e1'), { edit: 'e1', hypothesize: 'h1', diagnose: 'd1' });
  assert.equal(links.get('d1'), links.get('e1'));
  assert.deepEqual(links.get('d2'), { diagnose: 'd2' }, 'a diagnosis with no hypothesis is its own, unfinished experiment');
  const outcomes = new Map([['e1', { approved: true, reason: 'accepted' }]]);
  assert.equal(stageAdmission({ stage: invocations[0].stage, invocation: invocations[0], experiment: links.get('d1'), outcomes }).approved, true);
  assert.equal(stageAdmission({ stage: invocations[3].stage, invocation: invocations[3], experiment: links.get('d2'), outcomes }).approved, false);
  assert.equal(stageAdmission({ stage: invocations[1].stage, invocation: { value: { kind: 'none' } }, experiment: undefined, outcomes }).approved, true, 'an honest hypothesis of no edit');
});

test('exact host captures win over bounded state events, and a redacted value cannot be exported', () => {
  const capture = (name, value, extra = {}) => ({ kind: 'host_capture', call_id: 'c1', capture_kind: name === 'return' ? 'invocation_output' : 'invocation_input', name, complete: true, value, ...extra });
  const trace = { callId: 'c1', events: [capture('request', { a: 1 }), capture('facts', undefined, { complete: false, reason: 'missing' }), capture('return', { b: 2 })] };
  assert.deepEqual(invocationFields(trace), { args: { request: { a: 1 } }, value: { b: 2 }, captured: true });
  const redacted = { callId: 'c2', events: [{ kind: 'state', phase: 'initial', value: { $lambda: { args: { request: { $diagnostic_preview: 'cut' } } } } }, { kind: 'state', phase: 'final', value: { $lambda: { return: {} } } }] };
  assert.throws(() => invocationFields(redacted), /redacted/);
});

test('rewriteProgram generation: a recorded run keeps exporting, keyed on the old editor, with its parent step admitting it', async () => {
  const legacy = fileURLToPath(new URL('./fixtures/legacy-improver/', import.meta.url));
  const files = { 'types.ts': readFileSync(join(legacy, 'types.ts'), 'utf8'), 'improveStep/rewriteProgram.nl': readFileSync(join(legacy, 'improveStep/rewriteProgram.nl'), 'utf8') };
  for (const accepted of [true, false]) {
    const root = mkdtempSync(join(tmpdir(), 'natlang-export-legacy-'));
    const traces = [], exchanges = [];
    const model = scriptedModel(() => write(1).replace('preserves:', 'changed:["main.ts"],preserves:'));
    const driver = async (request, signal) => { const turn = await finishing(model)(request, signal); exchanges.push({ command: 'native', role: 'optimizer', request: structuredClone(request), turn: structuredClone(turn) }); return turn; };
    const folder = Folder.fromFiles(baseline);
    const task = createNatlangRuntime({ model: { driver, maxTurns: 16, maxTokens: 24000, turnTokens: 2048, maxFailureRepairs: 4 }, seed: { mode: 'derived', root: 0 }, codeEdits: 'deny', network: false,
      // The model-run step's runtime offered the evaluator service, scoped to improveStep.nl; the editor saw it declared.
      services: { evaluator: {} }, serviceDeclarations: { evaluator: EVALUATOR_DECLARATION }, serviceScopes: { evaluator: ['improveStep.nl'] },
      trace: trace => traces.push(structuredClone(trace)) });
    const request = { brief: 'Goal: increment', goal: 'increment', mode: 'structural', hypothesis: '', sourceFiles: [{ path: 'main.ts', text: baseline['main.ts'] }], evidence: [], allowedFiles: ['main.ts'] };
    await task.run(() => folder.apply(loadVirtualNatlang(files, 'improveStep/rewriteProgram.nl'), request));
    const editor = traces.find(trace => trace.events.some(event => event.kind === 'manifest' && event.definition_source === 'improveStep/rewriteProgram.nl'));
    assert.ok(editor, 'the legacy editor ran and was traced');
    // The model-run step that called it recorded the measured outcome of the experiment in its final state.
    editor.parentCallId = 'recorded-step';
    traces.push({ callId: 'recorded-step', parentCallId: null, outcome: 'done', events: [{ kind: 'manifest', definition_source: 'improveStep.nl' },
      { kind: 'state', phase: 'final', value: { $lambda: { return: { history: [{ accepted, reason: accepted ? 'Measured candidate: improved.' : 'Measured candidate: worse.' }] } } } }] });
    mkdirSync(join(root, 'journal'), { recursive: true });
    const run = recording_({ root, exchanges, traces, journal: join(root, 'journal'), state: { quality: 1, history: [{ accepted: true }] }, authored: files, id: 'fixture-legacy' });
    const result = await exportCase(run.path);
    if (accepted) {
      assert.ok(result.rows.length > 0);
      for (const row of result.rows) {
        assert.equal(row.improver_stage.stage, 'rewriteProgram');
        assert.equal(row.improver_stage.generation, 'rewriteProgram/1');
        assert.equal(row.improver_stage.definition_source, 'improveStep/rewriteProgram.nl');
        assert.equal(row.task.program_ir.semantics.root, 'improveStep/rewriteProgram.nl');
        assert.equal(row.supervision.scope, 'whole-trajectory');
      }
    } else {
      assert.equal(result.rows.length, 0);
      assert.equal(result.failureInvocations.length, 1);
      assert.equal(result.failureInvocations[0].generation, 'rewriteProgram/1');
      assert.match(result.failureInvocations[0].disposition, /^context-only-rewriteProgram$/);
    }
  }
});

test('both generations are in the table, each definition once, and the editors are the ones the skill exporter accepts', () => {
  const generations = new Map();
  for (const [source, stage] of IMPROVER_DEFINITIONS) generations.set(stage.generation, [...(generations.get(stage.generation) ?? []), source]);
  assert.deepEqual([...generations.keys()].sort(), ['rewriteProgram/1', 'stages/1']);
  assert.ok(generations.get('rewriteProgram/1').includes('improveStep/rewriteProgram.nl'));
  for (const source of ['improveStep/diagnose.nl', 'improveStep/hypothesize.nl', 'improveStep/editSource.nl', 'improveStep/editSourceStructural.nl',
    'improveStep/findOpportunity.nl', 'improveStep/chooseParent.nl', 'improveStep/selectIncumbent.nl', 'improveStep/shouldStop.nl'])
    assert.ok(generations.get('stages/1').includes(source), source);
  for (const source of IMPROVER_DEFINITIONS.keys().filter(source => source.endsWith('.nl') && source.includes('/')))
    if (source.startsWith('improveStep/') && source !== 'improveStep/rewriteProgram.nl') assert.ok(AUTHORED_IMPROVER[source], `${source} exists in the authored improver`);
});
