/**
 * The program improver's decomposition (applications/program-improver/DECOMPOSITION.md): the diagnose, hypothesize and
 * edit stages, typed dispositions in place of magic strings, the pluggable search policies and the journaled plan.
 * Every model here is scripted.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Folder, improveProgram, OperationJournal } from '../dist/index.js';
import { AUTHORED_IMPROVER } from '../dist/improvement/authored-source.js';
import { compileVirtualProject } from '../dist/runtime/virtual-project.js';
import * as runtime from '../dist/runtime/node.js';
import { scriptedModel } from './support/natlang.mjs';
import { stagedImprover, finishing, STAGES } from './support/improver.mjs';

const contract = { entry: 'main.ts', exportName: 'solve', programId: 'stages-fixture', signature: 'solve(value: number): number' };
const cases = ['train', 'validation', 'test'].flatMap(split => [1, 2].map(value => ({ id: `${split}-${value}`, group: `${split}-${value}`, split, args: [value], expected: value + 1 })));
const baseline = () => Folder.fromFiles({ 'main.ts': 'export function solve(value: number): number { return value; }' });
const budget = { maxRollouts: 60, maxModelCalls: 80, maxProposals: 8 };
const write = delta => `await folder.file("main.ts").writeText("export function solve(value: number): number { return value + ${delta}; }"); return {summary:"add ${delta}",preserves:["number contract"]};`;
const policy = (extra = {}) => ({ maxExperiments: 3, maxPopulation: 3, strategy: 'adaptive', mode: 'structural', goal: 'increment', allowedFiles: ['main.ts'], ...extra });
const run = (model, options = {}) => improveProgram({ folder: baseline(), contract, cases, policy: policy(), improver: finishing(model), executor: () => { throw Error('no inference'); },
  executorId: 'exact', budget, ...options });

test('a hypothesis of kind none ends the search with the typed disposition and no edit', async () => {
  const model = scriptedModel(stagedImprover({ hypothesize: 'return {kind:"none",statement:"the evidence supports no edit",files:[],predictedChange:"none"};',
    edit: 'throw Error("the edit stage must not run");' }));
  const result = await run(model);
  assert.equal(result.state.history.length, 1);
  assert.equal(result.state.history[0].disposition, 'no-hypothesis');
  assert.match(result.state.history[0].reason, /the evidence supports no edit/);
  assert.equal(result.state.done, true);
  assert.equal(result.state.stopReason, 'No further evidenced change.');
  assert.equal(model.openings.filter(opening => opening.includes(STAGES.edit)).length, 0);
});

test('the stages receive the evidence, the diagnosis and the history, and the edit receives the hypothesis', async () => {
  const model = scriptedModel(stagedImprover({
    diagnose: 'if(request.opportunity.kind!=="quality"||!Array.isArray(request.evidence)||!request.brief.includes("Training")||request.goal!=="increment")throw Error("diagnose request incomplete");return {observations:[{caseId:"train-1",whatHappened:"returned its input",expected:"input plus one",category:"wrong-result"}],pattern:"the program returns its input"};',
    hypothesize: 'if(request.diagnosis.pattern!=="the program returns its input"||request.sourceFiles.length!==1||request.mode!=="structural"||!request.allowedFiles.includes("main.ts"))throw Error("hypothesize request incomplete");return {kind:"helper",statement:"The helper returns its input instead of adding one.",files:["main.ts"],predictedChange:"train quality rises"};',
    edit: `if(request.hypothesis.kind!=="helper"||request.diagnosis.observations[0].caseId!=="train-1"||request.mode!=="structural"||request.transformation!==undefined)throw Error("edit request incomplete");${write(1)}` }));
  const result = await run(model);
  assert.equal(result.disposition, 'improved', result.error);
  assert.equal(result.state.history[0].disposition, 'accepted');
  assert.equal(result.state.history[0].candidate !== undefined, true);
  assert.equal(result.state.stopReason, 'Declared objective satisfied.');
  assert.ok(model.openings.some(opening => opening.includes(STAGES.diagnose)) && model.openings.some(opening => opening.includes(STAGES.hypothesize)));
});

test('instruction mode runs the instruction editor and structural mode the structural one', async () => {
  for (const mode of ['instruction', 'structural']) {
    const model = scriptedModel(stagedImprover({ edit: write(1) }));
    const sources = baseline();
    const result = await improveProgram({ folder: sources, contract, cases, policy: policy({ mode, maxExperiments: 1 }), improver: finishing(model),
      executor: () => { throw Error('no inference'); }, executorId: 'exact', budget });
    // Instruction mode keeps prose only; this scripted edit changes code, which the source policy refuses in that mode.
    const edits = model.openings.filter(opening => opening.includes(STAGES.edit));
    assert.equal(edits.length, 1, mode);
    assert.equal(edits[0].includes('Reorganize the execution structure'), mode === 'structural', mode);
    assert.equal(edits[0].includes('Change prose only'), mode === 'instruction', mode);
    if (mode === 'structural') assert.equal(result.disposition, 'improved', result.error);
  }
});

test('a hypothesis naming a file outside the allowed ones is a typed invalid candidate, and the search continues', async () => {
  let hypotheses = 0;
  const model = scriptedModel(stagedImprover({
    hypothesize: () => { hypotheses++; return `return {kind:"helper",statement:"Fix the helper.",files:[${hypotheses === 1 ? '"elsewhere.ts"' : '"main.ts"'}],predictedChange:"quality rises"};`; },
    edit: write(1) }));
  const result = await run(model);
  assert.deepEqual(result.state.history.map(row => row.disposition), ['invalid-candidate', 'accepted']);
  assert.match(result.state.history[0].reason, /allowed ones \(main\.ts\); these are not: elsewhere\.ts/);
  assert.equal(result.disposition, 'improved', result.error);
});

test('a candidate that repeats an earlier outcome is a duplicate and is not evaluated again', async () => {
  let edits = 0;
  const model = scriptedModel(stagedImprover({ edit: () => { edits++; return edits < 3 ? write(2) : write(1); } }));
  const result = await run(model, { policy: policy({ maxExperiments: 4 }) });
  assert.deepEqual(result.state.history.map(row => row.disposition), ['rejected', 'duplicate', 'accepted']);
  assert.equal(result.state.history[1].candidate, result.state.history[0].candidate);
  assert.equal(result.disposition, 'improved', result.error);
});

test('an objective with no observed opportunity ends the search before any model stage', async () => {
  const model = scriptedModel(stagedImprover({ edit: 'throw Error("the edit stage must not run");', diagnose: 'throw Error("the diagnose stage must not run");' }));
  const passing = Folder.fromFiles({ 'main.ts': 'export function solve(value: number): number { return value + 1; }' });
  const result = await improveProgram({ folder: passing, contract, cases, policy: policy({ objective: 'model-calls', maxExperiments: 3 }), improver: finishing(model),
    executor: () => { throw Error('no inference'); }, executorId: 'exact', budget,
    executeCase: Object.assign(async (_folder, row) => ({ value: row.expected, modelCalls: 1 }), { identity: 'single-request', evaluationLevel: 1 }) });
  assert.equal(result.state.history[0].disposition, 'no-opportunity');
  assert.equal(result.state.stopReason, 'No further evidenced change.');
  assert.equal(result.state.done, true);
});

const NL = {
  stop: 'Decide whether the improvement search ends after this experiment',
  incumbent: 'Choose which candidate becomes the incumbent',
  opportunity: 'Say what the next experiment on the program should look at',
  parent: 'Choose which population member the next experiment starts from',
};

test('natural-language search policies decide within their crisp bounds', async () => {
  const model = scriptedModel(stagedImprover({
    other: opening => opening.includes(NL.stop) ? 'return {stop:true,reason:"the natural-language policy stopped"};' :
      opening.includes(NL.incumbent) ? 'return candidates[0].id;' :
      opening.includes(NL.opportunity) ? 'return {kind:"quality",reason:"chosen by the natural-language policy"};' :
      opening.includes(NL.parent) ? 'return members[0].id;' : null,
    edit: write(2) }));
  const result = await run(model, { policy: policy({ strategy: 'gepa', maxExperiments: 4,
    policies: { findOpportunity: 'nl', chooseParent: 'nl', selectIncumbent: 'nl', shouldStop: 'nl' } }) });
  assert.equal(result.state.stopReason, 'the natural-language policy stopped');
  assert.equal(result.state.history.length, 1);
  for (const phrase of Object.values(NL)) assert.ok(model.openings.some(opening => opening.includes(phrase)), phrase);
});

test('a natural-language policy outside its crisp bound is refused with the bound', async () => {
  const cases = [
    [{ [NL.stop]: 'return {stop:false,reason:"never"};' }, { maxExperiments: 1 }, /maxExperiments \(1\)/],
    [{ [NL.parent]: 'return "not-a-member";' }, { strategy: 'gepa' }, /population member/],
    [{ [NL.incumbent]: 'return "not-a-leader";' }, {}, /best-quality members/],
    [{ [NL.opportunity]: 'return {kind:"fixture",reason:"invented"};' }, {}, /exactly when a training row has failureKind/],
  ];
  const modes = { [NL.stop]: 'shouldStop', [NL.parent]: 'chooseParent', [NL.incumbent]: 'selectIncumbent', [NL.opportunity]: 'findOpportunity' };
  for (const [answers, extra, message] of cases) {
    const phrase = Object.keys(answers)[0];
    const model = scriptedModel(stagedImprover({ other: opening => opening.includes(phrase) ? answers[phrase] : null, edit: write(2) }));
    const result = await run(model, { policy: policy({ ...extra, policies: { [modes[phrase]]: 'nl' } }) });
    assert.equal(result.disposition, 'incomplete-search', phrase);
    assert.match(result.error, message, phrase);
  }
});

test('shadow mode serves the crisp decision and records the comparison', async () => {
  const traces = [];
  const model = scriptedModel(stagedImprover({
    other: opening => opening.includes(NL.stop) ? 'return {stop:true,reason:"the shadow disagrees"};' : null, edit: write(1) }));
  const result = await run(model, { policy: policy({ policies: { shouldStop: 'shadow' } }), trace: trace => traces.push(trace) });
  assert.equal(result.disposition, 'improved', result.error);
  assert.equal(result.state.stopReason, 'Declared objective satisfied.', 'the crisp decision was served');
  const shadows = traces.flatMap(trace => trace.events ?? []).filter(event => event.kind === 'pluggable_shadow');
  assert.equal(shadows.length, 1);
  assert.equal(shadows[0].name, 'shouldStop');
  assert.equal(shadows[0].agree, false);
  assert.equal(shadows[0].served, 'crisp');
});

test('the plan of an experiment is journaled before its edit and a resumed run reuses it', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'natlang-plan-journal-'));
  const first = scriptedModel(stagedImprover({
    other: opening => opening.includes(NL.parent) ? 'return members[0].id;' : null, edit: 'throw Error("interrupted during the edit");' }));
  const options = { policy: policy({ strategy: 'gepa', maxExperiments: 2, policies: { chooseParent: 'nl' } }), directory };
  const interrupted = await run(first, options);
  assert.equal(interrupted.state.iteration, 0);
  assert.notEqual(interrupted.disposition, 'improved');
  const planned = new OperationJournal(directory).read('plan:0')?.value;
  assert.equal(planned.parent, baseline().snapshot().digest);
  assert.deepEqual(planned.chose, { chooseParent: 'nl' });
  const second = scriptedModel(stagedImprover({
    other: opening => opening.includes(NL.parent) ? 'throw Error("the recorded plan is reused, so the parent is not chosen again");' : null, edit: write(1) }));
  const resumed = await run(second, options);
  assert.equal(resumed.disposition, 'improved', resumed.error);
  assert.equal(second.openings.filter(opening => opening.includes(NL.parent)).length, 0);
});

test('the transformation table holds one instruction sentence per kind of change, and an unknown name lists them', () => {
  const compiled = compileVirtualProject({ files: { 'main.ts': AUTHORED_IMPROVER['improveStep/transformations.ts'] } }, runtime, { constrained: true, target: 'node' });
  assert.equal(compiled.ok, true, JSON.stringify(compiled.diagnostics));
  const table = compiled.require('main.ts');
  assert.deepEqual(Object.keys(table.TRANSFORMATIONS).sort(), ['clarifyInstructions', 'extractHelper', 'implementProgram', 'removeDuplicatedGuidance', 'repairProgram', 'simplifyProgram', 'specializeForModel']);
  for (const [name, text] of Object.entries(table.TRANSFORMATIONS)) {
    assert.ok(text.length > 40 && text.endsWith('.'), name);
    assert.ok(!text.includes('\n'), `${name} is one paragraph of instruction`);
    assert.equal(table.transformationText(name), text);
  }
  assert.throws(() => table.transformationText('nonexistent'), /A transformation is one of: clarifyInstructions/);
  for (const stale of Object.keys(AUTHORED_IMPROVER).filter(name => name.startsWith('reducers/'))) assert.fail(`${stale} should be a row of the table`);
});

test('the authored program has no magic control strings and no caller-hint hypothesis', () => {
  const typescript = Object.entries(AUTHORED_IMPROVER).filter(([name]) => name.endsWith('.ts'));
  for (const [name, text] of typescript) {
    assert.doesNotMatch(text, /startsWith\('fixture-error:'\)|reason==='No supported hypothesis\.'|'fixture-error: '/, `${name} decides from dispositions`);
  }
  assert.doesNotMatch(AUTHORED_IMPROVER['types.ts'], /hypothesis: string/);
  assert.doesNotMatch(AUTHORED_IMPROVER['improveStep.nl'], /rewriteProgram/);
});
