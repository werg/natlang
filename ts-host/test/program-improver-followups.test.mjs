/**
 * Program-improver follow-ups: the counterexample stop is the shared pluggable `shouldStop` policy, the component edit is
 * checked exactly (gepa.componentProblem), and an experiment is crisp mechanism around the natural-language stages.
 * Every model here is scripted.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { Folder, counterexampleGuidedImprove, improveProgram } from '../dist/index.js';
import { crispCounterexampleStop, verifyCounterexampleStop } from '../dist/improvement/counterexample-stop.js';
import { componentProblem } from '../dist/gepa/index.js';
import { AUTHORED_IMPROVER } from '../dist/improvement/authored-source.js';
import { scriptedModel } from './support/natlang.mjs';
import { stagedImprover, finishing, STAGES } from './support/improver.mjs';

const contract = { entry: 'main.ts', exportName: 'solve', programId: 'followups' };
const cases = [{ id: 'train', group: 'train', split: 'train', args: [1], expected: 2 }, { id: 'validation', group: 'validation', split: 'validation', args: [2], expected: 3 },
  { id: 'test', group: 'test', split: 'test', args: [3], expected: 4 }];

test('the crisp counterexample stop is the rule counterexampleStep.nl stated in prose, in its order', () => {
  const facts = (extra = {}) => ({ search: 'counterexample', round: 1, admitted: 1, remainingChecks: 2, repair: { eligible: true, trainingQuality: 0.5, disposition: 'improved' }, ...extra });
  assert.deepEqual(crispCounterexampleStop(facts({ admitted: 0, repair: undefined })), { stop: true, reason: 'No counterexample was admitted; the source is retained.' });
  assert.deepEqual(crispCounterexampleStop(facts({ repair: { eligible: true, trainingQuality: 1, disposition: 'improved' } })), { stop: true, reason: 'The repaired training suite passes.' });
  assert.deepEqual(crispCounterexampleStop(facts({ remainingChecks: 0 })), { stop: true, reason: 'The independent oracle allowance is exhausted.' });
  assert.deepEqual(crispCounterexampleStop(facts({ repair: { eligible: false, trainingQuality: 0, disposition: 'no-eligible-promotion' } })), { stop: true, reason: 'The repair did not produce a completed eligible result.' });
  assert.deepEqual(crispCounterexampleStop(facts()), { stop: false, reason: 'Continue with further counterexamples.' });
  assert.throws(() => verifyCounterexampleStop({ stop: 'yes' }), /boolean/);
});

const loop = (model, policies, maxRounds = 3) => {
  const folder = Folder.fromFiles({ 'main.ts': 'export function solve(value:number):number{return value;}' });
  return counterexampleGuidedImprove({ folder, contract, cases, policy: { maxExperiments: 1, maxPopulation: 3, mode: 'structural', strategy: 'adaptive', goal: 'increment',
    allowedFiles: ['main.ts'], ...(policies ? { policies } : {}) }, improver: finishing(model), executor: () => { throw Error('no inference'); }, executorId: 'exact',
  budget: { maxModelCalls: 40, maxRollouts: 40, maxProposals: 6 }, oracle: { identity: 'independent-increment', expected: async args => args[0] + 1 }, maxRounds, maxChecks: 4 });
};
const roundCode = 'const observed=await counterexamples.evidence(folder.snapshot());const suggestion=await suggestCounterexamples(goal,observed);const admitted=await counterexamples.admit(suggestion.inputs);let repaired={folder:folder.snapshot(),quality:0};if(admitted.admitted>0){repaired=await counterexamples.repair(folder.snapshot());await folder.select(repaired.folder);}const decision=await counterexamples.stop();return {round:state.round+1,done:decision.stop,suite:admitted.suite,remainingChecks:admitted.remainingChecks,quality:repaired.quality,reason:decision.reason};';
const scripted = (extra = {}) => scriptedModel(stagedImprover({
  other: opening => opening.includes('Perform one counterexample-guided') ? roundCode : opening.includes('Suggest concrete deployment inputs') ? 'return {inputs:[[7]],reason:"probe a larger value"}' :
    extra.stop && opening.includes('Decide whether the improvement search ends') ? extra.stop : null,
  edit: 'await folder.file("main.ts").writeText("export function solve(value:number):number{return value+1;}");return {summary:"repair",preserves:["number signature"]};' }));

test('the counterexample loop stops through the shared policy: crisp by default, with the host-measured round facts', async () => {
  const model = scripted();
  const result = await loop(model);
  assert.equal(result.state.done, true);
  assert.equal(result.state.reason, 'The repaired training suite passes.');
  assert.equal(result.state.quality, 1);
  assert.ok(!model.openings.some(opening => opening.includes('Decide whether the improvement search ends')), 'the crisp policy calls no model');
});

test('policies.shouldStop selects the natural-language side for the counterexample search, within the same verifier', async () => {
  const model = scripted({ stop: 'if(facts.search!=="counterexample"||facts.admitted<1||facts.repair===undefined)throw Error("facts incomplete");return {stop:true,reason:"the natural-language policy stopped"};' });
  const result = await loop(model, { shouldStop: 'nl' });
  assert.equal(result.state.reason, 'the natural-language policy stopped');
  assert.ok(model.openings.some(opening => opening.includes('Counterexample search')), 'the shared function, with its counterexample steps, was asked');
});

test('shadow mode serves the crisp counterexample decision', async () => {
  const model = scripted({ stop: 'return {stop:false,reason:"the shadow would continue"};' });
  const result = await loop(model, { shouldStop: 'shadow' });
  assert.equal(result.state.reason, 'The repaired training suite passes.');
  assert.ok(model.openings.some(opening => opening.includes('Counterexample search')));
});

test('componentProblem names what to change: only the selected keys differ; kinds, slot ids and segment counts hold', () => {
  const parent = {
    a: { kind: 'lambda.instructions', template: { segments: ['one', 'two'], slotIds: ['x'] } },
    b: { kind: 'program.guidance', text: 'guide' },
    c: { kind: 'lambda.instructions', template: { segments: ['keep'], slotIds: [] } },
  };
  const edit = change => ({ ...structuredClone(parent), ...change });
  assert.equal(componentProblem(parent, edit({ a: { kind: 'lambda.instructions', template: { segments: ['1', '2'], slotIds: ['x'] } } }), ['a']), '');
  assert.equal(componentProblem(parent, edit({ b: { kind: 'program.guidance', text: 'new' } }), ['b']), '');
  assert.match(componentProblem(parent, edit({ c: { kind: 'program.guidance', text: 'new' } }), ['a']), /c is not selected/);
  assert.match(componentProblem(parent, edit({ a: { kind: 'program.guidance', text: 'x' } }), ['a']), /keeps its kind `lambda.instructions`/);
  assert.match(componentProblem(parent, edit({ a: { kind: 'lambda.instructions', template: { segments: 'one two', slotIds: ['x'] } } }), ['a']), /array of strings/);
  assert.match(componentProblem(parent, edit({ a: { kind: 'lambda.instructions', template: { segments: ['one'], slotIds: ['x'] } } }), ['a']), /keeps its 2 segments \(it has 1\)/);
  assert.match(componentProblem(parent, edit({ a: { kind: 'lambda.instructions', template: { segments: ['1', '2'], slotIds: ['y'] } } }), ['a']), /slot ids of a/);
  assert.match(componentProblem(parent, { a: parent.a }, ['a']), /missing: b, c/);
  assert.match(componentProblem(parent, { ...parent, d: parent.a }, ['a']), /not among them: d/);
  assert.match(componentProblem(parent, 'text', ['a']), /one JSON object/);
});

test('one experiment is crisp mechanism: no model request sequences it, and the application holds no model-run step', async () => {
  assert.doesNotMatch(AUTHORED_IMPROVER['improveStep.nl'], /lifecycle\.step\(|finish:\s*true/, 'the root file no longer asks a model to run the step');
  const model = scriptedModel(stagedImprover({ edit: 'await folder.file("main.ts").writeText("export function solve(value:number):number{return value+1;}");return {summary:"increment",preserves:["number contract"]};' }));
  const result = await improveProgram({ folder: Folder.fromFiles({ 'main.ts': 'export function solve(value:number):number{return value;}' }), contract, cases,
    policy: { maxExperiments: 1, maxPopulation: 2, strategy: 'adaptive', mode: 'structural', goal: 'increment', allowedFiles: ['main.ts'] }, improver: finishing(model),
    executor: () => { throw Error('no inference'); }, executorId: 'exact', budget: { maxModelCalls: 20, maxRollouts: 20, maxProposals: 3 } });
  assert.equal(result.disposition, 'improved', result.error);
  assert.deepEqual(model.openings.map(opening => [STAGES.diagnose, STAGES.hypothesize, STAGES.edit].find(phrase => opening.includes(phrase))).sort(),
    [STAGES.diagnose, STAGES.edit, STAGES.hypothesize].sort(), 'exactly the three stages ran');
});
