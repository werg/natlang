import assert from 'node:assert/strict';
import { test } from 'node:test';
import { NativeToolAgent } from '../dist/native/agent.js';
import { RefinementChecker } from '../dist/native/refinement.js';
import { interpreter, lambda } from './support/natlang.mjs';
import { executeProgram } from '../dist/teacher/collector.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';

const failure = { code: 'refinement-unsatisfied', path: 'return', predicate: 'declared child result lineage',
  message: 'Complete the declared named child decisions and carry their actual typed results into the return value.' };

for (const action of ['return_result', 'eval-finish']) test(`host return guard repairs a plain typed ${action}`, async () => {
  const seen = [];
  let checks = 0;
  const driver = request => {
    seen.push(String(request.messages.at(-1).content));
    return action === 'return_result' ? { calls: [['return_result', { status: 'success', value: 'ready' }]] } :
      { calls: [['eval', { code: 'return "ready";', finish: true }]] };
  };
  const agent = new NativeToolAgent(driver, { maxTurns: 4,
    refinement: { checker: new RefinementChecker(), returnGuard: async runId => {
      assert.equal(runId, 'binding-root');
      return ++checks === 1 ? [failure] : [];
    } } });
  const runtime = interpreter({ runId: 'binding-root', calls: false, agent: session => agent.run(session) });
  const result = await runtime.run(lambda({ type: '() => string', instructions: 'Return a string.' }));
  assert.equal(result.outcome.kind, 'done', result.outcome.detail);
  assert.equal(result.value, 'ready');
  assert.equal(checks, 2);
  assert.ok(seen.some(message => message.includes(failure.message)));
});

test('declared binding rejects a direct root answer before collection admission', async () => {
  const seen = [];
  const record = { version: 'natlang.program/2', id: 'binding-direct-answer', kind: 'lambda_source',
    semantics: { root: 'solve.nl', files: { 'solve.nl': '---\nargs: {}\nreturns: string\n---\nUse the named child.\n' },
      inputs: {}, expected: 'ready' },
    curriculum: { item_result_binding: { schema: 'semantic-actual-child-result-binding/1', sites: [{
      owner_source_path: 'solve/child.nl', inline_child_site: { return_type: 'string' },
      frontier_input_pointer: '/state', next_state_pointer: '', prior_result_array_pointer: '/results',
      new_slot_pointer: '/results', prior_completed_array_pointer: '/completed', next_completed_array_pointer: '/completed',
      pending_array_pointer: '/pending', consumed_input_selector: { kind: 'first_pending_item', pointer: '' },
      item_identity_fields: [], semantic_fields: [], owned_pending_update: 'remove_exact_first_item', stable_state_fields: [],
    }] } } };
  let turn = 0;
  const run = await executeProgram(record, async request => {
    seen.push(String(request.messages.at(-1).content));
    return ++turn === 1 ? { calls: [['return_result', { status: 'success', value: 'ready' }]] } :
      { calls: [['return_result', { status: 'blocked', reason: 'The named child was unavailable.' }]] };
  }, { rootSeed: 1, runId: 'binding-direct-answer', systemPrompt: TOOLS_PROMPT, contextTokens: 8192, maxTurns: 3 });
  assert.equal(run.outcome.accepted, false);
  assert.equal(run.outcome.checks.item_result_binding, false);
  assert.ok(seen.some(message => message.includes(failure.message)));
});

test('scripted source reference checks the canonical return without claiming child lineage', async () => {
  const record = { version: 'natlang.program/2', id: 'binding-source-reference', kind: 'lambda_source',
    semantics: { root: 'solve.nl', files: { 'solve.nl': '---\nargs: {}\nreturns: string\n---\nReturn the source reference.\n' },
      inputs: {}, expected: 'ready' },
    curriculum: { item_result_binding: { schema: 'semantic-actual-child-result-binding/1', sites: [{
      owner_source_path: 'solve/child.nl', inline_child_site: { return_type: 'string' },
      frontier_input_pointer: '/state', next_state_pointer: '', prior_result_array_pointer: '/results',
      new_slot_pointer: '/results', prior_completed_array_pointer: '/completed', next_completed_array_pointer: '/completed',
      pending_array_pointer: '/pending', consumed_input_selector: { kind: 'first_pending_item', pointer: '' },
      item_identity_fields: [], semantic_fields: [], owned_pending_update: 'remove_exact_first_item', stable_state_fields: [],
    }] } } };
  const run = await executeProgram(record, async () => ({ calls: [['return_result', { status: 'success', value: 'ready' }]] }),
    { rootSeed: 1, runId: 'binding-source-reference', systemPrompt: TOOLS_PROMPT,
      contextTokens: 8192, maxTurns: 2, sourceReferenceOnly: true });
  assert.equal(run.outcome.accepted, true);
  assert.equal(run.outcome.checks.item_result_binding, undefined);
});
