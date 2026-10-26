import assert from 'node:assert/strict';
import { test } from 'node:test';
import { honestBlockedRootInvocation } from '../dist/teacher/native-materializer.js';

function blockedFixture() {
  const program = { semantics: { operation: 'blocked' }, curriculum: { reference: { root: [
    ['eval', { code: 'assess(records)' }],
    ['return_result', { status: 'blocked', reason: 'the records are insufficient' }],
  ] } } };
  const row = {
    id: 'honest-root-block',
    outcome: {
      status: 'quiesced', accepted: true, oracle: { accepted: true, level: 'exact' },
      checks: { seeded_failure_requirement: true, expected_status: true, honest_stop: true,
        effects: true, files: true, world: true, file_return_consistency: true, answer: true },
      invocation_ledger: [
        { invocation_id: 'root', parent_invocation_id: null,
          completion_status: 'quiesced', completion_source: 'execution_graph' },
        { invocation_id: 'child', parent_invocation_id: 'root',
          completion_status: 'done', completion_source: 'execution_graph' },
      ],
      action_ledger: [{ call_id: 'root', seq: 8, name: 'return_result',
        arguments: { status: 'blocked', reason: 'No record resolves delivery.' }, outcome: 'blocked' }],
    },
    trajectory: [
      { invocation_id: 'root', assistant: { calls: [
        { tool: 'eval', source_tool: 'eval', arguments: { code: 'assess(records)' } },
      ] } },
      { invocation_id: 'child', assistant: { calls: [
        { tool: 'return_result', source_tool: 'return_result', arguments: { status: 'success', value: false } },
      ] } },
      { invocation_id: 'root', assistant: { calls: [
        { tool: 'return_result', source_tool: 'return_result',
          arguments: { status: 'blocked', reason: 'No record resolves delivery.' } },
      ] } },
    ],
  };
  return { program, row };
}

test('honest blocked root requires the expected terminal and complete child custody', () => {
  const { program, row } = blockedFixture();
  assert.equal(honestBlockedRootInvocation(row, program), 'root');
  const childBlocked = structuredClone(row);
  childBlocked.trajectory[1].assistant.calls[0].arguments.status = 'blocked';
  childBlocked.outcome.invocation_ledger[1].completion_status = 'quiesced';
  assert.equal(honestBlockedRootInvocation(childBlocked, program), undefined);
  const wrongExpected = structuredClone(program);
  wrongExpected.curriculum.reference.root.at(-1)[1].status = 'success';
  assert.equal(honestBlockedRootInvocation(row, wrongExpected), undefined);
  const dishonest = structuredClone(row);
  dishonest.outcome.checks.honest_stop = false;
  assert.equal(honestBlockedRootInvocation(dishonest, program), undefined);
  const unboundTerminal = structuredClone(row);
  unboundTerminal.outcome.action_ledger = [];
  assert.equal(honestBlockedRootInvocation(unboundTerminal, program), undefined);
});

test('a blocked reference cannot override a task requiring an answer', () => {
  const { program, row } = blockedFixture();
  program.semantics.operation = 'exact';
  assert.equal(honestBlockedRootInvocation(row, program), undefined);
});
