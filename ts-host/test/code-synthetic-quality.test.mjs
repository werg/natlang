import test from 'node:test';
import assert from 'node:assert/strict';
import { codeProposalTurns, compileCurriculum, syntheticCodeTasks } from '../scripts/code-corpus/curriculum.mjs';
import { checkSyntheticProperties, SYNTHETIC_CODE_FAMILIES } from '../scripts/code-corpus/synthetic-families.mjs';
import { replayIsolated } from '../scripts/code-corpus/replay.mjs';

test('synthetic code lane covers distinct, explicitly specified implementation families', async () => {
  assert.ok(SYNTHETIC_CODE_FAMILIES.length >= 14);
  assert.equal(new Set(SYNTHETIC_CODE_FAMILIES.map(family => family.name)).size, SYNTHETIC_CODE_FAMILIES.length);
  for (const family of SYNTHETIC_CODE_FAMILIES) {
    assert.ok(family.instruction.length >= 70, family.name);
    assert.equal(typeof family.ref, 'function', family.name);
    assert.equal(typeof family.make, 'function', family.name);
  }
  const rows = syntheticCodeTasks(883, 0, SYNTHETIC_CODE_FAMILIES.length);
  assert.deepEqual(rows.map(row => row.generation.family), SYNTHETIC_CODE_FAMILIES.map(family => family.name));
  assert.ok(rows.every(row => row.cases.length === 2 && row.cases.every(item => item.portable)));
  assert.ok(rows.some(row => row.function.parameters[0].type === 'number[][]'));
  assert.ok(rows.some(row => /Unicode|UTF-16|lexicographic/.test(row.instruction)));
  assert.ok(rows.some(row => /Parse|valid/.test(row.instruction)));
  const evidence = rows[0].behavioral_evidence;
  assert.equal(evidence.status, 'reference_checks_passed_pending_native_replay');
  assert.match(evidence.reference, /independence is not claimed/);
  assert.ok(evidence.property_checks.flatMap(item => item.checks).every(item => item.passed));
  assert.equal(compileCurriculum([rows[0]])[0].behavioral_evidence.status, evidence.status);
  assert.equal((await codeProposalTurns([rows[0]]))[0].behavioral_evidence.status, evidence.status);
  const proposalTurn = (await codeProposalTurns([rows[0]]))[0];
  assert.equal(proposalTurn.generation.family, rows[0].generation.family);
  assert.match(proposalTurn.implementation_sha256, /^[a-f0-9]{64}$/);
  const proposal = await codeProposalTurns([{ id:'syntax-only', language:'javascript', instruction:'Implement a task.', function:{source:'function f(x) { return x; }'} }]);
  assert.equal(proposal[0].behavioral_evidence.status, 'not_execution_verified');
});

test('each synthetic family implementation agrees with separate references under native replay', async () => {
  const rows = syntheticCodeTasks(1927, 0, SYNTHETIC_CODE_FAMILIES.length);
  for (const row of rows) {
    for (let index = 0; index < row.cases.length; index++) {
      const trajectory = await replayIsolated(row, index);
      assert.equal(trajectory.outcome.accepted, true, `${row.generation.family} case ${index}: ${JSON.stringify(trajectory.outcome)}`);
    }
  }
});

test('family repetition cap truncates visibly and chunked ranges remain index-stable', () => {
  const n = SYNTHETIC_CODE_FAMILIES.length;
  assert.equal(syntheticCodeTasks(2, 0, n * 2, 1).length, n);
  const cappedDefault = syntheticCodeTasks(2, 0, 1000);
  assert.equal(cappedDefault.length, n * 50);
  assert.equal(cappedDefault.at(-1).generation.index, n * 50 - 1);
  const full = syntheticCodeTasks(2, 0, n * 2, 2);
  const resumed = [...syntheticCodeTasks(2, 0, n, 2), ...syntheticCodeTasks(2, n, n, 2)];
  assert.deepEqual(resumed, full);
  assert.deepEqual(syntheticCodeTasks(2, n * 2, n, 2), []);
});

test('behavioral properties reject a wrong output even when checking reference fixtures', () => {
  assert.equal(checkSyntheticProperties('sorted_numbers', [1, 2], [2, 1])[0].passed, false);
  assert.equal(checkSyntheticProperties('stable_unique', ['b', 'a'], ['a', 'b'])[0].passed, false);
  assert.equal(checkSyntheticProperties('transpose_rectangular', [[1, 2], [3, 4]], [[1, 3]])[0].passed, false);
  assert.equal(checkSyntheticProperties('unicode_codepoint_count', '🪁', 2)[0].passed, false);
  const unsorted = [['b', 'first'], ['a', 'second']];
  assert.equal(checkSyntheticProperties('sort_pairs_by_key', unsorted, unsorted)[0].passed, false);
  assert.equal(checkSyntheticProperties('sort_pairs_by_key', [['a','one'], ['a','two']], [['a','two'], ['a','one']])[0].passed, false);
  assert.equal(checkSyntheticProperties('sort_pairs_by_key', unsorted, [['a','second'], ['b','first']])[0].passed, true);
  assert.equal(checkSyntheticProperties('parse_decimal', '42', 0)[0].passed, false);
});
