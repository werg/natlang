import test from 'node:test';
import assert from 'node:assert/strict';
import * as oracles from '../dist/evaluation/oracles.js';
import { registerRecordComparator, registeredRecordComparators } from '../dist/oracle-kit/oracle-registry.js';
import * as teacherOracle from '../dist/teacher/oracle.js';
import { sourceContractsFor, runtimeFailureRulesFor } from '../dist/benchmarks/registry.js';
import * as tatqaShim from '../dist/teacher/tatqa-unit-contract.js';
import * as musiqueShim from '../dist/teacher/musique-oklahoma-event-contract.js';
import * as tatqa from '../dist/benchmarks/tatqa/index.js';

const rec = (answer, scale = '') => JSON.stringify({ answer, scale });

test('teacher/oracle registers the bundled benchmark comparators under their historical names', async () => {
  assert.ok(teacherOracle.checkOracle);
  assert.deepEqual(registeredRecordComparators().filter(name => name.startsWith('tatqa')),
    ['tatqa-answer-record', 'tatqa-answer-record-exact']);
});

test('tatqa-answer-record resolves identically through checkOracle, checkFiles and checkFileReturn', async () => {
  const rounded = { level: 'normalized', normalization: 'tatqa-answer-record' };
  const exact = { level: 'normalized', normalization: 'tatqa-answer-record-exact' };
  assert.equal((await oracles.checkOracle(rec('1,234.50'), rec('1234.5'), rounded)).accepted, true);
  assert.equal((await oracles.checkOracle(rec('1.004'), rec('1.00'), rounded)).accepted, true);
  assert.equal((await oracles.checkOracle(rec('1.004'), rec('1.00'), exact)).accepted, false);
  assert.equal((await oracles.checkOracle(rec('5', 'million'), rec('5'), rounded)).accepted, false);
  assert.equal((await oracles.checkOracle(JSON.stringify({ answer: '5', scale: 'lakh' }), rec('5'), rounded)).accepted, false);
  const files = { 'answer.json': rec('12.0') };
  const verdict = oracles.checkFiles(files, { 'answer.json': rec('12') }, {}, { compare: 'tatqa-answer-record' });
  assert.equal(verdict.accepted, true);
  assert.equal(oracles.checkFileReturn(rec('12'), files, {}, { compare: 'tatqa-answer-record' }), true);
  assert.equal(oracles.checkFileReturn(rec('13'), files, {}, { compare: 'tatqa-answer-record-exact' }), false);
  assert.equal(oracles.checkFiles({ 'other.json': rec('12') }, { 'other.json': rec('12') }, {},
    { compare: 'tatqa-answer-record' }).accepted, false);
});

test('names that are neither built in nor registered are rejected, not silently treated as exact', async () => {
  await assert.rejects(oracles.checkOracle('a', 'a', { level: 'normalized', normalization: 'no-such-benchmark' }), /unknown normalization/);
  assert.throws(() => oracles.checkFiles({}, {}, {}, { compare: 'no-such-benchmark' }), /unknown compare/);
  registerRecordComparator('test-only-length', { answerFile: 'x.txt', equal: (a, b) => String(a).length === String(b).length });
  assert.equal((await oracles.checkOracle('abc', 'xyz', { level: 'normalized', normalization: 'test-only-length' })).accepted, true);
});

test('generic normalized answers keep exact-decimal equivalence but not TaTQA rounding', async () => {
  assert.equal((await oracles.checkOracle('1,000.0', '1000', 'normalized')).accepted, true);
  assert.equal((await oracles.checkOracle('1.004', '1.00', 'normalized')).accepted, false);
});

test('source contracts and failure rules are looked up by dataset name', () => {
  assert.deepEqual(sourceContractsFor('tatqa').map(c => c.sourceId), [tatqaShim.TATQA_LAKH_SOURCE_ID]);
  assert.deepEqual(sourceContractsFor('musique').map(c => c.sourceId), [musiqueShim.OKLAHOMA_EVENT_SOURCE_ID]);
  assert.deepEqual(sourceContractsFor('hotpotqa'), []);
  assert.deepEqual(runtimeFailureRulesFor('tatqa'), [tatqa.tatqaRuntimeFailureRule]);
  assert.equal(tatqaShim.isReviewedTatqaLakhVariant({}), false);
});
