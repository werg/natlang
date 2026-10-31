import assert from 'node:assert/strict';
import { test } from 'node:test';
import { checkReportCompleteness } from '../dist/teacher/report-completeness.js';

const spec = { schema: 'natlang.report-completeness/1', mutating_methods: ['turnAirplaneModeOff'],
  report_effects_path: 'effects', checkpoint_effects_path: 'checkpoint.receipts',
  report_diagnosis_path: 'diagnosis', diagnosis_function: 'advance/presentDiagnosis.nl' };
const diagnosis = { primaryHypothesis: 'airplane mode blocked service', evidence: ['r1'] };
const effect = { kind: 'effect', phase: 'completed', method: 'turnAirplaneModeOff', result: {
  receiptId: 'case:1', sequence: 1, toolName: 'turn_airplane_mode_off', requestor: 'user', arguments: {},
  value: 'Airplane Mode is now OFF.' } };
const receipt = { receiptId: 'case:1', sequence: 1, toolName: 'turn_airplane_mode_off', requestor: 'user',
  arguments: {}, result: 'Airplane Mode is now OFF.', actionId: 'repair-1' };
const ledger = [{ host_result: { capture_kind: 'invocation_output', complete: true,
  definition_source: 'advance/presentDiagnosis.nl', value: diagnosis } }];
const good = () => ({ effects: [structuredClone(receipt)], checkpoint: { receipts: [structuredClone(receipt)] },
  diagnosis: structuredClone(diagnosis) });

test('report completeness reconciles actual successful mutation and diagnosis receipts', () => {
  const result = checkReportCompleteness(spec, good(), [effect], ledger);
  assert.equal(result.accepted, true);
  assert.equal(result.actual_effect_receipts, 1);
  assert.equal(result.diagnosis_matches, true);
});

for (const [name, mutate] of [
  ['dropped mutation', value => { value.effects = []; }],
  ['forged receipt', value => { value.effects.push({ ...receipt, receiptId: 'invented' }); }],
  ['checkpoint omission', value => { value.checkpoint.receipts = []; }],
  ['altered diagnosis', value => { value.diagnosis.primaryHypothesis = 'unsupported'; }],
]) test(`report completeness rejects ${name}`, () => {
  const value = good(); mutate(value);
  const result = checkReportCompleteness(spec, value, [effect], ledger);
  assert.equal(result.accepted, false);
});

test('report completeness requires a completed diagnosis invocation', () => {
  const result = checkReportCompleteness(spec, good(), [effect], []);
  assert.equal(result.accepted, false);
  assert.ok(result.reasons.includes('diagnosis_invocation_missing'));
});

test('unconfigured programs retain existing collector behavior', () => {
  assert.deepEqual(checkReportCompleteness(undefined, {}, [], []), { required: false });
});
