import test from 'node:test';
import assert from 'node:assert/strict';
import { validateSourceValueBoundaries } from '../scripts/inline-curriculum/source-boundary-validation.mjs';

function sourceRow({ passName = 'approved window', evidencePath = 'evidence-01.md', field = 'window' } = {}) {
  const task = {
    output_contract: {
      fields: { window: 'string' },
      field_value_boundaries: {
        [field]: { pass_value_spans: [{ pass_name: passName, evidence_path: evidencePath,
          start_after: 'Custodian approves ', end_before: ' for the visit.', copy_exact_substring: true }] }
      }
    },
    passes: [{ name: 'approved window', evidence_path: 'evidence-01.md', allowed_fields: ['window'] }]
  };
  const taskText = JSON.stringify(task);
  return { id: 'fixture:boundary', semantics: { expected_files: { 'task.json': taskText },
    folder_files: { 'task.json': taskText, 'evidence-01.md': 'Custodian approves 09:00–11:00 for the visit.' } } };
}

test('valid boundary belongs to its declared field, pass, and current evidence', () => {
  assert.deepEqual(validateSourceValueBoundaries(sourceRow()), {
    status: 'validated', fields: ['window'], boundary_fields: ['window']
  });
});

test('rejects a same-named boundary copied from a different world pass', () => {
  const row = sourceRow({ passName: 'custodian window', evidencePath: 'evidence-06.md' });
  assert.throws(() => validateSourceValueBoundaries(row), /found 0 exact pass matches/);
});

test('rejects an anchor that appears only in another evidence file', () => {
  const row = sourceRow();
  row.semantics.folder_files['evidence-01.md'] = 'No approval is recorded here.';
  row.semantics.folder_files['evidence-06.md'] = 'Custodian approves 09:00–11:00 for the visit.';
  assert.throws(() => validateSourceValueBoundaries(row), /start_after anchor/);
});

test('ordinary nested task without boundary metadata needs no fields or passes', () => {
  const task = { instruction: 'Inspect the folder records and report the approved item.' };
  const taskText = JSON.stringify(task);
  const row = { id: 'fixture:nested', semantics: { folder_files: { 'task.json': taskText },
    expected_files: { 'task.json': taskText } } };
  assert.deepEqual(validateSourceValueBoundaries(row), {
    status: 'no_boundary_contract', fields: [], boundary_fields: []
  });
});

test('compares the actual FileHandle task contract with its expected copy', () => {
  const row = sourceRow();
  const expected = JSON.parse(row.semantics.expected_files['task.json']);
  expected.output_contract.fields.window = 'a conflicting expected definition';
  row.semantics.expected_files['task.json'] = JSON.stringify(expected);
  assert.throws(() => validateSourceValueBoundaries(row), /actual folder task.json differs from expected_files/);
});
