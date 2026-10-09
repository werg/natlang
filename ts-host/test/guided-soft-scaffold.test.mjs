import assert from 'node:assert/strict';
import { test } from 'node:test';
import { analyzeEvalSnippet } from '../dist/compiler/eval-check.js';
import { compileModule } from '../dist/runtime/modules.js';
import { compileScopeSnippet } from '../dist/scope-compiler.js';
import { neuraleseRef } from '../dist/native/neuralese.js';
import { makeGuidedSoftIterateCase } from '../scripts/inline-curriculum/semantic-iterate-worlds-v15-soft-guided-builder.mjs';
import { worlds } from '../scripts/inline-curriculum/semantic-iterate-worlds-v15-data.mjs';

test('generated guided soft scaffold compiles marker guidance as literal template text', () => {
  const record = makeGuidedSoftIterateCase(worlds[0], 0);
  assert.deepEqual(record.collection_guidance, {
    training_admission: false,
    review_scope: 'sampled-actions-require-independent-semantic-review',
  });
  const source = record.curriculum.reference.root.find(([kind]) => kind === 'eval')[1].code;
  const plans = [];
  const compiled = compileModule({ kind: 'module', id: record.id, name: 'guidedScaffold', source: record.source, revision: 'r1',
    text: source, types: {}, exports: {}, imports: [], codebase: {} }, {}, value => plans.push(...value));
  assert.ok(compiled.length > 0);
  const step = plans.find(plan => plan.strings[0]?.includes('This note child does not write files'));
  assert.ok(step, 'the actual generated step child reaches inline analysis');
  assert.match(step.strings[0], /Marker bodies are literal text and do not interpolate JavaScript variables\./);
  assert.deepEqual(step.interpolations, [], 'the `${...}` prompt example is not parsed as a JavaScript interpolation');

  const evalScope = { types: {}, inputs: [{ name: 'priorNotes', type: 'Neuralese<string>' }], locals: [], captures: [],
    imports: [], returns: 'Neuralese<string>' };
  const snippet = 'const next = String(priorNotes); return next;';
  const analysis = analyzeEvalSnippet(snippet, evalScope);
  assert.deepEqual(analysis.diagnostics, []);
  const executable = compileScopeSnippet(snippet, { inputBindings: ['priorNotes'], neuralese: true,
    analyze: text => analyzeEvalSnippet(text, evalScope) });
  assert.equal(executable.ok, true, JSON.stringify(executable.diagnostics));
  const run = new Function('__natlang_frozen', '__natlang_copy', '__natlang_settle', '__natlang_output', '__live',
    `${executable.program}; return __natlang_scope;`)(value => value, value => value, async value => value,
    output => output.result, { readNeuralese: async () => 'computed next note' });
  return run({ priorNotes: neuraleseRef('Neuralese<string>', 'nz1_ffffffffffffffffffff') }, {}, {}).then(value => {
    assert.equal(value, 'computed next note');
  });
});
