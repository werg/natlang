import assert from 'node:assert/strict';
import { test } from 'node:test';
import { session } from './support/natlang.mjs';

test('structured Neuralese marker concatenation gets actionable feedback while literal markers are not misclassified', async () => {
  const { session: computed } = session({ type: '() => Neuralese<{ hypothesis: string }>', instructions: 'Return a diagnosis.' });
  const rejected = await computed.applyAsync('eval', { code:
    'const result = { hypothesis: "SIM missing" }; return "<|neuralese|>" + JSON.stringify(result) + "<|/neuralese|>";', finish: true });
  assert.equal(rejected.kind, 'rejected');
  assert.match(rejected.text, /computed-neuralese-marker-concatenation/);
  assert.match(rejected.text, /Return the complete \{ hypothesis: string \} value directly/);

  const { session: template } = session({ type: '() => Neuralese<{ hypothesis: string }>', instructions: 'Return a diagnosis.' });
  const templateRejected = await template.applyAsync('eval', { code:
    'const result = { hypothesis: "SIM missing" }; return `<|neuralese|>${JSON.stringify(result)}<|/neuralese|>`;', finish: true });
  assert.equal(templateRejected.kind, 'rejected');
  assert.match(templateRejected.text, /computed-neuralese-marker-concatenation/);

  const { session: literal } = session({ type: '() => Neuralese<{ hypothesis: string }>', instructions: 'Return a diagnosis.' });
  const literalResult = await literal.applyAsync('eval', { code:
    'return "<|neuralese|>{\\"hypothesis\\":\\"literal\\"}<|/neuralese|>";', finish: true });
  assert.doesNotMatch(literalResult.text, /computed-neuralese-marker-concatenation/);
});
