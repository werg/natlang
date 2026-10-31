import assert from 'node:assert/strict';
import { test } from 'node:test';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const dist = process.env.NATLANG_TS_HOST_TEST_DIST;
const load = name => dist ? import(pathToFileURL(join(dist, name)).href) : import(new URL(`../dist/${name}`, import.meta.url));
const [{ NodeNativeRuntime }, { NativeSession }, { TypeEnv }, { buildPending }] = await Promise.all([
  load('node-runtime.js'), load('native/runtime.js'), load('native/types.js'), load('native/values.js'),
]);
const session = body => {
  const lam = buildPending({ $lambda: body }), runtime = new NodeNativeRuntime();
  return new NativeSession(runtime, lam, new TypeEnv().child(lam.types));
};

test('structured Neuralese marker concatenation gets actionable feedback while literal markers are not misclassified', async () => {
  const computed = session({ type: '() => Neuralese<{ hypothesis: string }>', instructions: 'Return a diagnosis.' });
  const rejected = await computed.applyAsync('eval', { code:
    'const result = { hypothesis: "SIM missing" }; return "<|neuralese|>" + JSON.stringify(result) + "<|/neuralese|>";', finish: true });
  assert.equal(rejected.kind, 'rejected');
  assert.match(rejected.text, /computed-neuralese-marker-concatenation/);
  assert.match(rejected.text, /Return the computed value directly according to its declared type/);

  const template = session({ type: '() => Neuralese<{ hypothesis: string }>', instructions: 'Return a diagnosis.' });
  const templateRejected = await template.applyAsync('eval', { code:
    'const result = { hypothesis: "SIM missing" }; return `<|neuralese|>${JSON.stringify(result)}<|/neuralese|>`;', finish: true });
  assert.equal(templateRejected.kind, 'rejected');
  assert.match(templateRejected.text, /computed-neuralese-marker-concatenation/);

  const stringResult = session({ type: '() => Neuralese<string>', instructions: 'Return a note.' });
  const stringRejected = await stringResult.applyAsync('eval', { code:
    'const note = "SIM missing"; return "<|neuralese|>" + note + "<|/neuralese|>";', finish: true });
  assert.equal(stringRejected.kind, 'rejected');
  assert.match(stringRejected.text, /computed-neuralese-marker-concatenation/);

  const consumer = { kind: 'natlang', id: 'consume', name: 'consume', source: '', revision: '', text: '', description: '',
    args: { note: 'Neuralese<{ hypothesis: string }>' }, returns: 'string', instructions: 'Read the note.', types: {},
    subtype: 'function', codebase: {} };
  const typedArgument = session({ type: '(text: string) => string', args: { text: 'SIM missing' },
    instructions: 'Call consume with a typed note.', codebase: { consume: consumer } });
  const argumentRejected = await typedArgument.applyAsync('eval', { code:
    'const payload = "<|neuralese|>" + JSON.stringify({ hypothesis: text }) + "<|/neuralese|>"; await consume(payload); return "done";', finish: true });
  assert.equal(argumentRejected.kind, 'rejected');
  assert.match(argumentRejected.text, /This argument is declared Neuralese<T>/);

  const literal = session({ type: '() => Neuralese<{ hypothesis: string }>', instructions: 'Return a diagnosis.' });
  const literalResult = await literal.applyAsync('eval', { code:
    'return "<|neuralese|>{\\"hypothesis\\":\\"literal\\"}<|/neuralese|>";', finish: true });
  assert.doesNotMatch(literalResult.text, /computed-neuralese-marker-concatenation/);

  const ordinaryString = session({ type: '() => string', instructions: 'Return the exact quoted string.' });
  const quoted = await ordinaryString.applyAsync('eval', { code:
    'const value = "SIM missing"; return "<|neuralese|>" + value + "<|/neuralese|>";', finish: true });
  assert.equal(quoted.kind, 'completed', 'marker-looking concatenation remains available as ordinary string/code data');
});
