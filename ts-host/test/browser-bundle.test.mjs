import assert from 'node:assert/strict';
import { browserTest as test } from './support/natlang.mjs';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { browserNodeImports } from '../scripts/browser-node-imports.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const NEURALESE_NAMES = ['neuraleseServerInfo', 'checkNeuraleseReader', 'serverViewer', 'VIEW_TYPE', 'createNeuraleseLibrary',
  'loadStandardLibrary', 'buildStandardLibrary', 'COMBINATORS', 'encodeBlockBody', 'decodeBlockBody', 'supportsNeuralese',
  'textToParts', 'registerBuiltinModule', 'ModuleUnavailableError', 'startBrowserNeuralese', 'neuraleseServerModelTurn'];
const bundle = (entry, options = {}) => build({ entryPoints: [entry], bundle: true, platform: 'browser', format: 'esm',
  target: 'es2022', splitting: true, outdir: '/browser-bundle-test', write: false, absWorkingDir: root, external: ['undici'],
  logLevel: 'silent', plugins: [browserNodeImports(options)] });

/** The browser bundle, loaded with Node's `process` hidden so any Node dependency fails. */
async function api() {
  const process = globalThis.process;
  try { globalThis.process = undefined; return await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = process; }
}

test('the browser entry point bundles under the Node built-in policy and exports the Neuralese API', async () => {
  const result = await bundle('src/browser/index.ts');
  const entry = Object.values(result.metafile.outputs).find(output => output.entryPoint === 'src/browser/index.ts');
  for (const name of NEURALESE_NAMES) assert.ok(entry.exports.includes(name), `the browser entry point exports ${name}`);
  assert.ok(!entry.exports.includes('createLearning'), 'natlang:learning stays out of the browser API');
  assert.ok(!Object.keys(result.metafile.inputs).some(path => /neuralese\/(learning|deltas|node-files)\.ts$/.test(path)),
    'Node-only Neuralese modules stay out of the browser graph');
});

test('the built browser bundle exports the Neuralese API', async () => {
  const browser = await api();
  for (const name of NEURALESE_NAMES) assert.notEqual(browser[name], undefined, name);
  await assert.rejects(browser.loadStandardLibrary('/lib.nz'), /takes the \.nz file's bytes here/);
  const meta = { id: 'x', dialect: 'd', width: 2, length: 1, dtype: 'f32' };
  assert.ok(browser.encodeBlockBody(meta, new Uint8Array(8)) instanceof Uint8Array);
});

test('browser programs import natlang:neuralese; natlang:learning fails with a teaching error', async () => {
  const browser = await api();
  const project = browser.compileVirtualProject({ files: {
    'soft.ts': `import { read } from 'natlang:neuralese';\nexport const reader = () => read;\n`,
    'learn.ts': `import { grad } from 'natlang:learning';\nexport const learner = () => grad;\n` } }, browser);
  assert.equal(project.ok, true, JSON.stringify(project.diagnostics));
  const soft = project.require('soft.ts');
  assert.throws(() => soft.reader(), /natlang:neuralese needs the task's neuralese service/);
  assert.throws(() => project.require('learn.ts'), error => error instanceof browser.ModuleUnavailableError &&
    error.code === 'module-unavailable' && /natlang:learning is not available on this platform: .*run on Node/.test(error.message));
});

test('a node: import reachable from a browser entry point fails the build and names the chain', async () => {
  const error = await bundle('test/fixtures/browser-node-import/entry.ts').then(() => undefined, failure => failure);
  assert.ok(error, 'the build is rejected');
  const text = error.errors.map(item => item.text).join('\n');
  assert.match(text, /node:fs, reached by\n {2}test\/fixtures\/browser-node-import\/entry\.ts\n {4}-> test\/fixtures\/browser-node-import\/helper\.ts\n {4}-> node:fs/);
});

test('a tolerated node: import gets its declared stand-in', async () => {
  const result = await bundle('test/fixtures/browser-node-import/entry.ts', { tolerated: [{ specifiers: ['node:fs'],
    importer: /browser-node-import\/helper\.ts$/, reason: 'test', contents: () => 'export const statSync = () => ({ size: 7 });' }] });
  assert.match(result.outputFiles.map(file => file.text).join('\n'), /size: 7/);
});
