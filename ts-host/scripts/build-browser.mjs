import { copyFileSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build as esbuild } from 'esbuild';
import { browserNodeImports } from './browser-node-imports.mjs';
import { STAMP, makeStamp, browserBuildProblems, staleMessage } from './browser-build-freshness.mjs';

const root = resolve(import.meta.dirname, '..');
// The build stamp (browser-build-freshness.mjs) is removed first and written last: an interrupted build has none.
rmSync(STAMP, { force: true });
const inputs = [fileURLToPath(import.meta.url), resolve(root, 'scripts/browser-node-imports.mjs'),
  resolve(root, 'scripts/browser-build-freshness.mjs'), fileURLToPath(import.meta.resolve('esbuild/package.json')),
  resolve(root, 'prelude.js')];
const outputs = [], assets = [];
// Every bundle records what esbuild read and wrote (paths in its metafile are relative to absWorkingDir). Inputs in
// another namespace ("ns:path": browser-node-imports.mjs stand-ins, esbuild's "(disabled):" for package.json browser
// fields) are not files; what they are comes from the plugin and the package files, inputs themselves.
async function build(options) {
  const result = await esbuild({ ...options, absWorkingDir: root, metafile: true });
  for (const path of Object.keys(result.metafile.inputs)) if (!/^[^/]*:/.test(path)) inputs.push(resolve(root, path));
  for (const path of Object.keys(result.metafile.outputs)) outputs.push(resolve(root, path));
  return result;
}
function copy(source, target) {
  copyFileSync(source, target);
  assets.push({ source, target });
}
// Library declarations for the in-browser TypeScript checker (eval `nl` analysis, virtual projects).
const libDir = dirname(fileURLToPath(import.meta.resolve('typescript/lib/lib.es2022.d.ts')));
const libs = Object.fromEntries(readdirSync(libDir).filter(name => /^lib\.(es5|es20\d\d(\.[a-z0-9]+)*|decorators(\.legacy)?|dom(\.[a-z]+)?|webworker\.importscripts|scripthost)\.d\.ts$/.test(name))
  .map(name => { inputs.push(join(libDir, name)); return [name, readFileSync(join(libDir, name), 'utf8')]; }));
await build({ entryPoints: { natlang: resolve(root, 'src/browser/index.ts') }, bundle: true,
  platform: 'browser', format: 'esm', target: 'es2022', splitting: true,
  outdir: resolve(root, 'dist/browser'), chunkNames: 'chunks/[name]-[hash]',
  // undici is only loaded under Node (fetchModel); browsers use plain fetch.
  external: ['undici'],
  plugins: [browserNodeImports()],
  define: { __NATLANG_PRELUDE__: JSON.stringify(readFileSync(resolve(root, 'prelude.js'), 'utf8')),
    __NATLANG_TS_LIBS__: JSON.stringify(libs) },
  legalComments: 'none' });
// The Neuralese service as WebAssembly (vendor/neuralese-wasm, built from the llama.cpp fork) and the worker that hosts
// it (startBrowserNeuralese).
await build({ entryPoints: { 'neuralese-worker': resolve(root, 'src/browser/neuralese-worker.ts') }, bundle: true,
  platform: 'browser', format: 'esm', target: 'es2022', outdir: resolve(root, 'dist/browser'), external: ['undici'],
  plugins: [browserNodeImports()], legalComments: 'none' });
// The call store's worker (src/browser/call-store-worker.ts) and SQLite's WebAssembly, which it loads from beside itself.
await build({ entryPoints: { 'call-store-worker': resolve(root, 'src/browser/call-store-worker.ts') }, bundle: true,
  platform: 'browser', format: 'esm', target: 'es2022', outdir: resolve(root, 'dist/browser'), external: ['undici'],
  plugins: [browserNodeImports()], legalComments: 'none' });
copy(fileURLToPath(import.meta.resolve('@sqlite.org/sqlite-wasm/sqlite3.wasm')), resolve(root, 'dist/browser/sqlite3.wasm'));
for (const asset of ['neuralese-wasm.mjs', 'neuralese-wasm.wasm', 'neuralese-wasm-mt.mjs', 'neuralese-wasm-mt.wasm',
  'neuralese-wasm-gpu.mjs', 'neuralese-wasm-gpu.wasm'])
  copy(resolve(root, 'vendor/neuralese-wasm', asset), resolve(root, 'dist/browser', asset));
copy(fileURLToPath(import.meta.resolve('@wllama/wllama/esm/wasm/wllama.wasm')),
  resolve(root, 'dist/browser/wllama.wasm'));
for (const [source, target] of [['wllama.js', 'wllama-compat.js'],
  ['wllama.wasm', 'wllama-compat.wasm']])
  copy(fileURLToPath(import.meta.resolve(`@wllama/wllama-compat/wasm/${source}`)),
    resolve(root, `dist/browser/${target}`));
// Pyodide is loaded on first Python use, from fixed local assets beside natlang.js.
const pyodideDir = resolve(root, 'dist/browser/pyodide');
mkdirSync(pyodideDir, { recursive: true });
for (const asset of ['pyodide.asm.mjs', 'pyodide.asm.wasm', 'python_stdlib.zip', 'pyodide-lock.json'])
  copy(fileURLToPath(import.meta.resolve(`pyodide/${asset}`)), join(pyodideDir, asset));
for (const asset of readdirSync(resolve(root, 'vendor/pyodide')).filter(name => name.endsWith('.whl')))
  copy(resolve(root, 'vendor/pyodide', asset), join(pyodideDir, asset));
// What this build read and wrote, and a check that it is consistent (e.g. the vendored wasm matches provenance.json).
writeFileSync(STAMP, JSON.stringify(makeStamp({ inputs, assets, outputs }), null, 1) + '\n');
const problems = browserBuildProblems();
if (problems.length) { console.error(staleMessage(problems, 'the browser build just written')); process.exit(1); }
