import { copyFileSync, mkdirSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const root = resolve(import.meta.dirname, '..');
// Library declarations for the in-browser TypeScript checker (eval `nl` analysis, virtual projects).
const libDir = dirname(fileURLToPath(import.meta.resolve('typescript/lib/lib.es2022.d.ts')));
const libs = Object.fromEntries(readdirSync(libDir).filter(name => /^lib\.(es5|es20\d\d(\.[a-z0-9]+)*|decorators(\.legacy)?|dom(\.[a-z]+)?|webworker\.importscripts|scripthost)\.d\.ts$/.test(name))
  .map(name => [name, readFileSync(join(libDir, name), 'utf8')]));
await build({ entryPoints: { natlang: resolve(root, 'src/browser/index.ts') }, bundle: true,
  platform: 'browser', format: 'esm', target: 'es2022', splitting: true,
  outdir: resolve(root, 'dist/browser'), chunkNames: 'chunks/[name]-[hash]',
  // undici is only loaded under Node (fetchModel); browsers use plain fetch.
  external: ['undici'],
  plugins: [{ name: 'browser-node-stubs', setup(build) {
    build.onResolve({ filter: /^node:/ }, args => ({ path: args.path.slice(5), namespace: 'browser-node-stub' }));
    build.onLoad({ filter: /.*/, namespace: 'browser-node-stub' }, args => ({ contents: args.path === 'zlib' ?
      'import { gunzipSync, gzipSync } from "fflate"; export { gunzipSync, gzipSync }; export const constants = {}; export default { gunzipSync, gzipSync, constants };' :
      'export default {}; export const fileURLToPath = () => { throw new Error("Node path API unavailable in browser"); };', loader: 'js', resolveDir: root }));
  } }],
  define: { __NATLANG_PRELUDE__: JSON.stringify(readFileSync(resolve(root, 'prelude.js'), 'utf8')),
    __NATLANG_TS_LIBS__: JSON.stringify(libs) },
  legalComments: 'none' });
// The Neuralese service as WebAssembly (vendor/neuralese-wasm, built from the llama.cpp fork) and the worker that hosts
// it (startBrowserNeuralese).
await build({ entryPoints: { 'neuralese-worker': resolve(root, 'src/browser/neuralese-worker.ts') }, bundle: true,
  platform: 'browser', format: 'esm', target: 'es2022', outdir: resolve(root, 'dist/browser'), external: ['undici'],
  plugins: [{ name: 'browser-node-stubs', setup(build) {
    build.onResolve({ filter: /^node:/ }, args => ({ path: args.path.slice(5), namespace: 'browser-node-stub' }));
    build.onLoad({ filter: /.*/, namespace: 'browser-node-stub' }, () => ({ contents: 'export default {};', loader: 'js', resolveDir: root }));
  } }], legalComments: 'none' });
// The call store's worker (src/browser/call-store-worker.ts) and SQLite's WebAssembly, which it loads from beside itself.
await build({ entryPoints: { 'call-store-worker': resolve(root, 'src/browser/call-store-worker.ts') }, bundle: true,
  platform: 'browser', format: 'esm', target: 'es2022', outdir: resolve(root, 'dist/browser'), external: ['undici'],
  plugins: [{ name: 'browser-node-stubs', setup(build) {
    build.onResolve({ filter: /^node:/ }, args => ({ path: args.path.slice(5), namespace: 'browser-node-stub' }));
    build.onLoad({ filter: /.*/, namespace: 'browser-node-stub' }, () => ({ contents: 'export default {};', loader: 'js', resolveDir: root }));
  } }], legalComments: 'none' });
copyFileSync(fileURLToPath(import.meta.resolve('@sqlite.org/sqlite-wasm/sqlite3.wasm')), resolve(root, 'dist/browser/sqlite3.wasm'));
for (const asset of ['neuralese-wasm.mjs', 'neuralese-wasm.wasm', 'neuralese-wasm-mt.mjs', 'neuralese-wasm-mt.wasm',
  'neuralese-wasm-gpu.mjs', 'neuralese-wasm-gpu.wasm'])
  copyFileSync(resolve(root, 'vendor/neuralese-wasm', asset), resolve(root, 'dist/browser', asset));
copyFileSync(fileURLToPath(import.meta.resolve('@wllama/wllama/esm/wasm/wllama.wasm')),
  resolve(root, 'dist/browser/wllama.wasm'));
for (const [source, target] of [['wllama.js', 'wllama-compat.js'],
  ['wllama.wasm', 'wllama-compat.wasm']])
  copyFileSync(fileURLToPath(import.meta.resolve(`@wllama/wllama-compat/wasm/${source}`)),
    resolve(root, `dist/browser/${target}`));
// Pyodide is loaded on first Python use, from fixed local assets beside natlang.js.
const pyodideDir = resolve(root, 'dist/browser/pyodide');
mkdirSync(pyodideDir, { recursive: true });
for (const asset of ['pyodide.asm.mjs', 'pyodide.asm.wasm', 'python_stdlib.zip', 'pyodide-lock.json'])
  copyFileSync(fileURLToPath(import.meta.resolve(`pyodide/${asset}`)), join(pyodideDir, asset));
for (const asset of readdirSync(resolve(root, 'vendor/pyodide')).filter(name => name.endsWith('.whl')))
  copyFileSync(resolve(root, 'vendor/pyodide', asset), join(pyodideDir, asset));
