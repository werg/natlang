import { copyFileSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const root = resolve(import.meta.dirname, '..');
await build({ entryPoints: [resolve(root, 'src/browser/index.ts')], bundle: true,
  platform: 'browser', format: 'esm', target: 'es2022',
  outfile: resolve(root, 'dist/browser/natlang.js'),
  plugins: [{ name: 'browser-eval', setup(build) {
    build.onResolve({ filter: /^\.\.\/environment\.js$/ }, args =>
      ['/native/runtime.ts', '/native/workspace.ts'].some(file => args.importer.endsWith(file)) ?
        { path: resolve(root, 'src/browser/environment.ts') } : null);
  } }],
  define: { __NATLANG_PRELUDE__: JSON.stringify(readFileSync(resolve(root, 'prelude.js'), 'utf8')) },
  legalComments: 'none' });
copyFileSync(fileURLToPath(import.meta.resolve('@wllama/wllama/esm/wasm/wllama.wasm')),
  resolve(root, 'dist/browser/wllama.wasm'));
for (const [source, target] of [['wllama.js', 'wllama-compat.js'],
  ['wllama.wasm', 'wllama-compat.wasm']])
  copyFileSync(fileURLToPath(import.meta.resolve(`@wllama/wllama-compat/wasm/${source}`)),
    resolve(root, `dist/browser/${target}`));
