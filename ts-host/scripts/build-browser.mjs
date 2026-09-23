import { copyFileSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const root = resolve(import.meta.dirname, '..');
// Library declarations for the in-browser TypeScript checker (eval `nl` analysis, virtual projects).
const libDir = dirname(fileURLToPath(import.meta.resolve('typescript/lib/lib.es2022.d.ts')));
const libs = Object.fromEntries(readdirSync(libDir).filter(name => /^lib\.(es5|es20\d\d(\.[a-z0-9]+)*|decorators(\.legacy)?|dom(\.[a-z]+)?|webworker\.importscripts|scripthost)\.d\.ts$/.test(name))
  .map(name => [name, readFileSync(join(libDir, name), 'utf8')]));
await build({ entryPoints: [resolve(root, 'src/browser/index.ts')], bundle: true,
  platform: 'browser', format: 'esm', target: 'es2022',
  outfile: resolve(root, 'dist/browser/natlang.js'),
  external: ['node:*'],
  define: { __NATLANG_PRELUDE__: JSON.stringify(readFileSync(resolve(root, 'prelude.js'), 'utf8')),
    __NATLANG_TS_LIBS__: JSON.stringify(libs) },
  legalComments: 'none' });
copyFileSync(fileURLToPath(import.meta.resolve('@wllama/wllama/esm/wasm/wllama.wasm')),
  resolve(root, 'dist/browser/wllama.wasm'));
for (const [source, target] of [['wllama.js', 'wllama-compat.js'],
  ['wllama.wasm', 'wllama-compat.wasm']])
  copyFileSync(fileURLToPath(import.meta.resolve(`@wllama/wllama-compat/wasm/${source}`)),
    resolve(root, `dist/browser/${target}`));
