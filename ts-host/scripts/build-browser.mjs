import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { build } from 'esbuild';

const root = resolve(import.meta.dirname, '..');
await build({ entryPoints: [resolve(root, 'src/browser/index.ts')], bundle: true,
  platform: 'browser', format: 'esm', target: 'es2022',
  outfile: resolve(root, 'dist/browser/natlang.js'),
  plugins: [{ name: 'browser-eval', setup(build) {
    build.onResolve({ filter: /^\.\.\/environment\.js$/ }, args =>
      args.importer.endsWith('/native/runtime.ts') ?
        { path: resolve(root, 'src/browser/environment.ts') } : null);
  } }],
  define: { __NATLANG_PRELUDE__: JSON.stringify(readFileSync(resolve(root, 'prelude.js'), 'utf8')) },
  legalComments: 'none' });
