#!/usr/bin/env node
/**
 * Build an application's browser entry: the application's `tsconfig.browser.json` compiled for the browser target, with
 * the runtime specifiers (`natlang:runtime`, `@natlang/browser`) bound to the browser runtime and typed against its
 * declarations, then bundled under the Node built-in policy (browser-node-imports.mjs) into APP/dist/browser.
 *
 * The runtime stays a module of its own, imported as `--runtime` (default `@natlang/browser`, which the page maps to
 * ts-host's dist/browser/natlang.js with an import map), so the application shares the page's runtime instance and the
 * runtime's own graph is checked by its build (build-browser.mjs). Run after `npm run build:node` and `build:browser`.
 *
 * Usage: node scripts/build-application-browser.mjs APP [--entry browser.ts] [--out DIR] [--runtime SPECIFIER|URL]
 */
import { basename, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { browserNodeImports } from './browser-node-imports.mjs';

const root = resolve(import.meta.dirname, '..');

/** Leaves the runtime's import as it is: the page provides that module. */
const runtimeModule = specifier => ({ name: 'natlang-runtime-module', setup(build) {
  build.onResolve({ filter: /.*/ }, args => args.path === specifier ? { path: specifier, external: true } : undefined);
} });

/**
 * Compile and bundle `app`'s browser entry. Returns esbuild's result (with its metafile) and the bundled entry's path.
 * `write: false` keeps the bundle in memory (`result.outputFiles`).
 */
export async function buildApplicationBrowser({ app, entry = 'browser.ts', outdir, runtime = '@natlang/browser', write = true,
  tolerated } = {}) {
  const project = resolve(app);
  const { buildProject, formatDiagnostics } = await import(pathToFileURL(join(root, 'dist', 'index.js')).href);
  const compiled = join(project, '.natlang', 'browser-build');
  const result = buildProject({ project: join(project, 'tsconfig.browser.json'), outDir: compiled, target: 'browser',
    writeDeclarations: false, runtimeModule: { specifiers: ['@natlang/browser'], url: runtime,
      types: join(root, 'dist', 'browser', 'index.d.ts') } });
  if (!result.ok) throw new Error(`${app}: the browser build does not type-check:\n${formatDiagnostics(result.diagnostics)}`);
  const name = basename(entry).replace(/\.ts$/, '');
  const out = resolve(outdir ?? join(project, 'dist', 'browser'));
  const bundled = await build({ entryPoints: { [name]: join(compiled, entry.replace(/\.ts$/, '.js')) }, bundle: true,
    platform: 'browser', format: 'esm', target: 'es2022', splitting: true, outdir: out, chunkNames: 'chunks/[name]-[hash]',
    absWorkingDir: project, write, logLevel: 'silent', legalComments: 'none',
    // undici is only loaded under Node (fetchModel); browsers use plain fetch.
    external: ['undici'],
    plugins: [runtimeModule(runtime), browserNodeImports(tolerated ? { tolerated } : {})] });
  return { result: bundled, entry: join(out, `${name}.js`) };
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  const args = process.argv.slice(2);
  const option = name => { const at = args.indexOf(name); return at >= 0 ? args[at + 1] : undefined; };
  const app = args.find((arg, i) => !arg.startsWith('--') && !['--entry', '--out', '--runtime'].includes(args[i - 1] ?? ''));
  if (!app) { console.error('usage: build-application-browser.mjs APP [--entry browser.ts] [--out DIR] [--runtime SPECIFIER|URL]'); process.exit(2); }
  try {
    const { entry } = await buildApplicationBrowser({ app, entry: option('--entry'), outdir: option('--out'), runtime: option('--runtime') });
    console.log(`built ${entry}`);
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  }
}
