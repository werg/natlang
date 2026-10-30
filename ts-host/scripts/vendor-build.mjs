/**
 * The bundles natlang vendors into other repositories (scripts/stage-vibestudio.mjs), and the workerd core's own build
 * (scripts/build-core.mjs). One definition per target, one build function:
 *
 *   core   the platform-neutral runtime for workerd (src/core/index.ts): an ESM bundle that loads with `nodejs_compat`
 *   workerd  the same core with TypeScript bundled in, for hosts that cannot install TypeScript 5 as a package (see below)
 *   llama  the managed llama.cpp package (src/llama/index.ts), for Node
 *   eval   the evaluation, adaptation and optimization APIs (src/entry/evaluation.ts), for Node, with the evaluation worker
 *   dashboard  the dashboard's embeddable page modules and styles (scripts/dashboard-assets.mjs): copied files, not a bundle
 *
 * npm packages are always external: the consumer installs them (`packages: 'external'`). The Node built-ins the core
 * may import are named in CORE_NODE_IMPORTS, each with its reason; any other fails the build with its import chain (as
 * the browser build does, browser-node-imports.mjs). Declarations come from `tsc` over the target's entry point alone,
 * so the declarations always describe the sources the bundle was built from.
 */
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build as esbuild } from 'esbuild';
import { importChains } from './browser-node-imports.mjs';

export const root = resolve(import.meta.dirname, '..');

/** Node built-ins the core may import, which workerd provides under `nodejs_compat`. Importer paths are relative to ts-host. */
export const CORE_NODE_IMPORTS = [
  { specifier: 'node:async_hooks', importer: 'src/core/workerd.ts', reason: 'AsyncLocalStorage carries the natlang task through async control flow' },
];

/** Library declarations for the in-isolate TypeScript checker (ES libraries only: workerd has no DOM). */
function embeddedLibs() {
  const libDir = dirname(fileURLToPath(import.meta.resolve('typescript/lib/lib.es2022.d.ts')));
  return Object.fromEntries(readdirSync(libDir).filter(name => /^lib\.(es5|es20\d\d(\.[a-z0-9]+)*|decorators(\.legacy)?)\.d\.ts$/.test(name))
    .map(name => [name, readFileSync(join(libDir, name), 'utf8')]));
}

/** The plugin enforcing CORE_NODE_IMPORTS: an allowed import stays an import, any other fails the build with its import chain. */
function coreNodeImports(entry) {
  return { name: 'core-node-imports', setup(build) {
    const forbidden = new Set();
    build.onStart(() => { forbidden.clear(); });
    build.onResolve({ filter: /^node:/ }, args => {
      const importer = relative(root, args.importer).split('\\').join('/');
      if (CORE_NODE_IMPORTS.some(rule => rule.specifier === args.path && rule.importer === importer)) return { path: args.path, external: true };
      forbidden.add(args.path);
      return { path: args.path, namespace: 'core-node-forbidden' };
    });
    build.onLoad({ filter: /.*/, namespace: 'core-node-forbidden' }, () => ({ contents: 'module.exports = {};', loader: 'js' }));
    build.onEnd(result => {
      if (!forbidden.size || !result.metafile) return;
      const found = [...forbidden].sort().map(specifier => `${specifier}, reached by\n` +
        importChains(result.metafile, `core-node-forbidden:${specifier}`, [entry]).map(chain => '  ' + [...chain, specifier].join('\n    -> ')).join('\n'));
      return { errors: [{ text: `Node built-ins reachable from the workerd core (${entry}):\n${found.join('\n')}\n` +
        'Keep the Node-only module out of the core graph (behind a Node entry point or an injected platform hook), or, when workerd ' +
        'provides it and the core genuinely needs it, add a rule to CORE_NODE_IMPORTS in scripts/vendor-build.mjs.' }] };
    });
  } };
}

/**
 * What a self-contained workerd bundle does with the CommonJS compiler it carries. TypeScript decides at load whether it
 * runs on Node by reading `process.browser` (defined true for this target), so the Node built-ins it can `require` stay
 * unused (and just-bash's Node-only paths are unreachable): they become empty modules (`process` the isolate's own). `undici` is the Node model backend's long-request
 * dispatcher, which the core's model drivers never need in an isolate: its one use is stubbed to a class with no behavior.
 */
function selfContainedHost() {
  const builtins = /^(?:node:)?(?:fs|path|os|crypto|util|events|buffer|stream|zlib|assert|string_decoder|url|process|inspector|perf_hooks|module|readline|tty|net|vm|child_process|worker_threads)$/;
  return { name: 'self-contained-host', setup(build) {
    build.onResolve({ filter: builtins }, args => args.importer.includes('/node_modules/') ? { path: args.path, namespace: 'self-contained-builtin' } : undefined);
    // The shell's gzip, tar and zip commands need node:zlib, which an eval-linked module cannot import: they report it.
    build.onResolve({ filter: /^(?:node:)?zlib$/ }, args => args.importer.includes('/node_modules/') ? { path: args.path, namespace: 'self-contained-zlib' } : undefined);
    build.onLoad({ filter: /.*/, namespace: 'self-contained-zlib' }, () => ({ loader: 'js', contents:
      `const unavailable = () => { throw new Error('zlib is unavailable in this isolate'); };\n` +
      `export const gunzipSync = unavailable, gzipSync = unavailable, inflateSync = unavailable, deflateSync = unavailable, constants = {};\n` +
      `export default { gunzipSync, gzipSync, inflateSync, deflateSync, constants };` }));
    // The Python runner needs WebAssembly code generation, which an isolate disallows (workerd rejects it even through UNSAFE_EVAL).
    build.onResolve({ filter: /\/folder-python\.js$/ }, () => ({ path: 'folder-python', namespace: 'self-contained-python' }));
    build.onLoad({ filter: /.*/, namespace: 'self-contained-python' }, () => ({ loader: 'js', contents:
      `export async function runFolderPython() { throw new Error('Python is unavailable in this isolate: WebAssembly code generation is disabled'); }` }));
    build.onLoad({ filter: /.*/, namespace: 'self-contained-builtin' }, args => ({ loader: 'js',
      contents: /process$/.test(args.path) ? 'module.exports = globalThis.process ?? {};' : 'module.exports = {};' }));
    build.onResolve({ filter: /^undici$/ }, args => ({ path: args.path, namespace: 'self-contained-undici' }));
    build.onLoad({ filter: /.*/, namespace: 'self-contained-undici' }, () => ({ loader: 'js', contents: 'export class Agent {}' }));
  } };
}

/** `entries`: bundle name -> source path (the first is the target's entry point; the others are files it loads at run time). */
export const TARGETS = {
  core: { directory: 'core', entries: { 'natlang-core': 'src/core/index.ts' }, platform: 'neutral',
    plugins: entry => [coreNodeImports(entry)], define: () => ({ __NATLANG_TS_LIBS__: JSON.stringify(embeddedLibs()) }) },
  // Self-contained: TypeScript (the compiler API natlang runs) is bundled, so the host declares no `typescript` package, which
  // a host that ships TypeScript 7 (no JavaScript compiler API) could not satisfy. `yaml` and `@noble/hashes` stay external.
  workerd: { directory: 'workerd', entries: { 'natlang-workerd': 'src/core/workerd-bundle.ts' }, platform: 'neutral', selfContained: true,
    plugins: entry => [selfContainedHost(), coreNodeImports(entry)],
    define: () => ({ __NATLANG_TS_LIBS__: JSON.stringify(embeddedLibs()), 'process.browser': 'true' }) },
  llama: { directory: 'llama', entries: { 'natlang-llama': 'src/llama/index.ts' }, platform: 'node' },
  // The evaluation runner starts `./worker-entry.js` beside its own file (evaluation/worker.ts).
  dashboard: { directory: 'dashboard', assets: true },
  eval: { directory: 'eval', entries: { 'natlang-eval': 'src/entry/evaluation.ts', 'worker-entry': 'src/evaluation/worker-entry.ts' }, platform: 'node' },
};

/** The `.d.ts` files `entry` reaches through relative imports, as paths under `declarations`. */
function declarationClosure(declarations, entry) {
  const seen = new Set(), queue = [entry];
  const reference = /(?:\bfrom\s*|\bimport\s*\(\s*|\bimport\s+)['"](\.{1,2}\/[^'"]*)['"]/g;
  while (queue.length) {
    const file = queue.pop();
    if (seen.has(file)) continue;
    seen.add(file);
    const text = readFileSync(join(declarations, file), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
    for (const match of text.matchAll(reference)) {
      const target = join(dirname(file), match[1]).replace(/\.js$/, '');
      const candidate = [`${target}.d.ts`, join(target, 'index.d.ts')].find(path => existsSync(join(declarations, path)));
      if (!candidate) throw new Error(`${file} imports ${match[1]}, which tsc did not declare`);
      queue.push(candidate);
    }
  }
  return [...seen].sort();
}

/** Declarations for the closure of `entry` (a source path under src/), emitted by tsc into `outDir`. */
function emitDeclarations(entry, outDir) {
  const scratch = mkdtempSync(join(tmpdir(), 'natlang-dts-'));
  try {
    const config = join(scratch, 'tsconfig.json');
    writeFileSync(config, JSON.stringify({ extends: join(root, 'tsconfig.json'), include: [], exclude: [],
      files: [join(root, entry)], compilerOptions: { rootDir: join(root, 'src'), outDir, declaration: true, emitDeclarationOnly: true, noEmit: false,
        typeRoots: [join(root, 'node_modules/@types'), join(root, '../node_modules/@types')], types: ['node'] } }));
    const result = spawnSync(process.execPath, [join(root, 'node_modules/typescript/bin/tsc'), '-p', config], { encoding: 'utf8' });
    if (result.status !== 0) throw new Error(`declarations for ${entry} did not compile:\n${result.stdout}${result.stderr}`);
  } finally { rmSync(scratch, { recursive: true, force: true }); }
}

/**
 * Build target `name` into `outdir`: its bundles (`<bundle>.js`), `<entry bundle>.d.ts` and the declarations it re-exports
 * (`types/`). Returns the paths relative to `outdir` and every source file the result was built from (relative to ts-host).
 */
export async function buildTarget(name, outdir) {
  const target = TARGETS[name];
  if (!target) throw new Error(`unknown target ${name}; one of ${Object.keys(TARGETS).join(', ')}`);
  if (target.assets) {
    // Files copied as they are; the module says which and from where.
    const { buildDashboardAssets } = await import('./dashboard-assets.mjs');
    mkdirSync(outdir, { recursive: true });
    const built = buildDashboardAssets(outdir);
    return { files: built.files, sources: [...built.sources, 'scripts/dashboard-assets.mjs'] };
  }
  const [entryName, entrySource] = Object.entries(target.entries)[0];
  // Only what this build writes is replaced: `outdir` may be dist/core, which tsc also writes to.
  mkdirSync(outdir, { recursive: true });
  rmSync(join(outdir, 'types'), { recursive: true, force: true });
  for (const bundle of [...Object.keys(target.entries)]) rmSync(join(outdir, `${bundle}.js`), { force: true });
  const bundled = await esbuild({ entryPoints: Object.fromEntries(Object.entries(target.entries).map(([bundle, source]) => [bundle, join(root, source)])),
    bundle: true, format: 'esm', platform: target.platform, target: target.platform === 'node' ? 'node22' : 'es2022',
    ...(target.selfContained ? { external: ['yaml', '@noble/hashes', '@noble/hashes/*'], mainFields: ['module', 'main'], conditions: ['workerd', 'worker', 'browser'] }
      : { packages: 'external' }),
    outdir, absWorkingDir: root, metafile: true, legalComments: 'none', logLevel: 'silent',
    plugins: target.plugins?.(entrySource) ?? [], define: target.define?.() ?? {} });
  // esbuild names each bundled module in a comment by its path relative to the working directory; a package that resolves
  // through the checkout's node_modules link sits outside it, so the comment would depend on where the build ran.
  if (target.selfContained) for (const bundle of Object.keys(target.entries)) {
    const file = join(outdir, `${bundle}.js`);
    writeFileSync(file, readFileSync(file, 'utf8').replace(/^\/\/ (?:\.\.\/)+(?:[^\n]*?\/)?(node_modules\/)/gm, '// $1'));
  }
  const declarations = mkdtempSync(join(tmpdir(), 'natlang-types-'));
  try {
    emitDeclarations(entrySource, declarations);
    const entryDeclaration = entrySource.replace(/^src\//, '').replace(/\.ts$/, '.d.ts');
    const emitted = declarationClosure(declarations, entryDeclaration);
    for (const file of emitted) {
      mkdirSync(dirname(join(outdir, 'types', file)), { recursive: true });
      cpSync(join(declarations, file), join(outdir, 'types', file));
    }
    writeFileSync(join(outdir, `${entryName}.d.ts`), `export * from './types/${entryDeclaration.replace(/\.d\.ts$/, '.js')}';\n`);
    const sources = new Set(Object.keys(bundled.metafile.inputs).filter(path => !/^[^/]*:/.test(path) && !path.startsWith('node_modules/') && !path.includes('/node_modules/')));
    for (const file of emitted) sources.add(`src/${file.replace(/\.d\.ts$/, '.ts')}`);
    const files = [...Object.keys(target.entries).map(bundle => `${bundle}.js`), `${entryName}.d.ts`, ...emitted.map(file => `types/${file}`)];
    return { files: files.sort(), sources: [...sources].sort(), metafile: bundled.metafile };
  } finally { rmSync(declarations, { recursive: true, force: true }); }
}

/** Modules (paths relative to src) that must stay out of the core's bundle: Node-only by nature, not by import. */
const CORE_EXCLUDED = [/^cli\//, /^dashboard\//, /^teacher\//, /^tune\//, /^llama\//, /^browser\//, /^calls\/(store|feedback)\.ts$/,
  /^native\/folder-(python|shell)\.ts$/, /^model\/(local-server|llama-runtime|native-managed|shared-server)\.ts$/, /^runtime\/node\.ts$/,
  /^environment\.ts$/, /^machine-store\.ts$/, /^node-runtime\.ts$/, /^index\.ts$/];

/** What the core's bundle reaches that a workerd isolate cannot run: Node-only modules, Node built-ins beyond CORE_NODE_IMPORTS. */
export async function coreGraphProblems() {
  const target = TARGETS.core;
  const entry = Object.values(target.entries)[0];
  try {
    const result = await esbuild({ entryPoints: [join(root, entry)], bundle: true, format: 'esm', platform: target.platform, packages: 'external',
      write: false, metafile: true, absWorkingDir: root, logLevel: 'silent', plugins: target.plugins(entry), define: target.define() });
    const modules = Object.keys(result.metafile.inputs).filter(path => path.startsWith('src/')).map(path => path.slice('src/'.length));
    return { modules: modules.length, problems: modules.filter(path => CORE_EXCLUDED.some(pattern => pattern.test(path)))
      .map(path => `${path} is Node-only and must stay out of the core graph`) };
  } catch (error) { return { modules: 0, problems: error.errors ? error.errors.map(item => item.text) : [String(error)] }; }
}
