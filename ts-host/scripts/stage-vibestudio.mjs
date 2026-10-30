#!/usr/bin/env node
/**
 * Stage natlang for vendoring into Vibestudio (and any host that vendors rather than installs): generated ESM bundles
 * from one natlang commit, with a pin that lets the host verify its copy.
 *
 *   node scripts/stage-vibestudio.mjs --out DIR [--only core|workerd|pi|llama|eval|dashboard]
 *       Build and write, under DIR, `core/natlang-core.js` (+ `.d.ts`, `types/`; workerd, npm packages external), `workerd/natlang-workerd.js` (the same core with TypeScript bundled in, for hosts that cannot install TypeScript 5), `llama/natlang-llama.js`
 *       (+ `.d.ts`; Node), `eval/natlang-eval.js` and `eval/worker-entry.js` (+ `.d.ts`; Node), `dashboard/` (the embeddable
 *       page modules, styles and `types.d.ts`; copied files) and `natlang-pin.json`:
 *       { schema, commit, dirty: false, targets, toolchain, files: { "<path under DIR>": "<sha256>" } }.
 *       Refuses when any source file a bundle or its declarations were built from differs from HEAD (or is untracked), so
 *       the pin's commit names exactly the content that was built. npm packages stay external: the host installs them.
 *   node scripts/stage-vibestudio.mjs --check DIR
 *       Verify a vendored DIR: its files match the pin, and a fresh build of the pinned commit's content (built by that
 *       commit's own build scripts) reproduces them byte for byte. Exit 1 with every difference otherwise.
 *
 * Never edit the staged output; stage again.
 */
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { TARGETS, buildTarget, root as tsHost } from './vendor-build.mjs';

const PIN = 'natlang-pin.json';
const SCHEMA = 'natlang.vendor-pin/1';
/** Files besides the bundles' own inputs that decide what a build produces. Paths relative to ts-host. */
const BUILD_CONFIGURATION = ['scripts/vendor-build.mjs', 'scripts/browser-node-imports.mjs', 'tsconfig.json'];

const gitRaw = (...args) => {
  const result = spawnSync('git', ['-C', tsHost, ...args], { encoding: 'utf8', maxBuffer: 1 << 28 });
  if (result.status !== 0) throw new Error(`git ${args.join(' ')} failed: ${result.stderr.trim()}`);
  return result.stdout;
};
const git = (...args) => gitRaw(...args).trim();
const sha256 = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const repositoryRoot = git('rev-parse', '--show-toplevel');
const prefix = relative(repositoryRoot, tsHost).split('\\').join('/');

/** Every file under `directory`, as paths relative to it. */
function filesUnder(directory) {
  const found = [];
  (function walk(at) {
    for (const name of readdirSync(at)) {
      const path = join(at, name);
      if (statSync(path).isDirectory()) walk(path); else found.push(relative(directory, path).split('\\').join('/'));
    }
  })(directory);
  return found.sort();
}

/** Source files (paths relative to ts-host) whose worktree content differs from HEAD, or that HEAD does not have. */
function differingFromHead(paths) {
  const unique = [...new Set(paths)].sort();
  const tracked = new Set(git('ls-files', '--', ...unique).split('\n').filter(Boolean));
  const changed = new Set(gitRaw('status', '--porcelain', '--', ...unique).split('\n').filter(Boolean).map(line => relative(prefix, line.slice(3)).split('\\').join('/')));
  return unique.filter(path => !tracked.has(path) || changed.has(path));
}

function toolchain() {
  const version = name => JSON.parse(readFileSync(join(tsHost, 'node_modules', name, 'package.json'), 'utf8')).version;
  return { node: process.version, esbuild: version('esbuild'), typescript: version('typescript') };
}

async function stage(out, only) {
  const names = only ? [only] : Object.keys(TARGETS);
  for (const name of names) if (!TARGETS[name]) throw new Error(`--only is one of ${Object.keys(TARGETS).join(', ')}`);
  const scratch = mkdtempSync(join(tmpdir(), 'natlang-stage-'));
  try {
    const builds = await Promise.all(names.map(async name => ({ name, built: await buildTarget(name, join(scratch, TARGETS[name].directory)) })));
    const sources = new Set(BUILD_CONFIGURATION), files = {};
    for (const { name, built } of builds) {
      for (const source of built.sources) sources.add(source);
      for (const file of built.files) files[`${TARGETS[name].directory}/${file}`] = sha256(join(scratch, TARGETS[name].directory, file));
    }
    const dirty = differingFromHead([...sources]);
    if (dirty.length) throw new Error(`refusing to stage: ${dirty.length} source file(s) the bundles were built from differ from HEAD ` +
      `(commit them first):\n${dirty.map(path => `  ${prefix}/${path}`).join('\n')}`);
    // Replace what an earlier staging wrote (its pin lists it), then write this one.
    if (existsSync(join(out, PIN))) for (const path of Object.keys(JSON.parse(readFileSync(join(out, PIN), 'utf8')).files ?? {})) rmSync(join(out, path), { force: true });
    mkdirSync(out, { recursive: true });
    for (const name of names) cpSync(join(scratch, TARGETS[name].directory), join(out, TARGETS[name].directory), { recursive: true });
    const pin = { schema: SCHEMA, commit: git('rev-parse', 'HEAD'), dirty: false, targets: names, toolchain: toolchain(),
      files: Object.fromEntries(Object.entries(files).sort(([a], [b]) => a < b ? -1 : 1)) };
    writeFileSync(join(out, PIN), JSON.stringify(pin, null, 2) + '\n');
    console.log(`staged ${names.join(', ')} at ${pin.commit} into ${out}: ${Object.keys(files).length} files`);
  } finally { rmSync(scratch, { recursive: true, force: true }); }
}

/** Verify `dir` against its pin and a fresh build of the pinned commit. Returns the differences found. */
async function check(dir) {
  const pinPath = join(dir, PIN);
  if (!existsSync(pinPath)) throw new Error(`${pinPath} does not exist`);
  const pin = JSON.parse(readFileSync(pinPath, 'utf8'));
  if (pin.schema !== SCHEMA) throw new Error(`${PIN} has schema ${pin.schema}; expected ${SCHEMA}`);
  const problems = [];
  const present = filesUnder(dir).filter(path => path !== PIN);
  for (const path of Object.keys(pin.files)) {
    if (!existsSync(join(dir, path))) problems.push(`missing: ${path}`);
    else if (sha256(join(dir, path)) !== pin.files[path]) problems.push(`edited since staging: ${path}`);
  }
  for (const path of present) if (!(path in pin.files)) problems.push(`not in the pin: ${path}`);
  // A fresh build from the pinned commit's content, by that commit's own build scripts.
  const scratch = mkdtempSync(join(tmpdir(), 'natlang-verify-'));
  try {
    git('cat-file', '-e', `${pin.commit}^{commit}`);
    const archive = spawnSync('sh', ['-c', `git -C ${JSON.stringify(repositoryRoot)} archive ${pin.commit} ${prefix || '.'} | tar -x -C ${JSON.stringify(scratch)}`], { encoding: 'utf8' });
    if (archive.status !== 0) throw new Error(`cannot export ${pin.commit}: ${archive.stderr}`);
    const checkout = join(scratch, prefix);
    // Installed packages are shared, as in scripts/check-main.sh; nothing else of the working tree is.
    symlinkSync(join(tsHost, 'node_modules'), join(checkout, 'node_modules'));
    if (existsSync(join(repositoryRoot, 'node_modules'))) symlinkSync(join(repositoryRoot, 'node_modules'), join(scratch, 'node_modules'));
    const vendor = await import(pathToFileURL(join(checkout, 'scripts/vendor-build.mjs')).href);
    for (const name of pin.targets) {
      const built = join(scratch, `built-${name}`);
      await vendor.buildTarget(name, built);
      const directory = vendor.TARGETS[name].directory;
      for (const file of filesUnder(built)) {
        const path = `${directory}/${file}`;
        if (!(path in pin.files)) problems.push(`a fresh build of ${pin.commit.slice(0, 12)} writes ${path}, which the pin does not list`);
        else if (sha256(join(built, file)) !== pin.files[path]) problems.push(`a fresh build of ${pin.commit.slice(0, 12)} differs from the pin: ${path}`);
      }
      for (const path of Object.keys(pin.files)) if (path.startsWith(`${directory}/`) && !existsSync(join(built, path.slice(directory.length + 1))))
        problems.push(`a fresh build of ${pin.commit.slice(0, 12)} does not write ${path}`);
    }
  } finally { rmSync(scratch, { recursive: true, force: true }); }
  return problems;
}

const args = process.argv.slice(2);
const option = name => { const at = args.indexOf(name); return at >= 0 ? args[at + 1] : undefined; };
try {
  if (args.includes('--check')) {
    const problems = await check(resolve(option('--check') ?? ''));
    if (problems.length) { console.error(`vendored natlang does not match its pin:\n${problems.map(problem => `  ${problem}`).join('\n')}`); process.exit(1); }
    console.log('vendored natlang matches its pin and a fresh build of its commit');
  } else if (option('--out')) {
    await stage(resolve(option('--out')), option('--only'));
  } else {
    console.error('usage: stage-vibestudio.mjs --out DIR [--only core|workerd|pi|llama|eval|dashboard] | --check DIR');
    process.exit(2);
  }
} catch (error) {
  console.error(error.errors ? error.errors.map(item => item.text).join('\n') : error.message ?? error);
  process.exit(1);
}
