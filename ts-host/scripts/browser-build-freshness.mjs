#!/usr/bin/env node
/**
 * Whether dist/browser holds the browser build of the current sources (build-browser.mjs), and not a stale or partly
 * overwritten one. build-browser.mjs records a stamp (dist/browser/build-stamp.json): the sha256 of every file esbuild
 * read for the bundles (its metafiles), of the build's own configuration and of the assets it copied (the vendored
 * Neuralese wasm, sqlite-wasm, wllama, Pyodide), and of every file it wrote. The bundle is fresh when all of them still
 * match and the vendored Neuralese wasm matches its provenance.json. Anything else fails with what changed.
 *
 *   node scripts/browser-build-freshness.mjs             exit 1 with the reasons when dist/browser is not fresh
 *   node scripts/browser-build-freshness.mjs --rebuild   rebuild (build-browser.mjs) when it is not fresh, then recheck
 *
 * Browser test scripts call requireFreshBrowserBuild() first; their npm scripts run `--rebuild` before them.
 */
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(import.meta.dirname, '..');
export const STAMP = resolve(root, 'dist/browser/build-stamp.json');
export const STAMP_SCHEMA = 'natlang.browser-build/1';
const VENDOR = resolve(root, 'vendor/neuralese-wasm');

export const sha256 = path => createHash('sha256').update(readFileSync(path)).digest('hex');
/** A path as the stamp records it: relative to ts-host, with forward slashes. */
export const stampPath = path => relative(root, resolve(root, path)).split('\\').join('/');

/** The stamp's sections: `inputs` (read by the build), `assets` (copied: source -> target), `outputs` (written). */
export function makeStamp({ inputs, assets, outputs }) {
  const hashes = paths => Object.fromEntries([...new Set(paths.map(stampPath))].sort().map(path => [path, sha256(resolve(root, path))]));
  return { schema: STAMP_SCHEMA, built: new Date().toISOString(), inputs: hashes(inputs),
    assets: Object.fromEntries(assets.map(({ source, target }) => [stampPath(target), { source: stampPath(source), sha256: sha256(source) }])
      .sort(([a], [b]) => a.localeCompare(b))),
    outputs: hashes(outputs) };
}

/** Why the vendored Neuralese wasm is not what provenance.json pins (empty when it is). */
export function vendoredWasmProblems() {
  const provenance = JSON.parse(readFileSync(resolve(VENDOR, 'provenance.json'), 'utf8'));
  const problems = [];
  for (const [name, expected] of Object.entries(provenance.sha256 ?? {})) {
    const path = resolve(VENDOR, name);
    if (!existsSync(path)) problems.push(`vendor/neuralese-wasm/${name} is missing`);
    else if (sha256(path) !== expected) problems.push(`vendor/neuralese-wasm/${name} does not match provenance.json ` +
      `(fork ${provenance.fork_commit}); re-vendor it or update provenance.json, then run ` +
      'scripts/verify_neuralese_wasm_provenance.py');
  }
  return problems;
}

/** Why dist/browser is not the build of the current sources (empty when it is). */
export function browserBuildProblems() {
  const problems = vendoredWasmProblems();
  if (!existsSync(STAMP)) {
    problems.push(`${stampPath(STAMP)} is missing: dist/browser was not built by build-browser.mjs, or its build was interrupted`);
    return problems;
  }
  let stamp;
  try { stamp = JSON.parse(readFileSync(STAMP, 'utf8')); } catch (error) { problems.push(`${stampPath(STAMP)}: ${error.message}`); return problems; }
  if (stamp.schema !== STAMP_SCHEMA) { problems.push(`${stampPath(STAMP)}: schema ${stamp.schema}, expected ${STAMP_SCHEMA}`); return problems; }
  const differs = (path, expected) => !existsSync(resolve(root, path)) || sha256(resolve(root, path)) !== expected;
  const changed = Object.entries(stamp.inputs).filter(([path, hash]) => differs(path, hash)).map(([path]) => path);
  if (changed.length) problems.push(`sources changed since the browser build (${changed.length}): ${list(changed)}`);
  const assets = Object.values(stamp.assets).filter(({ source, sha256: hash }) => differs(source, hash)).map(({ source }) => source);
  if (assets.length) problems.push(`copied assets changed since the browser build (${assets.length}): ${list(assets)}`);
  const written = [...Object.entries(stamp.outputs), ...Object.entries(stamp.assets).map(([target, { sha256: hash }]) => [target, hash])]
    .filter(([path, hash]) => differs(path, hash)).map(([path]) => path);
  if (written.length) problems.push(`files of the browser build were removed or overwritten after it (${written.length}; ` +
    `e.g. by an older build:node writing tsc output into dist/browser): ${list(written)}`);
  return problems;
}

const list = paths => paths.slice(0, 6).join(', ') + (paths.length > 6 ? `, … (${paths.length - 6} more)` : '');

/** The message for a stale build: the reasons and how to fix it. */
export function staleMessage(problems, what = 'this') {
  return `${what} needs a fresh browser build (ts-host dist/browser), and it is not:\n` +
    problems.map(problem => `  - ${problem}`).join('\n') +
    '\nRebuild it: (cd ts-host && npm run build:browser), or node ts-host/scripts/browser-build-freshness.mjs --rebuild.';
}

/** Throw unless dist/browser is the build of the current sources. */
export function requireFreshBrowserBuild(what) {
  const problems = browserBuildProblems();
  if (problems.length) throw new Error(staleMessage(problems, what));
}

/** For a script run from the command line: print why and exit 1 unless dist/browser is fresh. */
export function exitUnlessFreshBrowserBuild(what) {
  const problems = browserBuildProblems();
  if (problems.length) { process.stderr.write(staleMessage(problems, what) + '\n'); process.exit(1); }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  const rebuild = process.argv.includes('--rebuild');
  let problems = browserBuildProblems();
  // A vendoring mismatch is not fixed by rebuilding the bundle.
  if (rebuild && problems.length && !vendoredWasmProblems().length) {
    process.stderr.write(`dist/browser is not fresh; rebuilding:\n${problems.map(problem => `  - ${problem}`).join('\n')}\n`);
    const built = spawnSync(process.execPath, [resolve(root, 'scripts/build-browser.mjs')], { stdio: 'inherit' });
    if (built.status !== 0) process.exit(built.status ?? 1);
    problems = browserBuildProblems();
  }
  if (problems.length) { process.stderr.write(staleMessage(problems, 'ts-host') + '\n'); process.exit(1); }
  process.stderr.write('dist/browser is fresh\n');
}
