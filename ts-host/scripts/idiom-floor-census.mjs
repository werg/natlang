#!/usr/bin/env node
/**
 * The idiom floor of `natlang check` (src/compiler/idiom.ts) over the repo's programs: every package (a directory with
 * natlang.json) under the given roots, plus the playground gallery's projects. Prints JSON counts per rule and per
 * directory; `--sites` adds every warning. Nothing is written into the programs (check runs with write: false).
 * The corpus migration (plans/semantic-fp/07_CORPUS_MIGRATION.md) reads these counts.
 *
 *   node scripts/idiom-floor-census.mjs [--sites] [ROOT...]    (default roots: applications codebases examples
 *   training/semantic_applications, and the playground gallery)
 */
import { existsSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { IDIOM_CODES } from '../dist/compiler/idiom.js';
import { checkProject } from '../dist/compiler/node-project.js';
import { newPlaygroundProject, validatePlaygroundProject } from '../dist/app/playground.js';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const args = process.argv.slice(2);
const sites = args.includes('--sites');
const roots = args.filter(arg => !arg.startsWith('--'));
const codes = [...IDIOM_CODES, 'untrusted-instruction'];
const SKIP = new Set(['node_modules', 'dist', 'build', '.natlang']);

/** Package roots: every natlang.json directory below `dir` (a nested package is left out of its parent's check). */
function packages(dir, out = []) {
  if (existsSync(join(dir, 'natlang.json'))) out.push(dir);
  for (const entry of readdirSync(dir)) {
    if (SKIP.has(entry) || entry.startsWith('.')) continue;
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) packages(path, out);
  }
  return out;
}

/**
 * Programs without natlang.json: each `.nl` file outside every package is checked as part of its nearest enclosing
 * directory that holds a tsconfig.json, package.json or application.json, else of the root's child that holds it.
 */
function looseUnits(root, pkgs) {
  const units = new Set();
  const inPackage = path => pkgs.some(pkg => path === pkg || path.startsWith(pkg + '/'));
  const walk = dir => {
    for (const entry of readdirSync(dir)) {
      if (SKIP.has(entry) || entry.startsWith('.')) continue;
      const path = join(dir, entry);
      if (statSync(path).isDirectory()) { if (!inPackage(path)) walk(path); continue; }
      if (!entry.endsWith('.nl')) continue;
      let unit;
      for (let at = dir; at.startsWith(root + '/'); at = dirname(at))
        if (['tsconfig.json', 'package.json', 'application.json'].some(name => existsSync(join(at, name)))) { unit = at; break; }
      unit ??= join(root, relative(root, dir).split('/')[0]);
      units.add(unit);
    }
  };
  walk(root);
  return [...units].filter(unit => ![...units].some(other => other !== unit && unit.startsWith(other + '/')));
}

const result = { schema: 'natlang.idiom-floor-census/1', rules: Object.fromEntries(codes.map(code => [code, 0])),
  directories: {}, failed: [], ...(sites ? { sites: [] } : {}) };
const count = (directory, code, site) => {
  result.rules[code]++;
  const row = result.directories[directory] ??= { packages: 0, ...Object.fromEntries(codes.map(item => [item, 0])) };
  row[code]++;
  if (sites) result.sites.push(site);
};
for (const root of roots.length ? roots.filter(root => root !== 'playground') : ['applications', 'codebases', 'examples', 'training/semantic_applications']) {
  const base = resolve(repo, root), pkgs = packages(base);
  for (const pkg of [...pkgs, ...looseUnits(base, pkgs)]) {
    const where = relative(repo, pkg);
    const directory = where.split('/').slice(0, root.split('/').length + 1).join('/');
    (result.directories[directory] ??= { packages: 0, ...Object.fromEntries(codes.map(item => [item, 0])) }).packages++;
    let diagnostics;
    try { diagnostics = checkProject(pkg, { write: false, adaptation: false }).diagnostics; }
    catch (error) { result.failed.push({ package: where, error: String(error?.message ?? error).split('\n')[0] }); continue; }
    for (const item of diagnostics) if (codes.includes(item.code))
      count(directory, item.code, { file: join(where, item.file), line: item.line, code: item.code });
  }
}
if (!roots.length || roots.includes('playground')) {
  const { examples } = await import('../playground/examples.mjs');
  const { naturalExamples } = await import('../playground/examples/natural.mjs');
  const all = [...new Map([...examples, ...naturalExamples].map(item => [item.id, item])).values()];
  const directory = 'ts-host/playground';
  (result.directories[directory] ??= { packages: 0, ...Object.fromEntries(codes.map(item => [item, 0])) }).packages += all.length;
  for (const item of all) {
    const project = newPlaygroundProject(item.name, item.root, item.files, item.inputs, item.expected);
    for (const diagnostic of validatePlaygroundProject(project)) {
      const code = codes.includes(diagnostic.code) ? diagnostic.code : undefined;
      if (code) count(directory, code, { file: `${item.id}:${diagnostic.file}`, code });
    }
  }
}
process.stdout.write(JSON.stringify(result, null, 2) + '\n');
