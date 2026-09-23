import { readFile, readdir, stat } from 'node:fs/promises';
import { basename, relative, resolve, sep } from 'node:path';

async function walk(dir) {
  const out = [];
  let entries;
  try { entries = await readdir(dir, { withFileTypes: true }); }
  catch (error) { if (error.code === 'ENOENT') return out; throw error; }
  for (const entry of entries) {
    if (entry.name === '.git' || entry.name === 'node_modules') continue;
    const path = resolve(dir, entry.name);
    if (entry.isDirectory()) out.push(...await walk(path));
    else if (entry.isFile()) out.push(path);
  }
  return out;
}

function toPatterns(value) {
  if (!value) return [];
  return (Array.isArray(value) ? value : [value]).filter((item) => typeof item === 'string').map((item) => item.replaceAll('\\', '/'));
}

function globRegex(pattern) {
  let output = '^';
  for (let i = 0; i < pattern.length; i += 1) {
    const ch = pattern[i];
    if (ch === '*' && pattern[i + 1] === '*') {
      i += 1;
      if (pattern[i + 1] === '/') { i += 1; output += '(?:.*/)?'; }
      else output += '.*';
    } else if (ch === '*') output += '[^/]*';
    else if (ch === '?') output += '[^/]';
    else output += ch.replace(/[|\\{}()[\]^$+?.]/g, '\\$&');
  }
  return new RegExp(`${output}$`);
}

function expandPatterns(patterns, files, exerciseDir) {
  const results = new Set();
  for (const original of patterns) {
    const pattern = original.replaceAll('%{slug}', basename(exerciseDir)).replaceAll('%{kebab_slug}', basename(exerciseDir));
    const regex = globRegex(pattern);
    for (const file of files) {
      const rel = relative(exerciseDir, file).split(sep).join('/');
      if (regex.test(rel)) results.add(file);
    }
  }
  return [...results].sort();
}

async function parseConfig(path) {
  try { return JSON.parse(await readFile(path, 'utf8')); }
  catch (error) { if (error.code === 'ENOENT') return null; throw new Error(`Invalid Exercism config ${path}: ${error.message}`); }
}

/**
 * Inventory one or more Exercism track roots without executing track tooling.
 * descriptionOverrides can map exercise slug or relative exercise path to inline
 * instruction text or a file path; this lets a later extractor use canonical
 * problem-specifications text while preserving track-local instructions here.
 */
export async function enumerateExercism(root, { descriptionOverrides = {} } = {}) {
  const absoluteRoot = resolve(root);
  const files = await walk(absoluteRoot);
  const configs = files.filter((path) => path.split(sep).slice(-2).join('/') === '.meta/config.json').sort();
  const inventory = [];
  for (const configPath of configs) {
    const exerciseDir = resolve(configPath, '../..');
    const config = await parseConfig(configPath);
    if (!config) continue;
    const slug = config.slug || basename(exerciseDir);
    const relExercise = relative(absoluteRoot, exerciseDir).split(sep).join('/');
    const docsPath = resolve(exerciseDir, '.docs/instructions.md');
    const override = descriptionOverrides[relExercise] ?? descriptionOverrides[slug];
    let instructionPath = docsPath;
    let instruction;
    if (typeof override === 'string' && override.startsWith('@file:')) {
      instructionPath = resolve(absoluteRoot, override.slice(6));
      instruction = await readFile(instructionPath, 'utf8');
    } else if (typeof override === 'string') {
      instruction = override;
      instructionPath = null;
    } else {
      try { instruction = await readFile(docsPath, 'utf8'); }
      catch (error) { if (error.code !== 'ENOENT') throw error; instruction = null; }
    }
    const declared = config.files || {};
    // Track schemas use both `exemplar` and the common `example` key for
    // canonical reference implementations; never fall back to learner solutions.
    const solutions = expandPatterns(toPatterns(declared.exemplar ?? declared.example), files, exerciseDir);
    const tests = expandPatterns(toPatterns(declared.test), files, exerciseDir);
    const stubs = expandPatterns(toPatterns(declared.solution), files, exerciseDir);
    // Older tracks sometimes omit files.exemplar; use only the conventional
    // metadata exemplar names, never arbitrary student implementation files.
    if (!solutions.length) {
      for (const candidate of ['.meta/example.ts', '.meta/example.js', '.meta/exemplar.ts', '.meta/exemplar.js', '.meta/exemplar.mjs']) {
        const found = files.find((path) => path === resolve(exerciseDir, candidate));
        if (found) solutions.push(found);
      }
    }
    const instructionExists = instruction !== null;
    for (const sourcePath of solutions) {
      inventory.push({
        exercise: `${relExercise}${slug !== basename(exerciseDir) ? ` (${slug})` : ''}`,
        slug,
        exercisePath: relExercise,
        sourcePath: relative(absoluteRoot, sourcePath).split(sep).join('/'),
        instruction,
        instructionPath: instructionPath ? relative(absoluteRoot, instructionPath).split(sep).join('/') : null,
        testPaths: tests.map((path) => relative(absoluteRoot, path).split(sep).join('/')),
        stubPaths: stubs.map((path) => relative(absoluteRoot, path).split(sep).join('/')),
        configPath: relative(absoluteRoot, configPath).split(sep).join('/'),
        inventoryWarnings: [
          ...(!instructionExists ? ['missing .docs/instructions.md and no description override'] : []),
          ...(!tests.length ? ['no test files resolved from files.test'] : []),
        ],
      });
    }
  }
  return inventory;
}
