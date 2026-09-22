#!/usr/bin/env node
import { mkdir, rename, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { buildSelection, expandPatterns } from './teacher-coverage.mjs';

const args = process.argv.slice(2), output = args.find(value => !value.startsWith('--') &&
  !args[args.indexOf(value) - 1]?.startsWith('--'));
const values = flag => args.flatMap((value, index) => value === flag ? [args[index + 1]] : []);
const value = (flag, fallback) => values(flag).at(-1) ?? fallback;
if (!output) throw new Error('usage: build-teacher-coverage-selection.mjs OUT [--glob PATTERN] [--per-family N] [--seed N]');
const patterns = values('--glob');
const perFamily = Number(value('--per-family', 4)), seed = Number(value('--seed', 909));
if (!Number.isInteger(perFamily) || perFamily < 1 || !Number.isSafeInteger(seed)) throw new Error('invalid per-family or seed');
const paths = await expandPatterns(patterns.length ? patterns : ['data/external_pilot/synthetic-*-reviewed-*.ir.jsonl']);
if (!paths.length) throw new Error('corpus glob found no files');
const { rows, manifest } = await buildSelection(paths, { perFamily, seed });
const target = resolve(output), staged = `${target}.building`;
await mkdir(dirname(target), { recursive: true });
await writeFile(staged, rows.map(row => JSON.stringify(row)).join('\n') + '\n');
await rename(staged, target);
await writeFile(`${target}.manifest.json`, JSON.stringify(manifest, null, 2) + '\n');
console.log(`${rows.length} programs across ${manifest.families} families -> ${target}`);
