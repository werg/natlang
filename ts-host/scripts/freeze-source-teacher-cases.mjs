#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, rename, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { dumpNativeState, loadFunctionFile } from '../dist/index.js';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const args = process.argv.slice(2);
const output = resolve(args[0] ?? join(repo, 'data/teacher/source-cases-s909.ir.jsonl'));
const seeds = resolve(args[1] ?? join(repo, 'data/teacher/source-case-seeds.jsonl'));
const sources = {
  cb_dependency_plan: ['codebases/dependency_plan/plan.nl', 'lambda'],
  cb_highlighter: ['codebases/highlighter/highlight.nl', 'lambda'],
  cb_legal_move: ['codebases/legal_move/check_move.nl', 'lambda'],
  cb_mail_rules: ['codebases/mail_rules/process_mail.nl', 'lambda'],
  cb_moderation: ['codebases/moderation/moderate.nl', 'lambda'],
  cb_nlprolog: ['codebases/nlprolog/solve.nl', 'lambda'],
  cb_order_saga: ['codebases/order_saga/step.nl', 'fold'],
  cb_reconciliation: ['codebases/reconciliation/reconcile.nl', 'lambda'],
  cb_shopkeeper: ['codebases/shopkeeper/serve.nl', 'fold'],
  cb_webserver: ['codebases/webserver/handle.nl', 'fold'],
};
const hash = value => createHash('sha256').update(value).digest('hex');
async function tree(path) {
  const rows = [];
  for (const entry of await readdir(path, { withFileTypes: true })) {
    const child = join(path, entry.name);
    if (entry.isDirectory()) rows.push(...await tree(child));
    else if (entry.isFile()) rows.push(child);
  }
  return rows.sort();
}
const seedText = await readFile(seeds, 'utf8');
const cases = seedText.split(/\r?\n/).filter(Boolean).map(JSON.parse);
const rows = [];
for (const seed of cases) {
  const source = sources[seed.family];
  if (!source) throw new Error(`no current source registered for ${seed.family}`);
  const [relative, shape] = source, path = resolve(repo, relative);
  const loaded = dumpNativeState(loadFunctionFile(path));
  let root = loaded;
  if (shape === 'fold') {
    const old = seed.semantics.root_template?.$fold;
    if (!old) throw new Error(`${seed.id} needs a fold root_template`);
    const step = loaded.$lambda;
    root = { $fold: { type: old.type, types: step.types, over: [], init: old.init, step: loaded } };
  }
  const folder = dirname(path), files = await tree(folder);
  const revision = hash(Buffer.concat(await Promise.all(files.map(async file =>
    Buffer.concat([Buffer.from(file.slice(folder.length + 1)), Buffer.from([0]), await readFile(file), Buffer.from([0])])))));
  const semantics = structuredClone(seed.semantics);
  delete semantics.root_template;
  semantics.root = root;
  rows.push({ version: 'natlang.program/1', id: seed.id, kind: 'lambda_graph',
    family: seed.family, source: 'natlang-current-source', split: seed.split,
    source_ids: [relative], source_groups: [seed.family], source_revisions: [revision],
    license: 'project-generated', gold_sources: ['frozen-reference-case'],
    generation: { generator: 'natlang.source_case_freezer/1' }, semantics });
}
const counts = Object.fromEntries(Object.keys(sources).map(family =>
  [family, rows.filter(row => row.family === family).length]));
if (Object.values(counts).some(count => count < 4))
  throw new Error(`source families need four cases: ${JSON.stringify(counts)}`);
await mkdir(dirname(output), { recursive: true });
const staged = `${output}.building`;
await writeFile(staged, rows.map(row => JSON.stringify(row)).join('\n') + '\n');
await rename(staged, output);
await writeFile(`${output}.manifest.json`, JSON.stringify({
  schema: 'natlang.source_teacher_cases/1', cases: rows.length, counts,
  seeds_sha256: hash(seedText), output_sha256: hash(await readFile(output)),
}, null, 2) + '\n');
console.log(`${rows.length} current-source cases across ${Object.keys(counts).length} codebases -> ${output}`);
