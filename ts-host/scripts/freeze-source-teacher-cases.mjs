#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, rename, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PROGRAM_VERSION, programDefinition } from '../dist/teacher/program.js';
import { quarantineReason } from '../dist/teacher/curriculum-policy.js';

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
/** A stream family runs its step function over the events, as ordinary code would. */
const streamRoot = (stepName, state, event) => `---
description: Apply ${stepName} to every event in order and return the final state.
args:
  initial: ${JSON.stringify(state)}
  events: ${JSON.stringify(`${event}[]`)}
returns: ${JSON.stringify(state)}
---
Start from initial. For each event in events, in order, the state becomes
${stepName}(state, event). Return the final state.
`;
const rows = [], held = [];
for (const seed of cases) {
  const source = sources[seed.family];
  if (!source) throw new Error(`no current source registered for ${seed.family}`);
  const [relative, shape] = source, path = resolve(repo, relative);
  const folder = dirname(path), paths = (await tree(folder)).filter(file => /\.(nl|ts)$/.test(file));
  const revision = hash(Buffer.concat(await Promise.all(paths.map(async file =>
    Buffer.concat([Buffer.from(file.slice(folder.length + 1)), Buffer.from([0]), await readFile(file), Buffer.from([0])])))));
  const sourceFiles = Object.fromEntries(await Promise.all(paths.map(async file => [file.slice(folder.length + 1), await readFile(file, 'utf8')])));
  const rootName = relative.slice(relative.lastIndexOf('/') + 1);
  const { expected, effects } = seed.semantics;
  let semantics;
  if (shape === 'fold') {
    const fold = seed.semantics.root_template?.$fold;
    if (!fold || !Array.isArray(seed.semantics.events)) throw new Error(`${seed.id} needs a fold root_template and events`);
    // The codebase's types stay at the top so the stream root and the nested step both see them.
    const step = programDefinition({ semantics: { root: rootName, files: sourceFiles } });
    const [state, event] = step.params.map(param => param.type);
    const files = { 'run.nl': streamRoot(step.name, state, event) };
    for (const [file, text] of Object.entries(sourceFiles)) files[file === 'types.ts' ? file : `run/${file}`] = text;
    semantics = { root: 'run.nl', files, inputs: { initial: fold.init, events: seed.semantics.events }, expected, ...(effects ? { effects } : {}) };
  } else {
    semantics = { root: rootName, files: sourceFiles, inputs: seed.semantics.inputs, expected, ...(effects ? { effects } : {}) };
  }
  const row = { version: PROGRAM_VERSION, id: seed.id, kind: 'lambda_graph',
    family: seed.family, source: 'natlang-current-source', split: seed.split,
    source_ids: [relative], source_groups: [seed.family], source_revisions: [revision],
    license: 'project-generated', gold_sources: ['frozen-reference-case'],
    generation: { generator: 'natlang.source_case_freezer/3' }, semantics };
  programDefinition(row);
  const reason = quarantineReason(row);
  if (reason) { held.push({ reason, record: row }); continue; }
  rows.push(row);
}
const counts = Object.fromEntries(Object.keys(sources).map(family =>
  [family, rows.filter(row => row.family === family).length]));
if (Object.entries(counts).some(([family, count]) => count < 4 &&
  !(count === 0 && held.some(item => item.record.family === family))))
  throw new Error(`source families need four cases: ${JSON.stringify(counts)}`);
await mkdir(dirname(output), { recursive: true });
const staged = `${output}.building`;
await writeFile(staged, rows.map(row => JSON.stringify(row)).join('\n') + '\n');
await rename(staged, output);
await writeFile(`${output}.held.jsonl`, held.map(item => JSON.stringify(item)).join('\n') + (held.length ? '\n' : ''));
await writeFile(`${output}.manifest.json`, JSON.stringify({
  schema: 'natlang.source_teacher_cases/1', cases: rows.length, counts,
  held: held.length, held_reasons: Object.fromEntries([...new Set(held.map(item => item.reason))]
    .map(reason => [reason, held.filter(item => item.reason === reason).length])),
  seeds_sha256: hash(seedText), output_sha256: hash(await readFile(output)),
}, null, 2) + '\n');
console.log(`${rows.length} current-source cases across ${Object.values(counts).filter(count => count > 0).length} eligible codebases (${held.length} held) -> ${output}`);
