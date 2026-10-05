#!/usr/bin/env node
/** Build sealed, bounded-memory static trajectories through the real collector.
 * No model calls, no fabricated tool results, no implicit direct-answer admission.
 */
import { mkdir, open, rename, writeFile, readFile, readdir } from 'node:fs/promises';
import { resolve, join, basename } from 'node:path';
import { parseArgs } from 'node:util';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { FAMILIES, buildRecords } from './families.mjs';
import { referenceRow } from './references.mjs';
import { admitRow, verifyCases } from '../../dist/teacher/curriculum.js';
import { defaultToolSurfaceHash } from '../../dist/teacher/collector.js';
import { materializeNativeRows } from '../../dist/teacher/native-materializer.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';

const { values } = parseArgs({ options: {
  out: { type: 'string' }, families: { type: 'string' }, seed: { type: 'string' },
  shapes: { type: 'string', default: '10' }, start: { type: 'string', default: '0' },
  'authored-plans': { type: 'boolean', default: false }, 'supervise-reference-answers': { type: 'boolean', default: false },
  split: { type: 'string', default: 'train' }, 'chunk-shapes': { type: 'string', default: '4' },
} });
if (!values.out || !values.families || !values.seed) throw new Error('--out DIR --families a,b --seed N required');
const seed = Number(values.seed), shapes = Number(values.shapes), start = Number(values.start), chunk = Number(values['chunk-shapes']);
for (const n of [seed, shapes, start, chunk]) if (!Number.isSafeInteger(n) || n < 0) throw new Error('integer bounds required');
if (!shapes || !chunk) throw new Error('positive shapes/chunk required');
if (!['train', 'test'].includes(values.split)) throw new Error('split must be train or test');
if ([102,900].includes(seed) && values.split !== 'test') throw new Error('evaluation seed cannot train');
const families = values.families.split(',');
for (const name of families) if (!FAMILIES[name] || FAMILIES[name].track === 'authoring') throw new Error(`unsupported family ${name}`);
const sourceHashes = {};
for (const name of (await readdir(new URL('.', import.meta.url))).filter(name => name.endsWith('.mjs')).sort()) {
  sourceHashes[`ts-host/scripts/inline-curriculum/${name}`] = createHash('sha256').update(await readFile(new URL(name, import.meta.url))).digest('hex');
}
sourceHashes['training/task-recipes/semantic-folder-operations.json'] = createHash('sha256').update(await readFile(new URL('../../../training/task-recipes/semantic-folder-operations.json', import.meta.url))).digest('hex');
const output = resolve(values.out), staged = `${output}.building-${process.pid}`;
await mkdir(staged, { recursive: false });
const handles = new Map();
for (const name of ['cases.jsonl', 'reference.results.jsonl', 'native.jsonl', 'verification.jsonl']) handles.set(name, await open(join(staged,name), 'wx'));
const hashes = new Map([...handles.keys()].map(name => [name,createHash('sha256')]));
async function append(name, row) {
  const text = JSON.stringify(row) + '\n';
  hashes.get(name).update(text);
  await handles.get(name).writeFile(text);
}
const options = { modelId: 'static-lambda-reference', rootSeed: seed, systemPrompt: TOOLS_PROMPT,
  contextTokens: 65536, maxTurns: 60, toolSurfaceSha256: await defaultToolSurfaceHash(), collectionRole: 'reference',
  authoredActionPlans: values['authored-plans'], followCutoffPages: true, followEvalCutoffPages: true };
const stats = { cases: 0, native_decisions: 0, approved_decisions: 0, held_decisions: 0, child_decisions: 0,
  approved_inline_evals: 0, approved_iterate_evals: 0, families: {}, held_reasons: {} };
const invocations = new Set(), groups = new Map(), ids = new Set();
try {
  // Family weights apply to the global shape interval, not independently rounded chunks.
  for (const name of families) {
    const weight = FAMILIES[name].weight ?? 1;
    const count = Math.max(1,Math.round(shapes*weight));
    for (let offset = 0; offset < count; offset += chunk) {
      const amount = Math.min(chunk,count-offset);
      const records = buildRecords({ seed, shapes: amount/weight, start: start+offset, families: [name], split: values.split, hints: false });
      const checks = await verifyCases(records, TOOLS_PROMPT);
      const bad = checks.filter(check => !check.ok);
      if (bad.length) throw new Error(`source verification failed: ${JSON.stringify(bad)}`);
      for (const check of checks) await append('verification.jsonl', check);
      for (const record of records) {
        if (ids.has(record.id)) throw new Error(`duplicate id ${record.id}`);
        ids.add(record.id);
        const pair = record.curriculum.pair_group;
        if (pair) {
          const opening = checks.find(check => check.id === record.id).opening_sha256;
          if (groups.has(pair) && groups.get(pair) !== opening) throw new Error(`counterfactual opening differs ${pair}`);
          groups.set(pair,opening);
        }
        const row = await referenceRow(record, stats.cases, options);
        const admission = admitRow(row);
        if (!admission.admitted) throw new Error(`trajectory rejected ${record.id}: ${JSON.stringify(admission.reasons)}`);
        const native = materializeNativeRows([row], { directAnswers: values['supervise-reference-answers'] });
        if (native.acceptedRows !== 1 || native.unlinked.length) throw new Error(`unlinked native export ${record.id}`);
        await append('cases.jsonl', record);
        await append('reference.results.jsonl', row);
        for (const turn of native.turns) {
          await append('native.jsonl', turn);
          stats.native_decisions++;
          const approved = turn.training_admission.approved;
          stats[approved ? 'approved_decisions' : 'held_decisions']++;
          if (!approved) { const why = turn.training_admission.reason; stats.held_reasons[why] = (stats.held_reasons[why] ?? 0)+1; }
          const opening = turn.messages.find(message => message.role === 'user')?.content;
          const caller = typeof opening === 'string' ? /^You are inside this call: ([^\s(]+)\(/.exec(opening)?.[1] : null;
          if (caller && `${caller}.nl` !== basename(record.semantics.root)) {
            stats.child_decisions++;
            const invocation = turn.source_ref.invocation_id;
            if (!invocation) throw new Error(`missing child invocation identity ${turn.id}`);
            invocations.add(`${row.id}:${invocation}`);
          }
          if (approved) for (const call of turn.target.tool_calls ?? []) {
            if (call.function.name !== 'eval') continue;
            const code = JSON.parse(call.function.arguments).code ?? '';
            if (/\bnl(?:\.with\([^)]*\))?(?:<[^`]*?>)?`/.test(code)) stats.approved_inline_evals++;
            if (/\biterateOn\s*\(/.test(code)) stats.approved_iterate_evals++;
          }
        }
        stats.cases++;
        stats.families[record.curriculum.family] = (stats.families[record.curriculum.family] ?? 0)+1;
      }
      console.log(JSON.stringify({ ...stats, progress_family: name, family_shapes_complete: offset+amount }));
    }
  }
  for (const handle of handles.values()) { await handle.sync(); await handle.close(); }
  const report = { schema: 'natlang.static-lambda-corpus/1', seed, split: values.split, shapes, start,
    source_files_sha256: sourceHashes,
    source_commit: execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),
    options, counts: { ...stats, unique_child_invocations: invocations.size },
    sha256: Object.fromEntries([...hashes].map(([name,hash]) => [name,hash.digest('hex')])),
    admission: 'decision-filtered-static-reference; held targets remain held; apply source/split closure when combining corpora',
    synthetic_reasoning: values['authored-plans'] ? 'authored-action-plans/1; intended actions trained, no invented observations' : 'action-notes/1; not trained',
    reference_answers_supervised: values['supervise-reference-answers'], model_requests: 0 };
  await writeFile(join(staged,'summary.json'),JSON.stringify(report,null,2)+'\n');
  await rename(staged,output);
  console.log(JSON.stringify({ sealed: output, ...report.counts }));
} catch (error) {
  for (const handle of handles.values()) await handle.close().catch(() => {});
  await writeFile(join(staged,'failure.json'),JSON.stringify({ error: String(error), counts: stats },null,2)+'\n');
  throw error; // Keep partial evidence; never publish it as a sealed corpus.
}
