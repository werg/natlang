#!/usr/bin/env node
// Broad Neuralese producer/consumer programs from public sources (plans/neuralese/NEURALESE_BROAD_GRAPHS_20261031.md).
// Input: source-agnostic cases (natlang.broad-graph-case/1, scripts/neuralese_data/broad_graph_cases.py). Output:
// curriculum IR shards replayed by replay-port-record-calls.mjs (the generic reference replay).
//
// DECOMPOSITION (natlang-native first):
//   - The root (directory-reducer, one argument: the request) never reads a document. Its documents are folder files
//     (docs/NN.txt); its code reads each file and passes the text to a named view function, without printing it.
//   - The view function (named nl child, one per document) returns a view of its own document: the gold-derived
//     selection the case declares (supporting sentences, referenced schema, tool signature, event templates, cells),
//     or the case's exact "none" line. -> its own nl function; the producer of a Neuralese edge.
//   - The consumer is either the root's next turn (pattern fan-in/root: the root prints every view, then returns the
//     answer from them) or a named answering child that receives the views as an argument (pattern fan-in/child; the
//     root returns its result). -> instruction or own nl function, chosen per case by a seeded draw.
//   - Program text (root/child instructions) comes from the case: the source adapter owns the wording of its task
//     kind and its paraphrase variants. This builder is crisp plumbing: exact field mapping, no judgement.
//
//   node build-broad-graph-calls.mjs --input CASES.jsonl --out DIR [--shard-cases 250]
// Writes DIR/<source>/NNNNN.ir.jsonl, DIR/<source>/omissions.jsonl and DIR/<source>/build.json.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { jsonlLines } from '../jsonl-stream.mjs';
import { Random, curriculumCase, evalCall, nlFile, returnCall } from './lib.mjs';

export const CASE_SCHEMA = 'natlang.broad-graph-case/1';
export const BUILDER = 'natlang.broad_graph_calls/1';
const sha = value => createHash('sha256').update(value).digest('hex');
const IDENT = /^[a-z][a-z0-9_]*$/;

/** Why a case cannot be converted, or null. */
export function caseProblem(item) {
  if (item?.schema !== CASE_SCHEMA) return 'schema';
  if (!item.id || !item.source || !['train', 'test'].includes(item.split)) return 'identity-or-split';
  if (!Array.isArray(item.groups) || !item.groups.length) return 'no-source-groups';
  if (typeof item.request?.text !== 'string' || !item.request.text.trim()) return 'no-request';
  if (typeof item.answer !== 'string' || !item.answer.trim()) return 'empty-answer';
  const { view, consumer, root } = item;
  if (!IDENT.test(view?.name ?? '') || !IDENT.test(consumer?.name ?? '') || !IDENT.test(root?.name ?? '')) return 'function-names';
  if (new Set([view.name, consumer.name, root.name, item.request.name]).size !== 4) return 'function-name-collision';
  if (!IDENT.test(item.request.name ?? '')) return 'request-name';
  const docs = item.documents ?? [];
  if (!docs.length || docs.length > 99) return 'document-count';
  if (!docs.some(doc => doc.view !== view.none)) return 'no-informative-view';
  for (const doc of docs) {
    if (typeof doc.text !== 'string' || !doc.text.trim() || typeof doc.view !== 'string' || !doc.view.trim()) return 'empty-document-or-view';
    if (doc.view.length < 16) return 'view-under-edge-minimum';
  }
  if (new Set(docs.map(doc => doc.text)).size !== docs.length) return 'duplicate-document';
  return null;
}

/** A unique visible marker of each document for the reference driver (its JSON-escaped opening). */
function markers(docs) {
  const escaped = docs.map(doc => JSON.stringify(doc.text).slice(1, -1));
  for (let length = 48; length <= 400; length += 48) {
    const out = escaped.map(text => text.slice(0, length));
    if (new Set(out).size === out.length) return out;
  }
  return null;
}

export function broadGraphCase(item) {
  const rng = new Random(0, `broad-graph:${item.id}`);
  const pattern = item.pattern ?? (rng.next() < 0.5 ? 'fan-in/root' : 'fan-in/child');
  const order = rng.shuffle(item.documents.map((_, i) => i));
  const docs = order.map(i => item.documents[i]);
  const width = String(docs.length).length < 2 ? 2 : String(docs.length).length;
  const names = docs.map((_, i) => `docs/${String(i + 1).padStart(width, '0')}.txt`);
  const visible = markers(docs);
  if (!visible) return { problem: 'documents-not-distinguishable' };
  const { view, consumer, root, request } = item;
  const files = Object.fromEntries(docs.map((doc, i) => [names[i], doc.text.endsWith('\n') ? doc.text : doc.text + '\n']));
  const viewCall = `await ${view.name}(await file.readText(), ${request.name})`;
  let code, rootCalls, rootInstructions, extraFiles = {}, children;
  children = docs.map((doc, i) => ({ match: [`You are inside this call: ${view.name}`, visible[i]],
    source_binding: { path: names[i], sha256: sha(files[names[i]]) }, value: doc.view }));
  if (pattern === 'fan-in/root') {
    code = `for (const file of await folder.files('docs/*.txt')) {\n  console.log(${viewCall});\n}`;
    rootCalls = [evalCall(code), returnCall(item.answer)];
    rootInstructions = root.instructions_root;
  } else if (pattern === 'fan-in/child') {
    code = `const notes: string[] = [];\nfor (const file of await folder.files('docs/*.txt')) {\n  notes.push(${viewCall});\n}\n` +
      `return await ${consumer.name}(${request.name}, notes);`;
    rootCalls = [evalCall(code), returnCall(item.answer)];
    rootInstructions = root.instructions_child;
    extraFiles[`${root.name}/${consumer.name}.nl`] = nlFile({ args: { [request.name]: 'string', notes: 'string[]' },
      returns: 'string', description: consumer.description, instructions: consumer.instructions });
    children.push({ match: [`You are inside this call: ${consumer.name}`], value: item.answer });
  } else return { problem: `pattern-${pattern}` };
  const record = curriculumCase({ family: `broad_graph_${item.source}`, shape: `${item.source}-${sha(item.id).slice(0, 20)}`,
    variant: `${pattern}/${item.variant ?? 'v0'}`, splitGroup: item.groups[0], split: item.split, slice: 'nested_scoped',
    domain: item.domain ?? item.source, mode: 'single_call', inline: 'avoid', named: 'required',
    evidence: { world: [request.text], retrieved: docs.map(doc => doc.title ?? '').filter(Boolean), background: [] },
    minimumSequence: [`call ${view.name} on each document file with the ${request.name}`,
      pattern === 'fan-in/root' ? 'print every view and answer from the views alone' : `pass the views to ${consumer.name} and return its answer`],
    reference: { root: rootCalls, children },
    root: { name: root.name, args: { [request.name]: 'string' }, returns: 'string', kind: 'directory-reducer', instructions: rootInstructions },
    files: { [`${root.name}/${view.name}.nl`]: nlFile({ args: { document: 'string', [request.name]: 'string' }, returns: 'string',
      description: view.description, instructions: view.instructions }), ...extraFiles },
    inputs: { [request.name]: request.text }, folderFiles: files, expectedFiles: files, expected: item.answer });
  record.semantics.oracle = item.oracle ?? 'exact';
  record.task_modality = 'directory-reducer';
  record.source = `broad-graph:${item.source}`;
  record.source_ids = [item.id];
  record.source_groups = [...new Set(item.groups)];
  record.source_revisions = [item.revision ?? item.source];
  record.license = item.license?.spdx ?? 'NOASSERTION';
  record.gold_sources = [...new Set([item.answer_origin ?? 'dataset-answer', ...docs.map(doc => doc.view_origin ?? 'none-line')])];
  record.dataset = item.dataset ?? item.source;
  record.dataset_records = [item.upstream_id ?? item.id];
  record.external_source = { source: item.source, source_id: item.upstream_id ?? item.id, original_split: item.upstream_split,
    revision: item.revision, files: item.files ?? [], license: item.license ?? null, adaptation: item.adaptation ?? null,
    case_sha256: sha(JSON.stringify(item)) };
  record.broad_graph = { schema: BUILDER, pattern, producers: docs.length, informative_producers: docs.filter(doc => doc.view !== view.none).length,
    view_function: view.name, consumer: pattern === 'fan-in/root' ? `${root.name}:answer-turn` : consumer.name,
    task_kind: item.task_kind ?? null, answer_kind: item.answer_kind ?? 'text',
    view_origins: [...new Set(docs.map(doc => doc.view_origin ?? 'none-line'))] };
  record.generation = { generator: BUILDER, task_variant: false, independent_world_credit: 1, output_kind: item.answer_kind ?? 'text',
    target_values_visible_to_model: false, provider_calls: 0 };
  return { record };
}

async function main() {
  const { values } = parseArgs({ options: { input: { type: 'string' }, out: { type: 'string' }, 'shard-cases': { type: 'string', default: '250' } } });
  if (!values.input || !values.out) throw new Error('usage: build-broad-graph-calls.mjs --input CASES.jsonl --out DIR');
  const SHARD = Number(values['shard-cases']);
  const bySource = new Map();
  const state = source => {
    if (!bySource.has(source)) bySource.set(source, { shard: [], shards: 0, cases: 0, omissions: [], omitted: {}, splits: {}, patterns: {} });
    return bySource.get(source);
  };
  const flush = async (source, force) => {
    const s = state(source);
    if (!s.shard.length || (!force && s.shard.length < SHARD)) return;
    const dir = resolve(values.out, source);
    await mkdir(dir, { recursive: true });
    await writeFile(resolve(dir, `${String(s.shards).padStart(5, '0')}.ir.jsonl`), s.shard.map(r => JSON.stringify(r)).join('\n') + '\n');
    s.shards++; s.shard = [];
  };
  let lines = 0;
  for await (const line of jsonlLines(resolve(values.input))) {
    if (!line) continue;
    lines++;
    const item = JSON.parse(line);
    const s = state(item.source ?? 'unknown');
    const omit = reason => { s.omissions.push({ id: item.id ?? null, reason }); s.omitted[reason] = (s.omitted[reason] ?? 0) + 1; };
    const problem = caseProblem(item);
    if (problem) { omit(problem); continue; }
    const built = broadGraphCase(item);
    if (built.problem) { omit(built.problem); continue; }
    s.shard.push(built.record); s.cases++;
    s.splits[item.split] = (s.splits[item.split] ?? 0) + 1;
    const pattern = built.record.broad_graph.pattern;
    s.patterns[pattern] = (s.patterns[pattern] ?? 0) + 1;
    await flush(item.source, false);
  }
  const builder = sha(readFileSync(new URL(import.meta.url)));
  const summary = {};
  for (const [source, s] of bySource) {
    await flush(source, true);
    const dir = resolve(values.out, source);
    await mkdir(dir, { recursive: true });
    await writeFile(resolve(dir, 'omissions.jsonl'), s.omissions.map(row => JSON.stringify(row)).join('\n') + (s.omissions.length ? '\n' : ''));
    const { shard, omissions, ...counts } = s;
    summary[source] = counts;
    await writeFile(resolve(dir, 'build.json'), JSON.stringify({ schema: 'natlang.broad-graph-calls-build/1', input: resolve(values.input),
      builder_sha256: builder, shard_cases: SHARD, ...counts }, null, 2) + '\n');
  }
  console.log(JSON.stringify({ lines, sources: summary }));
}

if (process.argv[1] && resolve(process.argv[1]) === new URL(import.meta.url).pathname) await main();
