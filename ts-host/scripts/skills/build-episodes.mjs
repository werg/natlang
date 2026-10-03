#!/usr/bin/env node
/**
 * Build `natlang.skill-episode/1` records (plans/neuralese/S2_SKILL_AUTHORING.md §4–5.1) from existing material:
 *
 * - the optimizer-curriculum slate (`natlang.improvement-case/1`, 16 families × variants): one program per family,
 *   support and query drawn from disjoint variants, transfer from a related family;
 * - `natlang.program/2` records (S1 Natlang task adapters and frozen source cases): records that share a program and
 *   family, split into support and query by source group.
 *
 * Episodes start from an empty library and allow every authoring operation. Each is validated (shape and leakage) before
 * it is written; rejects are counted with their reasons. No model is called.
 *
 *   node scripts/skills/build-episodes.mjs --slate ../data/teacher/self-improvement/task-slate-v1/cases.jsonl \
 *     --program ../data/teacher/source-cases-s909.ir.jsonl --out <dir> [--limit N]
 */
import { createReadStream } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { AUTHORING_OPERATIONS, SKILL_EPISODE_SCHEMA, validateEpisode } from '../../dist/skills/episode.js';
import { hexDigest } from '../../dist/native/hash.js';

const SEMANTIC = ['active-urgency', 'explicit-consent', 'completed-delivery', 'active-license', 'final-cancellation',
  'overall-recommendation', 'authorized-access', 'resolved-support', 'confirmed-attendance', 'final-renewal'];
const STRUCTURAL = ['identity-join', 'latest-revision', 'ranked-selection', 'exact-multi-edit', 'corrected-extraction', 'semantic-routing'];
/** A related family for transfer: the next family of the same kind. */
export function relatedFamily(family) {
  for (const group of [SEMANTIC, STRUCTURAL]) {
    const index = group.indexOf(family);
    if (index >= 0) return group[(index + 1) % group.length];
  }
  return undefined;
}

async function* jsonl(path) {
  const lines = createInterface({ input: createReadStream(path), crlfDelay: Infinity });
  for await (const line of lines) if (line.trim()) yield JSON.parse(line);
}

const pick = ({ id, group, args, folder, expected, expectedFiles }) =>
  ({ id, group, ...(args ? { args } : {}), ...(folder ? { folder } : {}), ...(expected !== undefined ? { expected } : {}),
    ...(expectedFiles ? { expectedFiles } : {}) });

function slateTarget(task) {
  return { kind: 'improvement-case', files: task.files, entry: task.contract.entry, exportName: task.contract.exportName ?? 'default',
    source: { schema: task.version, id: task.id } };
}

/**
 * Slate episodes: variants are partitioned into train (all but the last third) and validation; within a partition,
 * consecutive pairs of variants alternate between support and query. Transfer cases come from the related family's
 * variants in the query role, so they share the split.
 */
export function slateEpisodes(tasks, { pair = 2 } = {}) {
  const byFamily = new Map();
  for (const task of tasks) byFamily.set(task.family, [...(byFamily.get(task.family) ?? []), task]);
  for (const list of byFamily.values()) list.sort((a, b) => a.provenance.variant - b.provenance.variant);
  const episodes = [];
  for (const [family, variants] of [...byFamily].sort(([a], [b]) => a.localeCompare(b))) {
    const cut = variants.length - Math.floor(variants.length / 3);
    for (const [split, list] of [['train', variants.slice(0, cut)], ['validation', variants.slice(cut)]]) {
      for (let start = 0; start + 2 * pair <= list.length; start += 2 * pair) {
        const support = list.slice(start, start + pair), query = list.slice(start + pair, start + 2 * pair);
        const related = relatedFamily(family);
        const relatedList = related ? (byFamily.get(related) ?? []) : [];
        const relatedSplit = split === 'train' ? relatedList.slice(0, relatedList.length - Math.floor(relatedList.length / 3))
          : relatedList.slice(relatedList.length - Math.floor(relatedList.length / 3));
        const transferTasks = relatedSplit.slice(start + pair, start + 2 * pair);
        const id = `skill-episode-slate-${family}-${split}-${start / (2 * pair)}`;
        const episode = {
          version: SKILL_EPISODE_SCHEMA, id, family, split,
          source_groups: [...new Set([...support, ...query, ...transferTasks].flatMap(task => task.sourceGroups))].sort(),
          license: 'project-generated',
          target: slateTarget(support[0]),
          library: { kind: 'empty', skills: {} },
          support: { cases: support.flatMap(task => task.cases.map(pick)) },
          query: { cases: query.flatMap(task => task.cases.map(pick)) },
          ...(transferTasks.length ? { transfer: { family: related, target: slateTarget(transferTasks[0]),
            cases: transferTasks.flatMap(task => task.cases.map(pick)) } } : {}),
          operations: [...AUTHORING_OPERATIONS],
          limits: { maxSteps: 8 },
          provenance: { generator: 'natlang.skill-episodes/slate-1', source: 'natlang.optimizer-training-slate/1',
            support_tasks: support.map(task => task.id), query_tasks: query.map(task => task.id),
            transfer_tasks: transferTasks.map(task => task.id), reference_hidden: true },
        };
        episodes.push(episode);
      }
    }
  }
  return episodes;
}

/** Program-record episodes: records sharing family and program, at least two source groups, split by group. */
export function programEpisodes(records) {
  const groups = new Map();
  const skipped = {};
  const skip = reason => { skipped[reason] = (skipped[reason] ?? 0) + 1; };
  for (const record of records) {
    const semantics = record.semantics;
    if (record.version !== 'natlang.program/2' || !semantics?.files || !semantics.root) { skip('not-a-program-record'); continue; }
    if (semantics.expected === undefined && !semantics.expected_files) { skip('no-checked-result'); continue; }
    const key = `${record.family}\u0000${hexDigest(JSON.stringify([semantics.root, semantics.files]))}`;
    groups.set(key, [...(groups.get(key) ?? []), record]);
  }
  const episodes = [];
  for (const list of groups.values()) {
    const bySource = new Map();
    for (const record of list) {
      const group = (record.source_groups ?? [record.id])[0];
      bySource.set(group, [...(bySource.get(group) ?? []), record]);
    }
    if (bySource.size < 2) { skip('single-source-group'); continue; }
    const ordered = [...bySource.keys()].sort();
    const half = Math.ceil(ordered.length / 2);
    const asCase = record => ({ id: record.id, group: (record.source_groups ?? [record.id])[0],
      args: Object.values(record.semantics.inputs ?? {}),
      ...(record.semantics.folder_files ? { folder: record.semantics.folder_files } : {}),
      ...(record.semantics.expected !== undefined ? { expected: record.semantics.expected } : {}),
      ...(record.semantics.expected_files ? { expectedFiles: record.semantics.expected_files } : {}) });
    const first = list[0];
    const splits = new Set(list.map(record => record.split ?? 'train'));
    episodes.push({
      version: SKILL_EPISODE_SCHEMA, id: `skill-episode-program-${hexDigest(first.family + ordered.join()).slice(0, 12)}`,
      family: first.family, split: splits.size === 1 ? [...splits][0] : 'train',
      source_groups: [...new Set(list.flatMap(record => record.source_groups ?? [record.id]))].sort(),
      license: [...new Set(list.map(record => record.license ?? 'unknown'))].join(' AND '),
      target: { kind: 'program', files: first.semantics.files, entry: first.semantics.root, source: { schema: first.version, id: first.id } },
      library: { kind: 'empty', skills: {} },
      support: { cases: ordered.slice(0, half).flatMap(group => bySource.get(group).map(asCase)) },
      query: { cases: ordered.slice(half).flatMap(group => bySource.get(group).map(asCase)) },
      operations: [...AUTHORING_OPERATIONS], limits: { maxSteps: 8 },
      provenance: { generator: 'natlang.skill-episodes/program-1', records: list.map(record => record.id) },
    });
    if (splits.size > 1) skip('mixed-split-merged-as-train');
  }
  return { episodes, skipped };
}

async function main(argv) {
  const args = { slate: [], program: [], out: undefined, limit: Infinity };
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i], value = argv[++i];
    if (flag === '--slate') args.slate.push(resolve(value));
    else if (flag === '--program') args.program.push(resolve(value));
    else if (flag === '--out') args.out = resolve(value);
    else if (flag === '--limit') args.limit = Number(value);
    else throw Error(`unknown flag ${flag}`);
  }
  if (!args.out) throw Error('--out is required');
  const tasks = [];
  for (const path of args.slate) for await (const row of jsonl(path)) tasks.push(row);
  const records = [];
  for (const path of args.program) for await (const row of jsonl(path)) records.push(row);
  const fromSlate = slateEpisodes(tasks);
  const fromPrograms = programEpisodes(records);
  const accepted = [], rejected = {};
  for (const episode of [...fromSlate, ...fromPrograms.episodes]) {
    if (accepted.length >= args.limit) break;
    const problems = validateEpisode(episode);
    if (problems.length) { for (const problem of problems) rejected[problem.code] = (rejected[problem.code] ?? 0) + 1; continue; }
    accepted.push(episode);
  }
  await mkdir(args.out, { recursive: true });
  const body = accepted.map(episode => JSON.stringify(episode)).join('\n') + (accepted.length ? '\n' : '');
  await writeFile(join(args.out, 'episodes.jsonl'), body);
  const manifest = { schema: SKILL_EPISODE_SCHEMA, generator: 'ts-host/scripts/skills/build-episodes.mjs', sources: [...args.slate, ...args.program],
    episodes: accepted.length, by_split: Object.fromEntries(['train', 'validation', 'test'].map(split => [split, accepted.filter(e => e.split === split).length])),
    by_source: { slate: accepted.filter(e => e.target.kind === 'improvement-case').length, program: accepted.filter(e => e.target.kind === 'program').length },
    families: [...new Set(accepted.map(e => e.family))].sort(), rejected, skipped: fromPrograms.skipped, sha256: hexDigest(body) };
  await writeFile(join(args.out, 'episodes.manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  console.log(JSON.stringify(manifest, null, 2));
}

if (import.meta.url === `file://${process.argv[1]}`) await main(process.argv.slice(2));
