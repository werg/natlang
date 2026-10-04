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
  ({ id: `case-${hexDigest(id).slice(0, 20)}`, group, ...(args ? { args } : {}), ...(folder ? { folder } : {}), ...(expected !== undefined ? { expected } : {}),
    ...(expectedFiles ? { expectedFiles } : {}) });

function slateTarget(task) {
  return { kind: 'improvement-case', files: task.files, entry: task.contract.entry, exportName: task.contract.exportName ?? 'default',
    source: { schema: task.version, id: `source-${hexDigest(task.id).slice(0, 20)}` } };
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
    const splitGroups = [['train', variants.filter(task => task.cases.some(row => row.split === 'train'))],
      ['validation', variants.filter(task => task.cases.some(row => row.split === 'validation'))]];
    for (const [split, list] of splitGroups) {
      const usable = split === 'train' ? list.filter(task => task.provenance.variant < 8) : list.filter(task => task.provenance.variant >= 8);
      const rolePair = split === 'train' ? pair : 1;
      if (usable.length < rolePair * 2) continue;
      const support = usable.slice(0, rolePair), query = usable.slice(rolePair, rolePair * 2);
      const related = relatedFamily(family);
      const relatedList = related ? (byFamily.get(related) ?? []) : [];
      const transferTasks = relatedList.filter(task => task.provenance.variant >= (split === 'train' ? 6 : 10))
        .slice(0, split === 'train' ? pair : 2);
      if (query.length < rolePair || transferTasks.length < (split === 'train' ? pair : 2)) continue;
      const caseRows = task => task.cases.filter(row => row.split === split).map(row => ({
        ...pick(row), group: `g-${hexDigest(row.group).slice(0, 20)}` }));
      if ([...support, ...query, ...transferTasks].some(task => task.cases.filter(row => row.split === split).length === 0)) continue;
      const id = `skill-episode-slate-${family}-${split}-v2`;
      const groups = [...support.flatMap(t => caseRows(t)), ...query.flatMap(t => caseRows(t)),
        ...transferTasks.flatMap(t => caseRows(t))].map(row => row.group).sort();
        const episode = {
          version: SKILL_EPISODE_SCHEMA, id, family, split,
          source_groups: [`group-commitment:sha256:${hexDigest(JSON.stringify(groups))}`],
          license: 'project-generated',
          target: slateTarget(support[0]),
          library: { kind: 'empty', skills: {} },
          support: { cases: support.flatMap(task => caseRows(task)) },
          query: { cases: query.flatMap(task => caseRows(task)) },
          ...(transferTasks.length ? { transfer: { family: related, target: slateTarget(transferTasks[0]),
            cases: transferTasks.flatMap(task => caseRows(task)) } } : {}),
          operations: [...AUTHORING_OPERATIONS],
          limits: { maxSteps: 8 },
          provenance: { generator: 'natlang.skill-episodes/slate-2', source: 'natlang.optimizer-training-slate/1', reference_hidden: true },
        };
        episodes.push(episode);
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
  programLoop: for (const list of groups.values()) {
    const bySource = new Map();
    for (const record of list) {
      const group = (record.source_groups ?? [record.id])[0];
      bySource.set(group, [...(bySource.get(group) ?? []), record]);
    }
    if ([...bySource.values()].some(recordsForGroup => new Set(recordsForGroup.map(r => r.split ?? 'train')).size > 1)) {
      skip('source-group-crosses-splits'); continue programLoop;
    }
    if (bySource.size < 2) { skip('single-source-group'); continue; }
    const ordered = [...bySource.keys()].sort();
    const half = Math.ceil(ordered.length / 2);
    const asCase = record => ({ id: `case-${hexDigest(record.id).slice(0, 20)}`, group: `g-${hexDigest((record.source_groups ?? [record.id])[0]).slice(0, 20)}`,
      args: Object.values(record.semantics.inputs ?? {}),
      ...(record.semantics.folder_files ? { folder: record.semantics.folder_files } : {}),
      ...(record.semantics.expected !== undefined ? { expected: record.semantics.expected } : {}),
      ...(record.semantics.expected_files ? { expectedFiles: record.semantics.expected_files } : {}) });
    const first = list[0];
    for (const split of [...new Set(list.map(record => record.split ?? 'train'))].sort()) {
      const splitSource = ordered.filter(group => bySource.get(group).some(r => (r.split ?? 'train') === split));
      if (splitSource.length < 2) { skip('single-source-group-in-split'); continue; }
      const splitHalf = Math.ceil(splitSource.length / 2);
      const casesFor = group => bySource.get(group).filter(r => (r.split ?? 'train') === split).map(asCase);
      const groups = splitSource.flatMap(group => casesFor(group)).map(r => r.group).sort();
      episodes.push({ version: SKILL_EPISODE_SCHEMA, id: `skill-episode-program-${hexDigest(first.family + split + splitSource.join()).slice(0, 12)}`,
        family: first.family, split,
        source_groups: [`group-commitment:sha256:${hexDigest(JSON.stringify(groups))}`],
        license: [...new Set(list.map(record => record.license ?? 'unknown'))].join(' AND '),
        target: { kind: 'program', files: first.semantics.files, entry: first.semantics.root, source: { schema: first.version, id: `source-${hexDigest(first.id).slice(0, 20)}` } },
        library: { kind: 'empty', skills: {} },
        support: { cases: splitSource.slice(0, splitHalf).flatMap(group => casesFor(group, 'support')) },
        query: { cases: splitSource.slice(splitHalf).flatMap(group => casesFor(group, 'query')) },
        operations: [...AUTHORING_OPERATIONS], limits: { maxSteps: 8 },
        provenance: { generator: 'natlang.skill-episodes/program-2', reference_hidden: true } });
    }
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
