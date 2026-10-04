#!/usr/bin/env node
/**
 * Constructed repair episodes (plans/neuralese/S2_SKILL_AUTHORING.md §5.3): starting libraries that are wrong in known
 * ways, built from seed skills and base episodes.
 *
 * - missing: the skill the family needs is removed from the library;
 * - irrelevant: a plausible skill from another family is bound next to the needed one;
 * - incorrect: the needed skill is corrupted (a knowledge value swapped, two procedure steps swapped, or a helper
 *   comparison changed).
 *
 * A seed skill says which families it serves in `natlang.provenance.families`. The defect is recorded as the episode's
 * label with `verified: false`: query evaluation must still confirm that the defect lowers performance before the
 * episode is admitted for training (§5.3). No model is called.
 *
 *   node scripts/skills/build-repair-episodes.mjs --episodes <episodes.jsonl> --library <context dir with skills/> \
 *     --out <dir> [--limit N]
 */
import { createReadStream } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { loadSkills } from '../../dist/skills/registry.js';
import { directorySkillSource } from '../../dist/skills/node.js';
import { AUTHORING_OPERATIONS, validateEpisode } from '../../dist/skills/episode.js';
import { hexDigest } from '../../dist/native/hash.js';

/** Deterministic choice from a seed string. */
const choose = (seed, n) => parseInt(hexDigest(seed).slice(0, 8), 16) % n;

/** Swap two distinct string values of the same key across entries of a JSON array (or values of an object). */
function corruptKnowledge(text, seed) {
  let data;
  try { data = JSON.parse(text); } catch { return undefined; }
  if (!Array.isArray(data) || data.length < 2) return undefined;
  const keys = Object.keys(data[0] ?? {}).filter(key => data.every(row => typeof row?.[key] === 'string'));
  for (const key of keys.slice(choose(seed, Math.max(keys.length, 1))).concat(keys)) {
    const values = data.map(row => row[key]);
    const i = values.findIndex((value, index) => values.some((other, j) => j > index && other !== value));
    if (i < 0) continue;
    const j = values.findIndex((other, index) => index > i && other !== values[i]);
    const swapped = data.map(row => ({ ...row }));
    [swapped[i][key], swapped[j][key]] = [swapped[j][key], swapped[i][key]];
    return { text: JSON.stringify(swapped, null, 2) + '\n', detail: `swapped ${key} of entries ${i} and ${j}` };
  }
  return undefined;
}

/** Swap two numbered procedure steps in SKILL.md, keeping the numbering. */
function corruptSteps(text, seed) {
  const lines = text.split('\n');
  const steps = lines.map((line, index) => /^\d+\.\s/.test(line) ? index : -1).filter(index => index >= 0);
  if (steps.length < 2) return undefined;
  const a = choose(seed, steps.length - 1), i = steps[a], j = steps[a + 1];
  const body = line => line.replace(/^\d+\.\s/, '');
  const number = line => line.match(/^(\d+\.\s)/)[1];
  const out = [...lines];
  out[i] = number(lines[i]) + body(lines[j]);
  out[j] = number(lines[j]) + body(lines[i]);
  return { text: out.join('\n'), detail: `swapped procedure steps ${a + 1} and ${a + 2}` };
}

/** Change one comparison or arithmetic operator in a helper. */
function corruptHelper(text) {
  const rules = [[/ <= /, ' < '], [/ < /, ' <= '], [/ >= /, ' > '], [/ > /, ' >= '], [/ === /, ' !== '], [/\.length/, '.length - 1']];
  for (const [pattern, replacement] of rules) if (pattern.test(text))
    return { text: text.replace(pattern, replacement), detail: `changed ${pattern.source.trim()} to ${replacement.trim()}` };
  return undefined;
}

/** Corrupt one file of a skill folder; returns the corrupted folder and what was changed. */
export function corruptSkill(files, seed) {
  const candidates = [
    ...Object.keys(files).filter(path => path.endsWith('.json')).map(path => [path, corruptKnowledge]),
    ['SKILL.md', corruptSteps],
    ...Object.keys(files).filter(path => path.startsWith('helpers/') && /\.(ts|nl)$/.test(path)).map(path => [path, corruptHelper]),
  ];
  const start = choose(seed + ':strategy', candidates.length);
  for (let k = 0; k < candidates.length; k++) {
    const [path, corrupt] = candidates[(start + k) % candidates.length];
    const result = corrupt(files[path], seed);
    if (result && result.text !== files[path]) return { files: { ...files, [path]: result.text }, detail: `${path}: ${result.detail}` };
  }
  return undefined;
}

/** Build the three repair variants of one base episode from seed skill folders. */
export function repairEpisodes(base, seeds) {
  const serves = (seed, family) => (seed.families ?? []).includes(family);
  const needed = seeds.filter(seed => serves(seed, base.family));
  if (!needed.length) return { episodes: [], skipped: 'no-seed-skill-for-family' };
  const target = needed[choose(base.id, needed.length)];
  const unrelated = seeds.filter(seed => !serves(seed, base.family));
  const others = seeds.filter(seed => seed !== target);
  const make = (kind, skills, defect) => ({
    ...structuredClone(base), id: `repair-${hexDigest(`${base.id}:${kind}`).slice(0, 24)}`,
    library: { kind: 'corrupted', skills, defect: { kind, verified: false, ...defect } },
    operations: [...AUTHORING_OPERATIONS],
    provenance: { generator: 'natlang.skill-episodes/repair-2', reference_hidden: true },
  });
  const folders = list => Object.fromEntries(list.map(seed => [seed.name, seed.files]));
  const episodes = [make('missing', folders(others), { skill: target.name, detail: `removed ${target.name}, which serves ${base.family}` })];
  if (unrelated.length) {
    const distractor = unrelated[choose(base.id + ':irrelevant', unrelated.length)];
    episodes.push(make('irrelevant', folders([target, distractor]), { skill: distractor.name,
      detail: `bound ${distractor.name}, which does not serve ${base.family}` }));
  }
  const corrupted = corruptSkill(target.files, base.id);
  if (corrupted) episodes.push(make('incorrect', { [target.name]: corrupted.files }, { skill: target.name, detail: corrupted.detail }));
  return { episodes };
}

async function* jsonl(path) {
  for await (const line of createInterface({ input: createReadStream(path), crlfDelay: Infinity })) if (line.trim()) yield JSON.parse(line);
}

/** Seed skills with their raw folder contents and the families they serve. */
export async function loadSeeds(contextDir) {
  const source = directorySkillSource(contextDir);
  const { set, diagnostics } = await loadSkills(source);
  const seeds = [];
  for (const skill of set.list()) {
    if (skill.format !== 'markdown') continue;
    const files = {};
    for (const file of ['SKILL.md', ...skill.files]) files[file] = new TextDecoder().decode(await source.read(`${skill.root}/${file}`));
    const families = skill.natlang.provenance?.families;
    seeds.push({ name: skill.name, files, families: Array.isArray(families) ? families : [] });
  }
  return { seeds, diagnostics };
}

async function main(argv) {
  const args = { episodes: undefined, library: undefined, out: undefined, limit: Infinity };
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i], value = argv[++i];
    if (flag === '--episodes') args.episodes = resolve(value);
    else if (flag === '--library') args.library = resolve(value);
    else if (flag === '--out') args.out = resolve(value);
    else if (flag === '--limit') args.limit = Number(value);
    else throw Error(`unknown flag ${flag}`);
  }
  if (!args.episodes || !args.library || !args.out) throw Error('--episodes, --library and --out are required');
  const { seeds, diagnostics } = await loadSeeds(args.library);
  if (diagnostics.some(d => d.severity === 'error')) throw Error('seed library has errors: ' + JSON.stringify(diagnostics));
  const accepted = [], rejected = {}, skipped = {}, byKind = {};
  for await (const base of jsonl(args.episodes)) {
    if (accepted.length >= args.limit) break;
    if (base.library?.kind !== 'empty') continue;
    const { episodes, skipped: reason } = repairEpisodes(base, seeds);
    if (reason) { skipped[reason] = (skipped[reason] ?? 0) + 1; continue; }
    for (const episode of episodes) {
      const problems = validateEpisode(episode);
      if (problems.length) { for (const p of problems) rejected[p.code] = (rejected[p.code] ?? 0) + 1; continue; }
      accepted.push(episode);
      byKind[episode.library.defect.kind] = (byKind[episode.library.defect.kind] ?? 0) + 1;
    }
  }
  await mkdir(args.out, { recursive: true });
  const body = accepted.map(episode => JSON.stringify(episode)).join('\n') + (accepted.length ? '\n' : '');
  await writeFile(join(args.out, 'repair-episodes.jsonl'), body);
  const manifest = { generator: 'ts-host/scripts/skills/build-repair-episodes.mjs', base: args.episodes, library: args.library,
    seeds: seeds.map(seed => ({ name: seed.name, families: seed.families.length })), episodes: accepted.length, by_kind: byKind,
    rejected, skipped, verified: 0, sha256: hexDigest(body),
    note: 'defects are unverified until query evaluation confirms they lower performance' };
  await writeFile(join(args.out, 'repair-episodes.manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  console.log(JSON.stringify(manifest, null, 2));
}

if (import.meta.url === `file://${process.argv[1]}`) await main(process.argv.slice(2));
