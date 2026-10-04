#!/usr/bin/env node
/** Build selection episodes from frozen crisp-task episodes without moving case data across roles. */
import { createReadStream } from 'node:fs';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { resolve, join } from 'node:path';
import { createHash } from 'node:crypto';
import { validateEpisode, SKILL_EPISODE_SCHEMA, AUTHORING_OPERATIONS } from '../../dist/skills/episode.js';

const sha = value => createHash('sha256').update(typeof value === 'string' ? value : JSON.stringify(value)).digest('hex');
const SEMANTIC = new Set(['active-urgency', 'explicit-consent', 'completed-delivery', 'active-license', 'final-cancellation',
  'overall-recommendation', 'authorized-access', 'resolved-support', 'confirmed-attendance', 'final-renewal']);
const SELECTED_SLATE = new Set(['active-urgency', 'explicit-consent', 'completed-delivery', 'active-license',
  'final-cancellation', 'overall-recommendation', 'authorized-access', 'resolved-support',
  'exact-multi-edit', 'identity-join', 'latest-revision', 'ranked-selection']);

async function* jsonl(path) {
  for await (const line of createInterface({ input: createReadStream(path), crlfDelay: Infinity })) if (line.trim()) yield JSON.parse(line);
}
const frontmatter = (name, description, families, body, extra = '') => `---\nname: ${name}\ndescription: ${description}\nnatlang:\n  provenance:\n    author: project-generated\n    families: [${families.join(', ')}]\n${extra}---\n\n${body}\n`;
function taskSkill(family) {
  if (family === 'optimization-knapsack') return { name: 'capacity-value-check', description: 'Select valuable items while checking a shared capacity.',
    body: 'Track total weight before adding an item. Compare value per unit as a heuristic, then verify the chosen set against the capacity. Keep the empty set available.' };
  if (family === 'optimization-bin-packing') return { name: 'capacity-packing-check', description: 'Place each item once while minimizing the number of capacity-limited bins.',
    body: 'Place large items first as a baseline. Check every item appears once and every bin load fits; improve by trying alternate placements when a bin remains poorly filled.' };
  if (family === 'optimization-weighted-tardiness') return { name: 'weighted-schedule-check', description: 'Order jobs to reduce weighted tardiness.',
    body: 'Compute completion time from the prefix. For each job add weight times positive lateness. Use a simple ratio ordering only as a starting point and compare nearby swaps.' };
  if (SEMANTIC.has(family)) return { name: 'decision-state-review', description: 'Resolve the final state of a record from its complete history.',
    body: 'Read the whole record. Later explicit corrections, withdrawals and superseding statements replace earlier states. Do not treat proposals or predictions as completed facts.' };
  if (family === 'exact-multi-edit') return { name: 'literal-edit-review', description: 'Apply exact requested file edits without changing unrelated text.',
    body: 'Match the complete requested text exactly. Apply only the listed replacements, preserve surrounding whitespace, and check every requested change independently.' };
  if (family === 'identity-join') return { name: 'identity-link-review', description: 'Join records only through stable identifying fields.',
    body: 'Normalize identifiers only when the task permits it. Confirm a unique match and keep display names separate from stable IDs.' };
  if (family === 'latest-revision') return { name: 'revision-order-review', description: 'Choose the latest applicable revision from ordered records.',
    body: 'Compare revision timestamps or sequence numbers, apply explicit supersession, and ignore stale copies after verifying the selected record is complete.' };
  if (family === 'ranked-selection') return { name: 'ranked-evidence-review', description: 'Rank candidates using the complete stated criteria.',
    body: 'Apply required criteria before tie breakers. Keep the full candidate set until every criterion has been checked.' };
  return { name: 'record-evidence-review', description: 'Check the complete record before making a selection.',
    body: 'Read the full available evidence, check constraints, and preserve uncertainty when the record does not decide the requested fact.' };
}
function targetSkill(family) {
  const skill = taskSkill(family);
  return { ...skill, files: { 'SKILL.md': frontmatter(skill.name, skill.description, [family], skill.body) } };
}
function unrelatedSkill(family) {
  const skill = family.startsWith('optimization-') ? taskSkill('exact-multi-edit') : taskSkill('optimization-knapsack');
  const other = family.startsWith('optimization-') ? 'exact-multi-edit' : 'optimization-knapsack';
  return { ...skill, families: [other], files: { 'SKILL.md': frontmatter(skill.name, skill.description, [other], skill.body) } };
}
function redundantSkill(family) {
  const helpful = taskSkill(family);
  const name = `second-${helpful.name}`;
  const description = `A second checklist for ${helpful.description.toLowerCase()}`;
  const body = `Before acting, use this compact checklist: ${helpful.body}`;
  return { name, description, body, families: [family], files: { 'SKILL.md': frontmatter(name, description, [family], body) } };
}
function makeVariant(base, condition) {
  const useful = targetSkill(base.family), distractor = unrelatedSkill(base.family), redundant = redundantSkill(base.family);
  let skills;
  if (condition === 'candidate-selection') skills = { [useful.name]: useful.files, [distractor.name]: distractor.files, [redundant.name]: redundant.files };
  else {
    // The skill names, descriptions and bodies remain identical: only frontmatter applicability edges need tuning.
    const overbroad = [base.family, base.family === 'exact-multi-edit' ? 'identity-join' : 'exact-multi-edit'];
    skills = {
      [useful.name]: { 'SKILL.md': frontmatter(useful.name, useful.description, [], useful.body) },
      [distractor.name]: { 'SKILL.md': frontmatter(distractor.name, distractor.description, overbroad, distractor.body) },
      [redundant.name]: redundant.files,
    };
  }
  const episode = structuredClone(base);
  episode.id = `selection-${sha(`${base.id}:${condition}`).slice(0, 24)}`;
  episode.library = { kind: 'existing', skills };
  episode.operations = [...AUTHORING_OPERATIONS];
  // Keep only non-identifying author-independent metadata; authorView strips it from requests as an extra boundary.
  episode.provenance = { generator: 'natlang.skill-selection-episodes/1', selection_design: condition,
    applicability_metadata: 'explicit-skill-frontmatter' };
  return episode;
}

export function buildSelectionEpisodes(bases) {
  const selected = bases.filter(row => SELECTED_SLATE.has(row.family) || row.family.startsWith('optimization-'));
  const episodes = selected.flatMap(base => [makeVariant(base, 'candidate-selection'), makeVariant(base, 'metadata-tuning')]);
  const seen = new Map(), diagnostics = [];
  for (const episode of episodes) {
    diagnostics.push(...validateEpisode(episode).map(d => ({ episode: episode.id, ...d })));
    for (const [role, rows] of [['support', episode.support.cases], ['query', episode.query.cases], ['transfer', episode.transfer?.cases ?? []]])
      for (const row of rows) {
        const roleKey = `${episode.split}/${role}`;
        if (seen.has(row.group) && seen.get(row.group) !== roleKey) diagnostics.push({ code: 'group-role-split-collision', group: row.group });
        seen.set(row.group, roleKey);
      }
  }
  if (diagnostics.length) throw Error('selection packet rejected: ' + JSON.stringify(diagnostics.slice(0, 30)));
  return { episodes, uniqueGroups: seen.size, diagnostics };
}

const parseArgs = argv => {
  const args = {};
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i]?.replace(/^--/, ''), value = argv[i + 1];
    if (!['slate', 'optimization', 'out'].includes(key) || !value) throw Error('Usage: build-selection-episodes.mjs --slate FILE --optimization FILE --out DIR');
    args[key] = resolve(value);
  }
  if (!args.slate || !args.optimization || !args.out) throw Error('slate, optimization and out are required');
  return args;
};

async function main(argv) {
  const args = parseArgs(argv), bases = [];
  for (const path of [args.slate, args.optimization]) for await (const row of jsonl(path)) bases.push(row);
  const { episodes, uniqueGroups, diagnostics } = buildSelectionEpisodes(bases);
  await mkdir(args.out, { recursive: true });
  const body = episodes.map(row => JSON.stringify(row)).join('\n') + '\n';
  await writeFile(join(args.out, 'selection-episodes.jsonl'), body, { flag: 'wx' });
  const manifest = { schema: 'natlang.skill-selection-episodes/1', episodes: episodes.length,
    by_split: Object.fromEntries(['train', 'validation', 'test'].map(split => [split, episodes.filter(e => e.split === split).length])),
    variants: { candidate_selection: episodes.filter(e => e.provenance.selection_design === 'candidate-selection').length,
      metadata_tuning: episodes.filter(e => e.provenance.selection_design === 'metadata-tuning').length },
    families: [...new Set(episodes.map(e => e.family))].sort(), library_profiles: ['helpful', 'irrelevant', 'redundant', 'incomplete-or-overbroad-applicability-metadata'],
    case_groups: uniqueGroups, group_role_collisions: 0, validation_errors: diagnostics.length, model_calls: 0,
    source_sha256: { slate: sha(await readFile(args.slate)), optimization: sha(await readFile(args.optimization)) },
    sha256: sha(body), note: 'Candidate data only; no provider outcomes or admission labels.' };
  await writeFile(join(args.out, 'selection-episodes.manifest.json'), JSON.stringify(manifest, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify(manifest, null, 2));
}
if (import.meta.url === `file://${process.argv[1]}`) await main(process.argv.slice(2));
