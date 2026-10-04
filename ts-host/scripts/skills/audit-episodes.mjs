#!/usr/bin/env node
/** The episode gate: provider-free checks every skill-episode packet passes before it is collected or published.
 *
 * Usage: audit-episodes.mjs PACKET... [--out REPORT.json] [--database-root DIR] [--require-transfer]
 *
 * Errors (any one fails the gate):
 * - schema and leakage rules (`validateEpisode`), duplicate episode IDs across packets;
 * - a source group or a canonical input on two sides of a split/role boundary, across all packets given;
 * - the packet's manifest (`<stem>.manifest.json`, or a directory manifest listing it) disagrees with its bytes or
 *   episode count;
 * - the target (with its starting library) or the transfer target does not load;
 * - a sealed case whose gold output, scored by the episode's own scorer, misses its best score or a gate
 *   (graded metrics; SQL needs --database-root); objective references differ from the independent bound;
 * - with --require-transfer, an episode without transfer cases.
 *
 * Warnings (reported, not failing): a constant answer taken from the support cases that already reaches most of
 * the query quality (the family may not need judgement), and checks skipped for lack of a database root.
 */
import { readFile, writeFile, access } from 'node:fs/promises';
import { basename, dirname, join } from 'node:path';
import { readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { validateEpisode } from '../../dist/skills/episode.js';
import { exactObjectiveBounds } from '../../dist/skills/objective.js';
import { goldOutput, goldQualityBound, scoreGraded } from '../../dist/skills/graded.js';
import { skillEpisodeFiles } from '../../dist/improvement/skill-authoring.js';
import { loadVirtualNatlang } from '../../dist/index.js';

const hash = value => createHash('sha256').update(value).digest('hex');
const canonical = value => Array.isArray(value) ? value.map(canonical) : value && typeof value === 'object'
  ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, item]) => [key, canonical(item)])) : value;
const files = [], flags = { out: null, 'database-root': null, 'require-transfer': false };
for (let i = 2; i < process.argv.length; i++) {
  const arg = process.argv[i];
  if (arg === '--require-transfer') flags['require-transfer'] = true;
  else if (arg === '--out' || arg === '--database-root') flags[arg.slice(2)] = process.argv[++i];
  else files.push(arg);
}
if (!files.length) throw Error('Usage: audit-episodes.mjs PACKET... [--out REPORT.json] [--database-root DIR] [--require-transfer]');

const groups = new Map(), inputs = new Map(), ids = new Set(), errors = [], warnings = [], packets = [];
let episodes = 0, cases = 0, references = 0, goldChecked = 0, targetsLoaded = 0;

async function manifestFor(file) {
  const stem = file.replace(/\.jsonl$/, '');
  try { await access(stem + '.manifest.json'); return { path: stem + '.manifest.json', manifest: JSON.parse(await readFile(stem + '.manifest.json', 'utf8')) }; }
  catch { /* fall back to a directory manifest that lists this file */ }
  for (const name of readdirSync(dirname(file)).filter(item => item.endsWith('.manifest.json') || item === 'manifest.json')) {
    const manifest = JSON.parse(await readFile(join(dirname(file), name), 'utf8'));
    if ((manifest.sha256 && typeof manifest.sha256 === 'object' && basename(file) in manifest.sha256) ||
        manifest.outputs?.[basename(file)] || manifest.artifacts?.[basename(file)] ||
        (name === 'manifest.json' && typeof manifest.sha256 === 'string')) return { path: join(dirname(file), name), manifest };
  }
  return null;
}

function loads(where, episode, target, files) {
  if (!target.entry.endsWith('.nl')) return;
  try { loadVirtualNatlang(files, target.entry); targetsLoaded++; }
  catch (error) { errors.push({ code: 'target_load_error', episode: episode.id, where, error: String(error?.message ?? error).slice(0, 300) }); }
}

for (const file of files) {
  const bytes = await readFile(file), rows = bytes.toString().split('\n').filter(x => x.trim()).map(JSON.parse);
  const sha = hash(bytes);
  packets.push({ path: file, sha256: sha, episodes: rows.length });
  const found = await manifestFor(file);
  if (!found) warnings.push({ code: 'manifest_missing', packet: file });
  else {
    const recorded = found.manifest.sha256 && typeof found.manifest.sha256 === 'object' ? found.manifest.sha256[basename(file)] :
      found.manifest.sha256 ?? found.manifest.outputs?.[basename(file)]?.sha256 ?? found.manifest.artifacts?.[basename(file)]?.sha256;
    if (recorded && recorded !== sha) errors.push({ code: 'manifest_sha_mismatch', packet: file, manifest: found.path });
    const recordedCount = found.manifest.episodes ?? found.manifest.audit?.episodes;
    if (typeof recordedCount === 'number' && recordedCount !== rows.length)
      errors.push({ code: 'manifest_count_mismatch', packet: file, manifest: recordedCount, actual: rows.length });
  }
  for (const episode of rows) {
    episodes++;
    const episodeId = episode?.id ?? null;
    if (typeof episodeId === 'string') {
      if (ids.has(episodeId)) errors.push({ code: 'duplicate_episode', episode: episodeId });
      ids.add(episodeId);
    }
    let schemaErrors;
    try { schemaErrors = validateEpisode(episode); }
    catch (error) { schemaErrors = [{ code: 'episode-validator-error', message: String(error) }]; }
    errors.push(...schemaErrors.map(error => ({ episode: episodeId, ...error })));
    if (schemaErrors.length) continue;
    if (flags['require-transfer'] && !episode.transfer?.cases?.length) errors.push({ code: 'transfer_missing', episode: episode.id });
    try { loads('target', episode, episode.target, skillEpisodeFiles(episode)); }
    catch (error) { errors.push({ code: 'target_load_error', episode: episode.id, where: 'target', error: String(error?.message ?? error).slice(0, 300) }); }
    if (episode.transfer) loads('transfer', episode, episode.transfer.target, episode.transfer.target.files);
    const roles = [['support', episode.support.cases, episode.provenance?.metric],
      ['query', episode.query.cases, episode.provenance?.metric],
      ['transfer', episode.transfer?.cases ?? [], episode.provenance?.transfer_metric]];
    for (const [role, list, metric] of roles) {
      const task = metric?.schema === 'natlang.skill-objective/1' ? metric.kind : (role === 'transfer' ? episode.transfer?.target : episode.target)?.files?.[(role === 'transfer' ? episode.transfer?.target : episode.target)?.entry];
      for (const row of list) {
        cases++;
        const boundary = episode.split + '/' + role;
        const membership = { episode: episode.id, role, split: episode.split, case: row.id };
        if (groups.has(row.group) && groups.get(row.group).boundary !== boundary)
          errors.push({ code: 'group_boundary_collision', group: row.group, first: groups.get(row.group).membership, second: membership });
        else groups.set(row.group, { boundary, membership });
        const signature = hash(JSON.stringify(canonical({ task, args: row.args, folder: row.folder })));
        const previous = inputs.get(signature);
        if (previous && previous.boundary !== boundary)
          errors.push({ code: 'input_alias_boundary_collision', input_sha256: signature, first: previous.membership, second: membership });
        else inputs.set(signature, { boundary, membership });
        if (metric?.schema === 'natlang.skill-objective/1') {
          try {
            const actual = exactObjectiveBounds(metric.kind, typeof row.args[0] === 'string' ? JSON.parse(row.args[0]) : row.args[0]);
            if (JSON.stringify(canonical(actual)) !== JSON.stringify(canonical(row.expected))) throw Error('reference bound mismatch');
            references++;
          } catch (error) { errors.push({ code: 'independent_reference_error', episode: episode.id, case: row.id, error: String(error) }); }
        }
        if (metric?.schema === 'natlang.skill-graded/1' && role !== 'support') {
          if (metric.kind === 'sql-result-f1' && !flags['database-root']) { warnings.push({ code: 'gold_unchecked_no_database_root', episode: episode.id }); continue; }
          const gold = goldOutput(metric.kind, row.expected);
          if (gold === undefined) continue;
          const graded = { ...metric, ...(flags['database-root'] ? { database_root: flags['database-root'] } : {}) };
          const score = scoreGraded(graded, gold, row.expected);
          const bound = goldQualityBound(metric.kind, row.expected);
          goldChecked++;
          if (Math.abs(score.quality - bound) > 1e-9 || !Object.values(score.gates).every(Boolean))
            errors.push({ code: 'gold_not_best', episode: episode.id, role, case: row.id, quality: score.quality, bound, gates: score.gates });
        }
      }
    }
    // A constant answer: the best single support gold output, scored on every query case.
    const metric = episode.provenance?.metric;
    if (metric?.schema === 'natlang.skill-graded/1' && !['python-tests', 'sql-result-f1'].includes(metric.kind)) {
      const candidates = [...new Map(episode.support.cases.map(row => goldOutput(metric.kind, row.expected))
        .filter(value => value !== undefined).map(value => [JSON.stringify(value), value])).values()];
      let best = 0;
      for (const candidate of candidates) {
        const mean = episode.query.cases.reduce((sum, row) => sum + scoreGraded(metric, candidate, row.expected).quality, 0) / episode.query.cases.length;
        best = Math.max(best, mean);
      }
      if (candidates.length && best >= 0.9) warnings.push({ code: 'constant_answer_baseline', episode: episode.id, family: episode.family, quality: +best.toFixed(4) });
    }
  }
}
const report = { schema: 'natlang.skill-episode-audit/2', provider_calls: 0, packets, episodes, cases, groups: groups.size,
  canonical_inputs: inputs.size, independent_objective_references_checked: references, gold_outputs_checked: goldChecked,
  targets_loaded: targetsLoaded, errors, warnings, passed: errors.length === 0 };
if (flags.out) await writeFile(flags.out, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify({ ...report, errors: errors.slice(0, 10), warnings: warnings.slice(0, 10),
  error_count: errors.length, warning_count: warnings.length }, null, 2));
if (errors.length) process.exitCode = 1;
