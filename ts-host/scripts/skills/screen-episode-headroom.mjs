#!/usr/bin/env node
/** Screen skill episodes for headroom with one executor before spending authoring calls on them.
 *
 * A self-improvement episode only teaches something when the executor neither always nor never succeeds with the
 * starting library. For each episode this runs the starting context once on its SUPPORT cases, which the author
 * sees anyway, and records the mean quality and pass share. Query and transfer cases are never executed here:
 * choosing episodes by their sealed baseline would select for low query scores and inflate measured gains through
 * regression to the mean.
 *
 * Usage: screen-episode-headroom.mjs --episodes FILE --out SCREEN.jsonl --executor-endpoint URL --executor-model ID
 *          [--database-root DIR] [--arena-root DIR] [--band LOW,HIGH] [--keep KEPT.jsonl] [--limit N] [--split train]
 *          [--concurrency 4] [--family-probe 6]
 * Episodes are screened `--concurrency` at a time. Once a family's first `--family-probe` episodes all lie on the
 * same side outside the band (all saturated or all at the floor), its remaining episodes are recorded as
 * `skipped: family-saturated` / `family-floor` without executor calls (0 screens every episode).
 * Screens are per executor: the same family can be saturated for the teacher and open for the student. The output
 * is resumable (episodes already screened for this executor are skipped). With --keep, episodes whose support
 * quality lies inside the band are written with `provenance.headroom` (host-only; authors never see provenance).
 */
import { createReadStream } from 'node:fs';
import { headroomIdentity, resumedHeadroomRows } from './headroom-identity.mjs';
import { appendFile, readFile, writeFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { createHash } from 'node:crypto';
import { Folder } from '../../dist/native/scoped-fs.js';
import { UsageGateway } from '../../dist/evaluation/usage.js';
import { SourceEvaluator, hasUnscoredEvaluationFailure } from '../../dist/improvement/host.js';
import { skillEpisodeFiles, supportSearchCases } from '../../dist/improvement/skill-authoring.js';
import { episodeScorings } from '../../dist/skills/scoring.js';
import { ARENA_CODE_FILES, arenaEpisodeExecutions } from '../../dist/self-play/evaluation.js';
import { openAICompatibleModelTurn } from '../../dist/model/openai-compatible.js';

const options = { band: '0.15,0.85', split: 'train', limit: '0', concurrency: '4', 'family-probe': '6' };
const KEYS = ['episodes', 'out', 'executor-endpoint', 'executor-model', 'database-root', 'arena-root', 'band', 'keep', 'limit',
  'split', 'concurrency', 'family-probe'];
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i].replace(/^--/, '');
  if (!KEYS.includes(key) || process.argv[i + 1] === undefined) throw Error('Usage: see the header of screen-episode-headroom.mjs');
  options[key] = process.argv[i + 1];
}
for (const key of ['episodes', 'out', 'executor-endpoint', 'executor-model'])
  if (!options[key]) throw Error(`--${key} is required`);
const [low, high] = options.band.split(',').map(Number);
if (!(low >= 0 && high <= 1 && low < high)) throw Error('--band must be LOW,HIGH within [0,1]');
const sha = value => createHash('sha256').update(value).digest('hex');
const pins = {};
for (const file of ['skills/objective.js', 'skills/extended-objective.js', 'skills/efficiency-objective.js',
  'skills/code-objective.js','skills/visual-objective.js', 'skills/graded.js','skills/crossword-objective.js','skills/csp-objective.js','skills/translation-objective.js','skills/research-objective.js','skills/scifact-objective.js','skills/contractnli-objective.js', 'skills/scoring.js','skills/registry.js','skills/disclosure.js','skills/skill.js', 'improvement/host.js', ...ARENA_CODE_FILES])
  pins[file] = sha(await readFile(new URL('../../dist/' + file, import.meta.url)));
for (const file of ['screen-episode-headroom.mjs', 'headroom-identity.mjs'])
  pins[`scripts/skills/${file}`] = sha(await readFile(new URL(file, import.meta.url)));
const executorId = `${options['executor-endpoint']}:${options['executor-model']}`;
const executor = openAICompatibleModelTurn({ endpoint: options['executor-endpoint'], model: options['executor-model'],
  apiKey: process.env.NATLANG_IMPROVEMENT_API_KEY, request: { temperature: 0.2 } });
const inputSha256=sha(await readFile(options.episodes));
const screenIdentity = headroomIdentity({executorId, pins, inputSha256, options});

let done = new Map();
try {
  done = resumedHeadroomRows(await readFile(options.out, 'utf8'), screenIdentity);
} catch (error) { if (error.code !== 'ENOENT') throw error; }

const controller = new AbortController();
for (const signal of ['SIGINT', 'SIGTERM']) process.once(signal, () => controller.abort(new Error(signal)));
const kept = [], families = {};
let seen = 0;
const selected = [];
for await (const line of createInterface({ input: createReadStream(options.episodes) })) {
  if (!line.trim()) continue;
  const episode = JSON.parse(line);
  if (options.split !== 'all' && episode.split !== options.split) continue;
  if (Number(options.limit) && selected.length >= Number(options.limit)) break;
  selected.push(episode);
}
const probe = Number(options['family-probe']);
const outcomes = new Map();  // family -> support qualities of screened episodes, in screening order
const familyVerdict = family => {
  const qualities = outcomes.get(family) ?? [];
  if (!probe || qualities.length < probe) return null;
  const first = qualities.slice(0, probe);
  if (first.every(q => q > high)) return 'family-saturated';
  if (first.every(q => q < low)) return 'family-floor';
  return null;
};
/** Wait until the executor answers (a server restart includes loading its weights): up to an hour. */
async function executorReady() {
  for (let attempt = 0; attempt < 60; attempt++) {
    try { if ((await fetch(options['executor-endpoint'].replace(/\/$/, '') + '/v1/models', { signal: AbortSignal.timeout(5000) })).ok) return; }
    catch { /* not up yet */ }
    await new Promise(done => setTimeout(done, 60_000));
  }
  throw Error('executor did not come back within an hour');
}
const transient = error => /fetch failed|ECONNREFUSED|ECONNRESET|other side closed|socket hang up|HTTP 50[234]/i.test(String(error?.message ?? error) + String(error?.cause?.message ?? ''));
/** Screen one episode; an executor outage is waited out and the episode screened again, not recorded as an error. */
async function screenWithRetry(episode) {
  for (let attempt = 0; ; attempt++) {
    await executorReady();
    let row;
    try { row = await screen(episode); }
    catch (error) { row = { error: String(error).slice(0, 400), transient: transient(error) }; }
    const outage = row.transient || (row.error && (transient(row.error) || /model request to/i.test(JSON.stringify(row))));
    if (!outage || attempt >= 3) { const { transient: _, ...kept } = row; return kept; }
  }
}
let next = 0;
async function worker() {
  while (next < selected.length && !controller.signal.aborted) {
    const episode = selected[next++];
    let row = done.get(episode.id);
    if (!row) {
      const verdict = familyVerdict(episode.family);
      row = verdict ? { skipped: verdict } : await screenWithRetry(episode);
      row = { schema: 'natlang.episode-headroom/2', input_sha256:inputSha256, screen: screenIdentity, executor: executorId, episode: episode.id,
        family: episode.family, ...row };
      if (controller.signal.aborted) break;
      await appendFile(options.out, JSON.stringify(row) + '\n');
      console.log(JSON.stringify({ episode: row.episode, family: row.family, quality: row.support_quality,
        passed: row.support_passed, skipped: row.skipped, error: row.error }));
    }
    if (typeof row.support_quality === 'number')
      outcomes.set(episode.family, [...(outcomes.get(episode.family) ?? []), row.support_quality]);
    seen++;
    const inBand = !row.error && !row.skipped && row.support_quality >= low && row.support_quality <= high;
    const family = families[episode.family] ??= { episodes: 0, errors: 0, skipped: 0, quality: 0, in_band: 0, saturated: 0, floor: 0 };
    family.episodes++;
    if (row.error) family.errors++;
    else if (row.skipped) family.skipped++;
    else {
      family.quality += row.support_quality; family.in_band += inBand ? 1 : 0;
      family.saturated += row.support_quality > high ? 1 : 0; family.floor += row.support_quality < low ? 1 : 0;
    }
    if (inBand) kept.push({ ...episode, provenance: { ...episode.provenance, headroom: { executor: executorId,
      screen: screenIdentity, support_quality: row.support_quality, band: [low, high] } } });
  }
}
await Promise.all(Array.from({ length: Math.max(1, Number(options.concurrency)) }, worker));
const order = new Map(selected.map((episode, index) => [episode.id, index]));
kept.sort((a, b) => order.get(a.id) - order.get(b.id));
for (const family of Object.values(families)) {
  const scored = family.episodes - family.errors - family.skipped;
  family.quality = scored > 0 ? +(family.quality / scored).toFixed(4) : null;
}
if (options.keep && !controller.signal.aborted)
  await writeFile(options.keep, kept.map(row => JSON.stringify(row)).join('\n') + (kept.length ? '\n' : ''), { flag: 'wx' });
console.error(JSON.stringify({ screened: seen, kept: kept.length, band: [low, high], executor: executorId, families }, null, 2));

async function screen(episode) {
  const files = skillEpisodeFiles(episode);
  const cases = supportSearchCases(episode).map(row => ({ ...row, split: 'train' }));
  const { scoring } = episodeScorings(episode.provenance, !!episode.transfer,
    { pins, databaseRoot: options['database-root'] });
  const executions = await arenaEpisodeExecutions(episode, executor, { pins, executorId,
    arenaRoot: options['arena-root'], signal: controller.signal });
  const gateway = new UsageGateway({ maxModelCalls: 40 * cases.length, maxRollouts: cases.length, maxProposals: 0 });
  const contract = { entry: episode.target.entry, exportName: episode.target.exportName ?? 'default',
    programId: episode.target.source.id };
  const evaluator = new SourceEvaluator(contract, cases, executor, gateway, { executorId, signal: controller.signal,
    executeCase: executions.executeCase, scoring, excludeModelWaitFromTimeout: true, maxCasesPerRequest: cases.length });
  const report = await evaluator.evaluate(Folder.fromFiles(files).snapshot(), { split: 'train' });
  const outcomes = evaluator.page(report.evidence, 0, 100);
  if (hasUnscoredEvaluationFailure({outcomes})) return {
    error: 'unscored support screen: fixture failure or resource timeout',
    failures: outcomes.filter(row => row.failureKind === 'fixture' || row.failureKind === 'timeout')
      .map(row => ({id:row.caseId, failure_kind:row.failureKind, error:row.error})),
    cases:report.total, model_calls:report.modelCalls ?? null };
  return { support_quality: report.quality, support_passed: report.passed / report.total, cases: report.total,
    gates_passed: report.gatesPassed, model_calls: report.modelCalls ?? null,
    case_quality: outcomes.map(row => ({ id: row.caseId, quality: row.quality, passed: row.passed })) };
}
