#!/usr/bin/env node
/** S7 rollout driver (plans/neuralese/S7_RL.md §5, spec in plans/neuralese/S7_ROLLOUT_CONTRACT.md).
 *
 * Each rollout task is one support case of a skill episode (natlang.skill-episode/1): its natlang target runs through
 * the native runtime against an OpenAI-compatible model server (the reference server, vLLM, the llama.cpp fork), and
 * the episode's own host-only scorer (skills/scoring.ts) gives the reward. A group of `--group` rollouts per task
 * samples the current policy (`--adapters`, a JSON list of adapter blocks served per request) at `--temperature`.
 * Every model turn is recorded from the wire (the messages and tools as sent, the assistant message as returned), so
 * the trainer can replay it (`logLikelihood` terms of /v1/neuralese/grad). Query and transfer cases are never run:
 * they stay sealed for evaluation.
 *
 * Usage: rollout-episodes.mjs --episodes FILE[,FILE...] --out ROLLOUTS.jsonl --endpoint URL --model ID
 *          [--tasks TASKS.jsonl] [--round 0] [--group 4] [--temperature 1] [--adapters JSON] [--database-root DIR]
 *          [--families a,b] [--split train] [--limit-episodes N] [--cases-per-episode N] [--concurrency 4]
 *          [--max-model-calls 12] [--timeout-ms 600000]
 * The output is resumable: (round, task, sample) rows already written are skipped. `--tasks` also writes the task
 * records (natlang.rollout-task/1) the rollouts reference.
 */
import { appendFile, readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { Folder } from '../../dist/native/scoped-fs.js';
import { UsageGateway } from '../../dist/evaluation/usage.js';
import { SourceEvaluator, hasUnscoredEvaluationFailure } from '../../dist/improvement/host.js';
import { skillEpisodeFiles, supportSearchCases } from '../../dist/improvement/skill-authoring.js';
import { episodeScorings } from '../../dist/skills/scoring.js';
import { arenaEpisodeExecutions } from '../../dist/self-play/evaluation.js';
import { openAICompatibleModelTurn } from '../../dist/model/openai-compatible.js';
import { recordingModelDriver } from '../skills/record-model-turn.mjs';

const options = { round: '0', group: '4', temperature: '1', split: 'train', concurrency: '4', 'max-model-calls': '12',
  'timeout-ms': '600000', 'cases-per-episode': '0', 'limit-episodes': '0' };
const KEYS = ['episodes', 'out', 'endpoint', 'model', 'tasks', 'round', 'group', 'temperature', 'adapters', 'database-root',
  'families', 'split', 'limit-episodes', 'cases-per-episode', 'concurrency', 'max-model-calls', 'timeout-ms'];
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i].replace(/^--/, '');
  if (!KEYS.includes(key) || process.argv[i + 1] === undefined) throw Error('Usage: see the header of rollout-episodes.mjs');
  options[key] = process.argv[i + 1];
}
for (const key of ['episodes', 'out', 'endpoint', 'model']) if (!options[key]) throw Error(`--${key} is required`);

const sha = value => createHash('sha256').update(value).digest('hex');
const pins = {};
for (const file of ['skills/objective.js', 'skills/extended-objective.js', 'skills/efficiency-objective.js', 'skills/code-objective.js',
  'skills/visual-objective.js', 'skills/graded.js', 'skills/crossword-objective.js', 'skills/csp-objective.js',
  'skills/translation-objective.js', 'skills/research-objective.js', 'skills/scifact-objective.js', 'skills/contractnli-objective.js'])
  try { pins[file] = sha(await readFile(new URL('../../dist/' + file, import.meta.url))); } catch { /* scorer not built */ }
pins['scripts/rl/rollout-episodes.mjs'] = sha(await readFile(new URL(import.meta.url)));

const round = Number(options.round), group = Number(options.group), temperature = Number(options.temperature);
const adapters = options.adapters ? JSON.parse(options.adapters) : [];
const families = options.families ? new Set(options.families.split(',')) : null;
const policy = { endpoint: options.endpoint, model: options.model, adapters, temperature };

/** A rollout task: one support case of an episode, with the environment that resets, runs and checks it. */
export function rolloutTask(episode, row, source) {
  return { schema: 'natlang.rollout-task/1', id: `${episode.id}#${row.id}`, family: episode.family, split: episode.split,
    environment: { kind: 'natlang-episode', episodes_file: source.path, episodes_sha256: source.sha256, episode: episode.id,
      case: row.id, entry: episode.target.entry, export: episode.target.exportName ?? 'default' },
    fixture: { args: row.args ?? [] },
    check: { metric: episode.provenance?.metric ?? null, expected: 'episode-case' },
    budget: { max_model_calls: Number(options['max-model-calls']), timeout_ms: Number(options['timeout-ms']) },
    difficulty: episode.provenance?.headroom?.support_quality ?? null,
    provenance: { license: episode.license ?? null, source_groups: episode.source_groups ?? [], case_group: row.group } };
}

const selected = [];
for (const path of options.episodes.split(',')) {
  const text = await readFile(path, 'utf8'), source = { path, sha256: sha(text) };
  let taken = 0;
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    const episode = JSON.parse(line);
    if (options.split !== 'all' && episode.split !== options.split) continue;
    if (families && !families.has(episode.family)) continue;
    if (Number(options['limit-episodes']) && taken >= Number(options['limit-episodes'])) break;
    let cases;
    try { cases = supportSearchCases(episode); skillEpisodeFiles(episode); } catch { continue; }
    if (Number(options['cases-per-episode'])) cases = cases.slice(0, Number(options['cases-per-episode']));
    taken++;
    for (const row of cases) selected.push({ episode, row, task: rolloutTask(episode, row, source) });
  }
}
if (options.tasks) {
  let known = new Set();
  try { known = new Set((await readFile(options.tasks, 'utf8')).split('\n').filter(Boolean).map(l => JSON.parse(l).id)); } catch { /* new */ }
  for (const { task } of selected) if (!known.has(task.id)) await appendFile(options.tasks, JSON.stringify(task) + '\n');
}

let done = new Set();
try {
  for (const line of (await readFile(options.out, 'utf8')).split('\n'))
    if (line.trim()) { const row = JSON.parse(line); done.add(`${row.round}|${row.task_id}|${row.sample}`); }
} catch (error) { if (error.code !== 'ENOENT') throw error; }

const controller = new AbortController();
for (const signal of ['SIGINT', 'SIGTERM']) process.once(signal, () => controller.abort(new Error(signal)));

/** The assistant message as the replay consumes it: role, content and tool calls only. */
const assistant = message => ({ role: 'assistant', content: message?.content ?? '',
  ...(Array.isArray(message?.tool_calls) && message.tool_calls.length ? { tool_calls: message.tool_calls } : {}) });

async function rollout({ episode, row, task }, sample) {
  const turns = [];
  const seed = parseInt(sha(`${round}|${task.id}|${sample}`).slice(0, 8), 16);
  const executorId = `${options.endpoint}:${options.model}:${sha(JSON.stringify(adapters)).slice(0, 12)}`;
  const executor = recordingModelDriver({
    createDriver: onExchange => openAICompatibleModelTurn({ endpoint: options.endpoint, model: options.model,
      apiKey: process.env.NATLANG_IMPROVEMENT_API_KEY, onExchange,
      request: { temperature, seed, ...(adapters.length ? { x_natlang_adapters: adapters } : {}) } }),
    record: async () => {},
    recordWire: async exchange => {
      const message = exchange.wireResponse?.choices?.[0]?.message;
      if (!message) return;
      turns.push({ messages: exchange.wireRequest?.messages ?? [], tools: exchange.wireRequest?.tools ?? null,
        target: assistant(message), finish_reason: exchange.wireResponse?.choices?.[0]?.finish_reason ?? null });
    } });
  const started = Date.now();
  const cases = [{ ...row, split: 'train' }];
  const { scoring } = episodeScorings(episode.provenance, !!episode.transfer, { pins, databaseRoot: options['database-root'] });
  const executions = await arenaEpisodeExecutions(episode, executor, { pins, executorId, signal: controller.signal });
  const gateway = new UsageGateway({ maxModelCalls: task.budget.max_model_calls, maxRollouts: 1, maxProposals: 0 });
  const contract = { entry: episode.target.entry, exportName: episode.target.exportName ?? 'default', programId: episode.target.source.id };
  const evaluator = new SourceEvaluator(contract, cases, executor, gateway, { executorId, signal: controller.signal,
    executeCase: executions.executeCase, scoring, excludeModelWaitFromTimeout: true, maxCasesPerRequest: 1,
    timeoutMs: task.budget.timeout_ms });
  let report;
  try { report = await evaluator.evaluate(Folder.fromFiles(skillEpisodeFiles(episode)).snapshot(), { split: 'train' }); }
  catch (error) {
    // Running out of the task's model-call budget is the policy's failure (it did not return in time): reward 0,
    // with the turns it took. Other errors are the driver's and stay unscored.
    if (!/budget exhausted/i.test(String(error?.message ?? error))) throw error;
    return { schema: 'natlang.rollout/1', round, task_id: task.id, family: task.family, group: task.id, sample, seed, policy,
      turns, reward: { quality: 0, passed: 0, gates: { within_budget: false } }, unscored: null,
      model_calls: turns.length, wall_ms: Date.now() - started, pins };
  }
  const outcomes = evaluator.page(report.evidence, 0, 10);
  const unscored = hasUnscoredEvaluationFailure({ outcomes });
  const outcome = outcomes[0] ?? {};
  return { schema: 'natlang.rollout/1', round, task_id: task.id, family: task.family, group: task.id, sample, seed, policy,
    turns, reward: unscored ? null : { quality: report.quality, passed: report.passed / Math.max(1, report.total), gates: outcome.gates ?? null },
    unscored: unscored ? { failure_kind: outcome.failureKind ?? null, error: String(outcome.error ?? '').slice(0, 400) } : null,
    model_calls: report.modelCalls ?? turns.length, wall_ms: Date.now() - started, pins };
}

const jobs = [];
for (const item of selected) for (let sample = 0; sample < group; sample++)
  if (!done.has(`${round}|${item.task.id}|${sample}`)) jobs.push([item, sample]);
let next = 0, written = 0;
const summary = {};
async function worker() {
  while (next < jobs.length && !controller.signal.aborted) {
    const [item, sample] = jobs[next++];
    let row;
    try { row = await rollout(item, sample); }
    catch (error) {
      row = { schema: 'natlang.rollout/1', round, task_id: item.task.id, family: item.task.family, group: item.task.id, sample,
        policy, turns: [], reward: null, unscored: { failure_kind: 'driver', error: String(error?.message ?? error).slice(0, 400) }, pins };
    }
    if (controller.signal.aborted) break;
    await appendFile(options.out, JSON.stringify(row) + '\n');
    written++;
    const family = summary[row.family] ??= { rollouts: 0, scored: 0, quality: 0 };
    family.rollouts++;
    if (row.reward) { family.scored++; family.quality += row.reward.quality; }
    console.log(JSON.stringify({ task: row.task_id, sample, quality: row.reward?.quality ?? null, unscored: row.unscored?.failure_kind ?? null,
      turns: row.turns.length }));
  }
}
await Promise.all(Array.from({ length: Math.max(1, Number(options.concurrency)) }, worker));
for (const family of Object.values(summary)) family.quality = family.scored ? family.quality / family.scored : null;
console.log(JSON.stringify({ round, tasks: selected.length, rollouts_written: written, families: summary }));
