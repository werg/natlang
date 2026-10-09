/** Teacher-side stages of the refine-judge family and the teacher judge labeller, with injectable model drivers.
 *
 * `runStages` runs the natlang functions of applications/refine-data (exemplify, nearMiss, verifyNearMiss) through the
 * normal runtime, so every stage is a recorded call. `labelPairs` scores (value, predicate) pairs with the runtime's
 * own refinement judge (`decisionJudge`, native/refinement.ts) over a driver's `decide`, so the prompt the teacher scores
 * is the prompt the student is later trained on. Neither loads a model: the CLI wrappers pass a driver.
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createNatlangRuntime, loadVirtualNatlang } from '../../dist/index.js';
import { decisionJudge } from '../../dist/native/refinement.js';

const here = dirname(fileURLToPath(import.meta.url));
export const STAGE_DIR = join(here, '..', '..', '..', 'applications', 'refine-data');
export const STAGES = {
  exemplify: ['predicate', 'base', 'slot', 'count'],
  nearMiss: ['predicate', 'value'],
  verifyNearMiss: ['predicate', 'original', 'edited'],
};

export function stageFiles(dir = STAGE_DIR) {
  const files = { 'types.ts': readFileSync(join(dir, 'types.ts'), 'utf8') };
  for (const name of Object.keys(STAGES)) files[`${name}.nl`] = readFileSync(join(dir, `${name}.nl`), 'utf8');
  return files;
}

/**
 * Run `stage` over rows `{ id, args }`; a row's result is `{ id, stage, ok, value }` or `{ id, stage, ok: false, error }`.
 * `done` holds ids to skip (a resumed run). `onRow` receives each result as it completes. Calls are recorded in the
 * machine's call store unless `runtimeOptions.calls` is false (tests).
 */
export async function runStages({ stage, rows, driver, concurrency = 4, done = new Set(), onRow = () => {}, files = stageFiles(), runtimeOptions = {} }) {
  const order = STAGES[stage];
  if (!order) throw new Error(`unknown stage ${stage}; expected one of ${Object.keys(STAGES).join(', ')}`);
  const fn = loadVirtualNatlang(files, `${stage}.nl`);
  const runtime = createNatlangRuntime({ model: { driver }, seed: { mode: 'backend' }, ...runtimeOptions });
  const todo = rows.filter(row => !done.has(row.id));
  let next = 0;
  const results = [];
  const worker = async () => {
    for (let index = next++; index < todo.length; index = next++) {
      const row = todo[index];
      let result;
      try {
        const value = await runtime.run(() => fn(...order.map(name => row.args[name])));
        result = { id: row.id, stage, ok: true, value };
      } catch (error) {
        result = { id: row.id, stage, ok: false, error: String(error?.message ?? error).slice(0, 500) };
      }
      results.push(result);
      await onRow(result);
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, concurrency) }, worker));
  return results;
}

/**
 * Score pairs `{ id, value, predicate }` with a judge over `decide` (a DecisionScorer). Result rows are
 * `{ id, teacher, p_true }`; a pair the scorer fails on gets `{ id, teacher, error }` and is retried on the next run.
 */
export async function labelPairs({ pairs, decide, teacher, concurrency = 4, done = new Set(), onRow = () => {} }) {
  const judge = decisionJudge(decide, { id: teacher });
  const todo = pairs.filter(pair => !done.has(pair.id));
  let next = 0;
  const results = [];
  const worker = async () => {
    for (let index = next++; index < todo.length; index = next++) {
      const pair = todo[index];
      let result;
      try { result = { id: pair.id, teacher, p_true: await judge.probability(pair.value, pair.predicate) }; }
      catch (error) { result = { id: pair.id, teacher, error: String(error?.message ?? error).slice(0, 300) }; }
      results.push(result);
      await onRow(result);
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, concurrency) }, worker));
  return results;
}

/** The judge's messages for (value, predicate), as the scorer sees them (no model; used for parity checks). */
export async function judgeMessages(value, predicate) {
  let captured;
  const judge = decisionJudge(async request => { captured = request; return { log_probs: [0, 0] }; }, { id: 'capture' });
  await judge.probability(value, predicate);
  return captured.messages;
}
