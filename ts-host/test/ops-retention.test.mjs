import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { createNatlangRuntime } from '../dist/index.js';
import * as retention from '../../applications/dist/ops-retention/index.js';
import { main as classifyMain } from '../scripts/retention-classify.mjs';
import { scriptedModel } from './support/natlang.mjs';

const thresholds = { finished_after_days: 3, aborted_after_days: 7, leftover_after_days: 2 };
const facts = (over = {}) => ({ category: 'checkpoint', age_days: 10, run_age_days: 10, registered: false, named_by_reference: false,
  directory_named_by_reference: false, record_file: false, policy_keep_path: false, in_use: null, partial_write: false,
  run_marked_stopped: false, step: null, latest_step_in_directory: null, step_files_in_directory: 0, best_or_final: false, ...over });
const item = (id, over, crisp) => ({ id, path: `/runs/r/${id}.pt`, logical: `/runs/r/${id}.pt`, bytes: 2 ** 30, facts: facts(over), crisp });

/** A model that answers classifyRetention with the verdict `answer(opening)` gives. */
function runtimeFor(answer) {
  const { driver, openings } = scriptedModel(opening => opening.includes('facts describe one large file') ? `return ${JSON.stringify(answer(opening))};` : null);
  return { runtime: createNatlangRuntime({ model: driver }), openings };
}

test('refusal: a deleting verdict on protected facts or an unknown tier is refused; keeping is never refused', () => {
  const auto = { tier: 'PRUNE-AUTO', reason: 'r' };
  assert.equal(retention.refusal(facts(), thresholds, auto), null);
  assert.match(retention.refusal(facts({ registered: true }), thresholds, auto), /manifest pins/);
  assert.match(retention.refusal(facts({ in_use: 'open by a process' }), thresholds, { tier: 'PRUNE-ASK', reason: 'r' }), /ACTIVE/);
  assert.match(retention.refusal(facts({ run_age_days: 1 }), thresholds, auto), /within 3 d/);
  assert.equal(retention.refusal(facts({ registered: true }), thresholds, { tier: 'KEEP', reason: 'r' }), null);
  assert.match(retention.refusal(facts(), thresholds, { tier: 'DELETE', reason: 'r' }), /not one of/);
});

test('crisp mode returns the crisp verdicts and calls no model', async () => {
  const items = [item('a', {}, { tier: 'PRUNE-ASK', reason: 'large checkpoint file without a keep rule' })];
  const result = await retention.classifyItems(items, thresholds, 'crisp', () => { throw new Error('no model in crisp mode'); });
  assert.deepEqual(result.classified, [{ id: 'a', tier: 'PRUNE-ASK', reason: 'large checkpoint file without a keep rule', served: 'crisp' }]);
  assert.deepEqual(result.records, []);
});

test('shadow mode serves crisp and appends each disagreement to the shadow log, through the bridge script', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'ops-retention-'));
  const input = join(dir, 'items.json'), out = join(dir, 'verdicts.json'), log = join(dir, 'shadow.jsonl');
  const items = [
    item('agree', { registered: true }, { tier: 'KEEP', reason: 'registered corpus/artifact file' }),
    item('differ', { step: 1, latest_step_in_directory: 2, step_files_in_directory: 2 }, { tier: 'PRUNE-ASK', reason: 'crisp reason' }),
  ];
  writeFileSync(input, JSON.stringify({ thresholds, items }));
  const { runtime, openings } = runtimeFor(opening => opening.includes('"registered":true') || /registered: true/.test(opening)
    ? { tier: 'KEEP', reason: 'registered corpus or artifact file' } : { tier: 'PRUNE-AUTO', reason: 'intermediate checkpoint' });
  assert.equal(await classifyMain(['--input', input, '--out', out, '--mode', 'shadow', '--shadow-log', log, '--model', 'scripted'],
    { createRuntime: () => runtime }), 0);
  assert.equal(openings.length, 2, 'one call per file');
  assert.match(openings[0], /latest_step_in_directory/, 'the facts reach the model');
  const result = JSON.parse(readFileSync(out, 'utf8'));
  assert.deepEqual(result.classified.map(row => [row.id, row.tier, row.served]), [['agree', 'KEEP', 'crisp'], ['differ', 'PRUNE-ASK', 'crisp']]);
  assert.deepEqual({ ...result.summary }, { mode: 'shadow', items: 2, agree: 1, disagree: 1, refused: 0, errors: 0 });
  const records = readFileSync(log, 'utf8').trim().split('\n').map(line => JSON.parse(line));
  assert.equal(records.length, 1);
  assert.deepEqual([records[0].id, records[0].path, records[0].crisp_tier, records[0].nl_tier, records[0].nl_reason, records[0].served],
    ['differ', '/runs/r/differ.pt', 'PRUNE-ASK', 'PRUNE-AUTO', 'intermediate checkpoint', 'crisp']);
  assert.match(records[0].classifier, /^natlang@[0-9a-f]{16}$/);
});

test('nl mode serves the natural-language verdict, but a refused one falls back to crisp and is recorded', async () => {
  const items = [
    item('free', { step: 1, latest_step_in_directory: 2, step_files_in_directory: 2 }, { tier: 'PRUNE-AUTO', reason: 'intermediate checkpoint (step 1 < latest 2)' }),
    item('pinned', { registered: true }, { tier: 'KEEP', reason: 'registered corpus/artifact file' }),
  ];
  const verdicts = { free: { tier: 'PRUNE-AUTO', reason: 'intermediate checkpoint' }, pinned: { tier: 'PRUNE-AUTO', reason: 'old' } };
  const result = await retention.classifyItems(items, thresholds, 'nl', async f => f.registered ? verdicts.pinned : verdicts.free,
    { now: () => new Date(0) });
  assert.deepEqual(result.classified.map(row => [row.id, row.tier, row.served]), [['free', 'PRUNE-AUTO', 'nl'], ['pinned', 'KEEP', 'crisp']]);
  assert.equal(result.summary.refused, 1);
  assert.deepEqual(result.records.map(record => [record.id, record.nl_tier, record.served, Boolean(record.refused)]), [['pinned', 'PRUNE-AUTO', 'crisp', true]]);
  // A failing call is recorded as an error and crisp serves.
  const failed = await retention.classifyItems(items.slice(0, 1), thresholds, 'shadow', async () => { throw new Error('model down'); });
  assert.deepEqual([failed.summary.errors, failed.records[0].error, failed.classified[0].served], [1, 'model down', 'crisp']);
});
