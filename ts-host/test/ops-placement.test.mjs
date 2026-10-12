import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { createNatlangRuntime } from '../dist/index.js';
import * as placement from '../../applications/dist/ops-placement/index.js';
import { main as decideMain } from '../scripts/placement-decide.mjs';
import { scriptedModel } from './support/natlang.mjs';

const job = (over = {}) => ({ id: 'j', kind: 'eval', gpu_memory_gb: 34, hours: 2, needs_teacher: false, lineage_owner: 'dgx',
  moves_active_lineage: false, coreweave_cost_usd: 12, ...over });
const fleet = (over = {}) => ({ dgx_headroom_gb: 4, dgx_waiting_ahead: 10, coreweave_ready: true, coreweave_gpu_memory_gb: 96,
  coreweave_fit_margin_gb: 6, coreweave_active: 0, coreweave_max_parallel: 10, coreweave_budget_left_usd: null, ...over });

test('refusal bounds a natural-language placement exactly; queueing and asking are never refused', () => {
  const cw = { where: 'coreweave', reason: 'r' };
  assert.equal(placement.refusal(job(), fleet(), cw), null);
  assert.match(placement.refusal(job(), fleet({ coreweave_ready: false }), cw), /not ready/);
  assert.match(placement.refusal(job({ gpu_memory_gb: 95 }), fleet(), cw), /more than one sandbox GPU/);
  assert.match(placement.refusal(job({ needs_teacher: true }), fleet(), cw), /teacher/);
  assert.match(placement.refusal(job(), fleet({ coreweave_active: 10 }), cw), /limit 10/);
  assert.match(placement.refusal(job({ coreweave_cost_usd: null }), fleet({ coreweave_budget_left_usd: 50 }), cw), /unknown/);
  assert.match(placement.refusal(job(), fleet(), { where: 'dgx', reason: 'r' }), /queue-dgx/);
  assert.match(placement.refusal(job({ moves_active_lineage: true }), fleet(), cw), /owner's decision/);
  assert.match(placement.refusal(job(), fleet(), { where: 'pop', reason: 'r' }), /Pop runs only/);
  assert.match(placement.refusal(job(), fleet(), { where: 'elsewhere', reason: 'r' }), /not one of/);
  assert.equal(placement.refusal(job({ moves_active_lineage: true }), fleet(), { where: 'ask', reason: 'r' }), null);
  assert.equal(placement.refusal(job(), fleet({ coreweave_ready: false }), { where: 'queue-dgx', reason: 'r' }), null);
});

test('nl mode serves the model decision unless refused; shadow serves crisp and logs disagreements via the bridge', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'ops-placement-'));
  const input = join(dir, 'items.json'), out = join(dir, 'out.json'), log = join(dir, 'shadow.jsonl');
  const items = [
    { id: 'fits-cw', job: job(), fleet: fleet(), crisp: { where: 'coreweave', reason: 'DGX cannot take it now' } },
    { id: 'not-ready', job: job(), fleet: fleet({ coreweave_ready: false }), crisp: { where: 'queue-dgx', reason: 'CoreWeave is not ready' } },
  ];
  writeFileSync(input, JSON.stringify({ items }));
  const { driver, openings } = scriptedModel(opening => opening.includes('job describes one job')
    ? `return ${JSON.stringify({ where: 'coreweave', reason: 'model says coreweave' })};` : null);
  const runtime = createNatlangRuntime({ model: driver });
  assert.equal(await decideMain(['--input', input, '--out', out, '--mode', 'nl', '--shadow-log', log, '--model', 'scripted'],
    { createRuntime: () => runtime }), 0);
  assert.equal(openings.length, 2);
  assert.match(openings[0], /coreweave_fit_margin_gb/, 'the facts reach the model');
  const result = JSON.parse(readFileSync(out, 'utf8'));
  assert.deepEqual(result.placed.map(row => [row.id, row.where, row.served]), [['fits-cw', 'coreweave', 'nl'], ['not-ready', 'queue-dgx', 'crisp']]);
  assert.deepEqual({ ...result.summary }, { mode: 'nl', items: 2, agree: 1, disagree: 1, refused: 1, errors: 0 });
  const records = readFileSync(log, 'utf8').trim().split('\n').map(line => JSON.parse(line));
  assert.deepEqual([records[0].id, records[0].nl_where, records[0].served], ['not-ready', 'coreweave', 'crisp']);
  assert.match(records[0].refused, /not ready/);
});

test('crisp mode calls no model', async () => {
  const items = [{ id: 'a', job: job(), fleet: fleet(), crisp: { where: 'queue-dgx', reason: 'x' } }];
  const result = await placement.placeItems(items, 'crisp', () => { throw new Error('no model in crisp mode'); });
  assert.deepEqual(result.placed, [{ id: 'a', where: 'queue-dgx', reason: 'x', served: 'crisp' }]);
});
