/** `natlang.improvement-step/1` records and the converter for existing runs (LEARNING_CONTINUUM.md §8). */
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { improvementStep, stepView, validateImprovementStep } from '../dist/improvement/step-record.js';

const fields = () => ({
  episode: { id: 'ep-1', family: 'decision:sst5', split: 'train' }, facets: ['family:decision:sst5'],
  before: [{ kind: 'soft-skill', id: 'nzs1_' + 'a'.repeat(52) }],
  operator: { kind: 'soft-skill-tuned', version: 'test/1', regime: 'supervised', hyper: { lr: 0.01 } },
  view: { visibility: 'reward-blind-observations', evidence: ['trace-sha256:' + 'b'.repeat(64)] },
  proposal: { deltas: [{ artifact: { kind: 'soft-skill', id: 'nzs1_' + 'c'.repeat(52) }, delta: null, scale: 1 }] },
  after: [{ kind: 'soft-skill', id: 'nzs1_' + 'c'.repeat(52) }],
  outcome: { query: { before: 0.5, after: 0.7, effect: 0.2 } },
  trajectory: { id: 't-1', step: 0 },
});

test('a step is content-addressed, validates, and its view has no outcome', () => {
  const a = improvementStep(fields()), b = improvementStep(fields());
  assert.equal(a.id, b.id);
  assert.match(a.id, /^step-[0-9a-f]{32}$/);
  assert.deepEqual(validateImprovementStep(a), []);
  assert.deepEqual(validateImprovementStep({ ...a, outcome: { query: { before: 0, after: 1, effect: 1 } } }), ['id does not match the record content']);
  const view = stepView(a);
  assert.equal(view.outcome, undefined);
  assert.equal(view.id, a.id);
});

test('validation: visibility, regime, content IDs (null only when legacy)', () => {
  assert.throws(() => improvementStep({ ...fields(), view: { visibility: 'peeking', evidence: [] } }), /view.visibility/);
  assert.throws(() => improvementStep({ ...fields(), operator: { ...fields().operator, regime: 'magic' } }), /operator.regime/);
  assert.throws(() => improvementStep({ ...fields(), before: [{ kind: 'soft-skill', id: null }] }), /before\[0\].id/);
  assert.throws(() => improvementStep({ ...fields(), before: [{ kind: 'soft-skill', id: 'nz1_retired' }] }), /retired projected-block identity/);
  assert.throws(() => improvementStep({ ...fields(), before: [{ kind: 'soft-skill', id: 'source-digest' }] }), /archived native state/);
  assert.ok(improvementStep({ ...fields(), before: [{ kind: 'soft-skill', id: null }], provenance: { legacy: true } }));
});

test('the converter turns authoring results and soft-skill results into valid steps, once each', () => {
  const dir = mkdtempSync(join(tmpdir(), 'natlang-steps-'));
  const paired = (b, s) => ({ baseline: { quality: b, total: 4, modelCalls: 4 }, selected: { quality: s, modelCalls: 4 }, effect: s - b, wins: 1, losses: 0, ties: 3 });
  const authoring = { version: 'natlang.skill-authoring-trajectory/1', episode: 'ep-a', split: 'train', family: 'tools:x',
    baseline: 'aaa', selected: 'bbb', disposition: 'evaluated', identity: 'id-1', evaluation_ticket: { id: 'ticket-1' },
    searchDefinition: { version: 'natlang.improvement-case/1', policy: { strategy: 'gepa' }, authoredDigest: 'dig' },
    search: { baseline: { quality: 0.5 }, validation: { quality: 0.75 } }, query: paired(0.5, 0.75), transfer: null,
    authorExchanges: [{}, {}], author_identity: 'author-model' };
  const failed = { version: 'natlang.skill-authoring-trajectory/1', episode: 'ep-b', disposition: 'failed' };
  const soft = [{ arm: 'generic', block: 'nzs1_' + 'd'.repeat(52), trace: [1, 0.5] },
    { family: 'decision:sst5', scores: Object.fromEntries(['none', 'text-init', 'generic', 'tuned', 'specific'].map((arm, i) =>
      [arm, { query: { quality: 0.5 + i / 10 }, transfer: { quality: 0.5 } }])), traces: { tuned: [1, 0.4], specific: [1, 0.3] } }];
  writeFileSync(join(dir, 'a.json'), JSON.stringify(authoring));
  writeFileSync(join(dir, 'b.json'), JSON.stringify(failed));
  writeFileSync(join(dir, 'soft.jsonl'), soft.map(row => JSON.stringify(row)).join('\n') + '\n');
  const script = fileURLToPath(new URL('../scripts/skills/convert-improvement-steps.mjs', import.meta.url));
  const out = join(dir, 'steps.jsonl');
  const summary = JSON.parse(execFileSync(process.execPath, [script, '--out', out, join(dir, 'a.json'), join(dir, 'a.json'),
    join(dir, 'b.json'), join(dir, 'soft.jsonl')], { encoding: 'utf8' }));
  assert.deepEqual([summary.steps, summary.skipped], [4, 1]);
  const steps = readFileSync(out, 'utf8').trim().split('\n').map(JSON.parse);
  for (const step of steps) assert.deepEqual(validateImprovementStep(step), []);
  const crisp = steps.find(step => step.operator.kind === 'crisp-skill-search');
  assert.deepEqual([crisp.before[0].id, crisp.after[0].id, crisp.outcome.query.effect, crisp.outcome.support.effect], ['aaa', 'bbb', 0.25, 0.25]);
  const generic = steps.find(step => step.operator.kind === 'soft-skill-generic');
  assert.equal(generic.provenance.legacy, true);
  assert.equal(generic.before[0].id, null, 'historical projected IDs are not relabeled as native state');
  assert.equal(generic.proposal.deltas[0].artifact.id, null);
  const specific = steps.find(step => step.operator.kind === 'soft-skill-specific');
  assert.ok(Math.abs(specific.outcome.query.effect - 0.2) < 1e-9, 'the specific arm is measured from generic');
  assert.throws(() => execFileSync(process.execPath, [script, '--out', out, join(dir, 'a.json')], { stdio: 'pipe' }), 'never overwrites');
});
