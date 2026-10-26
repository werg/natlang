import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as host from '../dist/index.js';
import { activeNativeAdapters } from '../dist/native/model-recording.js';
import { scaffoldManifest } from '../dist/tune/collect.js';
import { sealManifest } from '../dist/adaptation/weights/manifest.js';
import { WeightBinding } from '../dist/adaptation/weights/binding.js';
import { releaseBehavior, runRollouts } from '../dist/tune/rollouts.js';
import { readRounds, runRounds } from '../dist/tune/rounds.js';
import { scriptedModel } from './support/natlang.mjs';

const base = { architecture: 'lfm2', checkpoint_sha256: 'c'.repeat(64), tokenizer_fingerprint: 't', embedding_weight_sha256: 'e'.repeat(64) };
const named = body => `---\nargs: { value: string }\nreturns: string\n---\n${body}\n`;

function project() {
  const root = mkdtempSync(join(tmpdir(), 'natlang-tune-rl-'));
  mkdirSync(join(root, 'outer'));
  writeFileSync(join(root, 'outer.nl'), named('Pass value to inner.'));
  writeFileSync(join(root, 'outer', 'inner.nl'), named('Return value in uppercase.'));
  writeFileSync(join(root, 'package.json'), '{"type":"module"}');
  const built = host.buildProject({ project: root, emit: false, write: false });
  assert.equal(built.ok, true, host.formatDiagnostics(built.diagnostics));
  return { root, program: built.manifest.adaptation };
}

function release(program) {
  const { digest, ...scaffold } = scaffoldManifest(program, base);
  const artifacts = scaffold.tree.map((node, index) => ({ node: node.id, artifact: 'nnt1_' + 'abcdefgh'[index].repeat(52),
    rank: 2, parameterization: 'native-subspace', requires: [] }));
  return sealManifest({ ...scaffold, artifacts });
}

/** A student stand-in on a native server: scripted answers, with the sampling receipt the server would attach. */
function student({ capture = true } = {}) {
  let inner = 0;
  const model = scriptedModel(opening => opening.includes('Pass value') ? 'return await inner(value)'
    : inner++ % 2 ? 'return value.toLowerCase()' : 'return value.toUpperCase()');
  const requests = [];
  const driver = async (request, ...rest) => {
    requests.push(request);
    const turn = await model.driver(request, ...rest);
    const adapters = activeNativeAdapters().map(item => ({ id: item.id, scale_hex: item.scale.toString(16) }));
    return { ...turn, raw_response: { choices: [{ message: { content: turn.text ?? '' } }],
      ...(capture ? { x_natlang_rl_capture: { schema: 'natlang.native-rl-sampling/4', prompt: [], samples: [],
        ...(adapters.length ? { adapter_binding: { adapters } } : {}) } } : {}) } };
  };
  return { driver, requests };
}

const examples = [
  { id: 'r1', function: 'outer.nl', input: { value: 'alpha' }, expected: 'ALPHA', component: 'c1', partition: 'train' },
  { id: 'r2', function: 'outer.nl', input: { value: 'beta' }, expected: 'BETA', component: 'c2', partition: 'train' },
  { id: 'r3', function: 'outer.nl', input: { value: 'gamma' }, component: 'c3', partition: 'train' },
  { id: 'r4', function: 'outer.nl', input: { value: 'delta' }, expected: 'DELTA', component: 'c4', partition: 'development' },
];

test('rollouts record every turn with its invocation, binding, seed and the root oracle reward', async () => {
  const { root, program } = project();
  const manifest = release(program);
  const weights = [new WeightBinding(manifest, { program, available: new Set(manifest.artifacts.map(item => item.artifact)), strict: true })];
  const { driver, requests } = student();
  const out = join(root, 'rollouts.jsonl');
  const options = { root, program, release: manifest, weights, driver, examples, samples: 2, temperature: .7, round: 1, out,
    policyCheckpointSha256: base.checkpoint_sha256 };
  const summary = await runRollouts(options);
  assert.equal(summary.written, 4);
  assert.deepEqual(summary.skipped.map(item => [item.root, item.reason]).sort(), [['r3', 'no root oracle'], ['r4', 'development root']]);
  const rows = readFileSync(out, 'utf8').trim().split('\n').map(line => JSON.parse(line));
  assert.ok(requests.every(request => request.temperature === .7 && Number.isInteger(request.seed)));
  assert.deepEqual(new Set(rows.map(row => row.reward.quality)), new Set([0, 1]), 'the oracle scores both outcomes');
  for (const row of rows) {
    assert.equal(row.schema, 'natlang.tune-rollout/v1');
    assert.equal(row.partition, 'train');
    assert.deepEqual(row.episode.reward, { quality: row.reward.quality });
    assert.equal(row.reward.source, 'root-oracle');
    assert.deepEqual(row.behavior.node_artifacts, releaseBehavior(manifest).node_artifacts);
    const calls = row.episode.turns.map(turn => turn.invocation_id);
    assert.equal(new Set(calls).size, 2, 'outer and inner invocations');
    assert.deepEqual(Object.keys(row.bindings).sort(), [...new Set(calls)].sort());
    assert.equal(new Set(Object.values(row.bindings)).size, 2, 'each invocation keeps its own binding');
    for (const turn of row.episode.turns) {
      assert.ok(turn.sampling.adapter_binding.adapters.length >= 2, 'the release stack served the turn');
      assert.equal(turn.replay.temperature, .7);
    }
    const turnNodes = row.episode.execution_graph.filter(event => event.kind === 'model_turn').map(event => event.node);
    for (const call of new Set(calls)) assert.ok(turnNodes.includes(`${call}#turn1`), 'graph turn names match the trainer');
  }
  const groups = new Set(rows.map(row => row.group));
  assert.equal(groups.size, 2, 'one group per root and round');
  const again = await runRollouts(options);
  assert.equal(again.written, 0);
  assert.equal(again.reused, 4, 'a rerun resumes from the file');
  await assert.rejects(runRollouts({ ...options, driver: student({ capture: false }).driver, out: join(root, 'none.jsonl') }),
    /x_natlang_rl_capture/);
  await assert.rejects(runRollouts({ ...options, samples: 1 }), /at least two/);
  const { NeuraleseUnsupportedError } = await import('../dist/native/neuralese.js');
  const broken = async () => { throw new NeuraleseUnsupportedError('consumer state missing for provenance'); };
  await assert.rejects(runRollouts({ ...options, driver: broken, out: join(root, 'invalid-infrastructure.jsonl') }),
    /consumer state missing for provenance/);
});

test('rounds keep the best development release, continue it, stop early and resume', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'natlang-tune-rounds-'));
  const journal = join(dir, 'rounds.json');
  const success = { 0: .5, 1: .7, 2: .6, 3: .65, 4: .9 };
  const calls = [];
  let fail = true;
  const options = {
    journal, schedule: ['sft', 'rl', 'rl', 'rl', 'rl'], patience: 2,
    train: async round => {
      calls.push(['train', round.index, round.kind, round.parent?.round ?? null, round.rollouts]);
      if (round.index === 2 && fail) { fail = false; throw new Error('interrupted'); }
      return { release: `release-${round.index}`, build: `build-${round.index}` };
    },
    rollouts: async round => {
      calls.push(['rollouts', round.index, round.parent.release]);
      return { path: `rollouts-${round.index}.jsonl`, written: 4, reused: 0, successes: 2, skipped: 0 };
    },
    evaluate: async release => { calls.push(['evaluate', release]); return { success: success[Number(release.split('-')[1])], roots: 10 }; },
  };
  await assert.rejects(runRounds(options), /interrupted/);
  const partial = readRounds(journal);
  assert.equal(partial.rounds[2].status, 'started', 'the sampled rollouts survive the interruption');
  calls.length = 0;
  const result = await runRounds(options);
  assert.deepEqual(calls[0], ['train', 2, 'rl', 1, 'rollouts-2.jsonl'], 'resumes inside the round without resampling');
  assert.equal(result.best.round, 1);
  assert.equal(result.best.release, 'release-1');
  assert.deepEqual(result.rounds.map(round => round.parent?.round ?? null), [null, 0, 1, 1]);
  assert.equal(result.rounds.length, 4, 'round 4 never runs');
  assert.match(result.stopped, /no development improvement in 2 rounds/);
  calls.length = 0;
  await runRounds(options);
  assert.deepEqual(calls, [], 'a finished journal does nothing');
  await assert.rejects(runRounds({ ...options, journal: join(dir, 'other.json'), schedule: ['rl'] }), /starts with an sft round/);
});
