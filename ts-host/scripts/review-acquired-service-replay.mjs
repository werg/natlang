#!/usr/bin/env node
/** Review exact replay of source observations; this is not a native trajectory or oracle. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, openSync, writeFileSync, fsyncSync, closeSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { ReplayServices } from '../dist/calls/replay.js';

const [fixturePath, expectedSha, outputPath, runtimeSourcePath] = process.argv.slice(2);
if (!fixturePath || !/^[0-9a-f]{64}$/.test(expectedSha ?? '') || !outputPath || !runtimeSourcePath)
  throw new Error('usage: review-acquired-service-replay FIXTURE FIXTURE_SHA256 OUTPUT RUNTIME_SOURCE');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const bind = path => { const bytes = readFileSync(path); return { path, sha256: sha(bytes), bytes: bytes.length }; };
const source = bind(fixturePath);
if (source.sha256 !== expectedSha) throw new Error('source observation fixture changed');
const fixture = JSON.parse(readFileSync(fixturePath, 'utf8'));
if (fixture.schema !== 'natlang.acquired-static-service-feasibility/1' || fixture.training_admission !== false ||
    fixture.structural_pairing_errors.length !== 0 || !fixture.source_observations_not_native_execution)
  throw new Error('source observation fixture lacks required custody');
const effects = fixture.observations.map(observation => {
  const [service, method, extra] = observation.name.split('.');
  if (!service || !method || extra) throw new Error('source method is not a two-part service name');
  return { service, method, args: [observation.arguments], result: observation.result, async: false, complete: true };
});
const replay = new ReplayServices(effects, [], { strictSequence: true });
for (const effect of effects) assert.equal(replay.services[effect.service][effect.method](...effect.args), effect.result);
assert.equal(replay.unused().length, 0);
assert.equal(replay.divergences.length, 0);
if (bind(fixturePath).sha256 !== expectedSha) throw new Error('source observation fixture changed during replay');
const receipt = { schema: 'natlang.acquired-static-service-replay-review/1', source,
  source_id: fixture.source_id, observations_replayed: effects.length, strict_sequence: true,
  compiled_replay: bind(fileURLToPath(new URL('../dist/calls/replay.js', import.meta.url))),
  runtime_source: bind(runtimeSourcePath), reviewer: bind(fileURLToPath(import.meta.url)),
  source_outcome_verified: false, native_graph_constructed: false, training_admission: false,
  scope: 'Strict replay of source-recorded observations only; no live service was called and no native trajectory was executed.' };
const fd = openSync(outputPath, 'wx');
try { writeFileSync(fd, JSON.stringify(receipt, null, 2) + '\n'); fsyncSync(fd); } finally { closeSync(fd); }
console.log(JSON.stringify({ source_id: receipt.source_id, observations_replayed: effects.length, training_admission: false }));
