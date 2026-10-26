import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as host from '../dist/index.js';
import { WeightBinding } from '../dist/adaptation/weights/binding.js';
import { currentSplit } from '../dist/tune/pipeline.js';
import { scaffoldManifest } from '../dist/tune/collect.js';
import { runRepairs } from '../dist/tune/repairs.js';
import { journaledServices, replayServices } from '../dist/tune/world.js';
import { scriptedModel } from './support/natlang.mjs';

test('a journaled world replays calls by arguments in order, replays errors, and flags unrecorded calls', async () => {
  let live = 0;
  const shape = { shop: { price(item) { live++; if (item === 'gone') throw new Error('no such item'); return item.length; },
    async stock(item) { live++; return { item, count: 3 }; } } };
  const world = { schema: 'natlang.tune-world/v1', calls: [] };
  const recorded = journaledServices(shape, world);
  assert.equal(recorded.shop.price('tea'), 3);
  assert.equal(recorded.shop.price('tea'), 3);
  assert.deepEqual(await recorded.shop.stock('tea'), { item: 'tea', count: 3 });
  assert.throws(() => recorded.shop.price('gone'), /no such item/);
  assert.equal(live, 4);
  assert.equal(world.calls.length, 4);
  const replay = replayServices(shape, world);
  assert.equal(replay.services.shop.price('tea'), 3);
  assert.deepEqual(await replay.services.shop.stock('tea'), { item: 'tea', count: 3 });
  assert.throws(() => replay.services.shop.price('gone'), /no such item/);
  assert.equal(replay.services.shop.price('tea'), 3, 'the second identical call is answered by the second record');
  assert.equal(replay.divergence(), null);
  assert.throws(() => replay.services.shop.price('tea'), /world-divergence/, 'a third identical call was never recorded');
  assert.match(replay.divergence(), /shop\.price/);
  assert.equal(live, 4, 'a replay never calls the live services');
});

const named = body => `---\nargs: { value: string }\nreturns: string\n---\n${body}\n`;
function project() {
  const root = mkdtempSync(join(tmpdir(), 'natlang-world-'));
  mkdirSync(join(root, 'outer'));
  writeFileSync(join(root, 'outer.nl'), named('Pass value to inner.'));
  writeFileSync(join(root, 'outer', 'inner.nl'), named('Look value up in the shop and return the name in uppercase.'));
  writeFileSync(join(root, 'package.json'), '{"type":"module"}');
  const examples = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta'].map(value => ({ function: 'outer.nl', input: { value }, expected: value.toUpperCase() }));
  writeFileSync(join(root, 'tune.examples.jsonl'), examples.map(example => JSON.stringify(example)).join('\n') + '\n');
  return root;
}

test('repairs replay the student\'s captured world: matching continuations are admitted without acting again, divergent ones rejected', async () => {
  const root = project();
  try {
    const built = host.buildProject({ project: root, emit: false, write: false });
    assert.equal(built.ok, true, host.formatDiagnostics(built.diagnostics));
    const program = built.manifest.adaptation, examples = currentSplit(root), release = scaffoldManifest(program, base());
    let live = 0;
    const services = () => ({ shop: { async lookup(name) { live++; return { name }; } } });
    // The student looks the value up but forgets to upper-case it.
    const student = scriptedModel(opening => opening.includes('Pass value') ? 'return await inner(value)' : 'return (await shop.lookup(value)).name');
    const run = async (teacherCode, out) => runRepairs({ root, program, scaffold: release, release, weights: new WeightBinding(release, { program, strict: true }),
      student: student.driver, teacher: scriptedModel(opening => opening.includes('Pass value') ? 'return await inner(value)' : teacherCode).driver,
      teacherId: 'teacher:test', examples, round: 1, records: join(root, out, 'records.jsonl'), rows: join(root, out, 'rows.jsonl'),
      checkpointSha256: 'c'.repeat(64), services });
    const train = examples.filter(example => example.partition === 'train').length;
    const same = await run('return (await shop.lookup(value)).name.toUpperCase()', 'same');
    assert.equal(same.admitted, train, 'a repair making the student\'s own service call replays it');
    assert.equal(live, train, 'only the student runs reached the live service; every repair replayed the captured world');
    live = 0;
    const divergent = await run('return (await shop.lookup(value + "!")).name.slice(0, -1).toUpperCase()', 'divergent');
    assert.equal(divergent.admitted, 0);
    const reasons = readFileSync(join(root, 'divergent', 'records.jsonl'), 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line).reason);
    assert.ok(reasons.some(reason => /left the captured world: shop\.lookup/.test(reason)), reasons.join('\n'));
    assert.equal(live, train, 'the divergent continuation never acted on the live world');
  } finally { rmSync(root, { recursive: true, force: true }); }
});

function base() { return { architecture: 'lfm2', checkpoint_sha256: 'c'.repeat(64), tokenizer_fingerprint: 't', embedding_weight_sha256: 'e'.repeat(64) }; }
