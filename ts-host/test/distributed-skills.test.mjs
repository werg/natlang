import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createNatlangRuntime, loadNatlang } from '../dist/index.js';
import { scriptedModel } from './support/natlang.mjs';

const path = fileURLToPath(new URL('../../skills/natlang-authoring/assets/review/review.nl', import.meta.url));
const decisions = { one: 'supported', two: 'contradicted', three: 'uncertain' };

for (const observations of [[], ['one', 'two', 'three']]) {
  test(`distributed authoring example: ${observations.length} inputs, scripted semantics`, async () => {
    // Without a scorer the verdict decision runs as an ordinary call, so the script answers it too.
    const model = scriptedModel(opening => opening.includes('Assess every observation') ?
      'const assessments = await Promise.all(observations.map(observation => assess(observation, criterion)));\nreturn await summarize(assessments)' :
      opening.includes('Decide the verdict with verdict') ? 'return { verdict: await verdict(observation, criterion), reason: "Scripted wiring fixture" }' :
      `return ${JSON.stringify(decisions)}[observation]`);
    const review = loadNatlang(path);
    const report = await createNatlangRuntime({ model: model.driver }).run(() => review(observations, 'Fixture criterion'));
    assert.equal(report.assessments.length, observations.length);
    assert.deepEqual(['supported', 'contradicted', 'uncertain'].map(key => report[key]), observations.length ? [1, 1, 1] : [0, 0, 0]);
  });
}

test('examples/review is the skill asset, file for file, plus its README', async () => {
  const { readdirSync, readFileSync, statSync } = await import('node:fs');
  const { join } = await import('node:path');
  const asset = fileURLToPath(new URL('../../skills/natlang-authoring/assets/review/', import.meta.url));
  const example = fileURLToPath(new URL('../../examples/review/', import.meta.url));
  const generated = name => name.endsWith('.d.nl.ts') || name === '.natlang' || name === 'node_modules';
  const files = (root, prefix = '') => readdirSync(join(root, prefix)).filter(name => !generated(name)).flatMap(name =>
    statSync(join(root, prefix, name)).isDirectory() ? files(root, join(prefix, name)) : [join(prefix, name)]).sort();
  assert.deepEqual(files(example).filter(path => path !== 'README.md'), files(asset));
  for (const path of files(asset)) assert.equal(readFileSync(join(example, path), 'utf8'), readFileSync(join(asset, path), 'utf8'),
    `examples/review/${path} differs from skills/natlang-authoring/assets/review/${path}; copy the asset again`);
});
