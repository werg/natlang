import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createNatlangRuntime, loadNatlang } from '../dist/index.js';
import { scriptedModel } from './support/natlang.mjs';

const path = fileURLToPath(new URL('../../skills/natlang-authoring/assets/review/review.nl', import.meta.url));
const decisions = { one: 'supported', two: 'contradicted', three: 'uncertain' };

for (const observations of [[], ['one', 'two', 'three']]) {
  test(`distributed authoring example: ${observations.length} inputs, scripted semantics`, async () => {
    const model = scriptedModel(opening => opening.includes('Assess every observation') ?
      'const assessments = await Promise.all(observations.map(observation => assess(observation, criterion)));\nresult = await summarize(assessments)' :
      `result = { verdict: ${JSON.stringify(decisions)}[observation], reason: "Scripted wiring fixture" }`);
    const review = loadNatlang(path);
    const report = await createNatlangRuntime({ model: model.driver }).run(() => review(observations, 'Fixture criterion'));
    assert.equal(report.assessments.length, observations.length);
    assert.deepEqual(['supported', 'contradicted', 'uncertain'].map(key => report[key]), observations.length ? [1, 1, 1] : [0, 0, 0]);
  });
}
