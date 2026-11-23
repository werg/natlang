import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createNatlangRuntime, loadNatlang } from '../dist/index.js';
import { scriptedModel } from './support/natlang.mjs';

const path = fileURLToPath(new URL('../../skills/natlang-authoring/assets/review/review.nl', import.meta.url));
const decisions = { one: 'supported', two: 'contradicted', three: 'uncertain' };

for (const observations of [[], ['one', 'two', 'three']]) {
  test(`distributed authoring example: ${observations.length} inputs, scripted semantics`, async () => {
    // Ordinary stages use scripted eval; the finite verdict uses the actual tool-free JSON decision surface.
    const model = scriptedModel(opening => opening.includes('Call summarize(observations, criterion)') ?
      'const tally = await summarize(observations, criterion);\nreturn { ...tally, conclusion: tally.contradicted ? "The criterion is contradicted by observation two." : "No evidence contradicts the criterion." };' :
      opening.includes('Decide the verdict with verdict') ? 'return { verdict: await verdict(observation, criterion), reason: "Scripted wiring fixture" }' :
      `return ${JSON.stringify(decisions)}[observation]`);
    const review = loadNatlang(path);
    const driver = request => {
      if (request.tools.length) return model.driver(request);
      const wire = request.messages.filter(message => message.role === 'tool').map(message =>
        typeof message.content === 'string' ? message.content : JSON.stringify(message.content)).join('\n');
      const observation = Object.keys(decisions).find(value => new RegExp('\\b' + value + '\\b').test(wire));
      assert.ok(observation, `the decision opening must carry its actual observation: ${wire}`);
      return { text: JSON.stringify(decisions[observation]), calls: [] };
    };
    driver.decide = async request => {
      const all = request.messages.map(m => typeof m.content === 'string' ? m.content : JSON.stringify(m.content)).join('\n');
      if (/Property: the value is/.test(all)) return { log_probs: [Math.log(0.95), Math.log(0.05)] };
      const wire = request.messages.filter(m => m.role === 'tool').map(m => typeof m.content === 'string' ? m.content : JSON.stringify(m.content)).join('\n');
      const observation = Object.keys(decisions).find(value => new RegExp('\\b' + value + '\\b').test(wire));
      const expected = decisions[observation];
      return { log_probs: request.options.map(option => Math.log(JSON.parse(option) === expected ? 0.9 : 0.05)) };
    };
    const report = await createNatlangRuntime({ model: driver }).run(() => review(observations, 'Fixture criterion'));
    assert.equal(report.assessments.length, observations.length);
    assert.deepEqual(['supported', 'contradicted', 'uncertain'].map(key => report[key]), observations.length ? [1, 1, 1] : [0, 0, 0]);
    assert.equal(typeof report.conclusion, 'string');
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
