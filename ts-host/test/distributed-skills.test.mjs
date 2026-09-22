import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { NatlangHost } from '../dist/index.js';

const path = fileURLToPath(new URL('../../skills/natlang-authoring/assets/review/review.nl', import.meta.url));
for (const observations of [[], ['one', 'two', 'three']]) {
  test(`distributed authoring example: ${observations.length} inputs, scripted semantics`, async () => {
    const host = new NatlangHost();
    const decisions = { one: 'supported', two: 'contradicted', three: 'uncertain' };
    let child = 0;
    try {
      const result = await host.run({ source: { kind: 'file', path },
        inputs: { observations, criterion: 'Fixture criterion' }, modelTurn: request => {
          const root = request.tools.some(tool => tool.function.name === 'read_function');
          const worked = request.messages.some(message => message.role === 'assistant' &&
            message.tool_calls?.some(call => call.function.name === 'eval'));
          if (root) {
            if (worked) return { calls: [['mark_lines', { start: 2, end: 4 }]] };
            return { calls: [['eval', { code: 'const assessments = await Promise.all(observations.map(observation => assess(observation, criterion))); await summarize(assessments)' }]] };
          }
          if (worked) return { calls: [['mark_lines', { start: 1, end: 4 }]] };
          return { calls: [['eval', { code: `({ verdict: ${JSON.stringify(decisions[observations[child++]])}, reason: "Scripted wiring fixture" })` }]] };
        } });
      assert.equal(result.outcome.kind, 'done', result.outcome.detail);
      assert.equal(result.value.assessments.length, observations.length);
      assert.deepEqual(['supported', 'contradicted', 'uncertain'].map(k => result.value[k]), observations.length ? [1, 1, 1] : [0, 0, 0]);
    } finally { host.close(); }
  });
}
