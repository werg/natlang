import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { NatlangHost } from '../dist/index.js';

const path = fileURLToPath(new URL('../../skills/natlang-authoring/assets/review/review.nl', import.meta.url));
for (const observations of [[], ['one', 'two', 'three']]) {
  test(`distributed authoring example: ${observations.length} inputs, scripted semantics`, async () => {
    const host = new NatlangHost();
    const decisions = { one: 'supported', two: 'contradicted', three: 'uncertain' };
    let child = 0, parent = 0;
    try {
      const result = await host.run({ source: { kind: 'file', path },
        inputs: { observations, criterion: 'Fixture criterion' }, modelTurn: request => {
          const calls = request.tools.some(tool => tool.function.name === 'run_function');
          const worked = request.messages.some(message => message.role === 'assistant' && message.tool_calls?.some(call => call.function.name === 'write_value'));
          if (!calls) return worked ? { calls: [] } : { calls: [['write_value', { destination: 'return', type: 'Assessment',
            value: { verdict: decisions[observations[child++]], reason: 'Scripted wiring fixture' } }]] };
          if (parent++ === 0) return { calls: [['for_each', { function: 'assess', save_as: 'let/assessments',
            items: 'args/observations', inputs: ['args/criterion'] }]] };
          if (parent === 2) return { calls: [['run_function', { function: 'summarize', save_as: 'return',
            inputs: ['let/assessments'] }]] };
          return { calls: [] };
        } });
      assert.equal(result.outcome.kind, 'done', result.outcome.detail);
      assert.equal(result.value.assessments.length, observations.length);
      assert.deepEqual(['supported', 'contradicted', 'uncertain'].map(k => result.value[k]), observations.length ? [1, 1, 1] : [0, 0, 0]);
    } finally { host.close(); }
  });
}
