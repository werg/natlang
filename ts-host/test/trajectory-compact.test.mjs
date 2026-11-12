import assert from 'node:assert/strict';
import { test } from 'node:test';
import { compactCaptures } from '../dist/runtime/trajectory.js';

const call = (invocation, ids, sampled) => ({ phase: 'action', invocation_id: invocation, model_response: { rl_sampling: {
  schema: 's', prompt: [{ kind: 'tokens', token_ids: ids, producing: ids.map(() => false) }],
  samples: sampled.map((token, index) => ({ index, token_id: token, logp: -0.5 * index, input_role: 'ordinary',
    terminal: index === sampled.length - 1, query_own_position: ids.length + index })),
  generated_inputs: sampled.slice(0, -1).map((token, index) => ({ sample_index: index, token_id: token, input_role: 'ordinary' })) } } });

test('compactCaptures stores prompt deltas against the longest earlier prefix and ordinary samples as columns', () => {
  const system = Array.from({ length: 100 }, (_, i) => i);
  const first = call('a', [...system, 500], [7, 8, 9]);
  const second = call('a', [...system, 500, 7, 8, 9, 600], [10, 11]);
  const short = call('b', [1, 2, 3], [4]);
  const trajectory = [first, second, short];
  const compact = compactCaptures(trajectory);
  assert.deepEqual(trajectory[1], second, 'input is not modified');
  assert.deepEqual(compact[0].model_response.rl_sampling.prompt, first.model_response.rl_sampling.prompt);
  assert.deepEqual(compact[1].model_response.rl_sampling.prompt,
    [{ kind: 'tokens_delta', turn: 0, keep: 101, token_ids: [7, 8, 9, 600], producing_at: [] }]);
  assert.deepEqual(compact[1].model_response.rl_sampling.samples_columns,
    { token_id: [10, 11], logp: [-0, -0.5], query_own_start: 105, terminal_at: [1], generated_inputs: 1 });
  assert.equal(compact[1].model_response.rl_sampling.samples, undefined);
  assert.deepEqual(compact[2].model_response.rl_sampling.prompt, short.model_response.rl_sampling.prompt, 'short prefix stays whole');
  const topK = call('c', [1], [2]);
  topK.model_response.rl_sampling.samples[0].top_k = { ids: 'AA==', logps: 'AA==' };
  assert.deepEqual(compactCaptures([topK])[0], topK, 'top-k receipts stay as recorded');
});
