import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import test from 'node:test';
import { buildTarget } from '../scripts/vendor-build.mjs';

// The staged `pi` target is pure: no package import, so a host without a compiler or a platform can load it.
test('the pi bundle imports nothing and scores replies', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'natlang-pi-'));
  try {
    await buildTarget('pi', directory);
    const text = await readFile(join(directory, 'natlang-pi.js'), 'utf8');
    assert.equal(/^import\s/m.test(text), false, 'the bundle has a static import');
    assert.equal(/\bimport\s*\(/.test(text), false, 'the bundle has a dynamic import');
    const pi = await import(pathToFileURL(join(directory, 'natlang-pi.js')).href);
    const row = (id, logprob) => ({ [id]: { logprob } });
    // Two replies share their first token; the score is what follows it.
    const scores = pi.scoresFromPromptLogprobs([
      { prompt_logprobs: [null, row('1', -1), row('2', -0.5), row('3', -2)], usage: { prompt_tokens: 4 } },
      { prompt_logprobs: [null, row('1', -1), row('2', -0.5), row('9', -0.25)], usage: { prompt_tokens: 4 } },
    ]);
    assert.deepEqual(scores.log_probs, [-2, -0.25]);
    assert.equal(scores.prompt_tokens, 8);
    assert.deepEqual(pi.promptLogprobRequest([{ role: 'user', content: 'q' }], 'yes').messages.at(-1), { role: 'assistant', content: 'yes' });
    assert.ok(Math.abs(pi.softmax([0, 0]).reduce((a, b) => a + b) - 1) < 1e-12);
    assert.throws(() => pi.scoresFromPromptLogprobs([{}]), /decision-unsupported/);
    assert.throws(() => pi.scoresFromBatchResult({ error: 'no slot' }, 2), /no slot/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
