import assert from 'node:assert/strict';
import { test } from 'node:test';
import { modelOracleJudge } from '../dist/teacher/model-judge.js';
import { collectBatch, defaultSystemPrompt, defaultToolSurfaceHash, expectedProvenance,
  nativeJobRunner } from '../dist/teacher/collector.js';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'node:http';

test('judge requests an independent structured grade and validates it', async () => {
  let request;
  const judge = modelOracleJudge(async value => {
    request = value;
    return { calls: [['grade', { accepted: true, verdict: 'The required fact is present.' }]] };
  });
  assert.deepEqual(await judge({ actual: 'Paris', expected: 'Paris', rubric: 'Name the city.' }),
    { accepted: true, verdict: 'The required fact is present.' });
  assert.equal(request.tool_choice, 'required');
  assert.equal(request.temperature, 0);
  assert.deepEqual(JSON.parse(request.messages[1].content),
    { rubric: 'Name the city.', reference: 'Paris', candidate: 'Paris' });
  await assert.rejects(modelOracleJudge(async () => ({ calls: [['grade', { accepted: 'yes', verdict: '' }]] }))
    ({ actual: '', expected: '', rubric: '' }), /valid grade/);
});

test('judge identity is included in resumable job provenance', () => {
  const record = { version: 'natlang.program/2', id: 'judge-fixture', kind: 'lambda_source',
    semantics: { root: 'main.nl', files: { 'main.nl': '---\nargs: {}\nreturns: string\n---\nAnswer.\n' },
      inputs: {}, expected: 'ok' } };
  const config = { modelId: 'teacher', toolSurfaceSha256: 'surface', rootSeed: 1,
    systemPrompt: 'prompt', contextTokens: 16384,
    judgeModel: { modelId: 'judge-v1', provider: 'provider', piOptions: { reasoningEffort: 'low' } } };
  assert.deepEqual(expectedProvenance(record, config).judge,
    { model: 'judge-v1', transport: 'pi-provider', provider: 'provider',
      pi_options: { reasoningEffort: 'low' }, stream_observation: {version:'pi-stream-observation/1',detail:'aggregate-delta-counts',watchdog_refresh:false}, provider_cleanup: { timeout_ms: 15000, scope: 'session-close-best-effort' } });
});

test('native collector sends judged oracle outcomes to the separate model', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'natlang-judge-'));
  const seen = [];
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8');
    request.on('data', chunk => { body += chunk; });
    request.on('end', () => {
      const call = JSON.parse(body);
      seen.push(call);
      const judge = call.model === 'judge';
      const staged = call.messages.some(message => message.role === 'tool' &&
        String(message.content).includes('Staged'));
      const answer = judge ? ['grade', { accepted: true, verdict: 'The answer meets the rubric.' }] :
        staged ? null : ['eval', { code: 'return 1;' }];
      const message = answer ? { role: 'assistant', content: '', tool_calls: [{ id: 'c1', type: 'function',
        function: { name: answer[0], arguments: JSON.stringify(answer[1]) } }] } :
        { role: 'assistant', content: '' };
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ choices: [{ message }], usage: { prompt_tokens: 10, completion_tokens: 4 } }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const endpoint = `http://127.0.0.1:${server.address().port}`;
    const record = { version: 'natlang.program/2', id: 'judged-case', kind: 'lambda_source',
      source: 'fixture', split: 'test', source_ids: ['judged-case'], source_groups: ['judged-case'],
      license: 'test', semantics: { root: 'one.nl',
        files: { 'one.nl': '---\nargs: {}\nreturns: number\n---\nReturn one.\n' },
        inputs: {}, expected: 1, operation: 'exact', oracle: { level: 'judged', rubric: 'Return one.' } } };
    const options = { jobs: join(dir, 'jobs'), output: join(dir, 'results.jsonl'), workers: 1,
      modelId: 'teacher', rootSeed: 7, systemPrompt: defaultSystemPrompt, contextTokens: 16384,
      toolSurfaceSha256: await defaultToolSurfaceHash(), endpoint,
      judgeModel: { modelId: 'judge', endpoint }, transportRetries: 1, retryDelayMs: 0 };
    const result = await collectBatch([{ index: 0, record }], options, nativeJobRunner(options));
    assert.equal(result.completed, 1);
    const row = JSON.parse((await readFile(options.output, 'utf8')).trim());
    assert.equal(row.outcome.accepted, true);
    assert.deepEqual(row.outcome.oracle,
      { accepted: true, level: 'judged', verdict: 'The answer meets the rubric.' });
    const judgeRequest = seen.find(call => call.model === 'judge');
    assert.equal(judgeRequest.tool_choice, 'required');
    assert.equal(JSON.parse(judgeRequest.messages[1].content).rubric, 'Return one.');
  } finally { await new Promise(resolve => server.close(resolve)); }
});
