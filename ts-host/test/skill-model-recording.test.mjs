import test from 'node:test';
import assert from 'node:assert/strict';
import { recordingModelDriver } from '../scripts/skills/record-model-turn.mjs';
import { chatCompletionModelTurn } from '../dist/model/chat-completion.js';

const request = () => ({ messages: [{ role: 'user', content: 'Return seven' }],
  tools: [{ type: 'function', function: { name: 'eval', parameters: {
    type: 'object', properties: { code: { type: 'string' } }, required: ['code'] } } }],
  seed: 1, max_tokens: 100 });

test('record effective synthesized calls rather than inferring them from wire tools', async () => {
  const records = [], input = request();
  const drive = recordingModelDriver({ record: row => records.push(row),
    createDriver: onExchange => chatCompletionModelTurn(async () => ({ choices: [{
      finish_reason: 'stop', message: { content: '{"code":"return 7"}' } }] }), { onExchange }) });
  const turn = await drive(input);
  assert.deepEqual(records[0].turn.calls, [['eval', { code: 'return 7' }]]);
  assert.deepEqual(records[0].turn, turn);
  assert.equal(records[0].wireExchanges[0].wireResponse.choices[0].message.tool_calls, undefined);
  input.messages.push({ role: 'assistant', content: 'later' }); turn.calls[0][1].code = 'return 99';
  assert.equal(records[0].request.messages.length, 1);
  assert.equal(records[0].turn.calls[0][1].code, 'return 7');
});

test('retry attempts remain wire evidence, with one normalized aliased runtime turn', async () => {
  const records = [], wires = []; let attempts = 0;
  const drive = recordingModelDriver({ record: row => records.push(row), recordWire: row => wires.push(row),
    createDriver: onExchange => chatCompletionModelTurn(async () => ({
      usage: { prompt_tokens: 10, completion_tokens: 5 }, choices: [{ finish_reason: 'tool_calls',
        message: { content: '', tool_calls: [{ id: 'a', type: 'function', function: {
          name: 'execute', arguments: ++attempts === 1 ? '{' : '{"code":"return 7"}' } }] } }] }),
    { onExchange, toolAliases: { eval: 'execute' } }) });
  await drive(request());
  assert.equal(records.length, 1); assert.equal(wires.length, 2);
  assert.equal(records[0].wireExchanges.length, 2);
  assert.deepEqual(records[0].turn.calls, [['eval', { code: 'return 7' }]]);
  assert.equal(records[0].turn.prompt_tokens, 20);
  assert.equal(records[0].turn.completion_tokens, 10);
});

test('failed turns preserve wire diagnostics without inventing an effective turn', async () => {
  const records = [], wires = [];
  const drive = recordingModelDriver({ record: row => records.push(row), recordWire: row => wires.push(row),
    createDriver: onExchange => async req => {
      await onExchange({ request: req, wireResponse: { failure: 'malformed' } });
      throw Error('bad response');
    } });
  await assert.rejects(drive(request()), /bad response/);
  assert.equal(records.length, 0); assert.equal(wires.length, 1);
});

test('concurrent invocations keep their wire evidence separate', async () => {
  const records = []; let release;
  const firstRecorded = new Promise(resolve => { release = resolve; });
  const drive = recordingModelDriver({ record: row => records.push(row),
    createDriver: onExchange => async req => {
      const id = req.invocation_id;
      await onExchange({ request: req, wireResponse: { id } });
      if (id === 'first') { release(); await Promise.resolve(); }
      else await firstRecorded;
      return { text: id };
    } });
  await Promise.all(['first', 'second'].map(invocation_id => drive({ ...request(), invocation_id })));
  assert.equal(records.length, 2);
  for (const row of records) {
    assert.equal(row.wireExchanges.length, 1);
    assert.equal(row.wireExchanges[0].wireResponse.id, row.request.invocation_id);
    assert.equal(row.turn.text, row.request.invocation_id);
  }
});

test('actual nested socket failures are recorded separately and rethrown unchanged', async () => {
  const records = [], failures = [];
  const error = new Error('fetch failed', { cause: Object.assign(new Error('other side closed'), {code:'UND_ERR_SOCKET'}) });
  const drive = recordingModelDriver({ record: row => records.push(row), recordFailure: row => failures.push(row),
    createDriver: () => async () => { throw error; } });
  await assert.rejects(drive(request()), caught => caught === error);
  assert.equal(records.length, 0);
  assert.equal(failures[0].diagnostic.retryable, true);
  assert.equal(failures[0].diagnostic.causes[1].code, 'UND_ERR_SOCKET');
  assert.equal(failures[0].turn, undefined);
});

test('semantic error text cannot masquerade as typed transport failure; stops are not retried', async () => {
  const {modelFailureDiagnostic} = await import('../scripts/skills/record-model-turn.mjs');
  assert.equal(modelFailureDiagnostic(Error('HTTP 429 UND_ERR_SOCKET')).retryable, false);
  assert.equal(modelFailureDiagnostic(Object.assign(Error('busy'), {status:429})).retryable, true);
  const controller = new AbortController(); controller.abort();
  assert.equal(modelFailureDiagnostic(Object.assign(Error('socket'), {code:'UND_ERR_SOCKET'}), controller.signal).retryable, false);
});

test('absorbed transport errors cannot become semantic negatives or positive demonstrations', async () => {
  const {withModelFailures} = await import('../scripts/skills/record-model-turn.mjs');
  const failures = [{diagnostic:{retryable:true}}];
  for (const disposition of ['incomplete', 'evaluated']) {
    const result = {disposition, positive:disposition === 'evaluated', query:{effect:1}};
    const classified = withModelFailures(result, failures);
    assert.equal(classified.disposition, 'provider_failure');
    assert.equal(classified.positive, false);
    assert.equal(classified.originalDisposition, disposition);
    assert.deepEqual(classified.query, result.query);
    assert.equal(withModelFailures(result, [], undefined), result);
    assert.equal(withModelFailures(result, failures, {aborted:true}), result);
  }
});
