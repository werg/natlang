import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createServer } from 'node:http';
import { pathToFileURL } from 'node:url';
import { test } from 'node:test';
import { buildCollectorPrograms } from '../../scripts/prepare_paired_guidance_collector_ir.mjs';

const dist = process.env.NATLANG_TEST_DIST;
const collector = dist ? await import(pathToFileURL(join(dist, 'teacher/collector.js'))) :
  await import('../dist/teacher/collector.js');

const options = Array.from({ length: 151 }, (_, index) => `option-${index}`);
const makeRequest = (pairIndex, arm) => ({ protocol: 'One independent inline natural-language function call; return exactly one option string. No shared history between requests.',
  request_id: `pair-${String(pairIndex).padStart(2, '0')}-${arm}`, pair_id: `pair-${String(pairIndex).padStart(2, '0')}`,
  arm, case_id: `source-case-${pairIndex}`,
  call: { capture_schema: { options: 'string[]' }, captures: { options }, input: { state: 'a visible state' },
    instruction_template: 'Question: ${question}\nAnswer from the scoped state.', question_value: 'Which intent?',
    result_type: 'string', signature: 'nl.with<{ options: string[] }, string>({ options })`instruction`(state)' } });

test('collector IR preserves exact typed A/B prompt inputs and hides the evaluator key', () => {
  const requests = [], matchedPairs = [];
  for (let index = 1; index <= 24; index++) {
    requests.push(makeRequest(index, 'A'), makeRequest(index, 'B'));
    matchedPairs.push({ pair_id: `pair-${String(index).padStart(2, '0')}`,
      case_id: `source-case-${index}`, matched_case_id: `unexposed-match-${index}`,
      source_group: `g-original-source-group-${index}` });
  }
  const a = requests[0], b = requests[1];
  b.call.instruction_template += '\nUse the exact options.';
  const programs = buildCollectorPrograms(requests, matchedPairs);
  assert.equal(programs.length, 48);
  const [programA, programB] = programs;
  assert.equal(programA.source_groups[0], 'g-original-source-group-1');
  assert.equal(programB.source_groups[0], 'g-original-source-group-1');
  assert.equal(programA.semantics.inputs.instruction, 'Question: Which intent?\nAnswer from the scoped state.');
  assert.equal(programB.semantics.inputs.instruction, `${programA.semantics.inputs.instruction}\nUse the exact options.`);
  for (const program of programs) {
    assert.equal(program.semantics.inputs.options.length, 151);
    assert.equal(program.semantics.expected, null);
    assert.equal(program.protocol.training_admission, false);
    assert.equal(program.curriculum.trajectory_contract.authored_root_eval, true);
    assert.match(program.curriculum.reference.root[0][1].code, /nl\.with<\{ options: string\[\] \}, string>/);
    assert.doesNotMatch(JSON.stringify(program), /expected-label|gold-label|unexposed-match/);
  }
});

test('collector IR rejects evaluator answers added to model-facing requests', () => {
  const requests = [], matchedPairs = [];
  for (let index = 1; index <= 24; index++) {
    requests.push(makeRequest(index, 'A'), makeRequest(index, 'B'));
    matchedPairs.push({ pair_id: `pair-${String(index).padStart(2, '0')}`,
      case_id: `source-case-${index}`, source_group: `g-source-${index}` });
  }
  const a = requests[0];
  a.expected = 'hidden-answer';
  assert.throws(() => buildCollectorPrograms(requests, matchedPairs), /evaluator material/);
});

test('native collector samples only the fixed typed child and keeps source labels out of provider input', async () => {
  const requests = [], matchedPairs = [];
  for (let index = 1; index <= 24; index++) {
    requests.push(makeRequest(index, 'A'), makeRequest(index, 'B'));
    matchedPairs.push({ pair_id: `pair-${String(index).padStart(2, '0')}`,
      case_id: `source-case-${index}`, source_group: `g-source-${index}` });
  }
  const record = buildCollectorPrograms(requests, matchedPairs)[0];
  const directory = await mkdtemp(join(tmpdir(), 'paired-guidance-collector-'));
  let providerRequests = 0, seenMessages = '';
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8'); request.on('data', chunk => { body += chunk; });
    request.on('end', () => {
      providerRequests++;
      const payload = JSON.parse(body);
      seenMessages = JSON.stringify(payload.messages);
      const message = { role: 'assistant', content: '', tool_calls: [{ id: 'child-result', type: 'function',
        function: { name: 'return_result', arguments: JSON.stringify({ status: 'success', value: 'option-0' }) } }] };
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ choices: [{ message }], usage: { prompt_tokens: 10, completion_tokens: 1 } }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const output = join(directory, 'out.jsonl');
    const options = { jobs: join(directory, 'jobs'), output, workers: 1, modelId: 'fixture-teacher',
      endpoint: `http://127.0.0.1:${server.address().port}`, rootSeed: 7,
      systemPrompt: collector.defaultSystemPrompt, contextTokens: 8192,
      toolSurfaceSha256: await collector.defaultToolSurfaceHash(), maxTurns: 8,
      transportRetries: 0, requestRetries: 0, executionPlans: false };
    const collected = await collector.collectBatch([{ index: 0, record }], options, collector.nativeJobRunner(options));
    const row = JSON.parse((await readFile(output, 'utf8')).trim());
    assert.equal(collected.completed, 1);
    assert.equal(providerRequests, 1, 'only one leaf child reaches the provider; root eval is authored and fixed');
    assert.match(seenMessages, /a visible state/);
    assert.match(seenMessages, /Answer from the scoped state\./);
    assert.match(seenMessages, /option-150/);
    assert.doesNotMatch(seenMessages, /expected|gold|source_group|matched_case_id/);
    assert.equal(row.outcome.value, 'option-0', 'the native collector captures the typed root result');
    const child = row.outcome.invocation_ledger.find(entry => entry.parent_invocation_id);
    assert.equal(child?.host_result?.value, 'option-0', 'the typed child result is preserved in the invocation ledger');
    assert.equal(row.outcome.accepted, false, 'a null expected value never claims correctness or admission');
    assert.equal(row.trajectory.length, 2, 'retain authored root provenance and the one sampled child turn');
    assert.equal(row.trajectory[0].action_provenance?.sampled, false);
    assert.equal(row.trajectory[1].assistant.calls[0].tool, 'return_result');
  } finally {
    await new Promise(resolve => server.close(resolve));
    await rm(directory, { recursive: true, force: true });
  }
});
