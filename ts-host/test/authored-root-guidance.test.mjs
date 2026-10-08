import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createServer } from 'node:http';
import { createHash } from 'node:crypto';
import { test } from 'node:test';

const dist = process.env.NATLANG_TEST_DIST;
const collector = dist ? await import(pathToFileURL(join(dist, 'teacher/collector.js'))) :
  await import('../dist/teacher/collector.js');
const { collectBatch, defaultSystemPrompt, defaultToolSurfaceHash, nativeJobRunner } = collector;
const materializer = dist ? await import(pathToFileURL(join(dist, 'teacher/native-materializer.js'))) :
  await import('../dist/teacher/native-materializer.js');

const rootCode = [
  "type Progress = { step: number; notes: Neuralese<string> };",
  "const seed: Neuralese<string> = await nl<Neuralese<string>>`Seed a short note.`();",
  "const revise = async (state: Progress): Promise<Progress> => {",
  "  const notes: Neuralese<string> = await nl.with<Neuralese<string>>({})`Revise the supplied note once.`(state.notes);",
  "  return { step: state.step + 1, notes };",
  "};",
  "const result = await iterateOn(revise, { step: 0, notes: seed }).withLimit({ maxSteps: 1 }).until(state => state.step === 1);",
  "return result.step;",
].join('\n');
const record = {
  version: 'natlang.program/2', id: 'fixture:authored-root-guidance', kind: 'lambda_source', source: 'fixture',
  split: 'train', source_ids: ['fixture:known-world'], source_groups: ['fixture:known-world'], license: 'test',
  curriculum: { trajectory_contract: { version: 'natlang.trajectory_contract/1', authored_root_eval: true,
    authored_root_sha256: createHash('sha256').update(rootCode).digest('hex'),
    min_child_invocations: 2, min_state_edges_from_children: 1, min_iteration_steps: 3,
    required_child_roles: [{ role: 'revision', min_invocations: 1, instruction_contains: 'Revise the supplied note once',
      return_type_contains: 'Neuralese<string>' }] },
    reference: { root: [['eval', { code: rootCode }]] } },
  semantics: { root: 'one.nl', files: { 'one.nl': '---\nargs: {}\nreturns: number\n---\nReturn one.\n' },
    inputs: {}, expected: 1, operation: 'exact' },
};

test('declared authored root eval runs once while child NL turns stay provider sampled and linked', async () => {
  const dir = process.env.NATLANG_FIXTURE_ARTIFACT_DIR ? resolve(process.env.NATLANG_FIXTURE_ARTIFACT_DIR) :
    await mkdtemp(join(tmpdir(), 'authored-root-guidance-'));
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, 'source.cases.jsonl'), JSON.stringify(record) + '\n');
  let requests = 0, sawNestedInvocation = false;
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8'); request.on('data', chunk => { body += chunk; });
    request.on('end', () => {
      requests++;
      const requestBody = JSON.parse(body);
      sawNestedInvocation ||= JSON.stringify(requestBody.messages).includes('You are inside this call: nl@');
      const message = { role: 'assistant', content: '', tool_calls: [{ id: `child-${requests}`, type: 'function',
        function: { name: 'return_result', arguments: JSON.stringify({ status: 'success', value: 1 }) } }] };
      response.writeHead(200, { 'content-type': 'application/json' });
      const value = requests === 1 ? 'seed note' : 'revised note';
      message.tool_calls[0].function.arguments = JSON.stringify({ status: 'success', value });
      response.end(JSON.stringify({ choices: [{ message }], usage: { prompt_tokens: 10, completion_tokens: 4 } }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const item = { index: 0, record };
    const options = { jobs: join(dir, 'jobs'), output: join(dir, 'out.jsonl'), workers: 1, modelId: 'fixture-teacher',
      endpoint: `http://127.0.0.1:${server.address().port}`, rootSeed: 41, systemPrompt: defaultSystemPrompt,
      contextTokens: 8192, toolSurfaceSha256: await defaultToolSurfaceHash(), maxTurns: 20, textNeuraleseEmulation: true };
    const collected = await collectBatch([item], options, nativeJobRunner(options));
    assert.equal(collected.completed, 1);
    const row = JSON.parse((await readFile(options.output, 'utf8')).trim());
    assert.equal(row.outcome.accepted, true);
    assert.equal(requests, 2, 'only the two typed child calls reach the provider');
    assert.equal(sawNestedInvocation, true);
    assert.equal(row.trajectory[0].action_provenance.kind, 'authored_reference_root_eval');
    assert.equal(row.collection_guidance.root_action.sampled, false);
    assert.equal(row.collection_guidance.child_actions.sampled, true);
    assert.equal(row.trajectory_review.answer_accepted, true);
    assert.equal(row.trajectory_review.topology_qualified, true);
    assert.equal(row.trajectory_review.observed_child_invocations, 2);
    assert.equal(row.trajectory_review.observed_child_roles.revision.qualified, true);
    assert.ok(row.trajectory_review.observed_state_edges_from_children >= 1);
    assert.ok(row.trajectory_review.observed_iteration_steps >= 3);
    const converted = materializer.materializeNativeRows([row]);
    assert.equal(converted.acceptedRows, 1);
    assert.equal(converted.turns.length, row.trajectory.length - 1, 'authored root action remains provenance only');
    assert.equal(converted.authored_actions.length, 1);
    assert.equal(converted.authored_actions[0].target.tool_calls[0].function.name, 'eval');
    assert.equal(converted.authored_actions[0].outcome.accepted, true, 'keep answer attestation separate from action admission');
    assert.equal(converted.authored_actions[0].training_admission.approved, false);
    assert.ok(converted.turns.every(turn => turn.source_ref.invocation_id !== row.trajectory[0].invocation_id));
    assert.ok(converted.turns.every(turn => turn.collection_guidance.root_action.sampled === false));
    assert.ok(converted.turns.every(turn => turn.training_admission.approved === false));
    assert.ok(converted.turns.every(turn => turn.trace_admission.admitted === false));
    assert.ok(converted.turns.every(turn => turn.training_admission.kind === 'authored-root-guided-pending-review'));
  } finally { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
});
