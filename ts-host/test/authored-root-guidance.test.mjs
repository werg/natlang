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
    assert.equal(row.outcome.oracle?.accepted, true, 'retain exact-oracle attestation provisionally');
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
    assert.ok(converted.authored_actions[0].actions[0].outcome, 'retain the authored root action outcome');
    assert.equal(converted.authored_actions[0].outcome.accepted, true, 'keep answer attestation separate from action admission');
    assert.equal(converted.authored_actions[0].training_admission.approved, false);
    assert.ok(converted.turns.every(turn => turn.source_ref.invocation_id !== row.trajectory[0].invocation_id));
    assert.ok(converted.turns.every(turn => turn.collection_guidance.root_action.sampled === false));
    assert.ok(converted.turns.every(turn => turn.training_admission.approved === false));
    assert.ok(converted.turns.every(turn => turn.outcome.accepted === true && turn.outcome.oracle?.accepted === true),
      'retain exact-oracle attestation while collection-review holds block admission');
    assert.ok(converted.turns.every(turn => turn.trace_admission.admitted === false));
    assert.ok(converted.turns.every(turn => turn.training_admission.kind === 'authored-root-guided-pending-review'));
  } finally { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
});

test('authored root rejection stops sampled root recovery while retaining provider child evidence', async () => {
  const failureCode = [
    "const seed: Neuralese<string> = await nl<Neuralese<string>>`Write one short starting note.`();",
    "throw new Error('fixture authored root rejection');",
  ].join('\n');
  const failureRecord = structuredClone(record);
  failureRecord.id = 'fixture:authored-root-guidance-rejection';
  failureRecord.curriculum.trajectory_contract = { version: 'natlang.trajectory_contract/1', authored_root_eval: true,
    authored_root_sha256: createHash('sha256').update(failureCode).digest('hex'), min_child_invocations: 1,
    min_state_edges_from_children: 0, min_iteration_steps: 1, required_child_roles: [] };
  failureRecord.curriculum.reference.root = [['eval', { code: failureCode }]];

  const dir = await mkdtemp(join(tmpdir(), 'authored-root-rejection-'));
  await writeFile(join(dir, 'source.cases.jsonl'), JSON.stringify(failureRecord) + '\n');
  let requests = 0, planRequests = 0, actionRequests = 0, sawNested = false;
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8'); request.on('data', chunk => { body += chunk; });
    request.on('end', () => {
      requests++;
      const requestBody = JSON.parse(body);
      sawNested ||= JSON.stringify(requestBody.messages).includes('You are inside this call: nl@');
      const offered = (requestBody.tools ?? []).map(tool => tool.function?.name);
      const isPlan = offered.length === 1 && offered[0] === 'execution_plan';
      if (isPlan) planRequests++; else actionRequests++;
      const call = isPlan ? { name: 'execution_plan', arguments: { plan: 'Return the requested note.' } } :
        { name: 'return_result', arguments: { status: 'success', value: 'Starting note.' } };
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({
        choices: [{ message: { role: 'assistant', content: '', tool_calls: [{
          id: `fixture-${requests}`, type: 'function',
          function: { name: call.name, arguments: JSON.stringify(call.arguments) },
        }] } }],
        usage: { prompt_tokens: 10, completion_tokens: 4 },
      }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const options = { jobs: join(dir, 'jobs'), output: join(dir, 'out.jsonl'), workers: 1, modelId: 'fixture-teacher',
      endpoint: `http://127.0.0.1:${server.address().port}`, rootSeed: 42, systemPrompt: defaultSystemPrompt,
      contextTokens: 8192, toolSurfaceSha256: await defaultToolSurfaceHash(), maxTurns: 20,
      executionPlans: true, textNeuraleseEmulation: true };
    const collected = await collectBatch([{ index: 0, record: failureRecord }], options, nativeJobRunner(options));
    assert.equal(collected.completed, 1);
    const row = JSON.parse((await readFile(options.output, 'utf8')).trim());
    assert.equal(row.outcome.status, 'quiesced');
    assert.equal(row.outcome.accepted, false);
    assert.equal(row.collection_guidance.root_action.status, 'did_not_finish');
    assert.equal(row.collection_guidance.root_action.recovery, 'stopped_before_sampled_root_continuation');
    assert.equal(row.trajectory_review.answer_accepted, false);
    assert.equal(row.trajectory_review.observed_child_invocations, 1);
    assert.equal(row.trajectory_review.topology_qualified, false);
    const rootId = row.trajectory[0].invocation_id;
    assert.equal(row.trajectory.filter(turn => turn.invocation_id === rootId).length, 1,
      'only the authored root frame is recorded; no sampled root recovery frame is sent');
    assert.equal(row.trajectory[0].action_provenance.sampled, false);
    assert.ok(row.trajectory.some(turn => turn.invocation_id !== rootId), 'keep the completed sampled child turn');
    assert.equal(requests, 2, 'execution plan and action were requested only for the nested child');
    assert.equal(planRequests, 1);
    assert.equal(actionRequests, 1);
    assert.equal(sawNested, true);
  } finally { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
});

const fullAuthoredSource = process.env.NATLANG_FULL_AUTHORED_SOURCE;
test('full authored root source runs typed child and soft-note paths through planned fake-provider turns',
  { skip: !fullAuthoredSource && 'set NATLANG_FULL_AUTHORED_SOURCE to the pinned one-case source-v2 JSONL' }, async () => {
  const fullRecord = JSON.parse((await readFile(resolve(fullAuthoredSource), 'utf8')).trim());
  const files = fullRecord.semantics.folder_files;
  const task = JSON.parse(files['task.json']);
  const reviewText = files[task.passes[0].evidence_path];
  const authText = files[task.passes[3].evidence_path];
  const scoreFacts = new Map();
  const eligibilityFacts = new Map();
  let requests = 0, planRequests = 0, actionRequests = 0;
  const finalExpected = () => {
    const eligible = [...eligibilityFacts.values()].filter(item => item.eligible)
      .sort((a, b) => scoreFacts.get(b.candidateId).score - scoreFacts.get(a.candidateId).score ||
        a.candidateId.localeCompare(b.candidateId));
    const selected = eligible[0];
    const reviewId = /review ([A-Z]+-\d+)/.exec(reviewText)?.[1];
    const deadline = /decision deadline is ([^.]+)\./.exec(reviewText)?.[1];
    const signedAt = /disposition on ([^.]+)\./.exec(authText)?.[1];
    const authorized = authText.includes(`Authorized reviewer signed ${reviewId}`) && signedAt === deadline;
    return { reviewId, selectedId: selected?.candidateId ?? 'none',
      eligibleIds: eligible.length ? eligible.map(item => item.candidateId).join('; ') : 'none',
      priorityScore: selected ? String(scoreFacts.get(selected.candidateId).score) : '0',
      decision: selected && authorized ? 'approve' : 'hold' };
  };
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8'); request.on('data', chunk => { body += chunk; });
    request.on('end', () => {
      requests++;
      const requestBody = JSON.parse(body);
      const text = JSON.stringify(requestBody.messages);
      const offered = (requestBody.tools ?? []).map(tool => tool.function?.name);
      const isPlan = offered.length === 1 && offered[0] === 'execution_plan';
      let call;
      if (isPlan) { planRequests++; call = ['execution_plan', { plan: 'Read the stated evidence, return the requested typed result, and preserve the note.' }]; }
      else {
        actionRequests++;
        const candidateId = /(?:priority|eligibility)-(ABD-1[A-D])\.json/.exec(text)?.[1] ?? null;
        if (text.includes('Write a short readable starting note'))
          call = ['return_result', { status: 'success', value: 'The draft is a placeholder; no evidence has been reviewed.' }];
        else if (text.includes('review-notice FileHandle'))
          call = ['return_result', { status: 'success', value: `Review notice: ${reviewText}` }];
        else if (text.includes('single-candidate priority FileHandle')) {
          assert.ok(candidateId, 'priority child receives one candidate file path');
          const fact = JSON.parse(files[`priority-${candidateId}.json`]);
          scoreFacts.set(fact.candidateId, { candidateId: fact.candidateId, score: fact.score });
          call = ['return_result', { status: 'success', value: { candidateId: fact.candidateId, score: fact.score } }];
        } else if (text.includes('four scored candidate records')) {
          assert.equal(scoreFacts.size, task.candidates.length);
          call = ['return_result', { status: 'success', value: `Priority scores: ${[...scoreFacts.values()]
            .map(fact => `${fact.candidateId}=${fact.score}`).join('; ')}. The highest is provisional until eligibility is known.` }];
        } else if (text.includes('single-candidate eligibility FileHandle')) {
          assert.ok(candidateId, 'eligibility child receives one candidate file path');
          const fact = JSON.parse(files[`eligibility-${candidateId}.json`]);
          const eligible = /step-free vehicle/i.test(fact.evidence) && /clearance is current/i.test(fact.evidence) &&
            /ramp inspection passed/i.test(fact.evidence);
          const result = { candidateId: fact.candidateId, eligible, evidence: fact.evidence };
          eligibilityFacts.set(fact.candidateId, result);
          call = ['return_result', { status: 'success', value: result }];
        } else if (text.includes('four candidate eligibility records')) {
          assert.equal(eligibilityFacts.size, task.candidates.length);
          const eligible = [...eligibilityFacts.values()].filter(fact => fact.eligible).map(fact => fact.candidateId);
          call = ['return_result', { status: 'success', value: `Eligibility facts: ${[...eligibilityFacts.values()]
            .map(fact => `${fact.candidateId}=${fact.eligible}`).join('; ')}. Eligible candidates by priority: ${eligible.join('; ')}.` }];
        } else if (text.includes('authorization FileHandle'))
          call = ['return_result', { status: 'success', value: `Authorization record: ${authText}` }];
        else if (text.includes('accumulated evidence note'))
          call = ['return_result', { status: 'success', value: finalExpected() }];
        else assert.fail(`unexpected fake-provider child request ${text.slice(-300)}`);
      }
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({
        choices: [{ message: { role: 'assistant', content: '', tool_calls: [{
          id: `full-${requests}`, type: 'function',
          function: { name: call[0], arguments: JSON.stringify(call[1]) },
        }] } }],
        usage: { prompt_tokens: 20, completion_tokens: 8 },
      }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const dir = process.env.NATLANG_FULL_AUTHORED_ARTIFACT_DIR ? resolve(process.env.NATLANG_FULL_AUTHORED_ARTIFACT_DIR) :
      await mkdtemp(join(tmpdir(), 'authored-root-full-source-'));
    await mkdir(dir, { recursive: true });
    const options = { jobs: join(dir, 'jobs'), output: join(dir, 'out.jsonl'), workers: 1, modelId: 'fixture-teacher',
      endpoint: `http://127.0.0.1:${server.address().port}`, rootSeed: 43, systemPrompt: defaultSystemPrompt,
      contextTokens: 32768, toolSurfaceSha256: await defaultToolSurfaceHash(), maxTurns: 60,
      executionPlans: true, textNeuraleseEmulation: true };
    const collected = await collectBatch([{ index: 0, record: fullRecord }], options, nativeJobRunner(options));
    assert.equal(collected.completed, 1);
    const row = JSON.parse((await readFile(options.output, 'utf8')).trim());
    assert.equal(row.outcome.accepted, true, 'source-visible computed draft matches the independent source oracle');
    assert.equal(row.trajectory_review.observed_child_invocations, 14);
    assert.equal(row.trajectory_review.observed_child_roles.priority_readers.observed, 4);
    assert.equal(row.trajectory_review.observed_child_roles.eligibility_readers.observed, 4);
    assert.ok(row.trajectory_review.observed_state_edges_from_children >= 4);
    assert.ok(row.trajectory_review.observed_iteration_steps >= 6);
    assert.equal(row.trajectory_review.topology_qualified, true);
    assert.equal(planRequests, 14, 'execution_plans are routed for every real nested provider turn');
    assert.equal(actionRequests, 14);
    assert.equal(requests, 28);
    const graph = row.outcome.execution_graph;
    const writes = graph.filter(event => event.kind === 'block_write' && event.result_type === 'Neuralese<string>');
    const reads = graph.filter(event => event.kind === 'block_read');
    const iterationNodes = new Set(graph.filter(event => event.kind === 'iteration_step' && typeof event.node === 'string')
      .map(event => `${event.trace_invocation_id ?? ''}\0${event.node}`));
    const scopeFailures = graph.filter(event => event.kind === 'scope_failure');
    assert.ok(writes.length >= 5, 'soft child final text is converted into typed Neuralese blocks');
    assert.ok(reads.length >= 4, 'later typed child arguments consume the Neuralese notes');
    assert.ok(iterationNodes.size >= 6, 'iteration evidence uses distinct scoped graph nodes');
    assert.equal(scopeFailures.length, 0, 'full-source typed returns and arguments produce no scope failures');
  } finally { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
});
