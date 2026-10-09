import test from 'node:test';
import assert from 'node:assert/strict';
import { chmod, lstat, mkdtemp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { stagedImprover } from './support/improver.mjs';
import { exportCollection } from '../scripts/skills/export-training.mjs';
import { stagePublication } from '../scripts/skills/stage-training-publication.mjs';
import { recordedDriver } from '../scripts/self-improvement/replay-followup-study.mjs';

const sha = bytes => createHash('sha256').update(bytes).digest('hex');
async function makeTreeWritable(path) {
  const stat = await lstat(path);
  if (stat.isSymbolicLink()) return;
  await chmod(path, stat.isDirectory() ? 0o700 : 0o600);
  if (stat.isDirectory()) for (const name of await readdir(path)) await makeTreeWritable(join(path, name));
}
function scriptedModel(respond) {
  const openings = [];
  const driver = async ({ messages }) => {
    const opening = [String(messages[1]?.content), ...messages.slice(2).flatMap(message =>
      message.role === 'assistant' ? (message.tool_calls ?? []).filter(call => String(call.id).startsWith('scope_'))
        .map(call => JSON.parse(call.function.arguments).code ?? '') :
      message.role === 'tool' && String(message.tool_call_id).startsWith('scope_') ? [String(message.content)] : [])].join('\n');
    const turns = messages.filter(message => message.role === 'assistant' &&
      !((message.tool_calls ?? []).some(call => String(call.id).startsWith('scope_')))).length;
    if (turns === 0) {
      openings.push(opening);
      const code = await respond(opening);
      return code === null ? { calls: [['return_result', { status: 'failed', reason: 'Fixture has no response.' }]] } :
        { calls: [['eval', { code }]] };
    }
    const last = messages.at(-1);
    if (last.role === 'tool' && /^(?:rejected|error)|\nerror|Nothing else from this eval was kept/.test(String(last.content)))
      return { calls: [['return_result', { status: 'failed', reason: `Fixture eval failed: ${String(last.content).slice(0, 300)}` }]] };
    return { text: 'done' };
  };
  return { driver, openings };
}
const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const parentRuntimePath = join(repo, 'runs/dgx-development-generated/crisp-skill-self-improvement-20261004/pop-runtime-v3-21b7439');

function episode() {
  const baselineSkill = '---\nname: task-procedure\ndescription: Use for unrelated text formatting.\n---\n\nReturn value plus one.\n';
  return { version: 'natlang.skill-episode/1', id: 'positive-authoring-e2e-v1', family: 'increment', split: 'train',
    license: 'project-generated', source_groups: ['support-a', 'support-b', 'query-private-sentinel-9901'],
    target: { kind: 'improvement-case', entry: 'solve.nl', source: { schema: 'test', id: 'increment' },
      files: { 'solve.nl': '---\nargs:\n  value: number\nreturns: number\n---\nIncrement the value.\n' } },
    library: { kind: 'existing', skills: { 'task-procedure': { 'SKILL.md': baselineSkill } } },
    support: { cases: [{ id: 'support-a1', group: 'support-a', args: [1], expected: 2 },
      { id: 'support-b1', group: 'support-b', args: [2], expected: 3 }] },
    query: { cases: [{ id: 'query-private-sentinel-9901', group: 'query-private-sentinel-9901', args: [17], expected: 18 }] },
    operations: ['create', 'revise'], limits: { maxSteps: 2 }, provenance: { selection_design: 'metadata-tuning' } };
}

function recordingDriver(model, exchanges) {
  let serial = 0;
  return async (request, signal) => {
    const response = await model.driver(request, signal);
    const rawCalls = (response.calls ?? []).map(([name, args]) => ({
      id: `fixture-call-${++serial}`, type: 'function', function: { name, arguments: JSON.stringify(args) },
    }));
    // The runtime mutates its conversation arrays after each response; preserve the exact per-turn request.
    const requestSnapshot = structuredClone(request);
    const turn = { ...response, raw_calls: rawCalls };
    exchanges.push({ recording_version: 'natlang.effective-model-turn/1', request: requestSnapshot,
      turn: structuredClone(turn), wireExchanges: [] });
    return turn;
  };
}

function recordedTurns(exchanges) {
  return exchanges.map(exchange => {
    if (exchange.recording_version === 'natlang.effective-model-turn/1')
      return { request: exchange.request, turn: exchange.turn };
    const message = exchange.wireResponse.choices[0].message;
    const calls = (message.tool_calls ?? []).map(call => [call.function.name, JSON.parse(call.function.arguments)]);
    return { request: exchange.request, turn: { calls, text: message.content ?? null,
      reasoning: message.reasoning_content ?? null, raw_calls: message.tool_calls ?? [] } };
  });
}
function differingCaptureFields(left, right) {
  const get = trace => trace?.events?.find(event => event.kind === 'host_capture' && event.capture_kind === 'invocation_input' && event.name === 'request')?.value;
  const a = get(left), b = get(right), pending = [['request', a, b]], result = [];
  while (pending.length) {
    const [path, x, y] = pending.pop();
    if (JSON.stringify(x) === JSON.stringify(y)) continue;
    if (Array.isArray(x) && Array.isArray(y)) {
      if (x.length !== y.length) result.push({ path: `${path}.length`, saved: x.length, replayed: y.length });
      for (let i = 0; i < Math.min(x.length, y.length); i++) pending.push([`${path}[${i}]`, x[i], y[i]]);
    } else if (x && y && typeof x === 'object' && typeof y === 'object' && !Array.isArray(x) && !Array.isArray(y)) {
      for (const key of new Set([...Object.keys(x), ...Object.keys(y)])) pending.push([`${path}.${key}`, x[key], y[key]]);
    } else result.push({ path, saved: sha(Buffer.from(JSON.stringify(x) ?? 'undefined')).slice(0, 12), replayed: sha(Buffer.from(JSON.stringify(y) ?? 'undefined')).slice(0, 12) });
  }
  return result.slice(0, 40);
}

test('positive skill episode exactly replays through ablations, materializer, and staged publication', {
  timeout: 360_000,
}, async t => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-skill-positive-e2e-'));
  const runtimePath = process.env.NATLANG_SKILL_AUTHORING_TEST_RUNTIME ?? join(root, 'runtime');
  if (!process.env.NATLANG_SKILL_AUTHORING_TEST_RUNTIME) {
    try { await readFile(join(parentRuntimePath, 'frozen-runtime.json')); }
    catch { await rm(root, { recursive: true, force: true }); t.skip(`sealed dependency parent unavailable at ${parentRuntimePath}`); return; }
    const prepared = spawnSync('python3', [join(repo, 'scripts/prepare_skill_authoring_runtime.py'),
      '--parent', parentRuntimePath, '--source', join(repo, 'ts-host'), '--output', runtimePath], { encoding: 'utf8' });
    assert.equal(prepared.status, 0, `failed to derive isolated test runtime: ${prepared.stderr || prepared.stdout}`);
  }
  let frozen;
  try { frozen = JSON.parse(await readFile(join(runtimePath, 'frozen-runtime.json'))); }
  catch { await rm(root, { recursive: true, force: true }); t.skip(`sealed authoring runtime not available at ${runtimePath}`); return; }

  const input = join(root, 'episodes.jsonl'), collectionPath = join(root, 'collection');
  const output = join(root, 'export'), episodeRow = episode();
  const episodeBytes = Buffer.from(JSON.stringify(episodeRow) + '\n'); await writeFile(input, episodeBytes);
  const runtime = { path: runtimePath, manifest_sha256: sha(await readFile(join(runtimePath, 'frozen-runtime.json'))) };
  const codePins = {};
  for (const name of ['improvement/skill-authoring.js', 'improvement/program.js', 'improvement/host.js',
    'improvement/source-worker.js', 'improvement/authored-source.js', 'skills/objective.js', 'skills/graded.js',
    'skills/scoring.js', 'skills/registry.js', 'runtime/kernel.js', 'native/agent.js', 'native/prompt.js']) {
    codePins[name] = frozen.files[`dist/${name}`];
    assert.match(codePins[name] ?? '', /^[a-f0-9]{64}$/, `sealed runtime must pin dist/${name}`);
  }
  codePins.collector = frozen.files['scripts/skills/collect-episodes.mjs'];
  assert.match(codePins.collector ?? '', /^[a-f0-9]{64}$/);
  const options = { limit: 1, experiments: 1, ablations: 2, endpoint: 'scripted-fixture', model: 'no-provider',
    episodes: input, out: collectionPath };
  const identity = { runtime, codePins, node: process.version, version: 'natlang.skill-collection/1',
    input_sha256: sha(episodeBytes), episode_ids: [episodeRow.id], options };
  await mkdir(collectionPath);
  await writeFile(join(collectionPath, 'collection.json'), JSON.stringify(identity, null, 2) + '\n');

  const runtimeModule = path => import(pathToFileURL(join(runtimePath, path)).href);
  const { authorSkillEpisode } = await runtimeModule('dist/improvement/skill-authoring.js');
  const authorModel = scriptedModel(stagedImprover({edit:`await folder.file("solve/skills/task-procedure/SKILL.md").writeText(${JSON.stringify('---\nname: task-procedure\ndescription: Use when adding one to a number; not for text formatting.\n---\n\nReturn value plus one.\n')}); return {summary:"clarify applicability",preserves:["support discovery failure"]};`}));
  const targetModel = scriptedModel(opening => opening.includes('Use when adding one') ? 'return value + 1' : 'return value');
  const authorExchanges = [], executorExchanges = [];
  const author = recordingDriver(authorModel, authorExchanges), executor = recordingDriver(targetModel, executorExchanges);
  const directory = await mkdtemp(join(root, 'author-run-'));
  try {
    const result = await authorSkillEpisode({ episode: episodeRow, directory, author, executor,
      executorId: 'scripted-frozen-runtime-e2e', maxExperiments: 1, maxAblations: 2,
      searchBudget: { maxModelCalls: 400, maxRollouts: 160, maxProposals: 2 },
      evaluationBudget: { maxModelCalls: 400, maxRollouts: 160, maxProposals: 0 } });
    assert.equal(result.disposition, 'evaluated', result.search.error);
    assert.equal(result.positive, true, 'positive paired query gate must remain required');
    assert.equal(result.ablations.length, 2);
    assert.ok(result.ablations.every(item => item.paired && item.disposition !== 'incomplete'));
    assert.ok(executorExchanges.length > 0, 'recorded executor stream must include sealed ablation calls');
    const replayAuthor = recordedDriver(recordedTurns(authorExchanges), { compareSeed: false });
    const replayExecutor = recordedDriver(recordedTurns(executorExchanges), { compareSeed: false });
    const replayDirectory = await mkdtemp(join(root, 'direct-replay-'));
    try {
      const replayResult = await authorSkillEpisode({ episode: episodeRow, directory: replayDirectory,
        author: replayAuthor, executor: replayExecutor, executorId: 'scripted-frozen-runtime-e2e',
        maxExperiments: 1, maxAblations: 2,
        searchBudget: { maxModelCalls: 400, maxRollouts: 160, maxProposals: 2 },
        evaluationBudget: { maxModelCalls: 400, maxRollouts: 160, maxProposals: 0 } });
      const replayAudit = { author: replayAuthor.audit(), executor: replayExecutor.audit() };
      const summarizeMismatch = mismatch => {
        if (!mismatch) return null;
        const expected = mismatch.expected?.[0], actual = mismatch.actual;
        const contents = request => (request?.messages ?? []).map(message => `${message.role}:${sha(Buffer.from(String(message.content ?? '').replace(/task-\d+-[a-z0-9]+/g, 'task-ID'))).slice(0, 10)}:${String(message.content ?? '').length}`);
        return { actual: { id: actual?.invocation_id, messages: contents(actual), tools: sha(Buffer.from(JSON.stringify(actual?.tools ?? []))) },
          expected: { id: expected?.invocation_id, messages: contents(expected), tools: sha(Buffer.from(JSON.stringify(expected?.tools ?? []))) } };
      };
      const replayDiagnostic = JSON.stringify({ replayAudit, searchError: replayResult.search?.error,
        authorMismatch: summarizeMismatch(replayAuthor.mismatches?.[0]),
        executorMismatch: summarizeMismatch(replayExecutor.mismatches?.[0]),
        captureDifference: differingCaptureFields(result.traces.find(trace => trace.events?.some(event => event.kind === 'host_capture' && event.capture_kind === 'invocation_input')),
          replayResult.traces.find(trace => trace.events?.some(event => event.kind === 'host_capture' && event.capture_kind === 'invocation_input'))) });
      const originalChild = result.traces.find(trace => trace.events?.some(event => event.kind === 'host_capture' && event.capture_kind === 'invocation_input'));
      const replayChild = replayResult.traces.find(trace => trace.events?.some(event => event.kind === 'host_capture' && event.capture_kind === 'invocation_input'));
      assert.deepEqual(differingCaptureFields(originalChild, replayChild), [], `child capture differs: ${replayDiagnostic}`);
      assert.equal(replayResult.positive, true, replayDiagnostic);
      assert.equal(replayAudit.author.unconsumedRequests, 0, replayDiagnostic);
      assert.equal(replayAudit.executor.unconsumedRequests, 0, replayDiagnostic);
    } finally { await rm(replayDirectory, { recursive: true, force: true }); }
    const episodeDirectory = join(collectionPath, sha(Buffer.from(episodeRow.id)).slice(0, 20));
    await mkdir(episodeDirectory);
    const artifact = { ...result, runtime, codePins, collection_sha256: sha(Buffer.from(JSON.stringify(identity))),
      author_identity: 'scripted-author', executor_identity: 'scripted-frozen-runtime-e2e', authorExchanges,
      executorExchanges, traces: result.traces };
    await writeFile(join(episodeDirectory, 'result.json'), JSON.stringify(artifact, null, 2) + '\n');

    const exported = await exportCollection(collectionPath, output);
    const traceSummary = result.traces.map(trace => ({ id: trace.callId, parent: trace.parentCallId, outcome: trace.outcome,
      definition: trace.events?.find(event => event.kind === 'manifest')?.definition_source,
      captures: trace.events?.filter(event => event.kind === 'host_capture'),
      initialRequest: (() => { const request = trace.events?.find(event => event.kind === 'state' && event.phase === 'initial')?.value?.$lambda?.args?.request;
        return request && { keys: Object.keys(request), sourceFilesIsArray: Array.isArray(request.sourceFiles),
          sourceFilesCount: request.sourceFiles?.length, preview: request.$diagnostic_preview, complete: request.complete, holder: request.holder }; })(),
      final: trace.events?.findLast(event => event.kind === 'state' && event.phase === 'final')?.value?.$lambda?.return }));
    const state = result.search?.state;
    const searchSummary = { incumbent: state?.incumbent, history: state?.history,
      lastExperiment: state?.lastExperiment && { source: state.lastExperiment.source,
        accepted: state.lastExperiment.accepted, selected: state.lastExperiment.selected,
        sourceFiles: state.lastExperiment.sourceFiles } };
    assert.equal(exported.rows, 2, JSON.stringify({ cases: exported.cases, traceSummary, searchSummary }));
    assert.equal(exported.cases.length, 1);
    assert.equal(exported.cases[0].disposition, 'verified-support-sft');
    assert.equal(exported.cases[0].paired_replay, true);
    const trainingText = await readFile(join(output, 'verified-turns.jsonl'), 'utf8');
    assert.equal(trainingText.includes('query-private-sentinel-9901'), false);
    assert.equal(trainingText.includes('ablation'), false);
    assert.equal(trainingText.includes('Observed query'), false);

    const manifestBytes = await readFile(join(output, 'manifest.json'));
    const evidencePath = join(root, 'source-evidence.txt'); await writeFile(evidencePath, 'fixture source policy evidence');
    const reviewPath = join(root, 'review.json');
    await writeFile(reviewPath, JSON.stringify({ schema: 'natlang.skill-authoring-source-review/1', decision: 'approve',
      reviewer: 'offline-e2e-fixture', candidate_manifest_sha256: sha(manifestBytes),
      source_policy: { decision: 'approved', evidence: [{ path: 'source-evidence.txt', sha256: sha(await readFile(evidencePath)) }] } }));
    const proposal = await stagePublication({ repo: root, exportDirectory: output, reviewPath,
      outputPath: join(root, 'publication-proposal.json') });
    assert.equal(proposal.proposed_registry_entry.rows, 2);
    assert.equal(proposal.status, 'staged_for_independent_review');
  } finally {
    await rm(directory, { recursive: true, force: true });
    if (!process.env.NATLANG_SKILL_AUTHORING_TEST_RUNTIME) await makeTreeWritable(runtimePath);
    await rm(root, { recursive: true, force: true });
  }
});
