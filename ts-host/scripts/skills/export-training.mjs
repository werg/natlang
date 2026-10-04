#!/usr/bin/env node
/** Verify recorded skill-authoring editor calls and stage approved support-only SFT rows. */
import { readFile, readdir, mkdir, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import { join, resolve, relative } from 'node:path';
import { pathToFileURL } from 'node:url';
import { tmpdir } from 'node:os';
import { recordedDriver } from '../self-improvement/replay-followup-study.mjs';

const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const jsonSha = value => sha(Buffer.from(JSON.stringify(value)));
const isObject = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const filesObject = files => Object.fromEntries(files.map(file => [file.path, file.text]));
const fileDigest = async path => sha(await readFile(path));

/** The sealed query/transfer verdict is consulted as a boolean gate only. */
export function finalPairGate(artifact) {
  if (artifact?.disposition !== 'evaluated' || artifact.positive !== true) return 'episode_not_positive';
  const query = artifact.query, transfer = artifact.transfer;
  if (!query || query.selected?.gatesPassed !== true || !Number.isFinite(query.effect) || query.effect <= 0)
    return 'final_query_gate_not_positive';
  if (transfer !== null && transfer !== undefined &&
      (transfer.selected?.gatesPassed !== true || !Number.isFinite(transfer.effect) || transfer.effect < 0))
    return 'final_transfer_gate_regressed';
  return null;
}

/** Compare only the replayed gate facts; never place those facts in task contexts or SFT rows. */
export function pairedReplayMatches(saved, replayed) {
  if (replayed?.positive !== true || finalPairGate(replayed) ||
      replayed.identity !== saved.identity || replayed.baseline !== saved.baseline ||
      replayed.selected !== saved.selected || !isDeepStrictEqual(replayed.selectedFiles, saved.selectedFiles)) return false;
  const summary = value => value == null ? null : {
    effect: value.effect, gatesPassed: value.selected?.gatesPassed,
  };
  return isDeepStrictEqual(summary(replayed.query), summary(saved.query)) &&
    isDeepStrictEqual(summary(replayed.transfer), summary(saved.transfer));
}

export function materializeVerifiedTrajectory(row, materializer) {
  const input = structuredClone(row);
  input.outcome.action_ledger = input.outcome_actions;
  delete input.outcome_actions;
  const result = materializer.materializeNativeRows([input]);
  if (result.acceptedRows !== 1 || result.rejectedRows !== 0 || result.unlinked.length ||
      !result.turns.length || result.turns.some(turn => turn.training_admission?.approved !== true))
    throw new Error('general native-teacher materializer did not approve every replayed decision');
  return result.turns;
}

/** Construct the sole task definition the exporter permits: only the support search definition. */
export function supportTaskDefinition(definition, selectedFiles) {
  if (!isObject(definition) || definition.version !== 'natlang.improvement-case/1' ||
      !Array.isArray(definition.cases) || !Array.isArray(definition.sourceGroups) ||
      !isObject(definition.authoredFiles) || typeof definition.authoredDigest !== 'string')
    throw new Error('incomplete support search definition');
  if (definition.cases.length === 0 || definition.cases.some(row => !['train', 'validation'].includes(row.split)))
    throw new Error('search definition contains non-support cases');
  const groups = [...new Set(definition.cases.map(row => row.group))].sort();
  if (groups.length < 2 || !isDeepStrictEqual([...definition.sourceGroups].sort(), groups))
    throw new Error('support source groups do not match support cases');
  if (definition.seed !== 0) throw new Error('unrecognized support seed');
  if (!isObject(selectedFiles) || !Object.values(selectedFiles).every(value => typeof value === 'string'))
    throw new Error('selected candidate files are missing');
  // Copy by allowlist. Episode, query, transfer, ticket and executor data never enter the task.
  return {
    version: 'natlang.program/2', id: `skill-authoring:${definition.id}`, kind: 'directory-reducer',
    family: 'skill-authoring', split: 'train', source_groups: groups,
    source_ids: definition.cases.map(row => row.id), source: 'project-generated', license: 'project-generated',
    semantics: {
      root: 'improveStep/rewriteProgram.nl', files: structuredClone(definition.authoredFiles),
      inputs: {}, expected: null, folder_files: structuredClone(definition.files), expected_files: structuredClone(selectedFiles),
      effects: { kind: 'support-selected-skill-files', support_only: true },
    },
  };
}

export function recordedRequestTurn(exchange) { return responseTurn(exchange); }

function responseTurn(exchange) {
  const message = exchange?.wireResponse?.choices?.[0]?.message;
  if (!isObject(exchange?.request) || !Array.isArray(exchange.request.messages) || !Array.isArray(exchange.request.tools) || !isObject(message))
    throw new Error('malformed recorded author exchange');
  const rawCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  const calls = rawCalls.map((call, index) => {
    const name = call?.function?.name, raw = call?.function?.arguments;
    if (typeof name !== 'string' || typeof raw !== 'string') throw new Error(`malformed raw tool call ${index}`);
    let args;
    try { args = JSON.parse(raw); } catch { throw new Error(`invalid JSON arguments in raw tool call ${index}`); }
    if (!isObject(args)) throw new Error(`tool call ${index} arguments must be an object`);
    return { tool: name, source_tool: name, arguments: args, call_id: call.id ?? null };
  });
  const usage = exchange.wireResponse.usage ?? {};
  return {
    context: structuredClone(exchange.request.messages), tools_offered: structuredClone(exchange.request.tools),
    invocation_id: exchange.request.invocation_id,
    assistant: { content: message.content ?? '', reasoning: message.reasoning_content ?? null, calls,
      raw_calls: structuredClone(rawCalls) },
    model_response: {
      ...(Number.isFinite(usage.prompt_tokens) ? { prompt_tokens: usage.prompt_tokens } : {}),
      ...(Number.isFinite(usage.completion_tokens) ? { completion_tokens: usage.completion_tokens } : {}),
      raw_calls: structuredClone(rawCalls),
    },
  };
}

function childTraceFields(trace) {
  const manifest = trace.events?.find(event => event.kind === 'manifest');
  const initial = trace.events?.find(event => event.kind === 'state' && event.phase === 'initial');
  const final = trace.events?.findLast(event => event.kind === 'state' && event.phase === 'final');
  if (trace.outcome !== 'done' || manifest?.definition_source !== 'improveStep/rewriteProgram.nl' ||
      !initial?.value?.$lambda?.args?.request || !final?.value?.$lambda || !Object.hasOwn(final.value.$lambda, 'return'))
    throw new Error('rewrite invocation did not complete with a replayable request and return');
  return { request: initial.value.$lambda.args.request, value: final.value.$lambda.return };
}

function actionsFor(trace) {
  return (trace.events ?? []).filter(event => event.kind === 'action').map((event, index) => ({
    call_id: trace.callId, seq: event.seq ?? index, name: event.name,
    arguments: structuredClone(event.arguments ?? {}), outcome: event.outcome ?? null,
    result_text: event.result_text ?? null, diagnostics: structuredClone(event.diagnostics ?? []),
  }));
}

function parentSelectionProof(parentTrace, selectedDigest, selectedFiles) {
  const final = parentTrace?.events?.findLast(event => event.kind === 'state' && event.phase === 'final');
  const value = final?.value?.$lambda?.return;
  const experiment = value?.lastExperiment;
  const history = value?.history;
  return !!(value?.incumbent === selectedDigest && Array.isArray(history) && history.some(item => item.source === selectedDigest && item.accepted === true && item.selected === true) &&
    Array.isArray(experiment?.sourceFiles) && isDeepStrictEqual(filesObject(experiment.sourceFiles), selectedFiles));
}

/** Verify physical runtime closure and all compiled/source pins before importing it. */
export async function verifySealedRuntime(descriptor) {
  if (!isObject(descriptor) || typeof descriptor.path !== 'string' || !/^[a-f0-9]{64}$/.test(descriptor.manifest_sha256 ?? ''))
    throw new Error('collection has no sealed physical runtime descriptor');
  const root = resolve(descriptor.path), manifestPath = join(root, 'frozen-runtime.json');
  if (await fileDigest(manifestPath) !== descriptor.manifest_sha256) throw new Error('physical runtime manifest hash mismatch');
  const manifestBytes = await readFile(manifestPath), manifest = JSON.parse(manifestBytes);
  if (manifest.schema !== 'natlang.skill-authoring-runtime/1' || !isObject(manifest.files) || !Array.isArray(manifest.symlinks))
    throw new Error('unrecognized physical runtime manifest');
  for (const [path, expected] of Object.entries(manifest.files)) {
    if (await fileDigest(join(root, path)) !== expected) throw new Error(`sealed runtime file changed: ${path}`);
  }
  for (const entry of manifest.symlinks) {
    const path = join(root, entry.path);
    const { lstat, readlink, realpath } = await import('node:fs/promises');
    if (!(await lstat(path)).isSymbolicLink() || await readlink(path) !== entry.target || !(await realpath(path)).startsWith(root + '/'))
      throw new Error(`sealed runtime symlink changed: ${entry.path}`);
  }
  const foundFiles = new Set(), foundLinks = new Map();
  async function walk(directory, prefix = '') {
    const { readdir, lstat, readlink } = await import('node:fs/promises');
    for (const name of await readdir(directory)) {
      const path = join(directory, name), rel = prefix ? `${prefix}/${name}` : name;
      if (rel === 'frozen-runtime.json') continue;
      const stat = await lstat(path);
      if (stat.isSymbolicLink()) foundLinks.set(rel, await readlink(path));
      else if (stat.isDirectory()) await walk(path, rel);
      else if (stat.isFile()) foundFiles.add(rel);
      else throw new Error(`unsupported filesystem entry in sealed runtime: ${rel}`);
    }
  }
  await walk(root);
  if (!isDeepStrictEqual([...foundFiles].sort(), Object.keys(manifest.files).sort()) ||
      !isDeepStrictEqual([...foundLinks].map(([path, target]) => ({ path, target })).sort((a, b) => a.path.localeCompare(b.path)),
        [...manifest.symlinks].sort((a, b) => a.path.localeCompare(b.path))))
    throw new Error('physical runtime contains missing or unsealed entries');
  return { root, manifest, manifest_sha256: descriptor.manifest_sha256 };
}

function verifyCodePins(artifact, manifest) {
  for (const [name, expected] of Object.entries(artifact.codePins ?? {})) {
    const path = name === 'collector' ? 'scripts/skills/collect-episodes.mjs' : `dist/${name}`;
    if (manifest.files[path] !== expected) throw new Error(`collection source pin differs from physical runtime: ${name}`);
  }
}

function makeTrajectory({ artifact, definition, trace, exchanges, before, after, parentTrace, task }) {
  const turns = exchanges.map(responseTurn);
  if (!turns.length || turns.some(turn => turn.invocation_id !== trace.callId)) throw new Error('author exchange invocation identity mismatch');
  const id = `${task.id}:${sha(Buffer.from(trace.callId)).slice(0, 16)}`;
  return {
    version: 'natlang.teacher_trajectory.native/1', id,
    task: { kind: 'whole_program', program_ir: { ...task, semantics: { ...task.semantics,
      inputs: { request: childTraceFields(trace).request }, expected: childTraceFields(trace).value,
      folder_files: before, expected_files: after } } },
    provenance: {
      collection_role: 'teacher', source_replay: 'skill-authoring-child-replay/1',
      collection_sha256: artifact.collection_sha256, invocation_id: trace.callId,
      parent_invocation_id: trace.parentCallId, support_only: true,
      parent_candidate_verified: parentSelectionProof(parentTrace, artifact.selected, after),
      scoring_identity: definition.scoringIdentity ?? null,
    },
    outcome: { accepted: true, status: 'completed', oracle: 'exact' },
    trajectory: turns,
    outcome_actions: actionsFor(trace),
  };
}

async function replayWholeEpisode({ runtime, collection, artifact, episode }) {
  const module = path => import(pathToFileURL(join(runtime.root, path)).href);
  const { authorSkillEpisode } = await module('dist/improvement/skill-authoring.js');
  const { scoreSkillObjective } = await module('dist/skills/objective.js');
  const authorExchanges = artifact.authorExchanges ?? [], executorExchanges = artifact.executorExchanges ?? [];
  const recordedTurns = rows => rows.map(exchange => ({ request: exchange.request, turn: {
    calls: responseTurn(exchange).assistant.calls.map(call => [call.source_tool, call.arguments]),
    text: responseTurn(exchange).assistant.content, reasoning: responseTurn(exchange).assistant.reasoning,
    raw_calls: responseTurn(exchange).assistant.raw_calls,
    prompt_tokens: responseTurn(exchange).model_response.prompt_tokens,
    completion_tokens: responseTurn(exchange).model_response.completion_tokens,
  } }));
  const author = recordedDriver(recordedTurns(authorExchanges), { compareSeed: false });
  const executor = recordedDriver(recordedTurns(executorExchanges), { compareSeed: false });
  const metric = episode.provenance?.metric;
  if (metric && (metric.schema !== 'natlang.skill-objective/1' ||
      !['knapsack', 'bin-packing', 'weighted-tardiness'].includes(metric.kind))) throw new Error('unrecognized support metric');
  const scoring = metric ? { identity: `natlang.skill-objective/1:${metric.kind}:${artifact.codePins['skills/objective.js']}`,
    score: (row, output) => output.error ? { quality: 0, gates: { completed: false } } :
      scoreSkillObjective(metric.kind, row.args[0], output.value, row.expected) } : undefined;
  const transferMetric = episode.provenance?.transfer_metric;
  const transferScoring = transferMetric ? { identity: `natlang.skill-objective/1:${transferMetric.kind}:${artifact.codePins['skills/objective.js']}`,
    score: (row, output) => output.error ? { quality: 0, gates: { completed: false } } :
      scoreSkillObjective(transferMetric.kind, row.args[0], output.value, row.expected) } : undefined;
  const directory = await mkdtemp(join(tmpdir(), 'natlang-skill-authoring-replay-'));
  try {
    const result = await authorSkillEpisode({ episode, directory, author, executor,
      executorId: artifact.executor_identity, maxExperiments: collection.options.experiments,
      scoring, transferScoring, scoringDescriptor: metric ?? { kind: 'exact-return-and-files' },
      searchBudget: { maxModelCalls: 400, maxRollouts: 160, maxProposals: collection.options.experiments * 2 },
      evaluationBudget: { maxModelCalls: 400, maxRollouts: 160, maxProposals: 0 },
      trace: () => {},
    });
    const audit = { author: author.audit(), executor: executor.audit() };
    if (audit.author.unconsumedRequests || audit.executor.unconsumedRequests || audit.author.providerCalls || audit.executor.providerCalls)
      throw new Error('offline episode replay did not consume every recorded exchange');
    if (!pairedReplayMatches(artifact, result)) throw new Error('offline support plus sealed-pair replay differs from the saved positive result');
    return { audit, searchState: result.search.state };
  } finally { await rm(directory, { recursive: true, force: true }); }
}

async function replayAcceptedChild({ runtime, artifact, trace, exchanges, parentTrace, task }) {
  verifyCodePins(artifact, runtime.manifest);
  const module = path => import(pathToFileURL(join(runtime.root, path)).href);
  const { Folder, createNatlangRuntime } = await module('dist/index.js');
  const { loadVirtualNatlang } = await module('dist/runtime/virtual-project.js');
  const { EVALUATOR_DECLARATION } = await module('dist/improvement/services.js');
  const { request, value: expectedValue } = childTraceFields(trace);
  if (!Array.isArray(request.sourceFiles) || !isObject(request)) throw new Error('rewrite request lacks exact source files');
  const before = filesObject(request.sourceFiles);
  const folder = Folder.fromFiles(before);
  const replayExchanges = exchanges.map(exchange => ({ request: exchange.request, turn: {
    calls: responseTurn(exchange).assistant.calls.map(call => [call.source_tool, call.arguments]),
    text: responseTurn(exchange).assistant.content, reasoning: responseTurn(exchange).assistant.reasoning,
    raw_calls: responseTurn(exchange).assistant.raw_calls,
    prompt_tokens: responseTurn(exchange).model_response.prompt_tokens,
    completion_tokens: responseTurn(exchange).model_response.completion_tokens,
  } }));
  const driver = recordedDriver(replayExchanges, { compareSeed: false });
  const replay = createNatlangRuntime({
    model: { driver, maxTurns: 16, maxTokens: 24000, turnTokens: 2048, maxFailureRepairs: 4, contextTokens: 16384 },
    seed: { mode: 'derived', root: 0 }, codeEdits: 'deny', network: false,
    services: { evaluator: {} }, serviceDeclarations: { evaluator: EVALUATOR_DECLARATION },
    serviceScopes: { evaluator: ['improveStep.nl'] }, signal: AbortSignal.timeout(60000),
  });
  const value = await replay.run(() => folder.apply(loadVirtualNatlang(artifact.searchDefinition.authoredFiles,
    'improveStep/rewriteProgram.nl'), request));
  const after = Object.fromEntries(folder.snapshot().filePaths().map(path => [path, new TextDecoder().decode(folder.readBytesSync(path))]));
  const parentFiles = parentTrace?.events?.findLast(event => event.kind === 'state' && event.phase === 'final')?.value?.$lambda?.return?.lastExperiment?.sourceFiles;
  if (!isDeepStrictEqual(value, expectedValue) || driver.audit().unconsumedRequests !== 0 || driver.audit().requestsReplayed !== exchanges.length)
    throw new Error('recorded rewrite invocation did not replay exactly');
  if (!parentSelectionProof(parentTrace, artifact.selected, after) || !isDeepStrictEqual(after, task.semantics.expected_files) ||
      !Array.isArray(parentFiles)) throw new Error('rewrite effects do not match the measured, accepted support candidate');
  return { row: makeTrajectory({ artifact, definition: artifact.searchDefinition, trace, exchanges, before, after, parentTrace, task }),
    audit: { providerCalls: 0, requestsReplayed: driver.audit().requestsReplayed, selectedDigest: artifact.selected } };
}

function sidecar(record) { return JSON.stringify(record) + '\n'; }

export async function exportCollection(collectionPath, outputPath) {
  const input = resolve(collectionPath), output = resolve(outputPath);
  const collectionBytes = await readFile(join(input, 'collection.json'));
  const collection = JSON.parse(collectionBytes);
  const runtime = await verifySealedRuntime(collection.runtime);
  if (!collection.options?.episodes || !/^[a-f0-9]{64}$/.test(collection.input_sha256 ?? ''))
    throw new Error('collection does not pin its input episode file');
  const episodeBytes = await readFile(collection.options.episodes);
  if (sha(episodeBytes) !== collection.input_sha256) throw new Error('pinned episode input hash mismatch');
  const episodeRows = episodeBytes.toString().split('\n').filter(line => line.trim()).map(JSON.parse)
    .filter(row => row.split === 'train').slice(0, collection.options.limit);
  if (!isDeepStrictEqual(episodeRows.map(row => row.id), collection.episode_ids))
    throw new Error('collection episode list differs from its pinned input');
  const episodesById = new Map(episodeRows.map(row => [row.id, row]));
  await mkdir(output, { recursive: true });
  const acceptedRows = [], negatives = [], cases = [];
  const recordedEpisodes = new Set();
  const entries = (await readdir(input, { withFileTypes: true })).filter(entry => entry.isDirectory()).sort((a, b) => a.name.localeCompare(b.name));
  for (const entry of entries) {
    const resultPath = join(input, entry.name, 'result.json');
    let artifact;
    try { artifact = JSON.parse(await readFile(resultPath, 'utf8')); }
    catch (error) { if (error.code === 'ENOENT') continue; throw error; }
    const artifactSha = await fileDigest(resultPath);
    const reject = reason => {
      negatives.push({ episode: artifact.episode ?? null, artifact_sha256: artifactSha, reason });
      for (const trace of artifact.traces ?? []) negatives.push({ episode: artifact.episode ?? null, artifact_sha256: artifactSha,
        invocation: trace.callId, definition_source: trace.events?.find(event => event.kind === 'manifest')?.definition_source ?? null,
        outcome: trace.outcome, reason: 'attempt-retained-with-quarantined-artifact' });
      cases.push({ episode: artifact.episode ?? null, disposition: 'quarantined', reason });
    };
    try {
      if (!collection.episode_ids?.includes(artifact.episode) || entry.name !== sha(Buffer.from(artifact.episode)).slice(0, 20))
        throw new Error('result path or episode is outside the pinned collection');
      if (recordedEpisodes.has(artifact.episode)) throw new Error('duplicate episode result');
      recordedEpisodes.add(artifact.episode);
      if (artifact.collection_sha256 !== jsonSha(collection)) throw new Error('collection identity mismatch');
      if (!isDeepStrictEqual(artifact.runtime, collection.runtime)) throw new Error('artifact runtime differs from collection runtime');
      verifyCodePins(artifact, runtime.manifest);
      const gate = finalPairGate(artifact);
      if (gate) { reject(gate); continue; }
      const paired = await replayWholeEpisode({ runtime, collection, artifact, episode: episodesById.get(artifact.episode) });
      const definition = artifact.searchDefinition;
      if (!definition || definition.id !== artifact.episode) throw new Error('support search definition identity mismatch');
      const { Folder } = await import(pathToFileURL(join(runtime.root, 'dist/index.js')).href);
      if (Folder.fromFiles(definition.authoredFiles).snapshot().digest !== definition.authoredDigest)
        throw new Error('authored improver files do not match their digest');
      const traces = Array.isArray(artifact.traces) ? artifact.traces : [];
      const byCall = new Map(traces.map(trace => [trace.callId, trace]));
      const eligible = traces.filter(trace => trace.outcome === 'done' &&
        trace.events?.some(event => event.kind === 'manifest' && event.definition_source === 'improveStep/rewriteProgram.nl'));
      const matches = [];
      for (const trace of eligible) {
        const parent = byCall.get(trace.parentCallId);
        if (parentSelectionProof(parent, artifact.selected, artifact.selectedFiles)) matches.push({ trace, parent });
      }
      if (matches.length !== 1) throw new Error(`expected one child invocation linked to accepted selected candidate; found ${matches.length}`);
      const groups = artifact.authorExchanges.filter(exchange => exchange.request?.invocation_id === matches[0].trace.callId);
      if (!groups.length) throw new Error('selected rewrite has no recorded author exchanges');
      const task = supportTaskDefinition(definition, artifact.selectedFiles);
      if (Folder.fromFiles(artifact.selectedFiles).snapshot().digest !== artifact.selected)
        throw new Error('selected candidate files do not match their source digest');
      const verified = await replayAcceptedChild({ runtime, artifact,
        trace: matches[0].trace, exchanges: groups, parentTrace: matches[0].parent, task });
      const row = verified.row;
      const materializer = await import(pathToFileURL(join(runtime.root, 'dist/teacher/native-materializer.js')).href);
      const materialized = materializeVerifiedTrajectory(row, materializer);
      acceptedRows.push(...materialized);
      cases.push({ episode: artifact.episode, disposition: 'verified-support-sft', trace: matches[0].trace.callId,
        turns: materialized.length, requests: verified.audit.requestsReplayed, paired_replay: true,
        paired_replay_requests: paired.audit, providerCalls: 0, artifact_sha256: artifactSha });
      // Failed and superseded author attempts remain visible as non-training metadata, without their answers.
      for (const trace of traces.filter(item => item.callId !== matches[0].trace.callId)) negatives.push({
        episode: artifact.episode, artifact_sha256: artifactSha, invocation: trace.callId,
        definition_source: trace.events?.find(event => event.kind === 'manifest')?.definition_source ?? null,
        reason: 'nonselected-or-unlinked-search-attempt', outcome: trace.outcome,
      });
    } catch (error) { reject(String(error)); }
  }
  for (const episode of collection.episode_ids ?? []) if (!recordedEpisodes.has(episode)) {
    negatives.push({ episode, artifact_sha256: null, reason: 'collection episode has no completed result artifact' });
    cases.push({ episode, disposition: 'missing-result' });
  }
  const turnsText = acceptedRows.map(JSON.stringify).join('\n') + (acceptedRows.length ? '\n' : '');
  const negativesText = negatives.map(sidecar).join('');
  const turnsPath = join(output, 'verified-turns.jsonl'), negativePath = join(output, 'negative-evidence.jsonl');
  await writeFile(turnsPath, turnsText, { flag: 'wx' });
  await writeFile(negativePath, negativesText, { flag: 'wx' });
  const manifest = {
    schema: 'natlang.skill-authoring-training-candidate/1', lane: 'skill-authoring',
    collection_path: relative(process.cwd(), input), collection_sha256: sha(collectionBytes),
    runtime: collection.runtime, runtime_schema: runtime.manifest.schema,
    rows: acceptedRows.length, rows_sha256: sha(Buffer.from(turnsText)),
    negative_attempts: negatives.length, negative_sha256: sha(Buffer.from(negativesText)),
    cases, provider_calls: 0, dpo_pairs: 0,
    policy: 'Only independently replayed selected skill rewrite calls from support-only search contexts. Query/transfer results gate eligibility in memory and are never copied into SFT rows. Failed and superseded attempts are metadata-only negative evidence.',
    registry_entry_file: relative(process.cwd(), join(output, 'registry-entry.json')),
  };
  const manifestPath = join(output, 'manifest.json');
  const manifestBytes = Buffer.from(JSON.stringify(manifest, null, 2) + '\n');
  await writeFile(manifestPath, manifestBytes, { flag: 'wx' });
  const registryEntry = {
    path: relative(process.cwd(), turnsPath), sha256: manifest.rows_sha256, rows: manifest.rows,
    lane: 'skill-authoring', runtime_api: 'native-ordinary-runtime',
    verification_artifact: relative(process.cwd(), manifestPath), verification_sha256: sha(manifestBytes),
  };
  await writeFile(join(output, 'registry-entry.json'), JSON.stringify(registryEntry, null, 2) + '\n', { flag: 'wx' });
  return manifest;
}

if (process.argv[1] === new URL(import.meta.url).pathname) {
  const args = process.argv.slice(2);
  if (args.length !== 4 || args[0] !== '--collection' || args[2] !== '--out')
    throw Error('usage: node scripts/skills/export-training.mjs --collection DIR --out DIR');
  console.log(JSON.stringify(await exportCollection(args[1], args[3]), null, 2));
}
