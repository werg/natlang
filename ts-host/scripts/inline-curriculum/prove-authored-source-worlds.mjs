#!/usr/bin/env node
/** Replay authored worlds through the collector and preserve native action sidecars for review only. */
import { createHash, randomUUID } from 'node:crypto';
import { link, mkdir, open, readFile, unlink } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { referenceDriver } from '../../dist/teacher/curriculum.js';
import { defaultToolSurfaceHash, expectedProvenance, executeProgram, programRow, programRunId, trajectoryTurn } from '../../dist/teacher/collector.js';
import { openingLength, openingText, text } from '../../dist/teacher/opening.js';
import { materializeNativeRows } from '../../dist/teacher/native-materializer.js';
import { sourceConversionProblems } from '../../dist/teacher/source-conversion.js';
import { MemoryNeuraleseStore, StandInNeuralesePort, hashingEmbedder } from '../../dist/native/neuralese-store.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';

const { values } = parseArgs({ options: { source: { type: 'string' }, out: { type: 'string' } } });
if (!values.source || !values.out) throw new Error('usage: node prove-authored-source-worlds.mjs --source SOURCE_JSONL --out CANDIDATE_DIR');
const sourcePath = resolve(values.source), outPath = resolve(values.out);
const sourceBytes = await readFile(sourcePath), sourceSha = createHash('sha256').update(sourceBytes).digest('hex');
const rows = sourceBytes.toString('utf8').trimEnd().split('\n').map(line => JSON.parse(line));
const toolSurfaceSha256 = await defaultToolSurfaceHash();
const proofCases = [], nativeRows = [], nativeTurns = [], actionReviews = [];
const canonical = value => Array.isArray(value) ? `[${value.map(canonical).join(',')}]` :
  value && typeof value === 'object' ? `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonical(value[key])}`).join(',')}}` : JSON.stringify(value);
const shown = (value, context) => {
  const last = [...context].reverse().find(message => message.role === 'tool');
  if (!last) return false;
  const squash = text => String(text).replace(/["'`\s]/g, ''), wanted = squash(JSON.stringify(value));
  const content = String(last.content ?? ''), staged = /^Staged ([\s\S]*?) as the result\./m.exec(content)?.[1];
  return squash(content.split('\n')[0]) === wanted || (staged !== undefined && squash(staged) === wanted);
};
const openingPlainText = context => context.slice(1, openingLength(context)).map(message => {
  const content = Array.isArray(message.content) ? message.content.map(part =>
    typeof part?.text === 'string' ? part.text : part?.type === 'neuralese' ? `${part.id}` : text(part)).join('') : text(message.content);
  return content + (message.tool_calls ?? []).map(call => call.function?.arguments ?? '').join('\n');
}).join('\n');

for (const [index, record] of rows.entries()) {
  const store = new MemoryNeuraleseStore();
  const port = new StandInNeuralesePort(store, hashingEmbedder(8), 8, 'nd:authored-source-reference@1');
  const reference = referenceDriver(record); reference.neuralese = true;
  const trajectory = [];
  const driver = async request => {
    const response = await reference(request);
    // The deterministic authored reference is the source of these model-call
    // arguments. Record them in provider-shaped form so materializer can bind
    // the original inline body to the host's sentinel-rewritten eval event.
    response.raw_calls = (response.calls ?? []).map(([tool, args]) => ({ function: {
      name: tool, arguments: JSON.stringify(args ?? {}) } }));
    const turn = trajectoryTurn(request, response);
    const [tool, args] = response.calls?.[0] ?? [];
    if (tool === 'return_result' && args?.status === 'success' && !shown(args.value, request.messages))
      turn.assistant.direct_answer = true;
    trajectory.push(turn);
    return response;
  };
  driver.neuralese = true;
  const options = { modelId: 'authored-source-static-reference', rootSeed: 909,
    systemPrompt: TOOLS_PROMPT, contextTokens: 65_536, maxTurns: 80, toolSurfaceSha256,
    collectionRole: 'authored-source-static-reference', neuralese: { store, port } };
  const provenance = { ...expectedProvenance(record, options), collection_role: options.collectionRole,
    synthetic_reasoning: 'action-notes/1', source_review_status: 'candidate-only' };
  const runId = programRunId(index, provenance);
  const run = await executeProgram(record, driver, { ...options, runId });
  const nativeRow = programRow(record, options.modelId, runId, provenance, run, trajectory);
  const result = materializeNativeRows([nativeRow], { directAnswers: true });
  const materializerAudit = { accepted_rows: result.acceptedRows, rejected_rows: result.rejectedRows,
    unlinked: result.unlinked, source_conversion_problems: sourceConversionProblems(nativeRow) };

  const ledger = run.outcome.invocation_ledger ?? [];
  const isIterate = record.curriculum.slice === 'iterate';
  const isDepth2Variant = record.curriculum.variant === 'static-per-folder-nl-reducer-plus-item-semantic-judge/1';
  const root = ledger.find(entry => entry.parent_invocation_id === null);
  if (!root) throw new Error(`${record.id}: no root invocation`);
  const children = ledger.filter(entry => entry.parent_invocation_id === root.invocation_id);
  const events = run.outcome.action_ledger ?? [];
  const failedActions = events.filter(event => !['ok', 'completed'].includes(String(event.outcome ?? '')));
  let childReads;
  let depthAudit = {};
  if (isDepth2Variant) {
    const childReadsByRoot = [];
    const outerIds = new Set(children.map(child => child.invocation_id));
    const itemChildren = ledger.filter(entry => outerIds.has(entry.parent_invocation_id));
    const expectedItems = record.curriculum.reference.children.filter(child =>
      typeof child.match === 'string' && child.match.endsWith('.md') && !child.match.includes('/'));
    const itemByPath = new Map(Object.entries(record.semantics.folder_files).filter(([path]) => path.includes('/items/')));
    if (itemChildren.length !== expectedItems.length || itemChildren.length !== itemByPath.size)
      throw new Error(`${record.id}: depth-2 item call count mismatch (${itemChildren.length}/${expectedItems.length}/${itemByPath.size})`);

    const outerAudit = [];
    for (const reducer of children) {
      const reducerTurn = trajectory.find(turn => turn.invocation_id === reducer.invocation_id);
      if (!reducerTurn) throw new Error(`${record.id}: missing outer reducer opening ${reducer.invocation_id}`);
      const outerText = openingPlainText(reducerTurn.context ?? []);
      const policyEntries = Object.entries(record.semantics.folder_files).filter(([path]) =>
        path.startsWith('teams/') && path.endsWith('/policy.json'));
      const matchedPolicy = policyEntries.filter(([path, text]) => {
        const policy = JSON.parse(text);
        const folder = path.slice(0, -'/policy.json'.length);
        const hasFolderItem = Object.keys(record.semantics.folder_files).some(itemPath =>
          itemPath.startsWith(`${folder}/items/`) && outerText.includes(itemPath.slice(folder.length + 1)));
        const captured = JSON.stringify(policy.inherited_rule).replaceAll('"', '\\"');
        return hasFolderItem && outerText.includes(`const inheritedRule: string = ${captured};`);
      });
      if (matchedPolicy.length !== 1)
        throw new Error(`${record.id}: outer reducer does not have one exact inherited policy snapshot`);
      const teamPath = matchedPolicy[0][0].slice(0, -'/policy.json'.length);
      const localOverride = record.semantics.folder_files[`${teamPath}/override.json`];
      if (!localOverride) throw new Error(`${record.id}: missing local override for ${teamPath}`);
      const expectedLocalRule = JSON.parse(localOverride).local_rule;
      const reducerEvents = events.filter(event => event.call_id === reducer.invocation_id);
      if (reducerEvents.some(event => !['ok', 'completed'].includes(String(event.outcome ?? ''))))
        throw new Error(`${record.id}: outer reducer ${reducer.invocation_id} has unsuccessful actions`);
      const evalAction = reducerEvents.find(event => event.name === 'eval');
      if (!evalAction || !String(evalAction.arguments?.code ?? '').includes("team.file('override.json').readText()") ||
          !String(evalAction.result_text ?? '').includes('Stored local localRule = '))
        throw new Error(`${record.id}: outer reducer did not read and carry its exact local override`);

      const teamItemRows = [...itemByPath].filter(([path]) => path.startsWith(`${teamPath}/items/`));
      const reducerChildren = itemChildren.filter(item => item.parent_invocation_id === reducer.invocation_id);
      if (reducerChildren.length !== teamItemRows.length)
        throw new Error(`${record.id}: ${teamPath} has ${reducerChildren.length} item calls for ${teamItemRows.length} source items`);
      const itemReceipts = [];
      for (const itemChild of reducerChildren) {
        const itemTurn = trajectory.find(turn => turn.invocation_id === itemChild.invocation_id);
        if (!itemTurn) throw new Error(`${record.id}: missing item judge opening ${itemChild.invocation_id}`);
        const itemOpening = openingPlainText(itemTurn.context ?? []);
        const matches = teamItemRows.filter(([, sourceText]) => itemOpening.includes(sourceText));
        const quotedLocalRule = JSON.stringify(expectedLocalRule).replaceAll('"', '\\"');
        if (matches.length !== 1)
          throw new Error(`${record.id}: item judge ${itemChild.invocation_id} does not expose exactly one complete source file`);
        if (!itemOpening.includes(`const localRule: string = ${quotedLocalRule};`))
          throw new Error(`${record.id}: item judge does not receive the exact local override snapshot`);
        const [sourcePath, sourceText] = matches[0];
        const itemId = sourcePath.split('/').at(-1).replace(/\.md$/, '');
        const target = expectedItems.find(candidate => candidate.match === `${itemId}.md`);
        const expectedAnswer = target?.value;
        const childEvents = events.filter(event => event.call_id === itemChild.invocation_id);
        if (childEvents.some(event => !['ok', 'completed'].includes(String(event.outcome ?? ''))))
          throw new Error(`${record.id}: item judge ${itemChild.invocation_id} has unsuccessful actions`);
        const answerEvent = childEvents.find(event => event.name === 'return_result' && event.arguments?.status === 'success');
        const observedAnswer = itemChild.host_result?.value ?? answerEvent?.arguments?.value;
        if (!target || canonical(observedAnswer) !== canonical(expectedAnswer))
          throw new Error(`${record.id}: item judge ${itemId} differs from its authored source reference`);
        itemReceipts.push({ invocation_id: itemChild.invocation_id, source_path: sourcePath,
          source_text_sha256: createHash('sha256').update(sourceText).digest('hex'),
          exact_complete_filehandle_opening: true, expected_answer: expectedAnswer,
          observed_answer: observedAnswer, answer_matches_authored_reference: true,
          successful_actions: childEvents.length });
      }
      childReadsByRoot.push(...itemReceipts);
      const expectedTeam = itemReceipts.filter(receipt => receipt.expected_answer === true)
        .map(receipt => receipt.source_path.split('/').at(-1).replace(/\.md$/, '')).sort();
      const reducerResult = reducer.host_result?.value ?? reducerEvents.find(event =>
        event.name === 'return_result' && event.arguments?.status === 'success')?.arguments?.value;
      if (canonical(reducerResult) !== canonical(expectedTeam))
        throw new Error(`${record.id}: outer reducer result differs from its source-grounded item judgments`);
      outerAudit.push({ invocation_id: reducer.invocation_id, team_path: teamPath,
        inherited_policy_snapshot: true, local_override_read: true,
        expected_result: expectedTeam, observed_result: reducerResult,
        item_judges: itemReceipts.length });
    }
    childReads = childReadsByRoot;
    depthAudit = { depth_layers: 2, outer_reducers: outerAudit.length,
      item_judges: itemChildren.length, outer_reducer_audit: outerAudit };
  } else childReads = children.map(child => {
    const childEvents = events.filter(event => event.call_id === child.invocation_id);
    if (childEvents.some(event => !['ok', 'completed'].includes(String(event.outcome ?? ''))))
      throw new Error(`${record.id}: child ${child.invocation_id} has an unsuccessful action`);
    const reads = childEvents.filter(event => event.name === 'read_file');
    if (reads.length !== 1) throw new Error(`${record.id}: child ${child.invocation_id} has ${reads.length} source reads`);
    const read = reads[0];
    const candidates = Object.entries(record.semantics.folder_files).filter(([path, text]) =>
      String(read.result_text ?? '').includes(text) && (isIterate ? path === 'packet.md' : path.includes('/items/')));
    if (candidates.length !== 1) throw new Error(`${record.id}: child ${child.invocation_id} did not read exactly one complete intended source file: ${JSON.stringify({ read, available: Object.keys(record.semantics.folder_files).filter(path => path.includes('/items/')) })}`);
    const answerEvent = childEvents.find(event => event.name === 'return_result' && event.arguments?.status === 'success');
    const childTarget = record.curriculum.reference.children?.find(candidate =>
      (Array.isArray(candidate.match) ? candidate.match : [candidate.match]).every(fragment => openingText(
        trajectory.find(turn => turn.invocation_id === child.invocation_id)?.context ?? []).includes(fragment)));
    const expectedAnswer = childTarget?.calls?.findLast(call => call[0] === 'return_result')?.[1]?.value ?? childTarget?.value;
    const actualAnswer = child.host_result?.value ?? answerEvent?.arguments?.value;
    if (!childTarget || canonical(actualAnswer) !== canonical(expectedAnswer))
      throw new Error(`${record.id}: child ${child.invocation_id} answer differs from its authored, source-bound reference`);
    return { invocation_id: child.invocation_id, source_path: candidates[0][0], source_text_sha256:
      createHash('sha256').update(candidates[0][1]).digest('hex'), exact_complete_source_read: true,
      expected_answer: expectedAnswer, observed_answer: actualAnswer,
      answer_matches_authored_reference: true, successful_actions: childEvents.length };
  });
  if (run.outcome.status !== 'done' || run.outcome.accepted !== true)
    throw new Error(`${record.id}: runtime rejected source reference: ${JSON.stringify(run.outcome.rejection_reasons)}`);
  if (canonical(run.outcome.value) !== canonical(record.semantics.expected) ||
      canonical(run.outcome.files) !== canonical(record.semantics.expected_files))
    throw new Error(`${record.id}: actual value or output files differ from declared source reference`);
  if (failedActions.length) throw new Error(`${record.id}: ${failedActions.length} unsuccessful action(s)`);

  nativeRows.push(nativeRow); nativeTurns.push(...result.turns);
  for (const turn of result.turns) {
    const decisionIndex = Number(turn.id.slice(turn.id.lastIndexOf(':decision:') + ':decision:'.length));
    const original = trajectory[decisionIndex];
    const actionCalls = turn.decision.assistant.calls ?? [];
    for (const call of actionCalls) {
      const status = call.outcome?.status ?? null;
      if (!['ok', 'completed'].includes(String(status)))
        throw new Error(`${record.id}:${decisionIndex}: materialized action is not linked to a successful host event`);
      actionReviews.push({ id: turn.id, trajectory_id: nativeRow.id, source_program_id: record.id,
        source_group: record.source_groups[0], split: record.split, decision_index: decisionIndex,
        invocation_id: original?.invocation_id ?? null, source_action: call.source_tool,
        source_arguments: call.arguments, exact_runtime_status: status,
        trace_sequence: call.outcome.trace_seq, trace_result_sha256: call.outcome.result == null ? null :
          createHash('sha256').update(String(call.outcome.result)).digest('hex'),
        authored_direct_answer: original?.assistant?.direct_answer === true,
        runtime_training_flag: turn.training_admission,
        source_review_status: 'pending independent semantic review' });
    }
  }
  proofCases.push({ id: record.id, source_group: record.source_groups[0], split: record.split,
    kind: isIterate ? 'iterateOn' : 'nested-FileHandle',
    accepted_by_runtime_oracles: true, value_matches_expected: true, files_match_expected: true,
    child_invocations: isDepth2Variant ? children.length + (depthAudit.item_judges ?? 0) : children.length,
    ...(isDepth2Variant ? { depth_audit: depthAudit } : {}), clean_child_reads: childReads, failed_actions: 0,
    materialized_native_decisions: result.turns.length,
    decisions_marked_approved_by_runtime_materializer: result.turns.filter(turn => turn.training_admission.approved).length,
    materializer_audit: materializerAudit });
}

await mkdir(outPath, { recursive: true });
const exclusive = async (name, data) => {
  const destination = resolve(outPath, name), temporary = `${destination}.${process.pid}.${randomUUID()}.tmp`;
  const handle = await open(temporary, 'wx');
  try { await handle.writeFile(data); await handle.sync(); }
  finally { await handle.close(); }
  try { await link(temporary, destination); }
  catch (error) { if (error.code === 'EEXIST') throw new Error(`refusing to overwrite proof artifact: ${destination}`); throw error; }
  finally { await unlink(temporary).catch(() => {}); }
};
await exclusive('reference-trajectories.jsonl', nativeRows.map(row => JSON.stringify(row)).join('\n') + '\n');
await exclusive('native-decisions.jsonl', nativeTurns.map(turn => JSON.stringify(turn)).join('\n') + '\n');
await exclusive('source-action-review.jsonl', actionReviews.map(row => JSON.stringify(row)).join('\n') + '\n');
const proof = { schema: 'natlang.authored-source-runtime-reference-proof/1', source_path: sourcePath,
  source_sha256: sourceSha, runtime: 'compiled shared TypeScript collector; CPU-only scripted referenceDriver and StandInNeuralesePort',
  model_calls: 0, provider_calls: 0, teacher_trajectories: 0, admission_granted: false,
  native_trajectory_rows: nativeRows.length, native_decisions: nativeTurns.length,
  source_action_reviews: actionReviews.length,
  direct_answer_decisions: actionReviews.filter(row => row.authored_direct_answer).length,
  decisions_approved_by_runtime_materializer_pending_source_review: nativeTurns.filter(turn => turn.training_admission.approved).length,
  runtime_cases: proofCases.length, successful_source_reads: proofCases.reduce((sum, item) => sum + item.clean_child_reads.length, 0),
  unsuccessful_actions: 0,
  materializer_unlinked_outcomes: proofCases.reduce((sum, item) => sum + item.materializer_audit.unlinked.reduce((n, entry) => n + entry.outcomes, 0), 0),
  interpretation: 'These are constructed-world scripted references, not teacher observations. Deterministic authored calls are recorded as provider-shaped raw arguments solely to bind the actual host trace. Native trajectory and decision sidecars are preserved for independent source/action review. Runtime materializer flags do not grant corpus admission.',
  cases: proofCases };
await exclusive('runtime-reference-proof.json', JSON.stringify(proof, null, 2) + '\n');
const artifactHashes = {};
for (const name of ['source.cases.jsonl', 'runtime-reference-proof.json', 'reference-trajectories.jsonl',
  'native-decisions.jsonl', 'source-action-review.jsonl'])
  artifactHashes[name] = createHash('sha256').update(await readFile(resolve(outPath, name))).digest('hex');
await exclusive('review-artifact-manifest.json', JSON.stringify({ schema: 'natlang.authored-source-static-review/1',
  source_sha256: sourceSha, artifact_sha256: artifactHashes, cases: proofCases.length,
  native_rows: nativeRows.length, native_decisions: nativeTurns.length, source_action_reviews: actionReviews.length,
  direct_answer_decisions: proof.direct_answer_decisions, successful_source_reads: proof.successful_source_reads,
  unsuccessful_actions: proof.unsuccessful_actions, unlinked_materializer_outcomes: proof.materializer_unlinked_outcomes,
  model_calls: 0, provider_calls: 0, admission_granted: false,
  status: 'constructed source reference artifact; pending independent root review' }, null, 2) + '\n');
console.log(JSON.stringify({ out: outPath, source_sha256: sourceSha, cases: proofCases.length,
  native_rows: nativeRows.length, native_decisions: nativeTurns.length, clean_reads: proof.successful_source_reads,
  approved_by_runtime_materializer_pending_review: proof.approved_by_runtime_materializer_pending_source_review,
  admission_granted: false }, null, 2));
