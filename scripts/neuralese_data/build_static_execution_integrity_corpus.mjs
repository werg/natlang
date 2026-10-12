#!/usr/bin/env node
/** Build exact train examples for prebound helper calls and honest host-effect recovery. */
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const CORPUS_ID = 'luna-authored-execution-integrity-20261012-v1';
const OUTPUT = join(ROOT, 'data/neuralese/corpora', CORPUS_ID);
const sha256 = value => createHash('sha256').update(value).digest('hex');
const jsonl = rows => rows.map(row => JSON.stringify(row)).join('\n') + '\n';
const summaryFile = async (path, relative) => {
  const bytes = await readFile(path);
  return { path: relative, bytes: bytes.length, sha256: sha256(bytes),
    ...(relative.endsWith('.jsonl') ? { lines: bytes.toString('utf8').trim().split(/\r?\n/).filter(Boolean).length } : {}) };
};

const rootHeader = (name, args, returns, instruction) => `---\nargs:\n${args.map(([key, type]) => `  ${key}: ${type}`).join('\n')}\nreturns: ${returns}\n---\n${instruction}\n`;
const trajectoryContract = code => ({ version: 'natlang.trajectory_contract/1', authored_root_eval: true,
  authored_root_sha256: sha256(code), min_child_invocations: 0, min_state_edges_from_children: 0, min_iteration_steps: 0 });

function helperRecord(index, input, topic) {
  const code = 'return await merge_1(task);';
  const id = `${CORPUS_ID}:prebound:${String(index).padStart(2, '0')}`;
  return { version: 'natlang.program/2', id, kind: 'lambda_source', source: 'authored-runtime-integrity-v1',
    split: 'train', source_ids: [id], source_groups: [id], license: 'project-generated',
    curriculum: { trajectory_contract: trajectoryContract(code), reference: { root: [['eval', { code }]] } },
    semantics: { root: 'review.nl', files: {
      'review.nl': rootHeader('review', [['task', 'unknown']], 'unknown',
        `Use the already available merge_1 helper once for this ${topic} task, passing task unchanged. Return its exact result.`),
      'review/merge_1.ts': 'export default function merge_1<T>(task: T): T { return task; }',
    }, inputs: { task: input }, expected: input, operation: 'exact' },
    static_example: { pattern: 'use-prebound-binding', topic } };
}

function effectRecord({ index, idPart, topic, input, expected, effects, code }) {
  const id = `${CORPUS_ID}:${idPart}:${String(index).padStart(2, '0')}`;
  const root = idPart === 'delivery' ? 'send.nl' : 'send.nl';
  return { version: 'natlang.program/2', id, kind: 'lambda_source', source: 'authored-runtime-integrity-v1',
    split: 'train', source_ids: [id], source_groups: [id], license: 'project-generated',
    curriculum: { trajectory_contract: trajectoryContract(code), reference: { root: [['eval', { code }]] } },
    semantics: { root, files: { [root]: rootHeader('send', [['command', 'unknown']], 'unknown',
      `Use the delivery service for the supplied ${topic} command. Report only the result or error actually returned by the service.`) },
      inputs: { command: input }, expected, operation: 'exact', effects },
    static_example: { pattern: 'report-host-effect', topic } };
}

function rejectionReportRecord(index, sourceRow, line, effectEvent, command) {
  const sourceProgram = sourceRow.task.program_ir;
  const sourceLineSha256 = sha256(line);
  const eventSha256 = sha256(JSON.stringify(effectEvent));
  const task = { key: command.key, error: effectEvent.error, observed_tool: 'mail.deliver' };
  const status = effectEvent.error === lostAck ? 'acknowledgement_lost' : 'rejected';
  const code = 'return await report_1(task);';
  const id = `${CORPUS_ID}:service-result:${String(index).padStart(2, '0')}`;
  return { version: 'natlang.program/2', id, kind: 'lambda_source', source: 'authored-runtime-integrity-v1',
    split: 'train', source_ids: [id], source_groups: [id], license: 'project-generated',
    curriculum: { trajectory_contract: trajectoryContract(code), reference: { root: [['eval', { code }]] } },
    semantics: { root: 'report.nl', files: {
      'report.nl': rootHeader('report', [['task', 'unknown']],
        '{ status: string, key: string, reason: string }',
        'Call the already available report_1 helper once with the captured service result. Preserve its exact status and reason.'),
      'report/report_1.ts': `export default function report_1(task: { key: string; error: string; observed_tool: string }): { status: string; key: string; reason: string } { return { status: task.error === ${JSON.stringify(lostAck)} ? "acknowledgement_lost" : "rejected", key: task.key, reason: task.error }; }`,
    }, inputs: { task }, expected: { status, key: command.key, reason: effectEvent.error }, operation: 'exact' },
    static_example: { pattern: 'report-captured-tool-outcome', topic: status,
      source_effect_failure: { source_program_id: sourceProgram.id, source_teacher_row_sha256: sourceLineSha256,
        source_effect_event_sha256: eventSha256, source_effect_event: effectEvent } } };
}

const mail = (key, to, text) => ({ key, to, text });
const lostAck = 'delivery succeeded but its acknowledgement was lost';
const duplicateReject = 'delivery key already belongs to a different command; retry the original command unchanged';
const invalidCommand = 'delivery requires a nonempty string key and string to and text';

function records() {
  const commandOne = mail('repair-job-41', 'night-ops@example.invalid', 'The backup job completed after a retry.');
  const commandTwo = mail('calendar-brief-12', 'dispatch@example.invalid', 'Move the inspection to Thursday at 10:30.');
  const conflict = mail(commandOne.key, commandOne.to, 'Replace the original repair note with this one.');
  return [
    helperRecord(1, { source: 'equipment-audit', task_id: 'audit-308', findings: [
      { unit: 'pump-2', result: 'seal intact' }, { unit: 'pump-4', result: 'recheck scheduled' } ] }, 'equipment audit'),
    helperRecord(2, { source: 'dispatch-board', task_id: 'route-917', stops: [
      { stop: 'north depot', window: '09:00-09:30' }, { stop: 'clinic', window: '10:15-10:45' } ] }, 'dispatch update'),
    effectRecord({ index: 1, idPart: 'delivery', topic: 'maintenance', input: commandOne,
      expected: { status: 'delivered', key: commandOne.key, attempts: 1, receipt: null },
      effects: { 'mail.deliver': { kind: 'deliver_once_ack_loss', expected_delivered: [commandOne], fail_key: 'unused-key' } },
      code: 'const receipt: null = await mail.deliver(command); return { status: "delivered", key: command.key, attempts: 1, receipt };' }),
    effectRecord({ index: 2, idPart: 'delivery', topic: 'appointment', input: commandTwo,
      expected: { status: 'delivered', key: commandTwo.key, attempts: 2, receipt: null },
      effects: { 'mail.deliver': { kind: 'deliver_once_ack_loss', expected_delivered: [commandTwo], fail_key: commandTwo.key } },
      code: `let attempts = 0; let receipt: null; try { receipt = await mail.deliver(command); attempts = 1; } catch (error) { if (!(error instanceof Error) || error.message !== ${JSON.stringify(lostAck)}) throw error; attempts = 2; receipt = await mail.deliver(command); } return { status: "delivered", key: command.key, attempts, receipt };` }),
    effectRecord({ index: 3, idPart: 'delivery', topic: 'conflicting retry', input: commandOne,
      expected: { status: 'rejected', key: commandOne.key, reason: duplicateReject },
      effects: { 'mail.deliver': { kind: 'deliver_once_ack_loss', expected_delivered: [commandOne], fail_key: commandOne.key } },
      code: `let acknowledgementLost = false; try { await mail.deliver(command); } catch (error) { acknowledgementLost = error instanceof Error && error.message === ${JSON.stringify(lostAck)}; if (!acknowledgementLost) throw error; } if (!acknowledgementLost) return { status: "unexpected", key: command.key }; const changed = { ...command, text: "A different message under the original key." }; try { await mail.deliver(changed); } catch (error) { if (!(error instanceof Error)) throw error; return { status: "rejected", key: command.key, reason: error.message }; } return { status: "unexpected", key: command.key };` }),
    effectRecord({ index: 4, idPart: 'delivery', topic: 'invalid command', input: mail('', 'dispatch@example.invalid', 'Confirm the site inspection.'),
      expected: { status: 'rejected', reason: invalidCommand },
      effects: { 'mail.deliver': { kind: 'deliver_once_ack_loss', expected_delivered: [], fail_key: 'unused-key' } },
      code: `try { await mail.deliver(command); } catch (error) { if (!(error instanceof Error)) throw error; return { status: "rejected", reason: error.message }; } return { status: "unexpected" };` }),
  ];
}

async function main() {
  const { collectBatch, defaultSystemPrompt, defaultToolSurfaceHash, nativeJobRunner } =
    await import('../../ts-host/dist/teacher/collector.js');
  const materializer = await import('../../ts-host/dist/teacher/native-materializer.js');
  const decisionReview = await import('../../ts-host/dist/native/decision-review.js');
  await mkdir(OUTPUT, { recursive: true });
  const probes = records();

  // Authored evals and TS helpers should be sufficient. Any provider request is a build failure.
  let providerRequests = 0;
  const server = createServer((_request, response) => {
    providerRequests++;
    response.writeHead(500, { 'content-type': 'text/plain' });
    response.end('static authored replay must not ask a model');
  });
  await new Promise(resolveListen => server.listen(0, '127.0.0.1', resolveListen));
  const work = join(OUTPUT, 'replay-work');
  const options = { jobs: join(work, 'jobs'), output: join(work, 'probe-results.jsonl'), workers: 1,
    modelId: 'authored-static-runtime-replay', endpoint: `http://127.0.0.1:${server.address().port}`,
    rootSeed: 12026, systemPrompt: defaultSystemPrompt, contextTokens: 4096,
    toolSurfaceSha256: await defaultToolSurfaceHash(), maxTurns: 4, requestRetries: 0,
    executionPlans: false, collectionRole: 'teacher' };
  let collected;
  collected = await collectBatch(probes.map((record, index) => ({ index, record })), options, nativeJobRunner(options));
  if (providerRequests !== 0) throw new Error(`unexpected model/provider requests: ${providerRequests}`);
  if (collected.completed !== probes.length || collected.missing.length) throw new Error('collector did not complete every static source row');
  const probeText = await readFile(options.output, 'utf8');
  const probeRows = probeText.trim().split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
  if (probeRows.length !== probes.length) throw new Error('effect probe row count differs from authored source count');
  const probeByProgram = new Map(probeRows.map((row, index) => [row.task.program_ir.id,
    { row, line: probeText.trim().split(/\r?\n/)[index] }]));
  const expectedFailures = new Map([
    [`${CORPUS_ID}:delivery:02`, mail('calendar-brief-12', 'dispatch@example.invalid', 'Move the inspection to Thursday at 10:30.')],
    [`${CORPUS_ID}:delivery:03`, mail('repair-job-41', 'night-ops@example.invalid', 'The backup job completed after a retry.')],
    [`${CORPUS_ID}:delivery:04`, mail('', 'dispatch@example.invalid', 'Confirm the site inspection.')],
  ]);
  const derived = [];
  for (const [programId, command] of expectedFailures) {
    const source = probeByProgram.get(programId);
    if (!source) throw new Error(`missing exact host-effect evidence source: ${programId}`);
    const events = (source.row.outcome.execution_graph ?? []).filter(event => event.kind === 'effect' && event.phase === 'failed');
    if (events.length !== 1 || events[0].capability !== 'mail.deliver' || typeof events[0].error !== 'string')
      throw new Error(`${programId}: expected one captured mail.deliver rejection`);
    derived.push(rejectionReportRecord(derived.length + 1, source.row, source.line, events[0], command));
  }
  const allSource = [...probes, ...derived];
  const sourceText = jsonl(allSource);
  await writeFile(join(OUTPUT, 'source-programs.jsonl'), sourceText);
  const derivedOptions = { ...options, jobs: join(work, 'derived-jobs'), output: join(work, 'derived-results.jsonl') };
  const derivedCollected = await collectBatch(derived.map((record, index) => ({ index, record })), derivedOptions,
    nativeJobRunner(derivedOptions));
  server.closeAllConnections();
  await new Promise(resolveClose => server.close(resolveClose));
  if (derivedCollected.completed !== derived.length || derivedCollected.missing.length)
    throw new Error('collector did not complete all source-bound result reports');
  const derivedText = await readFile(derivedOptions.output, 'utf8');
  const derivedRows = derivedText.trim().split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
  const teacherRows = [...probeRows.filter(row => row.outcome.accepted), ...derivedRows];
  for (const row of teacherRows) {
    if (!row.outcome.accepted || row.outcome.status !== 'done' || !row.outcome.oracle?.accepted ||
        Object.values(row.outcome.checks ?? {}).some(value => value !== true))
      throw new Error(`${row.task.program_ir.id}: exact runtime/effect replay did not pass: ${JSON.stringify(row.outcome.rejection_reasons)}`);
  }
  const teacherText = jsonl([...probeRows, ...derivedRows]);
  const pending = materializer.materializeNativeRows(teacherRows);
  if (pending.authored_actions.length !== teacherRows.length || pending.turns.length !== 0)
    throw new Error('expected one held authored action per exact source replay');
  const builderText = await readFile(fileURLToPath(import.meta.url), 'utf8');
  const review = { schema: 'natlang.authored-static-execution-integrity-review/1', corpus_id: CORPUS_ID,
    method: 'shared collectBatch/nativeJobRunner/executeProgram/effectHarness and native-materializer; all source roots are authored evals; local HTTP endpoint rejects any provider request',
    builder_script_sha256: sha256(builderText),
    source_programs_sha256: sha256(sourceText), teacher_results_sha256: sha256(teacherText),
    provider_requests: providerRequests, exact_source_rows: teacherRows.length,
    observed_rejected_effects_context_only: [...expectedFailures.keys()].map(programId => {
      const source = probeByProgram.get(programId).row;
      return { source_program_id: programId, accepted_as_target: false,
        effect_events: source.outcome.execution_graph.filter(event => event.kind === 'effect' && event.phase === 'failed') };
    }),
    authored_action_rows: pending.authored_actions.map(action => ({ trajectory_id: action.source_ref.trajectory_id,
      source_row_sha256: action.source_ref.source_row_sha256, target_sha256: decisionReview.nativeDecisionTargetDigest(action.target),
      source_program_id: action.source_ref.program_ir_id, outcome: action.outcome,
      effect_log: action.actions.map(item => item.outcome), static_example: action.task.program_ir.static_example })),
    decision: 'Admit each exact authored root action as static source SFT only after matching the authored source hash, successful exact result/effect checks, and the target digest below. Trace admission is false.' };
  const reviewText = JSON.stringify(review, null, 2) + '\n';
  await writeFile(join(OUTPUT, 'review.json'), reviewText);
  const reviewHash = sha256(reviewText);
  const approvals = pending.authored_actions.map(action => ({ schema: 'natlang.native-decision-approval/1',
    trajectory_id: action.source_ref.trajectory_id, source_row_sha256: action.source_ref.source_row_sha256,
    decision_index: 0, target_sha256: decisionReview.nativeDecisionTargetDigest(action.target),
    review_sha256: reviewHash, reason: 'exact authored source replay passed the shared runtime, typed return oracle, and declared host-effect checks',
    evidence: ['source eval code hash matches curriculum.reference.root', 'runtime outcome status is done and exact oracle accepted',
      'all declared result/effect checks passed', 'no provider requests were made'] }));
  const admitted = materializer.materializeNativeRows(teacherRows, { decisionApprovals: approvals });
  if (admitted.authored_actions.length || admitted.turns.length !== teacherRows.length ||
      admitted.turns.some(turn => turn.training_admission?.approved !== true || turn.trace_admission?.admitted !== false))
    throw new Error('native materializer did not admit only the reviewed static actions');
  const nativeText = jsonl(admitted.turns);
  await writeFile(join(OUTPUT, 'teacher-results.jsonl'), teacherText);
  await writeFile(join(OUTPUT, 'native-decisions.jsonl'), nativeText);
  await writeFile(join(OUTPUT, 'decision-approvals.json'), JSON.stringify({ schema: 'natlang.native-decision-approvals/1',
    review_sha256: reviewHash, approvals }, null, 2) + '\n');
  await writeFile(join(OUTPUT, 'delegation-controls.json'), JSON.stringify({ schema: 'natlang.existing-delegation-control-bindings/1',
    reused_without_duplication: true,
    source_family: 'decision_extract_chain',
    source_definition: 'runs/semantic-lambda-topology-v17-v8-corrected-review-v1/authored-root-v1/runtime-build-v6/frozen-runtime/scripts/inline-curriculum/decision-rich.mjs',
    source_family_behavior: 'nested scoped natural-language extractors and merge_1 call chain using an existing task binding',
    admitted_control_corpus: 'semantic-inline-curriculum-s72-teacher-turns-20261007-v1',
    source_parent_corpus: 'semantic-inline-curriculum-s72-teacher-20261007-v1',
    control_use: 'Existing train-only NL delegation examples remain the control; this packet adds no duplicate NL generations or protected prompts.' }, null, 2) + '\n');
  await writeFile(join(OUTPUT, 'README.md'), `# ${CORPUS_ID}\n\nTrain-only static source SFT for three observed patterns: use an existing helper binding; report a host effect only after its result; and report a rejection from the exact returned service error. Nine source programs were replayed through the shared native collector and materializer. The deterministic ` +
    '`mail.deliver`' + ` effect harness exercised ordinary success, lost acknowledgement, conflicting-key rejection, and invalid-command rejection. Three rejected tool probes are retained as evidence only; their failed actions are not training targets. Three exact typed result reports are derived from their captured errors and effect logs. All admitted rows passed their exact value oracle. No model/provider requests occurred.\n\nOrdinary nested natural-language delegation controls are already represented by ` +
    '`decision_extract_chain`' + ` in the admitted s72 train corpus and are referenced in ` + '`delegation-controls.json`' + ` instead of being duplicated here. This corpus admits native SFT rows only; it does not admit a sampled teacher trace or qualify recurrence.\n`);
  const files = ['README.md', 'source-programs.jsonl', 'teacher-results.jsonl', 'native-decisions.jsonl',
    'review.json', 'decision-approvals.json', 'delegation-controls.json'];
  const snapshots = [];
  for (const path of files) snapshots.push(await summaryFile(join(OUTPUT, path), path));
  const fileIndexText = JSON.stringify({ schema: 'natlang.corpus-files/1', corpus_id: CORPUS_ID, files: snapshots }, null, 2) + '\n';
  await writeFile(join(OUTPUT, 'FILES.sha256.json'), fileIndexText);
  const manifest = { schema: 'natlang.corpus-snapshot/1', id: CORPUS_ID, owner: 'pop',
    path: `data/neuralese/corpora/${CORPUS_ID}`,
    files: [...snapshots, await summaryFile(join(OUTPUT, 'FILES.sha256.json'), 'FILES.sha256.json')]
      .sort((a, b) => a.path.localeCompare(b.path)),
    bytes: 0 };
  manifest.bytes = manifest.files.reduce((sum, entry) => sum + entry.bytes, 0);
  const manifestText = JSON.stringify(manifest, null, 2) + '\n';
  const manifestPath = join(ROOT, 'training/corpus-manifests', `${CORPUS_ID}.json`);
  await writeFile(manifestPath, manifestText);
  console.log(JSON.stringify({ corpus_id: CORPUS_ID, output: OUTPUT, native_rows: admitted.turns.length,
    context_only_rejected_probe_rows: expectedFailures.size, provider_requests: providerRequests, review_sha256: reviewHash,
    manifest_sha256: sha256(manifestText), source_ids: allSource.map(row => row.id) }, null, 2));
}

await main();
