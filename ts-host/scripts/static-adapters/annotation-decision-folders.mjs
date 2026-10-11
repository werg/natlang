#!/usr/bin/env node
/**
 * Prepare held static folder reducers from an explicitly supplied dataset annotation file.
 * This path uses annotations only for hidden reference outputs; it makes no provider calls and
 * does not claim independent semantic adjudication or training admission.
 */
import { mkdir, open, readFile, readdir, stat } from 'node:fs/promises';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { folderDecisionScaffold } from './folder-decision-scaffold.mjs';
import { decisionCriteria, DECISION_TASK_CONTRACT_PATH, DECISION_TASK_CONTRACT_SHA256 } from './decision-task-contracts.mjs';
import { canonical } from '../advisory-file.mjs';
import { referenceRow } from '../inline-curriculum/references.mjs';
import { materializeNativeRows } from '../../dist/teacher/native-materializer.js';
import { defaultToolSurfaceHash } from '../../dist/teacher/collector.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';
import { fileHash, requireValue, equal, sha } from './common.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, '../../../');
const jsonSha = value => sha(canonical(value));
const parse = value => JSON.parse(value);
async function writeExclusive(dir, name, value) {
  const handle = await open(resolve(dir, name), 'wx');
  try { await handle.writeFile(value); } finally { await handle.close(); }
}

function visibleInputs(record) {
  const files = record.semantics.folder_files;
  const itemFiles = Object.entries(files ?? {}).filter(([path]) => path.startsWith('items/') && path.endsWith('.json'));
  const input = { items: itemFiles.map(([, text]) => parse(text)) };
  requireValue(input.items.length >= 2, 'unexpected_batch_item_count');
  const root = record.semantics.files[record.semantics.root];
  const scriptedRoot = record.curriculum.reference.root?.[0]?.[1]?.code;
  const task = parse(files['task.json']);
  const labels = task.labels;
  requireValue(typeof root === 'string' && root.includes('directory') && root.includes('decisions.json'), 'missing_shared_folder_root');
  requireValue(typeof scriptedRoot === 'string' && scriptedRoot.includes('nl<Decision>') &&
    scriptedRoot.includes('${question}') && scriptedRoot.includes('caseId, state, question, criteria, options, levels'),
    'missing_runtime_inline_item_reducer');
  requireValue(Array.isArray(labels) && labels.length >= 2, 'missing_visible_labels');
  const visibleFiles = { folder_files: files, types_ts: record.semantics.files['types.ts'] ?? null };
  const visible = JSON.stringify({ root, visible_files: visibleFiles, inputs: input });
  for (const item of input.items) {
    requireValue(typeof item.state === 'string' && item.state.length > 0 && typeof item.id === 'string' &&
      typeof item.question === 'string' && item.question.trim().length > 0, 'incomplete_visible_item');
    requireValue(!Object.hasOwn(item, 'answer') && !Object.hasOwn(item, 'gold'), 'annotation_in_visible_item');
    requireValue(!Object.keys(files).some(path => /(^|\/)answers?\.(json|txt)$/i.test(path)), 'annotation_answer_file_present');
  }
  requireValue(!Object.hasOwn(task, 'answer') && !Object.hasOwn(task, 'gold'), 'annotation_in_visible_task');
  return { root_path: record.semantics.root, root, visible_files: visibleFiles, inputs: input,
    model_visible_sha256: sha(visible), reference_topology: { controller_sha256: sha(scriptedRoot),
      controller_chars: scriptedRoot.length }, complete_state_chars: input.items.map(item => item.state.length) };
}

function captureVisibility(row, record) {
  const expectedItems = Object.entries(record.semantics.folder_files)
    .filter(([path]) => path.startsWith('items/') && path.endsWith('.json')).map(([, text]) => parse(text));
  const task = parse(record.semantics.folder_files['task.json']);
  const labelsText = JSON.stringify(task.labels);
  const itemChecks = expectedItems.map((item, index) => {
    const invocations = new Map();
    for (const turn of row.trajectory ?? []) {
      if (typeof turn.invocation_id !== 'string' || !/^task-\d+-[^/]+\/\d+$/.test(turn.invocation_id)) continue;
      const group = invocations.get(turn.invocation_id) ?? [];
      group.push(turn); invocations.set(turn.invocation_id, group);
    }
    const expectedEval = 'console.log(JSON.stringify({ caseId, state, question, criteria, options, levels }));';
    let matched;
    for (const [invocationId, turns] of invocations) {
      const context = turns.flatMap(turn => turn.context ?? []).map(message => String(message.content ?? '')).join('\n');
      const idLine = context.match(/(?:^|\n)caseId: unknown = ("(?:\\.|[^"\\])*")(?=\n|$)/)?.[1];
      if (!idLine || JSON.parse(idLine) !== item.id) continue;
      const evalObserved = turns.some(turn => (turn.assistant?.calls ?? []).some(call =>
        call.tool === 'eval' && call.arguments?.code === expectedEval));
      const returnObserved = turns.some(turn => (turn.assistant?.calls ?? []).some(call =>
        call.tool === 'return_result' && call.arguments?.status === 'success' &&
        equal(call.arguments?.value, record.semantics.expected[item.id])));
      if (evalObserved && returnObserved) { matched = { invocationId, context }; break; }
    }
    const context = matched?.context ?? '';
    const exactTypedInput = context.includes(item.id) && context.includes(JSON.stringify(item.state));
    const questionVisible = context.includes(item.question) && context.includes('Question:');
    const optionsLine = context.match(/(?:^|\n)options: unknown = ([^\n]+)/)?.[1];
    let capturedOptions;
    try { capturedOptions = optionsLine ? JSON.parse(optionsLine) : undefined; } catch { /* malformed capture */ }
    const labelDomainVisible = Array.isArray(capturedOptions) && equal(capturedOptions, task.labels) ||
      context.includes(labelsText);
    return { id: item.id, exact_typed_input_visible: exactTypedInput, question_visible: questionVisible,
      complete_label_domain_visible: labelDomainVisible,
      state_chars: item.state.length, state_sha256: sha(item.state), invocation_id: matched?.invocationId ?? null,
      matched_exact_child_eval_and_return: Boolean(matched) };
  });
  return { visible: itemChecks.every(item => item.exact_typed_input_visible && item.question_visible &&
    item.complete_label_domain_visible), items: itemChecks };
}

async function main() {
  const { values } = parseArgs({ options: { help: { type: 'boolean' }, cases: { type: 'string' }, out: { type: 'string' },
    'items-per-family': { type: 'string', default: '12' }, 'batch-size': { type: 'string', default: '4' },
    'exclude-cases': { type: 'string' } } });
  const usage = 'usage: annotation-decision-folders.mjs --cases SOURCE_CASES.jsonl --out NEW_DIR [--items-per-family N] [--batch-size N] [--exclude-cases PRIOR_CASES.jsonl]';
  if (values.help) { console.log(usage); return; }
  if (!values.cases || !values.out) throw new Error(usage);
  const itemsPerFamily = Number(values['items-per-family']), batchSize = Number(values['batch-size']);
  requireValue(Number.isSafeInteger(itemsPerFamily) && itemsPerFamily >= 2, 'items-per-family must be an integer >= 2');
  requireValue(Number.isSafeInteger(batchSize) && batchSize >= 2, 'batch-size must be an integer >= 2');
  const casesPath = resolve(values.cases), out = resolve(values.out);
  const sourceBytes = await readFile(casesPath), sourceSha = sha(sourceBytes);
  const adjacentSourceManifest = resolve(dirname(casesPath), 'source-manifest.json');
  const sourceManifestPath = await stat(adjacentSourceManifest).then(() => adjacentSourceManifest,
    () => resolve(dirname(casesPath), 'decision-data.manifest.json'));
  const sourceManifestBytes = await readFile(sourceManifestPath);
  const sourceManifest = parse(sourceManifestBytes.toString('utf8'));
  const declaredCaseCount = sourceManifest.cases?.rows ?? (Number.isSafeInteger(sourceManifest.cases) ? sourceManifest.cases : undefined) ??
    sourceManifest.selected_count ?? sourceManifest.selection?.selected_count;
  const declaredCaseHash = sourceManifest.cases?.sha256 ?? sourceManifest.selected_sha256 ?? sourceManifest.cases_sha256 ??
    sourceManifest.sha256?.[basename(casesPath)];
  requireValue(declaredCaseHash === sourceSha && Number.isSafeInteger(declaredCaseCount),
    'source_manifest_case_pin_mismatch');
  const policyPath = resolve(repo, 'training/decision_source_quality_holds.json');
  const policyBytes = await readFile(policyPath), policy = parse(policyBytes.toString('utf8'));
  requireValue(policy.schema === 'natlang.decision-source-quality-holds/1' && Array.isArray(policy.excluded_items),
    'invalid_canonical_hold_policy');
  requireValue(Array.isArray(policy.state_truncation_rules) && policy.state_truncation_rules.every(rule =>
    rule && typeof rule.rule_id === 'string' && Number.isSafeInteger(rule.state_codepoints) &&
    rule.state_codepoints > 0 && typeof rule.suffix === 'string' && rule.suffix.length > 0),
  'invalid_canonical_state_truncation_rules');
  const holds = new Map(policy.excluded_items.map(item => [item.item_id, item]));
  const rows = sourceBytes.toString('utf8').split(/\r?\n/).filter(Boolean).map((line, index) => ({ row: parse(line), index, raw: line }));
  requireValue(rows.length === declaredCaseCount && new Set(rows.map(({row}) => row.id)).size === rows.length, 'invalid_source_rows');
  const sourceById = new Map(rows.map(entry => [entry.row.id, entry]));
  const excludedIds = new Set(), excludedGroups = new Set(), exclusionCases = [];
  let exclusionBytes = Buffer.alloc(0), exclusionPath = null;
  if (values['exclude-cases']) {
    exclusionPath = resolve(values['exclude-cases']);
    exclusionBytes = await readFile(exclusionPath);
    const lines = exclusionBytes.toString('utf8').split(/\r?\n/).filter(Boolean);
    const priorPrograms = new Set();
    for (const [lineIndex, line] of lines.entries()) {
      const prior = parse(line), programId = prior.id;
      requireValue(typeof programId === 'string' && programId.length > 0 && !priorPrograms.has(programId),
        `invalid_or_duplicate_excluded_program:${lineIndex}`);
      priorPrograms.add(programId);
      const priorSource = prior.external_source;
      const origins = Array.isArray(prior.exclusion_origins) ? prior.exclusion_origins : [];
      requireValue(priorSource?.snapshot_sha256 === sourceSha &&
        priorSource?.source_manifest_sha256 === sha(sourceManifestBytes), 'excluded_source_snapshot_mismatch');
      requireValue(Array.isArray(prior.source_ids) && Array.isArray(prior.source_groups) &&
        Array.isArray(priorSource.source_rows) && prior.source_ids.length === priorSource.source_rows.length,
        'invalid_excluded_lineage_shape');
      const rowEntries = [];
      for (const [index, id] of prior.source_ids.entries()) {
        requireValue(typeof id === 'string' && !excludedIds.has(id), `duplicate_excluded_source_id:${id}`);
        const declared = priorSource.source_rows[index];
        const actual = sourceById.get(id);
        requireValue(actual && declared.id === id && declared.row_index === actual.index &&
          declared.row_sha256 === sha(actual.raw) && declared.group === actual.row.group &&
          declared.split === actual.row.role && actual.row.role === 'train' && actual.row.kind === 'choice',
        `excluded_source_lineage_mismatch:${id}`);
        requireValue(!excludedGroups.has(actual.row.group), `duplicate_excluded_source_group:${actual.row.group}`);
        excludedIds.add(id); excludedGroups.add(actual.row.group);
        rowEntries.push({ id, row_index: actual.index, row_sha256: sha(actual.raw), group: actual.row.group,
          split: actual.row.role });
      }
      requireValue(equal([...new Set(rowEntries.map(entry => entry.group))], prior.source_groups),
        `excluded_program_group_mismatch:${programId}`);
      exclusionCases.push({ program_id: programId, source_ids: prior.source_ids,
        source_groups: prior.source_groups, source_rows: rowEntries, origins });
    }
  }
  const omittedHeld = rows.filter(({row}) => holds.has(row.id));
  for (const {row} of omittedHeld) {
    const hold = holds.get(row.id);
    requireValue(hold.original_source_group === row.group && hold.split === row.role, 'hold_lineage_mismatch');
  }
  // Python's len(str) counts Unicode codepoints. Use Array.from here so this
  // shared policy has the same meaning for astral Unicode as the source filter.
  const truncationOmissions = rows.filter(({row}) => row.role === 'train' && row.kind === 'choice' &&
    typeof row.state === 'string' && policy.state_truncation_rules.some(rule =>
      Array.from(row.state).length === rule.state_codepoints && row.state.endsWith(rule.suffix)))
    .map(({row, index, raw}) => {
      const rule = policy.state_truncation_rules.find(candidate => Array.from(row.state).length === candidate.state_codepoints &&
        row.state.endsWith(candidate.suffix));
      return { id: row.id, family: row.family, group: row.group, split: row.role,
        state_chars: Array.from(row.state).length, state_sha256: sha(row.state), row_index: index,
        row_sha256: sha(raw), also_prior_case_or_group: excludedIds.has(row.id) || excludedGroups.has(row.group),
        also_canonical_hold: holds.has(row.id), omission_code: rule.rule_id, rule_id: rule.rule_id };
    });
  const truncatedIds = new Set(truncationOmissions.map(item => item.id));
  const eligibleBeforeTruncation = rows.filter(({row}) => !holds.has(row.id) && !excludedIds.has(row.id) &&
    !excludedGroups.has(row.group) && row.role === 'train' && row.kind === 'choice');
  const sourceFamilies = [...new Set(rows.filter(({row}) => row.role === 'train' && row.kind === 'choice')
    .map(({row}) => row.family).filter(family => typeof family === 'string'))].sort();
  const preTruncationByFamily = new Map(sourceFamilies.map(family => [family, []]));
  for (const entry of eligibleBeforeTruncation) {
    const list = preTruncationByFamily.get(entry.row.family) ?? [];
    list.push(entry); preTruncationByFamily.set(entry.row.family, list);
  }
  const eligible = eligibleBeforeTruncation.filter(({row}) => !truncatedIds.has(row.id));
  const byFamily = new Map(sourceFamilies.map(family => [family, []]));
  for (const entry of eligible) {
    const list = byFamily.get(entry.row.family) ?? [];
    list.push(entry); byFamily.set(entry.row.family, list);
  }
  const selected = [];
  const familySelectionSummary = [];
  for (const family of sourceFamilies) {
    const available = byFamily.get(family);
    const familyRows = available.slice(0, itemsPerFamily);
    if (familyRows.length < 2) {
      const preFilter = preTruncationByFamily.get(family) ?? [];
      familySelectionSummary.push({ family, source_eligible_before_truncation: preFilter.length,
        truncation_omissions_after_other_filters: truncationOmissions.filter(item => item.family === family &&
          !item.also_prior_case_or_group && !item.also_canonical_hold).length,
        eligible_after_truncation: available.length, requested_rows: itemsPerFamily, selected_rows: 0,
        shortfall: itemsPerFamily, skipped: true, skip_reason: 'fewer_than_two_eligible_rows_after_all_filters' });
      continue;
    }
    selected.push(...familyRows);
    const preFilter = preTruncationByFamily.get(family) ?? [];
    familySelectionSummary.push({ family, source_eligible_before_truncation: preFilter.length,
      truncation_omissions_after_other_filters: truncationOmissions.filter(item => item.family === family &&
        !item.also_prior_case_or_group && !item.also_canonical_hold).length,
      eligible_after_truncation: available.length, requested_rows: itemsPerFamily,
      selected_rows: familyRows.length, shortfall: Math.max(0, itemsPerFamily - familyRows.length), skipped: false });
  }
  const sourceIds = new Set(), sourceGroups = new Set(), tasksByFamily = new Map();
  for (const {row, index} of selected) {
    requireValue(typeof row.id === 'string' && row.id.length > 0 && typeof row.group === 'string' && row.group.length > 0,
      'missing_source_identity');
    requireValue(typeof row.question === 'string' && row.question.trim().length > 0, 'empty_source_question');
    requireValue(typeof row.state === 'string' && row.state.trim().length > 0, 'empty_source_state');
    requireValue(Array.isArray(row.options) && row.options.length >= 2 &&
      row.options.every(option => typeof option === 'string' && option.trim().length > 0) &&
      new Set(row.options).size === row.options.length, 'invalid_or_duplicate_options');
    requireValue(typeof row.answer === 'string' && row.options.includes(row.answer), 'annotation_not_in_options');
    requireValue(!sourceIds.has(row.id) && !sourceGroups.has(row.group), 'duplicate_source_identity');
    sourceIds.add(row.id); sourceGroups.add(row.group);
    const instruction = row.question;
    const { criteria, provenance: criteriaProvenance } = decisionCriteria({ family: row.family,
      source: row.source, kind: row.kind, labels: row.options, explicitCriteria: row.criteria, where: row.id });
    const task = { id: row.id, family: row.family, state: row.state, kind: 'choice', labels: row.options,
      criteria,
      question: row.question, instruction, gold: row.answer, gold_source: 'dataset-annotation', group_id: row.group,
      split: row.role, license: row.license, source_meta: { family_id: row.family, source_index: index,
        source_version: row.version, source_name: row.source } };
    const familyTasks = tasksByFamily.get(row.family) ?? [];
    familyTasks.push({ task, sourceRow: row, sourceIndex: index,
      criteriaProvenance,
      sourceRowRaw: selected.find(item => item.index === index).raw,
      sourceRowSha256: sha(selected.find(item => item.index === index).raw) }); tasksByFamily.set(row.family, familyTasks);
  }
  const sourceLabel = sourceManifest.source?.name ?? sourceManifest.source?.path ?? sourceManifest.source ?? basename(casesPath);
  const info = { source: `dataset-annotations:${basename(String(sourceLabel))}`, sha256: sourceSha, path: casesPath,
    license: 'source-record-required' };
  const records = [];
  const familySummary = [];
  for (const [family, entries] of tasksByFamily) {
    const first = entries[0].task;
    const tasks = entries.map(entry => entry.task);
    const instruction = first.instruction;
    requireValue(tasks.every(task => task.instruction === instruction &&
      JSON.stringify(task.labels) === JSON.stringify(first.labels) &&
      JSON.stringify(task.criteria) === JSON.stringify(first.criteria) && task.license === first.license),
      'family_contract_mismatch');
    const batchSizes = partitionBatchSizes(entries.length, batchSize);
    let offset = 0;
    for (const size of batchSizes) {
      const group = entries.slice(offset, offset + size);
      offset += size;
      const firstTask = group[0].task;
      const contract = { kind: 'choice', options: firstTask.labels, levels: [], labels: firstTask.labels,
        criteria: firstTask.criteria, signature: jsonSha({ kind: firstTask.kind, labels: firstTask.labels,
          criteria: firstTask.criteria, question: firstTask.instruction }) };
      const scaffoldEntries = group.map(entry => ({ task: { ...entry.task, gold: entry.sourceRow.answer } }));
      const originalGroups = group.map(entry => entry.sourceRow.group);
      const record = folderDecisionScaffold({ contract, entries: scaffoldEntries, sourceSha,
        sourceSnapshot: sha(sourceManifestBytes), sourceGroups: originalGroups,
        curriculumFamily: `annotation_decision_folders_${String(family).toLowerCase().replace(/[^a-z0-9]+/g, '_')}`,
        recordFamily: `curriculum_annotation_decision_folders_${String(family).toLowerCase().replace(/[^a-z0-9]+/g, '_')}`,
        splitGroup: JSON.stringify([...new Set(originalGroups)].sort()) });
      record.source = info.source;
      record.source_ids = group.map(entry => entry.sourceRow.id);
      record.source_groups = [...new Set(originalGroups)];
      record.license = group[0].sourceRow.license;
      record.gold_sources = [`dataset-annotation:${sourceSha}`];
      const datasetName = group[0].sourceRow.source;
      requireValue(group.every(entry => entry.sourceRow.source === datasetName), 'family_source_mismatch');
      record.dataset = { source: datasetName, family, source_split: 'train' };
      record.dataset_records = group.map(entry => `${datasetName}:${entry.sourceRow.id}`);
      record.external_source = { source: info.source, original_split: 'train', snapshot_sha256: sourceSha,
        original_row_sha256: sha(group.map(entry => entry.sourceRowRaw).join('\n')), source_path: casesPath,
        quality: { version: 'natlang.source_quality/1', status: 'held' } };
      // Keep annotations out of the converted case's generic provenance payload. They remain
      // only in semantics.expected/reference and in the separately pinned source artifact.
      delete record.external_source.original_row;
      record.external_source.quality = { version: 'natlang.source_quality/1', status: 'held',
        checks: ['source_annotation_reference_only', 'no_independent_semantic_adjudication',
          'canonical_source_quality_holds_applied', 'zero_provider_calls',
          'shared_decision_task_contract_applied_when_declared'] };
      record.external_source.source_manifest_sha256 = sha(sourceManifestBytes);
      record.external_source.source_rows = group.map(entry => ({ id: entry.sourceRow.id,
        row_index: entry.sourceIndex, row_sha256: entry.sourceRowSha256, group: entry.sourceRow.group,
        split: entry.sourceRow.role }));
      record.generation = { ...record.generation, mode: 'static_dataset_annotation_reference',
        provider_calls: 0, independent_new_worlds: 0, training_admission: false,
        decision_task_contract_sha256: DECISION_TASK_CONTRACT_SHA256,
        decision_task_contract_path: DECISION_TASK_CONTRACT_PATH,
        task_criteria_provenance: group.map(entry => ({ id: entry.task.id,
          ...(entry.criteriaProvenance ?? {}) })) };
      const visible = visibleInputs(record);
      records.push({ record, visible });
    }
    familySummary.push({ family, source_rows: entries.length, programs: batchSizes.length, batch_sizes: batchSizes,
      source_ids: entries.map(entry => entry.sourceRow.id), source_groups: entries.map(entry => entry.sourceRow.group) });
  }
  requireValue(records.length > 0, 'no_annotation_programs_selected');
  await mkdir(out, { recursive: true });
  requireValue((await readdir(out)).length === 0, 'refusing_nonempty_output');
  const options = { modelId: 'static-annotation-reference-no-provider', rootSeed: 7281,
    systemPrompt: TOOLS_PROMPT, contextTokens: 32768, maxTurns: 60,
    toolSurfaceSha256: await defaultToolSurfaceHash(), collectionRole: 'reference' };
  const componentPaths = [
    resolve(here, 'annotation-decision-folders.mjs'), resolve(here, 'folder-decision-scaffold.mjs'),
    resolve(here, 'decision-task-contracts.mjs'), DECISION_TASK_CONTRACT_PATH,
    resolve(here, 'teacher-decision-labels.mjs'), resolve(here, '../advisory-file.mjs'),
    resolve(repo, 'scripts/prepare_decision_provider_source.py'),
    policyPath,
    resolve(here, 'common.mjs'), resolve(here, '../inline-curriculum/lib.mjs'),
    resolve(here, '../inline-curriculum/references.mjs'), resolve(here, '../../dist/teacher/curriculum.js'),
    resolve(here, '../../dist/teacher/collector.js'), resolve(here, '../../dist/teacher/native-materializer.js'),
    resolve(here, '../../dist/teacher/program.js'), resolve(here, '../../dist/native/prompt.js'),
  ];
  const componentPins = Object.fromEntries(await Promise.all(componentPaths.map(async path => [path, await fileHash(path)])));
  const replay = [], materializer = [], failures = [];
  for (let i = 0; i < records.length; i++) {
    const { record, visible } = records[i];
    try {
      const row = await referenceRow(record, i, options);
      const visibility = captureVisibility(row, record);
      const materialized = materializeNativeRows([row]);
      const outputMatches = equal(row.outcome?.value, record.semantics.expected);
      materializer.push({ record_id: record.id, accepted_rows: materialized.acceptedRows,
        rejected_rows: materialized.rejectedRows, turns: materialized.turns.length,
        unlinked: materialized.unlinked.length,
        note: 'Held annotation-derived rows are intentionally not promoted by reference replay.' });
      replay.push({ row, visible_input_check: visible, context_visibility: visibility,
        output_matches_annotation: outputMatches });
      if (!visibility.visible) throw new Error('full_source_state_not_visible_in_reference_context');
      if (!outputMatches || row.outcome?.accepted !== true) throw new Error('reference_output_not_exactly_accepted');
    } catch (error) {
      failures.push({ record_id: record.id, reason: error instanceof Error ? error.message : String(error) });
    }
  }
  const casesText = records.map(({record}) => JSON.stringify(record)).join('\n') + '\n';
  const replayText = replay.map(item => JSON.stringify(item)).join('\n') + (replay.length ? '\n' : '');
  const materializerText = JSON.stringify({ accepted_rows: materializer.reduce((sum, item) => sum + item.accepted_rows, 0),
    rejected_rows: materializer.reduce((sum, item) => sum + item.rejected_rows, 0),
    emitted_turns: materializer.reduce((sum, item) => sum + item.turns, 0), per_program: materializer }, null, 2) + '\n';
  const familyText = JSON.stringify(familySummary, null, 2) + '\n';
  const exclusionReceipt = { schema: 'natlang.annotation-decision-folder-exclusions/1',
    prior_cases_path: exclusionPath, prior_cases_sha256: exclusionPath ? sha(exclusionBytes) : null,
    source_cases_sha256: sourceSha, source_manifest_sha256: sha(sourceManifestBytes),
    excluded_programs: exclusionCases.length, excluded_source_ids: [...excludedIds].sort(),
    excluded_source_groups: [...excludedGroups].sort(), programs: exclusionCases,
    receipt_sha256: null };
  exclusionReceipt.receipt_sha256 = jsonSha({ ...exclusionReceipt, receipt_sha256: null });
  const exclusionText = JSON.stringify(exclusionReceipt, null, 2) + '\n';
  const failuresText = failures.map(item => JSON.stringify(item)).join('\n') + (failures.length ? '\n' : '');
  const truncationText = truncationOmissions.map(item => JSON.stringify(item)).join('\n') +
    (truncationOmissions.length ? '\n' : '');
  const truncationCounts = new Map();
  for (const item of truncationOmissions) truncationCounts.set(item.family, (truncationCounts.get(item.family) ?? 0) + 1);
  const summary = { schema: 'natlang.static-annotation-decision-folders/1', status: 'prepared_held_reference_only',
    training_admission: false, provider_calls: 0, independent_new_worlds: 0,
    source: { path: casesPath, sha256: sourceSha, source_manifest_path: sourceManifestPath,
      source_manifest_sha256: sha(sourceManifestBytes), selected_rows: declaredCaseCount },
    task_contracts: { path: DECISION_TASK_CONTRACT_PATH, sha256: DECISION_TASK_CONTRACT_SHA256,
      schema: 'natlang.decision-task-contracts/1' },
    quality_policy: { path: policyPath, sha256: sha(policyBytes), policy_id: policy.policy_id,
      canonical_holds_present_in_source: omittedHeld.map(({row}) => row.id),
      state_truncation_rules: policy.state_truncation_rules },
    source_quality_omissions: { rules: policy.state_truncation_rules,
      total_source_matches: truncationOmissions.length,
      source_matches_by_family: Object.fromEntries([...truncationCounts].sort(([a], [b]) => a.localeCompare(b))),
      artifact: 'source-quality-omissions.jsonl', artifact_sha256: sha(truncationText),
      note: 'Omitted rows are listed with exact source row and state hashes plus prior-use/hold overlap flags.' },
    exclusions: { path: exclusionPath, sha256: exclusionPath ? sha(exclusionBytes) : null,
      programs: exclusionCases.length, source_ids: excludedIds.size, source_groups: excludedGroups.size,
      receipt_sha256: sha(exclusionText) },
    selection: { families: familySummary, family_availability: familySelectionSummary,
      selected_annotations: selected.length, programs: records.length,
      requested_items_per_family: itemsPerFamily, batch_size_limit: batchSize,
      source_groups_preserved: sourceGroups.size,
      note: 'Annotations appear only in hidden semantics.expected/reference outputs. Visible task files and typed item inputs contain no answer/gold property. No provider-authored labels, no independent semantic adjudication, no admission.' },
    replay: { attempts: replay.length,
      exact_accepted: replay.filter(item => item.row.outcome?.accepted && item.output_matches_annotation).length,
      failures: failures.length, materializer_rejected_held_rows: materializer.reduce((sum, item) => sum + item.rejected_rows, 0),
      tool_surface_sha256: options.toolSurfaceSha256,
      options_sha256: jsonSha({ ...options, systemPrompt_sha256: sha(TOOLS_PROMPT) }) },
    execution_components: componentPins,
    adapter: { path: fileURLToPath(import.meta.url), sha256: sha(await readFile(fileURLToPath(import.meta.url)) ) },
    artifacts: { cases_sha256: sha(casesText), replay_sha256: sha(replayText), materializer_review_sha256: sha(materializerText),
      families_sha256: sha(familyText), exclusions_sha256: sha(exclusionText),
      source_quality_omissions_sha256: sha(truncationText), failures_sha256: sha(failuresText) } };
  await writeExclusive(out, 'cases.jsonl', casesText);
  await writeExclusive(out, 'replay.results.jsonl', replayText);
  await writeExclusive(out, 'materializer.review.json', materializerText);
  await writeExclusive(out, 'families.json', familyText);
  await writeExclusive(out, 'exclusions.json', exclusionText);
  await writeExclusive(out, 'source-quality-omissions.jsonl', truncationText);
  await writeExclusive(out, 'replay.failures.jsonl', failuresText);
  await writeExclusive(out, 'summary.json', JSON.stringify(summary, null, 2) + '\n');
  console.log(JSON.stringify({ out, programs: records.length, annotations: selected.length,
    replayed: replay.length, exact_accepted: summary.replay.exact_accepted, failures: failures.length,
    provider_calls: 0, training_admission: false,
    cases_sha256: summary.artifacts.cases_sha256 }, null, 2));
  if (failures.length) process.exitCode = 2;
}

function partitionBatchSizes(count, limit) {
  requireValue(Number.isSafeInteger(count) && count >= 2, 'family_requires_at_least_two_selected_rows');
  const sizes = [];
  let remaining = count;
  while (remaining > limit) {
    if (remaining - limit === 1) {
      requireValue(limit >= 3, 'cannot_partition_family_without_singleton_or_oversized_batch');
      sizes.push(limit - 1);
      remaining = 2;
    } else {
      sizes.push(limit);
      remaining -= limit;
    }
  }
  requireValue(remaining >= 2, 'cannot_partition_family_without_singleton_batch');
  sizes.push(remaining);
  return sizes;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
