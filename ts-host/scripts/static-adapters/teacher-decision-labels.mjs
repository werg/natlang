#!/usr/bin/env node
/**
 * Convert already-saved typed decision labels into held, authored static Natlang IR.
 *
 * Labels are treated only as a crisp child-result source after strict schema,
 * exact source-top, and the existing scoreGraded gates all pass. This writes no
 * admission approval and does not claim teacher-authored code or reasoning.
 */
import { createHash } from 'node:crypto';
import { mkdir, open, readFile, readdir } from 'node:fs/promises';
import { dirname, resolve, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { scoreGraded, ordinalDistribution } from '../../dist/skills/graded.js';
import { admitRow } from '../../dist/teacher/curriculum.js';
import { defaultToolSurfaceHash } from '../../dist/teacher/collector.js';
import { markAuthoredStaticReferencePending, materializeNativeRows } from '../../dist/teacher/native-materializer.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';
import { renderValue } from '../../dist/native/agent.js';
import { folderDecisionScaffold } from './folder-decision-scaffold.mjs';
import { decisionCriteria, DECISION_TASK_CONTRACT_PATH, DECISION_TASK_CONTRACT_SHA256 } from './decision-task-contracts.mjs';
import { canonical } from '../advisory-file.mjs';
import { referenceRow } from '../inline-curriculum/references.mjs';
import { archiveDecisionExecutionComponents, verifyDecisionExecutionComponents } from './execution-components.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(here, '../../../');
const sha = value => createHash('sha256').update(value).digest('hex');
const jsonSha = value => sha(canonical(value));
const parse = text => JSON.parse(text);
const nonempty = value => typeof value === 'string' && value.trim().length > 0;
const lines = async path => {
  const text = await readFile(path, 'utf8');
  const rows = [];
  for (const [index, raw] of text.split(/\r?\n/).entries()) if (raw.trim()) {
    try { rows.push({ value: JSON.parse(raw), raw, line: index + 1 }); }
    catch (error) { throw new Error(`${path}:${index + 1}: invalid JSON: ${error.message}`); }
  }
  return { text, rows };
};
const hashFile = async path => sha(await readFile(path));
const exactSet = (object, expected) => object && !Array.isArray(object) && typeof object === 'object' &&
  Object.keys(object).length === expected.length && expected.every(key => Object.hasOwn(object, key));
const validDistribution = (weights, keys) => exactSet(weights, keys) && keys.every(key =>
  typeof weights[key] === 'number' && Number.isFinite(weights[key]) && weights[key] >= 0 && weights[key] <= 1) &&
  Math.abs(keys.reduce((sum, key) => sum + weights[key], 0) - 1) <= 1e-4;
const top = (weights, keys) => keys.reduce((best, key) => weights[key] > weights[best] ? key : best, keys[0]);
const stringLabel = value => typeof value === 'string' ? value : String(value);
const slug = value => String(value).toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '').slice(0, 60) || 'task';
const sourceSnapshot = 'generic-typed-decision-cases-v1';
const labelSchema = 'natlang.decision-labels/1';
const poolSchema = 'natlang.gemini-decision-pool/1';
const probabilityContract = 'natlang.typed-decision-probabilities/1';
const choiceConfidenceContract = 'natlang.choice-label-confidence/1';
const reviewedSourceFields = new Set(['id', 'family', 'kind', 'role', 'source', 'group', 'license', 'state',
  'question', 'options', 'levels', 'criteria', 'version', 'answer', 'source_refs',
  'identity_state_sha256', 'full_state_sha256']);
const pubmedReferenceFields = ['config', 'dataset', 'local_file_sha256', 'local_path', 'physical_row', 'pubmed_id',
  'row_group', 'row_in_group', 'source_row_sha256', 'split', 'upstream', 'upstream_metadata_status'];
const pubmedUpstreamFields = ['config', 'lfs_sha256', 'path', 'revision'];
const sha256Hex = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);

async function loadCanonicalSourceHolds() {
  const path = resolve(repositoryRoot, 'training/decision_source_quality_holds.json');
  const raw = await readFile(path);
  const policy = parse(raw.toString('utf8'));
  if (policy?.schema !== 'natlang.decision-source-quality-holds/1' ||
      !Array.isArray(policy.excluded_items))
    throw new Error(`${path}: invalid canonical decision source-quality hold policy`);
  const holds = new Map();
  for (const item of policy.excluded_items) {
    if (!nonempty(item?.item_id) || !nonempty(item?.original_source_group) ||
        !nonempty(item?.split) || !nonempty(item?.reason))
      throw new Error(`${path}: each hold requires item_id, original_source_group, split, and reason`);
    if (holds.has(item.item_id)) throw new Error(`${path}: duplicate hold for ${item.item_id}`);
    holds.set(item.item_id, item);
  }
  return { path, sha256: sha(raw), policy_id: policy.policy_id, holds };
}

function validateSourceProvenance(source) {
  const hasIdentityHash = Object.hasOwn(source, 'identity_state_sha256');
  const hasFullHash = Object.hasOwn(source, 'full_state_sha256');
  if (hasIdentityHash !== hasFullHash) return { error: 'incomplete_state_hash_provenance' };
  if (hasIdentityHash && (!sha256Hex(source.identity_state_sha256) || !sha256Hex(source.full_state_sha256) ||
      source.full_state_sha256 !== sha(source.state))) return { error: 'invalid_state_hash_provenance' };
  if (!Object.hasOwn(source, 'source_refs')) return {};
  if (source.source !== 'qiaojin/PubMedQA' || !Array.isArray(source.source_refs) || source.source_refs.length !== 1)
    return { error: 'unsupported_source_reference_provenance' };
  for (const ref of source.source_refs) {
    if (!exactSet(ref, pubmedReferenceFields) || ref.dataset !== source.source || ref.config !== 'pqa_labeled' ||
        ref.split !== source.role || ref.upstream_metadata_status !== 'pinned_and_byte_bound' ||
        ref.local_path !== 'source/pqa_labeled-train.parquet' || !Number.isSafeInteger(ref.physical_row) || ref.physical_row < 0 ||
        ref.row_group !== 0 ||
        !Number.isSafeInteger(ref.row_in_group) || ref.row_in_group < 0 ||
        ref.physical_row !== ref.row_in_group ||
        !Number.isSafeInteger(ref.pubmed_id) || ref.pubmed_id < 0 || !sha256Hex(ref.source_row_sha256) ||
        !sha256Hex(ref.local_file_sha256) || !exactSet(ref.upstream, pubmedUpstreamFields) ||
        ref.upstream.config !== ref.config || ref.upstream.path !== 'pqa_labeled/train-00000-of-00001.parquet' ||
        !/^[a-f0-9]{40}$/.test(ref.upstream.revision ?? '') || !sha256Hex(ref.upstream.lfs_sha256) ||
        ref.upstream.lfs_sha256 !== ref.local_file_sha256)
      return { error: 'invalid_pubmedqa_source_reference_provenance' };
  }
  return {};
}

function reject(rejections, id, reason, detail = {}) {
  rejections.push({ schema: 'natlang.teacher-decision-label-rejection/1', id: id ?? null, reason, ...detail,
    training_admission: false });
}

function metricAndDecision(source, answer, decisionContract) {
  if (source.kind === 'choice') {
    const options = source.options;
    if (!Array.isArray(options) || !options.length || !options.every(nonempty) || new Set(options).size !== options.length)
      return { error: 'invalid_source_options' };
    if (decisionContract === choiceConfidenceContract) {
      if (!answer || !exactSet(answer, ['choice', 'confidence']) || !options.includes(answer.choice) ||
          typeof answer.confidence !== 'number' || !Number.isFinite(answer.confidence) ||
          answer.confidence < 0 || answer.confidence > 1)
        return { error: 'strict_choice_confidence_schema' };
      return { predicted: answer.choice, sourceDecision: stringLabel(source.answer),
        raw: { choice: answer.choice, confidence: answer.confidence }, score: null };
    }
    if (decisionContract !== probabilityContract) return { error: 'unsupported_decision_contract' };
    if (!answer || !exactSet(answer, ['probabilities']) || !validDistribution(answer.probabilities, options))
      return { error: 'strict_probability_schema' };
    const predicted = top(answer.probabilities, options);
    const score = scoreGraded({ schema: 'natlang.skill-graded/1', kind: 'choice-brier' },
      { probabilities: answer.probabilities }, { kind: 'choice', answer: source.answer, options });
    return { predicted, sourceDecision: stringLabel(source.answer), raw: { probabilities: answer.probabilities }, score };
  }
  if (source.kind === 'noul') {
    if (decisionContract !== probabilityContract) return { error: 'unsupported_decision_contract' };
    const raw = answer?.noul;
    if (!exactSet(answer, ['noul']) || typeof raw !== 'number' || !Number.isFinite(raw) || raw < 0 || raw > 1)
      return { error: 'strict_probability_schema' };
    const predicted = raw >= 0.5;
    const target = typeof source.answer === 'boolean' ? source.answer :
      typeof source.answer === 'number' && Number.isFinite(source.answer) ? source.answer >= 0.5 : undefined;
    const score = scoreGraded({ schema: 'natlang.skill-graded/1', kind: 'binary-brier' }, raw,
      { kind: 'binary', answer: source.answer });
    return { predicted, sourceDecision: target, raw: { noul: raw }, score };
  }
  if (source.kind === 'score') {
    if (decisionContract !== probabilityContract) return { error: 'unsupported_decision_contract' };
    const levels = source.levels;
    if (!Array.isArray(levels) || levels.length < 2 || !levels.every(nonempty) || new Set(levels).size !== levels.length)
      return { error: 'invalid_source_levels' };
    if (!answer || !exactSet(answer, ['probabilities']) || !validDistribution(answer.probabilities, levels))
      return { error: 'strict_probability_schema' };
    const predicted = top(answer.probabilities, levels);
    // Numeric annotations are zero-based ordinal positions, not level names.
    // Fractional positions use the scorer's split mass; argmax chooses the
    // nearest level, with the first declared level winning an exact tie.
    const targetDistribution = ordinalDistribution(source.answer, levels);
    if (!targetDistribution) return { error: 'invalid_source_ordinal_answer' };
    const sourceDecision = top(Object.fromEntries(levels.map((name, i) => [name, targetDistribution[i]])), levels);
    const score = scoreGraded({ schema: 'natlang.skill-graded/1', kind: 'ordinal-rps' },
      { probabilities: answer.probabilities }, { kind: 'ordinal', levels, answer: source.answer });
    return { predicted, sourceDecision, raw: { probabilities: answer.probabilities }, score };
  }
  return { error: 'unsupported_source_kind' };
}

function checkedLabel(source, label) {
  if (!label || label.answer?.error) return { error: 'label_error', detail: { error: label?.answer?.error ?? 'missing_answer' } };
  const decisionContract = label.decision_contract ?? probabilityContract;
  const judged = metricAndDecision(source, label.answer, decisionContract);
  if (judged.error) return judged;
  if (decisionContract === probabilityContract &&
      (!judged.score || typeof judged.score.quality !== 'number' || !Object.values(judged.score.gates ?? {}).every(Boolean)))
    return { error: 'scoreGraded_gate_failure', detail: { score: judged.score ?? null } };
  if (judged.predicted !== judged.sourceDecision)
    return { error: 'top_decision_disagrees_with_source', detail: { top: judged.predicted, source: judged.sourceDecision } };
  return { ...judged, decisionContract };
}

function verifyManifestBinding(manifest, row, source, casesSha) {
  const provider = row?.provider;
  if (!provider || typeof provider !== 'object') return { error: 'provider_receipt_missing' };
  if (provider.cases_sha256 !== casesSha) return { error: 'provider_cases_hash_not_pinned', got: provider.cases_sha256 ?? null };
  if (provider.endpoint !== manifest.endpoint) return { error: 'provider_endpoint_not_in_manifest', got: provider.endpoint ?? null };
  const model = provider.model;
  if (!nonempty(model)) return { error: 'provider_model_missing' };
  const requestedChoiceContract = manifest.schema === labelSchema
    ? manifest.request_settings?.choice_contract
    : manifest.choice_contract;
  if (requestedChoiceContract !== undefined && !['label-confidence', 'probabilities'].includes(requestedChoiceContract))
    return { error: 'unsupported_manifest_choice_contract', got: requestedChoiceContract };
  // Confidence is a label format only for categorical choice rows. Mixed
  // manifests still carry probability distributions for score and noul rows.
  // If the source row is absent, the caller will reject its ID after provider
  // binding; there is no safe kind-specific contract to infer yet.
  if (source) {
    const expectedContract = source.kind === 'choice' && requestedChoiceContract === 'label-confidence'
      ? choiceConfidenceContract : probabilityContract;
    const declaredContract = row.decision_contract ?? probabilityContract;
    if (declaredContract !== expectedContract)
      return { error: 'label_contract_not_bound_to_manifest_choice_contract',
        got: declaredContract, expected: expectedContract, source_kind: source.kind };
  }
  if (manifest.schema === labelSchema) {
    if (!nonempty(manifest.model) || model !== manifest.model) return { error: 'provider_model_not_in_manifest', got: model };
    if (manifest.backend && provider.backend !== manifest.backend) return { error: 'provider_backend_not_in_manifest', got: provider.backend ?? null };
    if (row.teacher !== manifest.teacher) return { error: 'teacher_not_in_manifest', got: row.teacher ?? null };
    return { model, teacher: row.teacher, endpoint: provider.endpoint, cases_sha256: provider.cases_sha256,
      model_manifest: { id: manifest.model, backend: manifest.backend ?? null } };
  }
  if (manifest.schema === poolSchema) {
    const modelEntry = Array.isArray(manifest.models) ? manifest.models.find(entry => entry?.id === model) : undefined;
    if (!modelEntry) return { error: 'provider_model_not_in_manifest', got: model };
    if (row.teacher !== `google/${model}`) return { error: 'teacher_not_bound_to_manifest_model', got: row.teacher ?? null };
    return { model, teacher: row.teacher, endpoint: provider.endpoint, cases_sha256: provider.cases_sha256,
      model_manifest: structuredClone(modelEntry) };
  }
  return { error: 'unsupported_label_manifest_schema', got: manifest.schema ?? null };
}

function contractOf(source, decisionContract) {
  const options = source.kind === 'choice' ? source.options : source.kind === 'score' ? source.levels : ['false', 'true'];
  const choiceLabels = source.kind === 'noul' ? ['false', 'true'] : options;
  if (!Array.isArray(choiceLabels) || choiceLabels.length < 2 || !choiceLabels.every(nonempty)) return null;
  const defaultCriteria = source.kind === 'noul' ? { false: 'No', true: 'Yes' } :
    Object.fromEntries(choiceLabels.map(label => [label, label]));
  let criteria = defaultCriteria, taskCriteriaProvenance = null;
  if (Object.hasOwn(source, 'criteria') || source.family === 'decision:trec-question') {
    const selected = decisionCriteria({ family: source.family, source: source.source, kind: source.kind,
      labels: choiceLabels, explicitCriteria: source.criteria, where: source.id ?? source.family });
    criteria = selected.criteria;
    taskCriteriaProvenance = selected.provenance;
  }
  const signature = { kind: source.kind, family: source.family, source: source.source,
    options: source.options ?? [], levels: source.levels ?? [], criteria, license: source.license, role: source.role };
  if (decisionContract === choiceConfidenceContract) signature.decision_contract = decisionContract;
  return { kind: source.kind, decisionContract, family: source.family, source: source.source, options: source.options ?? [],
    levels: source.levels ?? [], labels: choiceLabels, criteria, taskCriteriaProvenance, license: source.license,
    signature: jsonSha(signature) };
}

function createRecord(contract, batch, sourceMeta) {
  const records = batch.map(entry => entry.source), ids = records.map(row => row.id);
  const groupIds = [...new Set(records.map(row => row.group))], first = records[0];
  const sourceRev = `${sourceSnapshot}:${sourceMeta.cases_sha256}`;
  const curriculumFamily = `teacher_decision_labels_${slug(first.family)}_${contract.kind}`;
  const record = folderDecisionScaffold({ contract, entries: batch, sourceSha: sourceMeta.cases_sha256,
    sourceSnapshot, sourceGroups: groupIds,
    curriculumFamily, recordFamily: `curriculum_${curriculumFamily}`,
    variant: contract.decisionContract === choiceConfidenceContract
      ? 'exact-choice-with-held-confidence-receipt-v1' : 'strict-top-label-v2-ordinal-source-mapping',
    splitGroup: `source-groups:${groupIds.join(',')}` });
  const childResults = Object.fromEntries(batch.map(entry => [entry.source.id, entry.decision]));
  record.source_revisions = [...new Set([sourceRev, `folder-decision-scaffold:${sourceMeta.folder_decision_scaffold_sha256}`,
    ...(contract.taskCriteriaProvenance ? [`decision-task-contracts:${contract.taskCriteriaProvenance.contract_sha256}`] : []),
    ...batch.map(entry => entry.label.file_sha256)])];
  record.license = first.license;
  record.gold_sources = [`source-decision-annotation:${sourceMeta.cases_sha256}`];
  record.dataset = { source: first.source, family: first.family, kind: first.kind, source_split: 'train' };
  record.dataset_records = ids.map(id => `${first.source}:${id}`);
  const transformations = contract.decisionContract === choiceConfidenceContract
    ? { kind: contract.kind, decision_contract: contract.decisionContract,
      from: 'typed exact choice with a bounded scalar confidence receipt',
      to: 'the exact teacher choice, admitted only when it equals the source annotation',
      probabilities_normalized: false, synthetic_probabilities_created: false,
      confidence_retained_as_scalar: true, gold_in_prompt: false, reasoning_inferred: false }
    : { kind: contract.kind, from: 'strict probability judgment',
      to: contract.kind === 'noul' ? 'Boolean threshold at p >= 0.5; true means yes is at least as likely as no' : 'highest-probability source option/level',
      probabilities_normalized: false, gold_in_prompt: false, reasoning_inferred: false };
  record.generation = {
    ...record.generation,
    generator: 'natlang.static_teacher_decision_labels_adapter/1',
    origin: 'authored-static-scaffold-from-held-labels',
    teacher_output_used_only_as_crisp_child_result: true,
    teacher_generated_scaffold: false,
    teacher_generated_reasoning: false,
    provider_calls: 0,
    training_admission: false,
    source_cases_sha256: sourceMeta.cases_sha256,
    folder_decision_scaffold_sha256: sourceMeta.folder_decision_scaffold_sha256,
    decision_task_contract: contract.taskCriteriaProvenance,
    folder_decision_scaffold_components: sourceMeta.folder_decision_scaffold_components,
    label_file_sha256s: [...new Set(batch.map(entry => entry.label.file_sha256))],
    label_manifest_sha256s: [...new Set(batch.map(entry => entry.label.artifact.manifest_sha256))],
    scorer_source_sha256: sourceMeta.scorer_source_sha256,
    scorer_dist_sha256: sourceMeta.scorer_dist_sha256,
    transformations,
    source_provenance_receipts: records.filter(row => Object.hasOwn(row, 'source_refs') ||
      Object.hasOwn(row, 'identity_state_sha256') || Object.hasOwn(row, 'full_state_sha256')).map(row => ({
      source_id: row.id,
      ...(Object.hasOwn(row, 'source_refs') ? { source_refs: structuredClone(row.source_refs) } : {}),
      ...(Object.hasOwn(row, 'identity_state_sha256') ? {
        identity_state_sha256: row.identity_state_sha256, full_state_sha256: row.full_state_sha256,
      } : {}),
    })),
    label_receipts: batch.map(entry => ({ id: entry.source.id, label_file_sha256: entry.label.file_sha256,
      label_manifest_sha256: entry.label.artifact.manifest_sha256, label_manifest_schema: entry.label.artifact.schema,
      manifest_binding: entry.label.binding, label_row_sha256: entry.label.row_sha256,
      response_sha256: entry.label.value.provider?.response_sha256 ?? null,
      response_content_sha256: entry.label.value.provider?.response_content_sha256 ?? null,
      request_sha256: entry.label.value.provider?.request_sha256 ?? null,
      decision: entry.decision,
      ...(entry.label.value.decision_contract === choiceConfidenceContract
        ? { decision_contract: choiceConfidenceContract, confidence_receipt: entry.raw }
        : { probabilities: entry.raw }), score: entry.score,
      teacher: entry.label.value.teacher ?? null, provider_model: entry.label.value.provider?.model ?? null })),
  };
  return record;
}

async function writeExclusive(dir, name, content) {
  const handle = await open(resolve(dir, name), 'wx');
  try { await handle.writeFile(content); } finally { await handle.close(); }
}

function replayCaptureVisibility(row, record) {
  const task = JSON.parse(record.semantics.folder_files['task.json']);
  const items = Object.entries(record.semantics.folder_files)
    .filter(([path]) => path.startsWith('items/') && path.endsWith('.json'))
    .map(([, text]) => JSON.parse(text));
  const results = [];
  for (const item of items) {
    const captures = { caseId: item.id, state: item.state, question: item.question,
      criteria: Object.hasOwn(item, 'criteria') ? item.criteria : task.criteria,
      options: item.options ?? task.options ?? [], levels: item.levels ?? task.levels ?? [] };
    const matching = row.trajectory.map((turn, index) => ({ turn, index })).find(({ turn }) => {
      const context = (turn.context ?? []).map(message => String(message.content ?? '')).join('\n');
      return context.includes('nl@eval') && context.includes(item.id) &&
        (turn.assistant?.calls ?? []).some(call => call.tool === 'return_result' &&
          canonical(call.arguments?.value) === canonical(record.semantics.expected[item.id]));
    });
    const context = matching ? (matching.turn.context ?? []).map(message => String(message.content ?? '')).join('\n') : '';
    const fields = Object.keys(captures);
    // Native scope declarations render values as typed expressions (for example,
    // `criteria: unknown = { ... }`), not as JSON.stringify output. Require the
    // entire exact value in either supported representation. Rendering with an
    // infinite budget intentionally makes a truncated native preview fail this
    // check; do not infer visibility from the host-side capture or partial fields.
    const visible = fields.filter(name => {
      const jsonValue = JSON.stringify(captures[name]);
      const nativeValue = `unknown = ${renderValue(captures[name], { budget: Infinity })}`;
      return context.includes(jsonValue) || context.includes(`${name}: ${nativeValue}`);
    });
    results.push({ id: item.id, fields, visible_fields: visible,
      missing_fields: fields.filter(name => !visible.includes(name)), decision_turn_index: matching?.index ?? null });
  }
  return { visible: results.every(result => result.missing_fields.length === 0), items: results };
}

async function main() {
  const usage = 'usage: teacher-decision-labels.mjs --cases CASES.jsonl --labels LABELS.jsonl [--labels ...] --out NEW_DIR [--batch-size 8] [--replay-limit 3]';
  const { values } = parseArgs({ options: {
    help: { type: 'boolean' },
    cases: { type: 'string' }, labels: { type: 'string', multiple: true }, out: { type: 'string' },
    'batch-size': { type: 'string', default: '8' }, 'replay-limit': { type: 'string', default: '3' },
  } });
  if (values.help) { console.log(usage); return; }
  if (!values.cases || !values.labels?.length || !values.out)
    throw new Error(usage);
  const casesPath = resolve(values.cases), labelPaths = values.labels.map(path => resolve(path)), outDir = resolve(values.out);
  const batchSize = Number(values['batch-size']), replayLimit = Number(values['replay-limit']);
  if (!Number.isSafeInteger(batchSize) || batchSize < 1 || batchSize > 64 || !Number.isSafeInteger(replayLimit) || replayLimit < 0)
    throw new Error('batch-size must be 1..64 and replay-limit must be a non-negative integer');
  const caseFile = await lines(casesPath), casesSha = sha(caseFile.text);
  const sourceManifestPath = resolve(dirname(casesPath), 'source-manifest.json');
  const sourceManifest = parse(await readFile(sourceManifestPath, 'utf8'));
  if (sourceManifest.selected_sha256 && sourceManifest.selected_sha256 !== casesSha ||
      sourceManifest.cases_sha256 && sourceManifest.cases_sha256 !== casesSha)
    throw new Error(`source manifest does not pin ${casesPath}`);
  const cases = new Map();
  for (const row of caseFile.rows) {
    if (!row.value || !nonempty(row.value.id)) throw new Error(`${casesPath}:${row.line}: missing id`);
    if (cases.has(row.value.id)) throw new Error(`duplicate source id ${row.value.id}`);
    cases.set(row.value.id, row.value);
  }
  const sourceQualityPolicy = await loadCanonicalSourceHolds();
  const artifacts = [];
  const labelById = new Map(), rejections = [];
  for (const path of labelPaths) {
    const labelFile = await lines(path), manifestPath = `${path}.manifest.json`;
    const labelManifest = parse(await readFile(manifestPath, 'utf8'));
    const labelsSha = sha(labelFile.text), manifestSha = sha(await readFile(manifestPath));
    if (![labelSchema, poolSchema].includes(labelManifest.schema) || labelManifest.cases_sha256 !== casesSha)
      throw new Error(`${manifestPath}: schema/cases hash mismatch`);
    if (labelManifest.schema === labelSchema && (!nonempty(labelManifest.model) || !nonempty(labelManifest.endpoint) || !nonempty(labelManifest.teacher)))
      throw new Error(`${manifestPath}: incomplete natlang.decision-labels/1 provider binding`);
    if (labelManifest.schema === poolSchema && (!Array.isArray(labelManifest.models) || !labelManifest.models.length || !nonempty(labelManifest.endpoint)))
      throw new Error(`${manifestPath}: incomplete natlang.gemini-decision-pool/1 provider binding`);
    const artifact = { path, labels_sha256: labelsSha, label_rows: labelFile.rows.length, schema: labelManifest.schema,
      manifest_path: manifestPath, manifest_sha256: manifestSha, manifest: labelManifest };
    artifacts.push(artifact);
    for (const row of labelFile.rows) {
      const binding = verifyManifestBinding(labelManifest, row.value, cases.get(row.value?.id), casesSha);
      if (binding.error) { reject(rejections, row.value?.id, binding.error, { label_file: path, line: row.line,
        detail: binding }); continue; }
      const label = { value: row.value, row_sha256: sha(row.raw), line: row.line, file: path,
        file_sha256: labelsSha, artifact, binding };
      if (!cases.has(row.value?.id)) { reject(rejections, row.value?.id, 'label_id_not_in_pinned_source', { label_file: path, line: row.line }); continue; }
      const previous = labelById.get(row.value.id);
      if (previous) {
        labelById.delete(row.value.id);
        labelById.set(row.value.id, { duplicate: true, labels: [previous, label] });
        reject(rejections, row.value.id, 'duplicate_label_id_across_or_within_inputs', { label_files: [previous.file, path] });
      } else labelById.set(row.value.id, label);
    }
  }
  const sourceMeta = { cases_path: casesPath, cases_sha256: casesSha, source_manifest_path: sourceManifestPath,
    source_manifest_sha256: sha(await readFile(sourceManifestPath)), source_manifest: sourceManifest,
    source_quality_holds: { path: sourceQualityPolicy.path, sha256: sourceQualityPolicy.sha256,
      merge_semantics: 'Canonical holds are always applied before strict-label eligibility; each held source ID is rejected regardless of annotation match.' },
    scorer: 'scoreGraded from ts-host/src/skills/graded.ts; schema natlang.skill-graded/1',
    scorer_source_sha256: await hashFile(resolve(here, '../../src/skills/graded.ts')),
    scorer_dist_sha256: await hashFile(resolve(here, '../../dist/skills/graded.js')) };
  const sourceFields = [...new Set([...cases.values()].flatMap(source => Object.keys(source)))].sort();
  const eligible = [];
  for (const [id, source] of cases) {
    const sourceHold = sourceQualityPolicy.holds.get(id);
    if (sourceHold) {
      if (source.group !== sourceHold.original_source_group || source.role !== sourceHold.split)
        throw new Error(`${id}: canonical source-quality hold group/split does not match pinned source row`);
      reject(rejections, id, 'canonical_source_quality_hold', {
        detail: sourceHold.reason,
        source_group: source.group,
        split: source.role,
        policy_path: sourceQualityPolicy.path,
        policy_sha256: sourceQualityPolicy.sha256,
        policy_id: sourceQualityPolicy.policy_id,
        evidence: sourceHold.evidence ?? {},
        decision_receipt_sha256: sourceHold.decision_receipt_sha256,
        adapter_code_path: fileURLToPath(import.meta.url),
        adapter_code_sha256: await hashFile(fileURLToPath(import.meta.url))
      });
      continue;
    }
    const unknownFields = Object.keys(source).filter(key => !reviewedSourceFields.has(key));
    if (unknownFields.length) { reject(rejections, id, 'unreviewed_source_fields_not_exposed', {
      fields: unknownFields, source_record_canonical_sha256: sha(JSON.stringify(canonical(source))) }); continue; }
    const sourceProvenance = validateSourceProvenance(source);
    if (sourceProvenance.error) { reject(rejections, id, sourceProvenance.error, {
      source_record_canonical_sha256: sha(JSON.stringify(canonical(source))) }); continue; }
    if (source.role !== 'train') { reject(rejections, id, 'source_split_not_train', { role: source.role }); continue; }
    const label = labelById.get(id);
    if (!label) { reject(rejections, id, 'missing_label_for_source_case'); continue; }
    if (label.duplicate) continue;
    const result = checkedLabel(source, label.value);
    if (result.error) { reject(rejections, id, result.error, { detail: result.detail ?? null,
      label_file: label.file, label_row_sha256: label.row_sha256 }); continue; }
    const contract = contractOf(source, result.decisionContract);
    if (!contract || !nonempty(source.state) || !nonempty(source.question) || !nonempty(source.group) ||
        !nonempty(source.source) || typeof source.license !== 'string') {
      reject(rejections, id, 'incomplete_or_incompatible_source_contract', { family: source.family, kind: source.kind }); continue;
    }
    const decision = source.kind === 'noul' ? result.predicted : stringLabel(result.predicted);
    eligible.push({ source, label, contract, decision, raw: result.raw, score: result.score });
  }
  if (!eligible.length) throw new Error('no source labels passed strict schema, scoreGraded gates, and exact-top checks');

  const grouped = new Map();
  for (const entry of eligible) {
    const list = grouped.get(entry.contract.signature) ?? [];
    list.push(entry); grouped.set(entry.contract.signature, list);
  }
  const batches = [];
  for (const entries of grouped.values()) {
    entries.sort((a, b) => a.source.id.localeCompare(b.source.id));
    const byGroup = new Map();
    for (const entry of entries) { const group = byGroup.get(entry.source.group) ?? []; group.push(entry); byGroup.set(entry.source.group, group); }
    let batch = [];
    for (const groupEntries of byGroup.values()) {
      // A source group stays indivisible, even if it makes a batch exceed the requested size.
      if (batch.length && batch.length + groupEntries.length > batchSize) { batches.push({ contract: entries[0].contract, entries: batch }); batch = []; }
      batch.push(...groupEntries);
    }
    if (batch.length) batches.push({ contract: entries[0].contract, entries: batch });
  }
  const adapterSha = await hashFile(fileURLToPath(import.meta.url));
  const folderDecisionScaffoldSha = await hashFile(new URL('./folder-decision-scaffold.mjs', import.meta.url));
  const scaffoldDependencyComponents = {
    [fileURLToPath(new URL('./folder-decision-scaffold.mjs', import.meta.url))]: folderDecisionScaffoldSha,
    [fileURLToPath(new URL('../advisory-file.mjs', import.meta.url))]: await hashFile(new URL('../advisory-file.mjs', import.meta.url)),
    [fileURLToPath(new URL('./decision-task-contracts.mjs', import.meta.url))]: await hashFile(new URL('./decision-task-contracts.mjs', import.meta.url)),
    [DECISION_TASK_CONTRACT_PATH]: await hashFile(DECISION_TASK_CONTRACT_PATH),
  };
  const records = batches.map(batch => createRecord(batch.contract, batch.entries,
    { ...sourceMeta, adapter_sha256: adapterSha, folder_decision_scaffold_sha256: folderDecisionScaffoldSha,
      folder_decision_scaffold_components: scaffoldDependencyComponents }));
  const toolSurfaceSha256 = await defaultToolSurfaceHash();
  const replayRows = [], replayFailures = [], nativeRows = [];
  const options = { modelId: 'held-decision-label-static-reference', rootSeed: 4041, systemPrompt: TOOLS_PROMPT,
    contextTokens: 32768, maxTurns: 128, toolSurfaceSha256, collectionRole: 'reference',
    authoredActionPlans: false };
  await mkdir(outDir, { recursive: true });
  if ((await readdir(outDir)).length) throw new Error(`refusing to overwrite non-empty output directory: ${outDir}`);
  const executionArchive = await archiveDecisionExecutionComponents({ outDir,
    adapterPath: fileURLToPath(import.meta.url), additional: [
      { path: resolve(here, 'common.mjs'), role: 'adapter-validation-and-hashing' },
      { path: resolve(here, 'decisions.mjs'), role: 'decision-contract-plumbing' },
      { path: resolve(here, '../../src/skills/graded.ts'), role: 'scorer-source-reference' },
      { path: resolve(here, '../../dist/skills/graded.js'), role: 'loaded-scorer-runtime' },
    ] });
  const replaySelection = [];
  for (const kind of ['choice', 'noul', 'score']) {
    const candidate = records.find(record => record.generation.transformations.kind === kind);
    if (candidate && replaySelection.length < replayLimit) replaySelection.push(candidate);
  }
  for (const record of records) if (replaySelection.length < replayLimit && !replaySelection.includes(record)) replaySelection.push(record);
  for (let index = 0; index < replaySelection.length; index++) {
    const record = replaySelection[index];
    try {
      const row = await referenceRow(record, index, options);
      const admission = admitRow(row);
      const captureVisibility = replayCaptureVisibility(row, record);
      replayRows.push({ row, admission, capture_visibility: captureVisibility });
      const materialized = materializeNativeRows([row]);
      nativeRows.push(...markAuthoredStaticReferencePending(materialized.turns));
      if (!admission.admitted || !row.outcome?.accepted || !captureVisibility.visible) replayFailures.push({ id: record.id,
        admitted: admission.admitted, reasons: admission.reasons, notes: admission.notes, outcome: row.outcome ?? null,
        capture_visibility: captureVisibility });
    } catch (error) {
      replayFailures.push({ id: record.id, error: error instanceof Error ? error.message : String(error),
        retained_case: true, training_admission: false });
    }
  }
  const replayOptsSha = jsonSha({ ...options, systemPrompt_sha256: sha(TOOLS_PROMPT) });
  const rootContext = `Label-derived static controller/reference generation; zero provider calls; no teacher-generated scaffold/reasoning. Gold annotations and teacher judgment receipts are absent from model-facing folder files and prompts.`;
  const summary = { schema: 'natlang.held-static-decision-adapter-summary/1', created_at: new Date().toISOString(),
    training_admission: false, cases: records.length, source_labels_accepted: eligible.length,
    source_cases: cases.size, label_rows: artifacts.reduce((sum, item) => sum + item.label_rows, 0),
    replay_attempted: replayRows.length, replay_failures: replayFailures.length, native_turns: nativeRows.length,
    rejection_rows: rejections.length, batch_size_limit: batchSize, group_preserving_oversized_batches: batches.filter(batch => batch.entries.length > batchSize).length,
    replay_options_sha256: replayOptsSha, replay_context: rootContext,
    files: { source_cases_sha256: casesSha, adapter_sha256: adapterSha,
      folder_decision_scaffold_sha256: folderDecisionScaffoldSha,
      folder_decision_scaffold_components: scaffoldDependencyComponents,
      scorer_source_sha256: sourceMeta.scorer_source_sha256, scorer_dist_sha256: sourceMeta.scorer_dist_sha256,
      tool_surface_sha256: toolSurfaceSha256 } };

  summary.execution_component_archive = executionArchive;
  await writeExclusive(outDir, 'cases.jsonl', records.map(row => JSON.stringify(row)).join('\n') + '\n');
  await writeExclusive(outDir, 'replay.results.jsonl', replayRows.map(entry => JSON.stringify(entry)).join('\n') + (replayRows.length ? '\n' : ''));
  await writeExclusive(outDir, 'native.held.jsonl', nativeRows.map(row => JSON.stringify(row)).join('\n') + (nativeRows.length ? '\n' : ''));
  await writeExclusive(outDir, 'replay.failures.jsonl', replayFailures.map(row => JSON.stringify(row)).join('\n') + (replayFailures.length ? '\n' : ''));
  await writeExclusive(outDir, 'rejections.jsonl', rejections.map(row => JSON.stringify(row)).join('\n') + (rejections.length ? '\n' : ''));
  const executionComponentVerification = await verifyDecisionExecutionComponents(executionArchive);
  summary.execution_component_verification = executionComponentVerification;
  await writeExclusive(outDir, 'source-manifest.json', JSON.stringify({ schema: 'natlang.held-static-decision-source-manifest/1',
    training_admission: false, ...sourceMeta, execution_component_archive: executionArchive,
    execution_component_verification: executionComponentVerification,
    folder_decision_scaffold_sha256: folderDecisionScaffoldSha,
    folder_decision_scaffold_components: scaffoldDependencyComponents,
    labels: artifacts.map(item => ({ path: item.path,
      labels_sha256: item.labels_sha256, label_rows: item.label_rows, manifest_path: item.manifest_path, manifest_sha256: item.manifest_sha256,
      manifest: item.manifest })), adapter_sha256: adapterSha,
    decision_contracts: [...new Set(eligible.map(entry => entry.label.value.decision_contract ?? probabilityContract))],
    transformations: 'Probability-contract rows retain raw probability receipts, use argmax (first declared option wins ties), and pass existing scoreGraded gates without normalization. Choice-label-confidence rows retain choice and bounded confidence as separate typed values, create no probability distribution, and pass only when choice exactly matches the source annotation. Binary noul uses p>=0.5. Gold annotations stay out of model-facing files and prompts.',
    model_visible_fields: [...new Set([...sourceFields.filter(key => key !== 'answer'), 'criteria'])],
    criteria_provenance: { source: 'explicit source criteria wins after exact label-key validation; otherwise the shared versioned family task contract is applied when declared; the historical label-name fallback is used only for families with no shared contract',
      contract_path: DECISION_TASK_CONTRACT_PATH, contract_sha256: DECISION_TASK_CONTRACT_SHA256 },
    hidden_fields: ['source answer annotation', 'teacher judgment payload', 'provider response content/hash receipt'],
    source_field_review: { observed_fields: sourceFields, reviewed_visible_fields: sourceFields.filter(key => key !== 'answer'),
      hidden_source_fields: sourceFields.filter(key => key === 'answer'),
      unreviewed_fields_rejected_without_prompt_exposure: true },
    source_quality_hold_policy: sourceMeta.source_quality_holds,
    source_cases_without_generated_static_row: cases.size - eligible.length,
    label_rejection_records: rejections.length }, null, 2) + '\n');
  await writeExclusive(outDir, 'label-manifest.json', JSON.stringify({ schema: 'natlang.held-static-decision-label-manifest/1',
    training_admission: false, label_schemas: [...new Set(artifacts.map(item => item.schema))], files: artifacts.map(item => ({ path: item.path,
      sha256: item.labels_sha256, manifest_path: item.manifest_path, manifest_sha256: item.manifest_sha256,
      schema: item.schema, source_cases_sha256: item.manifest.cases_sha256, teacher: item.manifest.teacher ?? null,
      model: item.manifest.model ?? null, models: item.manifest.models ?? null, endpoint: item.manifest.endpoint,
      request_settings: item.manifest.request_settings, retry_policy: item.manifest.retry_policy })),
    accepted_rows: eligible.map(entry => ({ id: entry.source.id, source_group: entry.source.group,
      label_file_sha256: entry.label.file_sha256, label_manifest_sha256: entry.label.artifact.manifest_sha256,
      label_manifest_schema: entry.label.artifact.schema, manifest_binding: entry.label.binding,
      label_row_sha256: entry.label.row_sha256, teacher: entry.label.value.teacher,
      provider_model: entry.label.value.provider?.model, provider_endpoint: entry.label.value.provider?.endpoint,
      provider_cases_sha256: entry.label.value.provider?.cases_sha256,
      response_sha256: entry.label.value.provider?.response_sha256 ?? null,
      response_content_sha256: entry.label.value.provider?.response_content_sha256 ?? null,
      request_sha256: entry.label.value.provider?.request_sha256 ?? null,
      decision_contract: entry.label.value.decision_contract ?? probabilityContract,
      strict_top: entry.decision, raw_judgment_receipt: entry.raw,
      ...((entry.label.value.decision_contract ?? probabilityContract) === probabilityContract
        ? { raw_probability_receipt: entry.raw } : {}),
      score: entry.score })),
    training_admission: false }, null, 2) + '\n');
  await writeExclusive(outDir, 'scorer-manifest.json', JSON.stringify({ schema: 'natlang.held-static-decision-scorer-manifest/1',
    training_admission: false, scorer: 'scoreGraded', scorer_schema: 'natlang.skill-graded/1',
    source_file: resolve(here, '../../src/skills/graded.ts'), source_sha256: sourceMeta.scorer_source_sha256,
    loaded_module: resolve(here, '../../dist/skills/graded.js'), module_sha256: sourceMeta.scorer_dist_sha256,
    strict_schema: 'Probability contract: exact answer/distribution keys, finite numeric [0,1] values and total within 1e-4 of 1, without normalization. Choice-label-confidence contract: exact choice/confidence keys, declared choice, finite confidence in [0,1].',
    gates: { choice: ['distribution', 'top_correct'], choice_label_confidence: ['strict choice and confidence schema', 'exact source-choice match'], noul: ['probability', 'correct_side'], score: ['distribution', 'within_half_level'] },
    exact_top: 'Probability choices use argmax in source option order (first on ties); choice-label-confidence uses the directly supplied declared choice; noul p>=0.5; ordinal must equal argmax of shared ordinalDistribution(source answer, levels), including zero-based numeric and fractional positions',
    invocation: `Imported scoreGraded directly by ${basename(fileURLToPath(import.meta.url))}; see source-manifest.json for adapter and source paths.`,
    accepted_score_details: eligible.filter(entry => entry.score).map(entry => ({ id: entry.source.id, kind: entry.source.kind,
      quality: entry.score.quality, gates: entry.score.gates, detail: entry.score.detail })),
    accepted_choice_confidence_details: eligible.filter(entry =>
      (entry.label.value.decision_contract ?? probabilityContract) === choiceConfidenceContract).map(entry => ({
      id: entry.source.id, choice: entry.raw.choice, confidence: entry.raw.confidence,
      exact_source_choice_match: entry.raw.choice === entry.source.answer })) }, null, 2) + '\n');
  await writeExclusive(outDir, 'summary.json', JSON.stringify(summary, null, 2) + '\n');
  console.log(JSON.stringify({ out: outDir, cases: records.length, eligible_labels: eligible.length,
    rejections: rejections.length, replay_attempted: replayRows.length, replay_failures: replayFailures.length,
    training_admission: false }));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
