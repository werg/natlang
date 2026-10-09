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
import { scoreGraded } from '../../dist/skills/graded.js';
import { admitRow } from '../../dist/teacher/curriculum.js';
import { defaultToolSurfaceHash } from '../../dist/teacher/collector.js';
import { markAuthoredStaticReferencePending, materializeNativeRows } from '../../dist/teacher/native-materializer.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';
import { curriculumCase, evalCall, returnCall } from '../inline-curriculum/lib.mjs';
import { referenceRow } from '../inline-curriculum/references.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const sha = value => createHash('sha256').update(value).digest('hex');
const canonical = value => Array.isArray(value) ? value.map(canonical) : value && typeof value === 'object' ?
  Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])])) : value;
const jsonSha = value => sha(JSON.stringify(canonical(value)));
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
const reviewedSourceFields = new Set(['id', 'family', 'kind', 'role', 'source', 'group', 'license', 'state',
  'question', 'options', 'levels', 'criteria', 'version', 'answer']);

function reject(rejections, id, reason, detail = {}) {
  rejections.push({ schema: 'natlang.teacher-decision-label-rejection/1', id: id ?? null, reason, ...detail,
    training_admission: false });
}

function metricAndDecision(source, answer) {
  if (source.kind === 'choice') {
    const options = source.options;
    if (!Array.isArray(options) || !options.length || !options.every(nonempty) || new Set(options).size !== options.length)
      return { error: 'invalid_source_options' };
    if (!answer || !exactSet(answer, ['probabilities']) || !validDistribution(answer.probabilities, options))
      return { error: 'strict_probability_schema' };
    const predicted = top(answer.probabilities, options);
    const score = scoreGraded({ schema: 'natlang.skill-graded/1', kind: 'choice-brier' },
      { probabilities: answer.probabilities }, { kind: 'choice', answer: source.answer, options });
    return { predicted, sourceDecision: stringLabel(source.answer), raw: { probabilities: answer.probabilities }, score };
  }
  if (source.kind === 'noul') {
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
    const levels = source.levels;
    if (!Array.isArray(levels) || levels.length < 2 || !levels.every(nonempty) || new Set(levels).size !== levels.length)
      return { error: 'invalid_source_levels' };
    if (!answer || !exactSet(answer, ['probabilities']) || !validDistribution(answer.probabilities, levels))
      return { error: 'strict_probability_schema' };
    const predicted = top(answer.probabilities, levels);
    const sourceDecision = stringLabel(source.answer);
    const score = scoreGraded({ schema: 'natlang.skill-graded/1', kind: 'ordinal-rps' },
      { probabilities: answer.probabilities }, { kind: 'ordinal', levels, answer: source.answer });
    return { predicted, sourceDecision, raw: { probabilities: answer.probabilities }, score };
  }
  return { error: 'unsupported_source_kind' };
}

function checkedLabel(source, label) {
  if (!label || label.answer?.error) return { error: 'label_error', detail: { error: label?.answer?.error ?? 'missing_answer' } };
  const judged = metricAndDecision(source, label.answer);
  if (judged.error) return judged;
  if (!judged.score || typeof judged.score.quality !== 'number' || !Object.values(judged.score.gates ?? {}).every(Boolean))
    return { error: 'scoreGraded_gate_failure', detail: { score: judged.score ?? null } };
  if (judged.predicted !== judged.sourceDecision)
    return { error: 'top_decision_disagrees_with_source', detail: { top: judged.predicted, source: judged.sourceDecision } };
  return judged;
}

function verifyManifestBinding(manifest, row, casesSha) {
  const provider = row?.provider;
  if (!provider || typeof provider !== 'object') return { error: 'provider_receipt_missing' };
  if (provider.cases_sha256 !== casesSha) return { error: 'provider_cases_hash_not_pinned', got: provider.cases_sha256 ?? null };
  if (provider.endpoint !== manifest.endpoint) return { error: 'provider_endpoint_not_in_manifest', got: provider.endpoint ?? null };
  const model = provider.model;
  if (!nonempty(model)) return { error: 'provider_model_missing' };
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

function contractOf(source) {
  const options = source.kind === 'choice' ? source.options : source.kind === 'score' ? source.levels : ['false', 'true'];
  const choiceLabels = source.kind === 'noul' ? ['false', 'true'] : options;
  if (!Array.isArray(choiceLabels) || choiceLabels.length < 2 || !choiceLabels.every(nonempty)) return null;
  const defaultCriteria = source.kind === 'noul' ? { false: 'No', true: 'Yes' } :
    Object.fromEntries(choiceLabels.map(label => [label, label]));
  const criteria = Object.hasOwn(source, 'criteria') ? source.criteria : defaultCriteria;
  return { kind: source.kind, family: source.family, source: source.source, options: source.options ?? [],
    levels: source.levels ?? [], labels: choiceLabels, criteria, license: source.license,
    signature: jsonSha({ kind: source.kind, family: source.family, source: source.source,
      options: source.options ?? [], levels: source.levels ?? [], criteria, license: source.license, role: source.role }) };
}

function createRecord(contract, batch, sourceMeta) {
  const records = batch.map(entry => entry.source), childResults = Object.fromEntries(batch.map(entry => [entry.source.id, entry.decision]));
  const ids = records.map(row => row.id), groupIds = [...new Set(records.map(row => row.group))];
  const first = records[0], dataDirectory = 'items';
  const task = {
    family: first.family, kind: first.kind, options: contract.options, levels: contract.levels,
    labels: contract.labels, criteria: contract.criteria,
    output_path: 'decisions.json',
  };
  const folderFiles = { 'task.json': JSON.stringify(task) };
  for (const row of records) {
    // Preserve every visible source fact; only source.answer is hidden from the model-facing folder.
    const { answer: _hiddenAnnotation, ...visible } = row;
    folderFiles[`${dataDirectory}/${row.id}.json`] = JSON.stringify(visible);
  }
  const expectedFiles = { ...folderFiles, 'decisions.json': JSON.stringify(childResults) };
  const union = contract.labels.map(JSON.stringify).join(' | ');
  const returnType = contract.kind === 'noul' ? 'boolean' : 'Decision';
  const outputType = contract.kind === 'noul' ? 'Record<string, boolean>' : 'Record<string, Decision>';
  const kindInstruction = contract.kind === 'noul'
    ? 'Return true when yes is at least as likely as no; otherwise return false. Do not return a numeric probability.'
    : 'Apply the scoped criteria and return exactly one declared option or level. Do not return a probability distribution.';
  const controller = `const task = await folder.file('task.json').readJson();
const decisions: ${outputType} = {};
for (const file of await folder.files('${dataDirectory}/*.json')) {
 const item = await file.readJson();
 const caseId = item.id;
 const state = item.state;
 const question = item.question;
 const criteria = Object.hasOwn(item, 'criteria') ? item.criteria : task.criteria;
 const options = item.options ?? task.options ?? [];
 const levels = item.levels ?? task.levels ?? [];
 const decision = await nl<${returnType}>\`Question: \${question}\\nAnswer from the scoped state. Apply the scoped criteria. ${kindInstruction}\`(caseId, state, question, criteria, options, levels);
 decisions[caseId] = decision;
}
await folder.file(task.output_path).writeText(JSON.stringify(decisions));
return decisions;`;
  const childRefs = batch.map(entry => ({
    match: [entry.source.id],
    calls: [
      evalCall('console.log(JSON.stringify({ caseId, state, question, criteria, options, levels }));'),
      returnCall(entry.decision),
    ],
  }));
  const sourceRev = `${sourceSnapshot}:${sourceMeta.cases_sha256}`;
  const batchKey = jsonSha({ signature: contract.signature, ids: ids.slice().sort(), groups: groupIds.slice().sort() }).slice(0, 24);
  const record = curriculumCase({
    family: `teacher_decision_labels_${slug(first.family)}_${contract.kind}`,
    shape: `source_${sourceMeta.cases_sha256.slice(0, 12)}_${batchKey}`,
    variant: 'strict-top-label-v1', split: 'train', splitGroup: `source-groups:${groupIds.join(',')}`,
    slice: 'inline_placement', domain: 'other', mode: 'single_call', inline: 'required',
    root: { name: 'reduce_decisions', kind: 'directory-reducer', args: {}, returns: outputType,
      instructions: 'Read the task contract and every item in the directory. For each item, bind its case ID, state, question, criteria, options, and levels as separate scope values. Compose the item question into the inline natural-language instruction at runtime, and also pass question, state, criteria, options, and levels as separate arguments. Do not combine the fields into a serialized prompt variable. Save decisions.json and return that exact map.' },
    files: { 'types.ts': `export type Decision = ${union};\n` },
    folderFiles, expectedFiles, expected: childResults,
    minimumSequence: ['read the runtime contract and directory items', 'capture case ID, state, question, criteria, options and levels as separate scope values',
      `compose the item question into the typed inline instruction and follow the ${contract.kind}-specific output rule`,
      'pass all captures separately for every item', 'save and return the exact aggregate map'],
    reference: { root: [evalCall(controller), returnCall(childResults)], children: childRefs },
  });
  record.family = `curriculum_teacher_decision_labels_${slug(first.family)}_${contract.kind}`;
  record.source = 'held-provider-decision-label-adapter';
  record.source_ids = ids.slice();
  record.source_groups = groupIds.slice();
  record.source_revisions = [...new Set([sourceRev, ...batch.map(entry => entry.label.file_sha256)])];
  record.license = first.license;
  record.gold_sources = [`source-decision-annotation:${sourceMeta.cases_sha256}`];
  record.dataset = { source: first.source, family: first.family, kind: first.kind, source_split: 'train' };
  record.dataset_records = ids.map(id => `${first.source}:${id}`);
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
    label_file_sha256s: [...new Set(batch.map(entry => entry.label.file_sha256))],
    label_manifest_sha256s: [...new Set(batch.map(entry => entry.label.artifact.manifest_sha256))],
    scorer_source_sha256: sourceMeta.scorer_source_sha256,
    scorer_dist_sha256: sourceMeta.scorer_dist_sha256,
    transformations: { kind: contract.kind, from: 'strict probability judgment', to: contract.kind === 'noul' ? 'Boolean threshold at p >= 0.5; true means yes is at least as likely as no' : 'highest-probability source option/level',
      probabilities_normalized: false, gold_in_prompt: false, reasoning_inferred: false },
    label_receipts: batch.map(entry => ({ id: entry.source.id, label_file_sha256: entry.label.file_sha256,
      label_manifest_sha256: entry.label.artifact.manifest_sha256, label_manifest_schema: entry.label.artifact.schema,
      manifest_binding: entry.label.binding, label_row_sha256: entry.label.row_sha256,
      response_sha256: entry.label.value.provider?.response_sha256 ?? null,
      response_content_sha256: entry.label.value.provider?.response_content_sha256 ?? null,
      request_sha256: entry.label.value.provider?.request_sha256 ?? null,
      decision: entry.decision, probabilities: entry.raw, score: entry.score,
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
    const visible = fields.filter(name => context.includes(JSON.stringify(captures[name])));
    results.push({ id: item.id, fields, visible_fields: visible,
      missing_fields: fields.filter(name => !visible.includes(name)), decision_turn_index: matching?.index ?? null });
  }
  return { visible: results.every(result => result.missing_fields.length === 0), items: results };
}

async function main() {
  const { values } = parseArgs({ options: {
    cases: { type: 'string' }, labels: { type: 'string', multiple: true }, out: { type: 'string' },
    'batch-size': { type: 'string', default: '8' }, 'replay-limit': { type: 'string', default: '3' },
  } });
  if (!values.cases || !values.labels?.length || !values.out)
    throw new Error('usage: teacher-decision-labels.mjs --cases CASES.jsonl --labels LABELS.jsonl [--labels ...] --out NEW_DIR [--batch-size 8] [--replay-limit 3]');
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
      const binding = verifyManifestBinding(labelManifest, row.value, casesSha);
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
    scorer: 'scoreGraded from ts-host/src/skills/graded.ts; schema natlang.skill-graded/1',
    scorer_source_sha256: await hashFile(resolve(here, '../../src/skills/graded.ts')),
    scorer_dist_sha256: await hashFile(resolve(here, '../../dist/skills/graded.js')) };
  const sourceFields = [...new Set([...cases.values()].flatMap(source => Object.keys(source)))].sort();
  const eligible = [];
  for (const [id, source] of cases) {
    const unknownFields = Object.keys(source).filter(key => !reviewedSourceFields.has(key));
    if (unknownFields.length) { reject(rejections, id, 'unreviewed_source_fields_not_exposed', {
      fields: unknownFields, source_record_canonical_sha256: sha(JSON.stringify(canonical(source))) }); continue; }
    if (source.role !== 'train') { reject(rejections, id, 'source_split_not_train', { role: source.role }); continue; }
    const label = labelById.get(id);
    if (!label) { reject(rejections, id, 'missing_label_for_source_case'); continue; }
    if (label.duplicate) continue;
    const result = checkedLabel(source, label.value);
    if (result.error) { reject(rejections, id, result.error, { detail: result.detail ?? null,
      label_file: label.file, label_row_sha256: label.row_sha256 }); continue; }
    const contract = contractOf(source);
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
  const records = batches.map(batch => createRecord(batch.contract, batch.entries,
    { ...sourceMeta, adapter_sha256: adapterSha }));
  const toolSurfaceSha256 = await defaultToolSurfaceHash();
  const replayRows = [], replayFailures = [], nativeRows = [];
  const options = { modelId: 'held-decision-label-static-reference', rootSeed: 4041, systemPrompt: TOOLS_PROMPT,
    contextTokens: 32768, maxTurns: 128, toolSurfaceSha256, collectionRole: 'reference',
    authoredActionPlans: false };
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
  const rootContext = `Label-derived static controller/reference generation; zero provider calls; no teacher-generated scaffold/reasoning. Gold/probability labels are absent from model-facing folder files and prompts.`;
  const summary = { schema: 'natlang.held-static-decision-adapter-summary/1', created_at: new Date().toISOString(),
    training_admission: false, cases: records.length, source_labels_accepted: eligible.length,
    source_cases: cases.size, label_rows: artifacts.reduce((sum, item) => sum + item.label_rows, 0),
    replay_attempted: replayRows.length, replay_failures: replayFailures.length, native_turns: nativeRows.length,
    rejection_rows: rejections.length, batch_size_limit: batchSize, group_preserving_oversized_batches: batches.filter(batch => batch.entries.length > batchSize).length,
    replay_options_sha256: replayOptsSha, replay_context: rootContext,
    files: { source_cases_sha256: casesSha, adapter_sha256: adapterSha,
      scorer_source_sha256: sourceMeta.scorer_source_sha256, scorer_dist_sha256: sourceMeta.scorer_dist_sha256,
      tool_surface_sha256: toolSurfaceSha256 } };

  await mkdir(outDir, { recursive: true });
  if ((await readdir(outDir)).length) throw new Error(`refusing to overwrite non-empty output directory: ${outDir}`);
  await writeExclusive(outDir, 'cases.jsonl', records.map(row => JSON.stringify(row)).join('\n') + '\n');
  await writeExclusive(outDir, 'replay.results.jsonl', replayRows.map(entry => JSON.stringify(entry)).join('\n') + (replayRows.length ? '\n' : ''));
  await writeExclusive(outDir, 'native.held.jsonl', nativeRows.map(row => JSON.stringify(row)).join('\n') + (nativeRows.length ? '\n' : ''));
  await writeExclusive(outDir, 'replay.failures.jsonl', replayFailures.map(row => JSON.stringify(row)).join('\n') + (replayFailures.length ? '\n' : ''));
  await writeExclusive(outDir, 'rejections.jsonl', rejections.map(row => JSON.stringify(row)).join('\n') + (rejections.length ? '\n' : ''));
  await writeExclusive(outDir, 'source-manifest.json', JSON.stringify({ schema: 'natlang.held-static-decision-source-manifest/1',
    training_admission: false, ...sourceMeta, labels: artifacts.map(item => ({ path: item.path,
      labels_sha256: item.labels_sha256, label_rows: item.label_rows, manifest_path: item.manifest_path, manifest_sha256: item.manifest_sha256,
      manifest: item.manifest })), adapter_sha256: adapterSha,
    transformations: 'Raw probability receipt retained. Selection is argmax (first declared option wins ties); binary noul uses p>=0.5. No normalization. Only rows passing existing scoreGraded gates and exact source top decision are converted.',
    model_visible_fields: [...new Set([...sourceFields.filter(key => key !== 'answer'), 'criteria'])],
    criteria_provenance: 'source criteria is preserved exactly when present; otherwise a criteria map naming each declared label is generated',
    hidden_fields: ['source answer annotation', 'teacher probability payload', 'provider response content/hash receipt'],
    source_field_review: { observed_fields: sourceFields, reviewed_visible_fields: sourceFields.filter(key => key !== 'answer'),
      hidden_source_fields: sourceFields.filter(key => key === 'answer'),
      unreviewed_fields_rejected_without_prompt_exposure: true },
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
      strict_top: entry.decision, raw_probability_receipt: entry.raw, score: entry.score })),
    training_admission: false }, null, 2) + '\n');
  await writeExclusive(outDir, 'scorer-manifest.json', JSON.stringify({ schema: 'natlang.held-static-decision-scorer-manifest/1',
    training_admission: false, scorer: 'scoreGraded', scorer_schema: 'natlang.skill-graded/1',
    source_file: resolve(here, '../../src/skills/graded.ts'), source_sha256: sourceMeta.scorer_source_sha256,
    loaded_module: resolve(here, '../../dist/skills/graded.js'), module_sha256: sourceMeta.scorer_dist_sha256,
    strict_schema: 'exact answer object and distribution keys; finite numeric [0,1] values; total within 1e-4 of 1; no normalization by adapter',
    gates: { choice: ['distribution', 'top_correct'], noul: ['probability', 'correct_side'], score: ['distribution', 'within_half_level'] },
    exact_top: 'argmax in source option/level order (first on ties); noul p>=0.5; must equal source answer label/side',
    invocation: `Imported scoreGraded directly by ${basename(fileURLToPath(import.meta.url))}; see source-manifest.json for adapter and source paths.`,
    accepted_score_details: eligible.map(entry => ({ id: entry.source.id, kind: entry.source.kind,
      quality: entry.score.quality, gates: entry.score.gates, detail: entry.score.detail })) }, null, 2) + '\n');
  await writeExclusive(outDir, 'summary.json', JSON.stringify(summary, null, 2) + '\n');
  console.log(JSON.stringify({ out: outDir, cases: records.length, eligible_labels: eligible.length,
    rejections: rejections.length, replay_attempted: replayRows.length, replay_failures: replayFailures.length,
    training_admission: false }));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
