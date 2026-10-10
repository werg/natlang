#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { worlds as v24Worlds, deriveWorldResult as deriveV24 } from './semantic-iterate-reducers-v24-data.mjs';
import { worlds as v25Worlds, deriveWorldResult as deriveV25 } from './semantic-iterate-reducers-v25-data.mjs';
import { worlds as v26Worlds, deriveWorldResult as deriveV26 } from './semantic-iterate-reducers-v26-data.mjs';
import { makeGuidedSoftIterateCase } from './semantic-iterate-worlds-v15-soft-guided-builder.mjs';

const args = process.argv.slice(2);
function option(name) { const index = args.indexOf(name); if (index < 0 || !args[index + 1] || args[index + 1].startsWith('--')) throw new Error(`missing ${name} VALUE`); if (args.lastIndexOf(name) !== index) throw new Error(`duplicate option ${name}`); return args[index + 1]; }
const optionNames = ['--out', '--source-pools', '--review-paths', '--specs', '--revision', '--source-version'];
if (args.length !== optionNames.length * 2 || optionNames.some(name => args.filter(value => value === name).length !== 1) || args.some((value, index) => index % 2 === 0 && !optionNames.includes(value)))
  throw new Error('usage: node build-semantic-iterate-counterfactuals.mjs --out FRESH_DIRECTORY --source-pools SOURCE_POOLS.json --review-paths REVIEW_PATHS.json --specs EXPLICIT_SPECS.json --revision REVISION --source-version FAMILY_VERSION');
const outDir = resolve(option('--out')), sourcePoolConfigPath = resolve(option('--source-pools'));
const reviewPathConfigPath = resolve(option('--review-paths')), specsPath = resolve(option('--specs'));
const revision = option('--revision'), sourceVersion = Number(option('--source-version'));
if (!Number.isInteger(sourceVersion) || sourceVersion < 1 || !revision) throw new Error('source-version must be a positive integer and revision must be nonempty');
const sourcePoolConfig = JSON.parse(await readFile(sourcePoolConfigPath, 'utf8'));
const reviewPathConfig = JSON.parse(await readFile(reviewPathConfigPath, 'utf8'));
const specConfig = JSON.parse(await readFile(specsPath, 'utf8'));
if (specConfig.source_version !== sourceVersion || specConfig.revision !== revision || !Array.isArray(specConfig.specs) || !['v5','v6'].includes(specConfig.case_shape_version ?? 'v5')) throw new Error('spec config must pin the same source_version and revision and contain a specs array');
const specs = specConfig.specs;
if (!Array.isArray(sourcePoolConfig.prior_campaign_sources) || sourcePoolConfig.prior_campaign_sources.length === 0) throw new Error('source-pools config must pin prior_campaign_sources for overlap checks');
const worldInputs = {
  24: { worlds: v24Worlds, derive: deriveV24, module: fileURLToPath(new URL('./semantic-iterate-reducers-v24-data.mjs', import.meta.url)) },
  25: { worlds: v25Worlds, derive: deriveV25, module: fileURLToPath(new URL('./semantic-iterate-reducers-v25-data.mjs', import.meta.url)) },
  26: { worlds: v26Worlds, derive: deriveV26, module: fileURLToPath(new URL('./semantic-iterate-reducers-v26-data.mjs', import.meta.url)) },
};
for (const version of new Set(specs.map(spec => String(spec.parent_version)))) {
  if (!worldInputs[version] || !sourcePoolConfig[version] || !reviewPathConfig[version]?.root || !reviewPathConfig[version]?.quality)
    throw new Error(`configs must provide source, root-review, and quality-review paths for parent V${version}`);
}
const caseShapeVersion = specConfig.case_shape_version ?? 'v5';
const shapeVersion = specConfig.shape_version ?? 'v16';


const inputs = Object.fromEntries([...new Set(specs.map(spec => Number(spec.parent_version)))].map(version => [String(version), {
  ...worldInputs[version], path: resolve(sourcePoolConfig[String(version)]), rootReview: resolve(reviewPathConfig[String(version)].root), qualityReview: resolve(reviewPathConfig[String(version)].quality),
}]));
const priorSourceRecords = sourcePoolConfig.prior_campaign_sources.map(item => typeof item === 'string' ? { path: resolve(item) } : { ...item, path: resolve(item.path) });
const reviewContextRecords = (sourcePoolConfig.review_context_pins ?? []).map(item => ({ ...item, path: resolve(item.path) }));


const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const jsonl = rows => Buffer.from(rows.map(row => JSON.stringify(row)).join('\n') + '\n');
for (const item of reviewContextRecords) { const digest = hash(await readFile(item.path)); if (item.sha256 && item.sha256 !== digest) throw new Error(`review context SHA mismatch for ${item.path}`); item.sha256 = digest; }

function ranked(world) {
  const descending = /greatest first|descending/i.test(world.decision_rule);
  return [...world.scenarioFacts].sort((a, b) => (descending ? b.metric - a.metric : a.metric - b.metric) || a.id.localeCompare(b.id));
}
function setPassStates(world, expected) {
  const count = world.selectionCardinality;
  const provisional = ranked(world).slice(0, count);
  const ids = rows => rows.length ? rows.map(row => row.id).join('; ') : 'none';
  const measures = rows => rows.length ? rows.map(row => String(row.metric)).join('; ') : 'none';
  const initial = { ...world.initial, caseId: world.source_summary.requestId };
  const pass2 = { ...initial, selectedItems: ids(provisional), measure: measures(provisional) };
  const pass3 = { ...initial, selectedItems: expected.selectedItems, measure: expected.measure };
  const pass4 = { ...pass3, decision: expected.decision };
  if (pass2.selectedItems === pass3.selectedItems && pass2.measure === pass3.measure)
    throw new Error(`${world.slug}: edit did not require pass-three eligibility reselection`);
  world.passStates = [initial, pass2, pass3, pass4];
  world.justified_revision = { pass: 3, field: 'selectedItems', reason: 'The current evidence pass filters the metric-ranked provisional list by every declared eligibility condition and recomputes the selected set.' };
  world.expectedFactDerivation = { selected: expected.selectedItems === 'none' ? [] : expected.selectedItems.split('; '), measure: expected.measure, decision: expected.decision };
}
function candidateField(candidate, field) {
  if (candidate.flags && Object.hasOwn(candidate.flags, field)) return candidate.flags[field];
  if (Object.hasOwn(candidate, field)) return candidate[field];
  throw new Error(`${candidate.id}: undeclared candidate field ${field}`);
}
function setCandidateField(candidate, field, value) {
  if (candidate.flags && Object.hasOwn(candidate.flags, field)) candidate.flags[field] = value;
  else if (Object.hasOwn(candidate, field)) candidate[field] = value;
  else throw new Error(`${candidate.id}: undeclared candidate field ${field}`);
}
function applyEdits(world, spec) {
  const declared = spec.edits ?? [{ candidate_id: spec.candidate, field: spec.field, after: spec.value, after_facts: spec.after_facts }];
  if (!Array.isArray(declared) || declared.length === 0) throw new Error(`${spec.parent_version}/${spec.index}: each static counterfactual needs one or more explicit edits`);
  const edits = [];
  const changedCandidates = new Map();
  for (const edit of declared) {
    const candidate = world.scenarioFacts.find(item => item.id === edit.candidate_id);
    if (!candidate || typeof edit.field !== 'string' || typeof edit.after !== 'boolean') throw new Error(`${world.slug}: malformed edit ${JSON.stringify(edit)}`);
    const before = candidateField(candidate, edit.field);
    if (typeof before !== 'boolean' || before === edit.after) throw new Error(`${world.slug}/${candidate.id}: edit must change an existing boolean field`);
    setCandidateField(candidate, edit.field, edit.after);
    edits.push({ candidate_id: edit.candidate_id, field: edit.field, before, after: edit.after, before_facts: candidate.facts });
    if (!changedCandidates.has(candidate.id)) changedCandidates.set(candidate.id, { candidate, explicitAfterFacts: edit.after_facts });
    else if (edit.after_facts) changedCandidates.get(candidate.id).explicitAfterFacts = edit.after_facts;
  }
  for (const { candidate, explicitAfterFacts } of changedCandidates.values()) {
    if (Array.isArray(world.conditionSchema)) candidate.facts = world.conditionSchema.map(condition => `${condition.label}: ${candidate.flags[condition.label] ? condition.yes : condition.no}.`).join(' ');
    else {
      if (typeof explicitAfterFacts !== 'string' || !explicitAfterFacts.trim()) throw new Error(`${world.slug}/${candidate.id}: explicit after_facts required when the parent has no conditionSchema`);
      candidate.facts = explicitAfterFacts;
    }
    const path = 'pass-03-conditions.md', content = world.evidence[path], prefix = `${candidate.id}: `;
    let matches = 0;
    world.evidence[path] = content.split('\n').map(line => {
      if (!line.startsWith(prefix)) return line;
      matches++;
      if (line !== prefix + edits.find(edit => edit.candidate_id === candidate.id).before_facts) throw new Error(`${world.slug}/${candidate.id}: visible evidence line differs from parent candidate facts`);
      return prefix + candidate.facts;
    }).join('\n');
    if (matches !== 1) throw new Error(`${world.slug}/${candidate.id}: expected exactly one condition evidence line, found ${matches}`);
  }
  return edits.map(edit => ({ candidate_id: edit.candidate_id, field: edit.field, before: edit.before, after: edit.after,
    before_facts: edit.before_facts, after_facts: world.scenarioFacts.find(item => item.id === edit.candidate_id).facts }));
}

function applyModelFacingReplacements(world, spec) {
  const applied = [];
  for (const change of spec.model_facing_replacements ?? []) {
    const { target, before, after } = change;
    if (typeof before !== 'string' || !before || typeof after !== 'string' || !after || before === after) throw new Error(`${world.slug}: malformed model-facing replacement`);
    let source;
    if (target === 'decision_rule') source = world.decision_rule;
    else if (target === 'instruction') source = world.instruction;
    else if (target === 'field:decision') source = world.fields.decision;
    else if (target.startsWith('evidence:')) source = world.evidence[target.slice('evidence:'.length)];
    else throw new Error(`${world.slug}: unsupported model-facing replacement target ${target}`);
    if (typeof source !== 'string' || source.split(before).length !== 2) throw new Error(`${world.slug}: expected one exact old phrase at ${target}`);
    const next = source.replace(before, after);
    if (target === 'decision_rule') world.decision_rule = next;
    else if (target === 'instruction') world.instruction = next;
    else if (target === 'field:decision') world.fields.decision = next;
    else world.evidence[target.slice('evidence:'.length)] = next;
    applied.push({ target, before, after });
  }
  return applied;
}

if (specs.length === 0 || specs.some(spec => !Number.isInteger(spec.parent_version) || !Number.isInteger(spec.index) || !Array.isArray(spec.edits) || !Number.isInteger(spec.worker) || spec.worker < 1 || spec.worker > 5 || !spec.purpose)) throw new Error('Expected a nonempty explicit parent-bound counterfactual case list');
try { await readFile(resolve(outDir, 'source.cases.jsonl')); throw new Error(`output already exists: ${outDir}`); } catch (error) { if (error.code !== 'ENOENT') throw error; }
const usedGroupSources = new Map();
for (const record of priorSourceRecords) {
  const bytes = await readFile(record.path), digest = hash(bytes);
  if (record.sha256 && record.sha256 !== digest) throw new Error(`prior source SHA mismatch for ${record.path}`);
  for (const line of bytes.toString('utf8').trimEnd().split('\n')) {
    const row = JSON.parse(line);
    for (const group of row.source_groups ?? []) {
      const rows = usedGroupSources.get(group) ?? [];
      rows.push({ path: record.path, sha256: digest, case_id: row.id });
      usedGroupSources.set(group, rows);
    }
  }
}
const generatedGroups = new Set();

for (const [version, input] of Object.entries(inputs)) {
  const rootReview = JSON.parse(await readFile(input.rootReview, 'utf8'));
  const qualityReview = JSON.parse(await readFile(input.qualityReview, 'utf8'));
  const parentBytes = await readFile(input.path), parentHash = hash(parentBytes);
  const rootBound = rootReview.source_cases_sha256 ?? rootReview.inputs?.['source.cases.jsonl'];
  if (rootBound !== parentHash) throw new Error(`V${version}: root review is not SHA-bound to its configured source pool`);
  if (rootReview.generation_authorized !== true) throw new Error(`V${version}: root review does not authorize source generation`);
  if (qualityReview.source_cases_sha256 !== parentHash) throw new Error(`V${version}: quality review is not SHA-bound to its configured source pool`);
}
const generated = [], derivations = [], parentPins = {};
for (let newIndex = 0; newIndex < specs.length; newIndex++) {
  const spec = specs[newIndex], parentVersion = String(spec.parent_version), input = inputs[parentVersion], parentBytes = await readFile(input.path);
  const rawLines = parentBytes.toString('utf8').split('\n'), parentRows = rawLines.slice(0, -1).map(line => JSON.parse(line));
  const parentRow = parentRows[spec.index], original = input.worlds[spec.index];
  if (!parentRow || !original || parentRow.split !== 'train' || original.split !== 'train') throw new Error(`V${parentVersion}/${spec.index} must be an authored train source row/world`);
  if (parentRow.source_groups?.length !== 1 || parentRow.source_groups[0] !== original.group) throw new Error(`V${parentVersion}/${spec.index}: source group mismatch`);
  const group = original.group, priorGroupEvidence = usedGroupSources.get(group) ?? [];
  if (priorGroupEvidence.length && (spec.allow_prior_group_reuse !== true || typeof spec.prior_group_reuse_reason !== 'string' || !spec.prior_group_reuse_reason)) throw new Error(`${group}: explicit allow_prior_group_reuse and a reason are required`);
  if (generatedGroups.has(group) && (spec.allow_sibling_group_reuse !== true || typeof spec.sibling_group_reuse_reason !== 'string' || !spec.sibling_group_reuse_reason)) throw new Error(`${group}: explicit allow_sibling_group_reuse and a reason are required for sibling counterfactual rows`);
  generatedGroups.add(group);
  const parentRawLine = Buffer.from(rawLines[spec.index] + '\n');
  const parentRowBytes = Buffer.from(JSON.stringify(parentRow));
  const world = structuredClone(original);
  const edits = applyEdits(world, spec);
  const expected = input.derive(world);
  world.source_summary = { ...world.source_summary, candidates: world.scenarioFacts.map(item => structuredClone(item)), selectedItems: expected.selectedItems, measure: expected.measure, decision: expected.decision };
  world.scenarioFacts = world.scenarioFacts.map(item => ({ ...item }));
  setPassStates(world, expected);
  const modelFacingReplacements = applyModelFacingReplacements(world, spec);
  const row = makeGuidedSoftIterateCase(world, newIndex, { revision, shapeVersion });
  // Successor specs may explicitly declare a measure as an ordered count list.
  // This is scoped to that authored field; no other strings or map fields normalize.
  if (spec.numeric_list_fields !== undefined) {
    const declarations = spec.numeric_list_fields;
    if (!Array.isArray(declarations) || declarations.length !== 1 ||
        Object.keys(declarations[0] ?? {}).sort().join(',') !== 'empty_value,key,separator' ||
        declarations[0]?.key !== 'measure' || declarations[0]?.separator !== '; ' || declarations[0]?.empty_value !== 'none')
      throw new Error(`${row.id}: numeric_list_fields must explicitly declare measure with separator "; " and empty_value "none"`);
    const task = JSON.parse(row.semantics.folder_files['task.json']);
    const measureFormat = task.output_contract?.fields?.measure;
    const measure = row.semantics.expected?.measure;
    const declaresListSeparator = typeof measureFormat === 'string' &&
      (/semicolon and one space/i.test(measureFormat) || /separated by\s+["'`](?:; )["'`]/i.test(measureFormat));
    const listValue = value => typeof value === 'string' && (value === 'none' ||
      value.split('; ').length > 0 && value.split('; ').every(item => /^(?:0|[1-9]\d*|[1-9]\d{0,2}(?:,\d{3})*)$/.test(item)));
    if (!declaresListSeparator ||
        !/none/i.test(measureFormat) || !listValue(measure))
      throw new Error(`${row.id}: authored measure field does not declare the numeric-list contract`);
    row.semantics.oracle = { level: 'normalized', normalization: 'json-string-record', numeric_list_fields: declarations };
    if (typeof task.output_path !== 'string' || !task.output_path)
      throw new Error(`${row.id}: numeric-list output contract must name output_path`);
    row.semantics.files_oracle = { compare: 'json-string-record', numeric_list_fields: declarations, return_path: task.output_path };
  }
  const beforeId = row.id;
  row.id = row.id.replaceAll('authored_semantic_iterate_worlds_v15', `authored_semantic_iterate_reducers_v${sourceVersion}`)
    .replace(/:v15-/g, `:v${sourceVersion}-`).replace(/:evidence-scoped-guided-soft-state-derived-decision-v\d+$/, `:evidence-scoped-guided-soft-state-derived-decision-${caseShapeVersion}`);
  row.id += `:counterfactual-${String(newIndex + 1).padStart(3, '0')}`;
  row.family = `authored_semantic_iterate_reducers_v${sourceVersion}`; row.family_version = sourceVersion;
  row.curriculum.family = row.family; row.curriculum.family_version = sourceVersion;
  row.curriculum.shape = `v${sourceVersion}-${world.slug}-four-pass-counterfactual-reselection`;
  row.curriculum.variant = 'same-inherited-train-group-counterfactual-data-variant/1';
  row.semantics.shape = row.curriculum.shape;
  row.source_ids = [group]; row.source_groups = [group]; row.split_group = group; row.split = 'train'; row.source_revisions = [revision];
  row.generation.generator = revision; row.generation.independent_world = group; row.generation.independent_world_credit = false;
  row.generation.source_quality = 'Project-authored static eligibility counterfactual derived from a pinned inherited train group. Facts, visible evidence, and oracle were recomputed and source-bound. This receives no independent-world credit.';
  row.generation.counterfactual_parent = { source_path: input.path, source_file_sha256: hash(parentBytes), source_index_zero_based: spec.index,
    source_row_sha256_with_lf: hash(parentRawLine), source_row_sha256_without_lf: hash(parentRowBytes), source_case_id: parentRow.id,
    source_group: group, split: 'train', authored_data_module: input.module, authored_data_module_sha256: hash(await readFile(input.module)),
    parent_source_review: input.rootReview, parent_source_review_sha256: hash(await readFile(input.rootReview)), parent_source_quality_review: input.qualityReview, parent_source_quality_review_sha256: hash(await readFile(input.qualityReview)) };
  row.generation.counterfactual_edits = edits.map(({ candidate_id, field, before, after }) => ({ candidate_id, field, before, after }));
  row.generation.source_version = sourceVersion; row.generation.counterfactual_kind = spec.counterfactual_kind ?? 'static_single_fact_eligibility_counterfactual';
  row.generation.counterfactual_purpose = spec.purpose;
  if (modelFacingReplacements.length) row.generation.model_facing_replacements = modelFacingReplacements;

  row.curriculum.source_group_lineage = { source_revision: revision, factual_group: group, inherited_from_source_case: parentRow.id, independent_world_credit: false };
  row.curriculum.worker_allocation = `worker-${String(spec.worker).padStart(2, '0')}`;
  row.collection_guidance = { training_admission: false, review_scope: 'root-review-before-any-generation; source, visibility, and native child-trajectory quality stay held' };
  row.semantics.expected = structuredClone(expected);
  const taskText = row.semantics.folder_files?.['task.json'];
  if (typeof taskText !== 'string' || taskText.includes(JSON.stringify(expected))) throw new Error(`${row.id}: visible task missing or contains exact expected object`);
  if (JSON.stringify(JSON.parse(row.semantics.folder_files['decision.json'])) === JSON.stringify(expected)) throw new Error(`${row.id}: visible decision file is the gold answer`);
  if (JSON.stringify(JSON.parse(row.semantics.expected_files['decision.json'])) !== JSON.stringify(expected)) throw new Error(`${row.id}: hidden expected file binding is stale`);
  const pass2 = world.passStates[1], pass3 = world.passStates[2];
  if (pass2.selectedItems === pass3.selectedItems && pass2.measure === pass3.measure) throw new Error(`${row.id}: pass-three selection did not change`);
  derivations.push({ new_index_zero_based: newIndex, source_case_id: row.id, worker_allocation: spec.worker,
    source_version: sourceVersion, counterfactual_kind: spec.counterfactual_kind ?? 'static_single_fact_eligibility_counterfactual', parent_version: Number(parentVersion), parent_source_index_zero_based: spec.index, parent_source_case_id: parentRow.id,
    parent_source_sha256: hash(parentBytes), parent_row_sha256_with_lf: hash(parentRawLine), parent_row_sha256_without_lf: hash(parentRowBytes),
    parent_module_sha256: row.generation.counterfactual_parent.authored_data_module_sha256, parent_source_review_sha256: row.generation.counterfactual_parent.parent_source_review_sha256, parent_source_quality_review_sha256: row.generation.counterfactual_parent.parent_source_quality_review_sha256,
    source_group: group, split: 'train', edits, model_facing_replacements: modelFacingReplacements, prior_group_evidence: priorGroupEvidence, allow_prior_group_reuse: spec.allow_prior_group_reuse === true, prior_group_reuse_reason: spec.prior_group_reuse_reason ?? null, allow_sibling_group_reuse: spec.allow_sibling_group_reuse === true, sibling_group_reuse_reason: spec.sibling_group_reuse_reason ?? null, purpose: spec.purpose,
    candidates_after_edit: world.scenarioFacts.map(item => ({ id: item.id, metric: item.metric, facts: item.facts })),
    provisional_register_selection: { selectedItems: pass2.selectedItems, measure: pass2.measure }, expected_recomputed_by_parent_deriveWorldResult: expected,
    pass3_reselection_changed: true, model_visible_task_contains_expected: false, model_visible_decision_is_gold: false,
    authored_passes: world.passes.length, declared_semantic_children: ['seed writer', 'four current-pass FileHandle + priorNotes writers', 'final typed Draft interpreter'],
    per_pass_instructions_use_readText_revision: false, prompt_derivative_applied: false, provider_calls: 0, model_calls: 0, training_admission: false, independent_world_credit: false,
    generated_program_id_before_rename: beforeId });
  parentPins[parentVersion] = { cases: input.path, cases_sha256: hash(parentBytes), data_module: input.module, data_module_sha256: hash(await readFile(input.module)), root_source_review: input.rootReview, root_source_review_sha256: row.generation.counterfactual_parent.parent_source_review_sha256, source_quality_review: input.qualityReview, source_quality_review_sha256: row.generation.counterfactual_parent.parent_source_quality_review_sha256 };
  generated.push(row);
}
if (generated.some((row,index) => derivations[index].prior_group_evidence.length && !derivations[index].allow_prior_group_reuse)) throw new Error('prior group reuse must be explicit');
const sourceBytes = jsonl(generated), sourceSha = hash(sourceBytes);
const groupOccurrences = Object.groupBy(derivations, item => item.source_group);
await mkdir(outDir, { recursive: false });
await writeFile(resolve(outDir, 'source.cases.jsonl'), sourceBytes, { flag: 'wx' });
const scriptPath = fileURLToPath(import.meta.url);
const scriptSha = hash(await readFile(scriptPath));
const proof = { schema: 'natlang.semantic-iterate-counterfactual-source-proof/1', revision, source_version: sourceVersion, counterfactual_kind: 'static-counterfactual',
  source_cases: 'source.cases.jsonl', source_cases_sha256: sourceSha, build_script: 'build-semantic-iterate-counterfactuals.mjs', build_script_sha256: scriptSha,
  counts: { cases: generated.length, train: generated.length, inherited_groups: new Set(generated.map(row => row.source_groups[0])).size, rows_with_repeated_inherited_groups: generated.length - new Set(generated.map(row => row.source_groups[0])).size, per_worker_cases: Object.fromEntries([1,2,3,4,5].map(worker => [String(worker), derivations.filter(x => x.worker_allocation === worker).length])), independent_new_worlds: 0, provider_calls: 0, teacher_calls: 0, training_admission: false },
  parent_sources: parentPins, source_pool_config: { path: sourcePoolConfigPath, sha256: hash(await readFile(sourcePoolConfigPath)) }, review_path_config: { path: reviewPathConfigPath, sha256: hash(await readFile(reviewPathConfigPath)) }, specs: { path: specsPath, sha256: hash(await readFile(specsPath)) }, prior_campaign_sources_checked_for_group_overlap: await Promise.all(priorSourceRecords.map(async record => ({ path: record.path, sha256: hash(await readFile(record.path)), allocation: record.allocation ?? null }))), review_context_pins: reviewContextRecords,
  derivation: 'Each case clones a SHA-pinned root-reviewed V24, V25, or V26 train source row and matching authored world, applies the explicitly listed boolean eligibility edit(s), updates only their exact evidence lines, derives expected fields through that parent version’s deriveWorldResult, and rebuilds the shared four-pass typed Neuralese scaffold. Repeated inherited groups are permitted only when a spec explicitly allows prior or sibling reuse; every occurrence is pinned in the proof. Optional model-facing replacements are explicit, exact before/after string edits and remain held for owner review and live measurement. No expected result or gold decision is exposed in model-visible task.json or decision.json.',
  runtime_guidance_derivative: null,
  claims: { all_parent_rows_train: true, all_prior_group_reuse_explicit: true, all_sibling_group_reuse_explicit: true, all_counterfactual_edits_explicit: true, all_expected_results_recomputed_by_pinned_authored_oracles: true, all_cases_require_pass3_reselection: true, no_test_rows: true, no_gold_in_model_visible_task: true, provider_calls: 0, teacher_calls: 0, training_admission: false, independent_world_credit: 0, model_facing_replacements_need_live_measurement_before_adoption: derivations.some(item => item.model_facing_replacements?.length) },
  worker_allocation: derivations.map(({ new_index_zero_based, source_case_id, worker_allocation, source_group, expected_recomputed_by_parent_deriveWorldResult }) => ({ case_index_zero_based: new_index_zero_based, source_case_id, worker: worker_allocation, source_group, expected_hidden_oracle_for_review_only: expected_recomputed_by_parent_deriveWorldResult })),
  inherited_group_occurrences: Object.fromEntries(Object.entries(groupOccurrences).map(([group, items]) => [group, items.map(item => ({ case_index: item.new_index_zero_based, prior_group_reuse: item.prior_group_evidence, sibling_group_reuse: item.allow_sibling_group_reuse }))])),
  cases: derivations };
await writeFile(resolve(outDir, 'source-proof.json'), JSON.stringify(proof, null, 2) + '\n', { flag: 'wx' });
const plan = { schema: 'natlang.root-review-only-teacher-case-allocation/1', status: 'source-review-required; not a launch authorization', source: 'source.cases.jsonl', source_sha256: sourceSha,
  provider_calls: 0, teacher_calls: 0, training_admission: false, runtime_qualification: false, new_independent_world_credit: 0,
  allocation: Object.fromEntries([1,2,3,4,5].map(worker => [`worker-${String(worker).padStart(2,'0')}`, derivations.filter(item => item.worker_allocation === worker).map(item => ({ source_index_zero_based: item.new_index_zero_based, case_id: item.source_case_id, group: item.source_group }))])),
  constraints: ['No worker launch is included or authorized by this plan.', 'All cases are inherited train groups; reserve evaluation from disjoint pre-existing held-out groups before any qualification claim.', 'Owner review of fact-to-oracle correspondence, captured task gold omission, licenses/lineage, prompt derivative, and group closure is required before any generation request.'] };
await writeFile(resolve(outDir, 'root-review-plan.json'), JSON.stringify(plan, null, 2) + '\n', { flag: 'wx' });
await writeFile(resolve(outDir, 'source-manifest.json'), JSON.stringify({ schema: 'natlang.semantic-iterate-counterfactual-source-manifest/1', revision, source_version: sourceVersion,
  source_cases: 'source.cases.jsonl', source_cases_sha256: sourceSha, source_proof: 'source-proof.json', root_review_plan: 'root-review-plan.json', counts: proof.counts,
  source_version: sourceVersion, counterfactual_kind: 'static-counterfactual', parent_source_versions: [...new Set(specs.map(spec => Number(spec.parent_version)))], training_admission: false, independent_new_worlds: 0 }, null, 2) + '\n', { flag: 'wx' });
const sourceReview = { schema: 'natlang.semantic-iterate-counterfactual-source-quality-review/1', revision, source_cases_sha256: sourceSha, review_type: 'static inherited-train counterfactual pool; root semantic review required', generation_authorized: false, training_admission: false, independent_new_world_credit: 0, cases: derivations.map(item => ({ case_index_zero_based: item.new_index_zero_based, case_id: item.source_case_id, parent_version: item.parent_version, parent_source_index_zero_based: item.parent_source_index_zero_based, parent_source_case_id: item.parent_source_case_id, parent_source_sha256: item.parent_source_sha256, parent_row_sha256_with_lf: item.parent_row_sha256_with_lf, parent_module_sha256: item.parent_module_sha256, parent_source_review_sha256: item.parent_source_review_sha256, parent_source_quality_review_sha256: item.parent_source_quality_review_sha256, source_group: item.source_group, prior_group_evidence: item.prior_group_evidence, edits: item.edits, model_facing_replacements: item.model_facing_replacements, candidates_after_edit: item.candidates_after_edit, provisional_register_selection: item.provisional_register_selection, expected_recomputed_by_parent_deriveWorldResult: item.expected_recomputed_by_parent_deriveWorldResult, model_visible_task_contains_expected: item.model_visible_task_contains_expected, model_visible_decision_is_gold: item.model_visible_decision_is_gold })) };
await writeFile(resolve(outDir, 'source-quality-review.json'), JSON.stringify(sourceReview, null, 2) + '\n', { flag: 'wx' });
const readableCases = generated.map((row, index) => { const task = JSON.parse(row.semantics.folder_files['task.json']); const evidence = Object.entries(row.semantics.folder_files).filter(([name]) => /^pass-.*\.md$/.test(name)).map(([name, text]) => `### ${name}\n\n${text}`).join('\n\n'); return `## ${row.id}\n\nParent V${derivations[index].parent_version} row ${derivations[index].parent_source_index_zero_based}; inherited group ${row.source_groups[0]}; split train.\n\n**Task instruction**\n\n${task.instruction}\n\n**Decision rule**\n\n${task.output_contract.decision_rule}\n\n**Output fields**\n\n${JSON.stringify(task.output_contract.fields)}\n\n${evidence}\n\n**Expected, hidden for review only:** ${JSON.stringify(row.semantics.expected)}\n`; }).join('\n\n');
await writeFile(resolve(outDir, 'readable-source-review.md'), `# Static counterfactual source pool — review required\n\nSource SHA-256: ${sourceSha}\n\nThese authored inherited-group variants are train-only and confer no independent-world credit. This rendering exposes all visible task rules and evidence next to each hidden expected result for root review. Any model-facing before/after edits require live measurement before adoption.\n\n${readableCases}` , { flag: 'wx' });
console.log(JSON.stringify({ outDir, source_sha256: sourceSha, script_sha256: scriptSha, counts: proof.counts,
  cases: derivations.map(item => ({ id: item.source_case_id, worker: item.worker_allocation, group: item.source_group, expected: item.expected_recomputed_by_parent_deriveWorldResult, edits: item.edits })) }, null, 2));
