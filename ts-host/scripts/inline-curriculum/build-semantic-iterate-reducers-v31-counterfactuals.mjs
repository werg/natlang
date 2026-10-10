#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { worlds as v25Worlds, deriveWorldResult as deriveV25 } from './semantic-iterate-reducers-v25-data.mjs';
import { worlds as v26Worlds, deriveWorldResult as deriveV26 } from './semantic-iterate-reducers-v26-data.mjs';
import { makeGuidedSoftIterateCase } from './semantic-iterate-worlds-v15-soft-guided-builder.mjs';

const args = process.argv.slice(2);
function option(name) { const index = args.indexOf(name); if (index < 0 || !args[index + 1] || args[index + 1].startsWith('--')) throw new Error(`missing ${name} PATH`); return resolve(args[index + 1]); }
if (args.length !== 6 || ['--out', '--source-pools', '--review-paths'].some(name => args.filter(value => value === name).length !== 1)) throw new Error('usage: node build-semantic-iterate-reducers-v31-counterfactuals.mjs --out FRESH_DIRECTORY --source-pools SOURCE_POOL_CONFIG.json --review-paths REVIEW_PATH_CONFIG.json');
const outDir = option('--out'), sourcePoolConfigPath = option('--source-pools'), reviewPathConfigPath = option('--review-paths');
const sourcePoolConfig = JSON.parse(await readFile(sourcePoolConfigPath, 'utf8'));
const reviewPathConfig = JSON.parse(await readFile(reviewPathConfigPath, 'utf8'));
for (const key of ['25', '26']) if (!sourcePoolConfig[key] || !reviewPathConfig[key]) throw new Error(`configs must provide parent source and review paths for V${key}`);
if (!Array.isArray(sourcePoolConfig.prior_campaign_sources) || sourcePoolConfig.prior_campaign_sources.length === 0) throw new Error('source-pools config must pin prior_campaign_sources for overlap checks');
const revision = 'authored-semantic-iterate-reducers-v31/2-unique-group-counterfactual-expansion';
const specs = [
  { family: 26, index: 0, candidate: 'CORR-A', field: 'Primary source', value: false, worker: 1, purpose: 'The current top-ranked correction remains unsupported; remove the next valid item so the third-ranked fully supported correction must be selected.' },
  { family: 25, index: 0, candidate: 'TREE-A1', field: 'Hazard status', value: false, worker: 1, purpose: 'Invalidate the currently selected work order on hazard status while preserving the lower-ranked fully eligible work order.' },
  { family: 26, index: 1, candidate: 'CORR-D', field: 'Primary source', value: false, worker: 2, purpose: 'Remove the only correction with current-edition, primary-source, and quotation-boundary support; test the empty eligible outcome.' },
  { family: 25, index: 1, candidate: 'TREE-B1', field: 'Closure permit', value: false, worker: 2, purpose: 'Remove the sole candidate with a qualified arborist and location-matched permit; test no-action after full conjunction.' },
  { family: 26, index: 2, candidate: 'FOLIO-2', field: 'Custody chain', value: false, worker: 3, purpose: 'Remove the second-highest fully supported archival transfer; the lower-ranked certified folio must replace it.' },
  { family: 25, index: 2, candidate: 'SHIP-C4', field: 'Custody seal', value: false, worker: 3, purpose: 'Remove one of the two eligible shipments while the metric-leading shipment still fails temperature; recompute the one-item shipment output.' },
  { family: 26, index: 3, candidate: 'FOLIO-4', field: 'Source identity', value: false, worker: 4, purpose: 'Remove one of two valid archive folios while the highest-metric folio remains ineligible on source identity.' },
  { family: 25, index: 4, candidate: 'VESSEL-TERN', field: 'Route clearance', value: false, worker: 4, purpose: 'Remove the highest eligible vessel on route clearance; select the next verified vessel without relaxing the conjunction.' },
  { family: 26, index: 6, candidate: 'DOC-A', field: 'Protected identifiers', value: false, worker: 5, purpose: 'Remove the higher-sequence record because one identifier remains visible; select only the lower-sequence record that passes all checks.' },
  { family: 25, index: 6, candidate: 'SIG-E12', field: 'Engineer assignment', value: false, worker: 5, purpose: 'Remove the higher-priority eligible signal section for lack of an assigned engineer; select the next fully supported repair.' },
  { family: 25, index: 7, candidate: 'SIG-F02', field: 'Engineer assignment', value: false, worker: 2, purpose: 'Remove the only repair section with a qualified assigned engineer and test the zero-item repair decision.' },
  { family: 25, index: 5, candidate: 'VESSEL-ASH', field: 'Observer credential', value: true, worker: 3, purpose: 'Restore the missing credential for the only otherwise eligible vessel and recompute the one-vessel assignment.' },
];
for (const spec of specs) { spec.source_version = 31; spec.counterfactual_kind = 'static_single_fact_eligibility_counterfactual'; }
const inputs = {
  25: { worlds: v25Worlds, derive: deriveV25, path: resolve(sourcePoolConfig['25']), module: fileURLToPath(new URL('./semantic-iterate-reducers-v25-data.mjs', import.meta.url)), review: resolve(reviewPathConfig['25']) },
  26: { worlds: v26Worlds, derive: deriveV26, path: resolve(sourcePoolConfig['26']), module: fileURLToPath(new URL('./semantic-iterate-reducers-v26-data.mjs', import.meta.url)), review: resolve(reviewPathConfig['26']) },
};
const usedSourcePaths = sourcePoolConfig.prior_campaign_sources.map(path => resolve(path));

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const jsonl = rows => Buffer.from(rows.map(row => JSON.stringify(row)).join('\n') + '\n');

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
function updateEvidence(world, candidate) {
  if (!Array.isArray(world.conditionSchema)) throw new Error(`${world.slug}: this source expansion requires a declared condition schema`);
  candidate.facts = world.conditionSchema.map(condition => `${condition.label}: ${candidate.flags[condition.label] ? condition.yes : condition.no}.`).join(' ');
  const path = 'pass-03-conditions.md', content = world.evidence[path], prefix = `${candidate.id}: `;
  let matches = 0;
  world.evidence[path] = content.split('\n').map(line => {
    if (!line.startsWith(prefix)) return line;
    matches++;
    return prefix + candidate.facts;
  }).join('\n');
  if (matches !== 1) throw new Error(`${world.slug}/${candidate.id}: expected exactly one condition evidence line, found ${matches}`);
}

if (specs.length !== 12 || new Set(specs.map(x => `${x.family}:${x.index}`)).size !== specs.length || specs.some(x => x.source_version !== 31 || x.counterfactual_kind !== 'static_single_fact_eligibility_counterfactual'))
  throw new Error('Expected twelve unique V31 static single-fact parent source rows');
try { await readFile(resolve(outDir, 'source.cases.jsonl')); throw new Error(`output already exists: ${outDir}`); } catch (error) { if (error.code !== 'ENOENT') throw error; }
for (const version of ['25', '26']) {
  const review = JSON.parse(await readFile(reviewPathConfig[version], 'utf8'));
  const parentBytes = await readFile(sourcePoolConfig[version]);
  const parentHash = hash(parentBytes);
  const declared = review.source_cases_sha256 ?? review.source_sha256 ?? review.source_cases?.sha256;
  if (!declared) throw new Error(`V${version}: parent review does not declare its source cases SHA-256`);
  if (declared !== parentHash) throw new Error(`V${version}: parent review is bound to ${declared}, not ${parentHash}`);
}
const usedGroups = new Set();
for (const path of usedSourcePaths) for (const line of (await readFile(path, 'utf8')).trimEnd().split('\n')) {
  const row = JSON.parse(line); for (const group of row.source_groups ?? []) usedGroups.add(group);
}
const generated = [], derivations = [], parentPins = {};
for (let newIndex = 0; newIndex < specs.length; newIndex++) {
  const spec = specs[newIndex], input = inputs[spec.family], parentBytes = await readFile(input.path);
  const rawLines = parentBytes.toString('utf8').split('\n'), parentRows = rawLines.slice(0, -1).map(line => JSON.parse(line));
  const parentRow = parentRows[spec.index], original = input.worlds[spec.index];
  if (!parentRow || !original || parentRow.split !== 'train' || original.split !== 'train') throw new Error(`V${spec.family}/${spec.index} must be an authored train source row/world`);
  if (parentRow.source_groups?.length !== 1 || parentRow.source_groups[0] !== original.group) throw new Error(`V${spec.family}/${spec.index}: source group mismatch`);
  const group = original.group;
  if (usedGroups.has(group)) throw new Error(`source group already allocated in V28/V29: ${group}`);
  usedGroups.add(group);
  const parentRawLine = Buffer.from(rawLines[spec.index] + '\n');
  const parentRowBytes = Buffer.from(JSON.stringify(parentRow));
  const world = structuredClone(original), candidate = world.scenarioFacts.find(item => item.id === spec.candidate);
  if (!candidate || !world.conditionSchema?.some(item => item.label === spec.field) || !Object.hasOwn(candidate.flags, spec.field)) throw new Error(`${world.slug}/${spec.candidate}: undeclared candidate field ${spec.field}`);
  const before = candidate.flags[spec.field];
  if (before === spec.value) throw new Error(`${world.slug}/${spec.candidate}: proposed edit does not change the field`);
  candidate.flags[spec.field] = spec.value; updateEvidence(world, candidate);
  const expected = input.derive(world);
  world.source_summary = { ...world.source_summary, candidates: world.scenarioFacts.map(item => structuredClone(item)), selectedItems: expected.selectedItems, measure: expected.measure, decision: expected.decision };
  world.scenarioFacts = world.scenarioFacts.map(item => ({ ...item }));
  setPassStates(world, expected);
  const row = makeGuidedSoftIterateCase(world, newIndex, { revision, shapeVersion: 'v16' });
  const beforeId = row.id;
  row.id = row.id.replaceAll('authored_semantic_iterate_worlds_v15', 'authored_semantic_iterate_reducers_v31')
    .replace(/:v15-/g, ':v31-').replace(':evidence-scoped-guided-soft-state-derived-decision-v2', ':evidence-scoped-guided-soft-state-derived-decision-v4');
  row.family = 'authored_semantic_iterate_reducers_v31'; row.family_version = 31;
  row.curriculum.family = row.family; row.curriculum.family_version = 31;
  row.curriculum.shape = `v31-${world.slug}-four-pass-counterfactual-reselection`;
  row.curriculum.variant = 'same-inherited-train-group-counterfactual-data-variant/1';
  row.semantics.shape = row.curriculum.shape;
  row.source_ids = [group]; row.source_groups = [group]; row.split_group = group; row.split = 'train'; row.source_revisions = [revision];
  row.generation.generator = revision; row.generation.independent_world = group; row.generation.independent_world_credit = false;
  row.generation.source_quality = 'Project-authored, single-fact counterfactual derived from a distinct inherited train group. Facts, visible evidence, and oracle were recomputed and source-bound. This receives no independent-world credit.';
  row.generation.counterfactual_parent = { source_path: input.path, source_file_sha256: hash(parentBytes), source_index_zero_based: spec.index,
    source_row_sha256_with_lf: hash(parentRawLine), source_row_sha256_without_lf: hash(parentRowBytes), source_case_id: parentRow.id,
    source_group: group, split: 'train', authored_data_module: input.module, authored_data_module_sha256: hash(await readFile(input.module)),
    parent_source_review: input.review, parent_source_review_sha256: await readFile(input.review).then(hash) };
  row.generation.counterfactual_edits = [{ candidate_id: spec.candidate, field: spec.field, before, after: spec.value }];
  row.generation.source_version = spec.source_version; row.generation.counterfactual_kind = spec.counterfactual_kind;
  row.generation.counterfactual_purpose = spec.purpose;
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
    source_version: spec.source_version, counterfactual_kind: spec.counterfactual_kind, parent_version: spec.family, parent_source_index_zero_based: spec.index, parent_source_case_id: parentRow.id,
    parent_source_sha256: hash(parentBytes), parent_row_sha256_with_lf: hash(parentRawLine), parent_row_sha256_without_lf: hash(parentRowBytes),
    parent_module_sha256: row.generation.counterfactual_parent.authored_data_module_sha256, parent_source_review_sha256: row.generation.counterfactual_parent.parent_source_review_sha256,
    source_group: group, split: 'train', edit: { candidate_id: spec.candidate, field: spec.field, before, after: spec.value }, purpose: spec.purpose,
    candidates_after_edit: world.scenarioFacts.map(item => ({ id: item.id, metric: item.metric, facts: item.facts })),
    provisional_register_selection: { selectedItems: pass2.selectedItems, measure: pass2.measure }, expected_recomputed_by_parent_deriveWorldResult: expected,
    pass3_reselection_changed: true, model_visible_task_contains_expected: false, model_visible_decision_is_gold: false,
    authored_passes: world.passes.length, declared_semantic_children: ['seed writer', 'four current-pass FileHandle + priorNotes writers', 'final typed Draft interpreter'],
    per_pass_instructions_use_readText_revision: false, prompt_derivative_applied: false, provider_calls: 0, model_calls: 0, training_admission: false, independent_world_credit: false,
    generated_program_id_before_rename: beforeId });
  parentPins[spec.family] = { cases: input.path, cases_sha256: hash(parentBytes), data_module: input.module, data_module_sha256: hash(await readFile(input.module)), source_review: input.review, source_review_sha256: row.generation.counterfactual_parent.parent_source_review_sha256 };
  generated.push(row);
}
if (new Set(generated.map(row => row.source_groups[0])).size !== generated.length) throw new Error('new source groups must be disjoint');
const sourceBytes = jsonl(generated), sourceSha = hash(sourceBytes);
await mkdir(outDir, { recursive: false });
await writeFile(resolve(outDir, 'source.cases.jsonl'), sourceBytes, { flag: 'wx' });
const scriptPath = fileURLToPath(new URL('./build-semantic-iterate-reducers-v31-counterfactuals.mjs', import.meta.url));
const scriptSha = hash(await readFile(scriptPath));
const proof = { schema: 'natlang.semantic-iterate-next-batch-counterfactual-source-proof/1', revision,
  source_cases: 'source.cases.jsonl', source_cases_sha256: sourceSha, build_script: 'build-semantic-iterate-reducers-v31-counterfactuals.mjs', build_script_sha256: scriptSha,
  counts: { cases: generated.length, train: generated.length, inherited_groups: generated.length, per_worker_cases: Object.fromEntries([1,2,3,4,5].map(worker => [String(worker), derivations.filter(x => x.worker_allocation === worker).length])), independent_new_worlds: 0, provider_calls: 0, teacher_calls: 0, training_admission: false },
  source_pool_config: { path: sourcePoolConfigPath, sha256: hash(await readFile(sourcePoolConfigPath)) }, review_path_config: { path: reviewPathConfigPath, sha256: hash(await readFile(reviewPathConfigPath)) }, parent_sources: parentPins, prior_campaign_sources_checked_for_group_overlap: await Promise.all(usedSourcePaths.map(async path => ({ path, sha256: hash(await readFile(path)) }))),
  source_version: 31, counterfactual_kind: 'static_single_fact_eligibility_counterfactual', derivation: 'Each case clones an immutable authored V25 or V26 train source row and corresponding typed world, applies one declared boolean eligibility edit, regenerates the affected candidate fact line through the parent condition schema, recomputes the exact result with that version’s authored deriveWorldResult, and regenerates the same four-pass semantic iterate scaffold using the shared makeGuidedSoftIterateCase builder. Each inherits one unique parent group and stays train-only. No expected value or final answer is placed in model-visible task.json/decision.json.',
  runtime_guidance_derivative: null,
  claims: { all_parent_rows_train: true, all_groups_disjoint_from_v28_and_v29_sources: true, all_groups_unique_in_new_batch: true, all_counterfactual_edits_explicit: true, all_expected_results_recomputed_by_pinned_authored_oracles: true, all_cases_require_pass3_reselection: true, no_test_rows: true, no_gold_in_model_visible_task: true, provider_calls: 0, teacher_calls: 0, training_admission: false, independent_world_credit: 0 },
  worker_allocation: derivations.map(({ new_index_zero_based, source_case_id, worker_allocation, source_group, expected_recomputed_by_parent_deriveWorldResult }) => ({ case_index_zero_based: new_index_zero_based, source_case_id, worker: worker_allocation, source_group, expected_hidden_oracle_for_review_only: expected_recomputed_by_parent_deriveWorldResult })),
  cases: derivations };
await writeFile(resolve(outDir, 'source-proof.json'), JSON.stringify(proof, null, 2) + '\n', { flag: 'wx' });
const plan = { schema: 'natlang.root-review-only-teacher-case-allocation/1', status: 'source-review-required; not a launch authorization', source: 'source.cases.jsonl', source_sha256: sourceSha,
  provider_calls: 0, teacher_calls: 0, training_admission: false, runtime_qualification: false, new_independent_world_credit: 0,
  allocation: Object.fromEntries([1,2,3,4,5].map(worker => [`worker-${String(worker).padStart(2,'0')}`, derivations.filter(item => item.worker_allocation === worker).map(item => ({ source_index_zero_based: item.new_index_zero_based, case_id: item.source_case_id, group: item.source_group }))])),
  constraints: ['No worker launch is included or authorized by this plan.', 'All cases are inherited train groups; reserve evaluation from disjoint pre-existing held-out groups before any qualification claim.', 'Owner review of fact-to-oracle correspondence, captured task gold omission, licenses/lineage, prompt derivative, and group closure is required before any generation request.'] };
await writeFile(resolve(outDir, 'root-review-plan.json'), JSON.stringify(plan, null, 2) + '\n', { flag: 'wx' });
await writeFile(resolve(outDir, 'source-manifest.json'), JSON.stringify({ schema: 'natlang.semantic-iterate-next-batch-counterfactual-source-manifest/1', revision,
  source_cases: 'source.cases.jsonl', source_cases_sha256: sourceSha, source_proof: 'source-proof.json', root_review_plan: 'root-review-plan.json', counts: proof.counts,
  source_version: 31, counterfactual_kind: 'static_single_fact_eligibility_counterfactual', parent_source_versions: [25,26], training_admission: false, independent_new_worlds: 0 }, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify({ outDir, source_sha256: sourceSha, script_sha256: scriptSha, counts: proof.counts,
  cases: derivations.map(item => ({ id: item.source_case_id, worker: item.worker_allocation, group: item.source_group, expected: item.expected_recomputed_by_parent_deriveWorldResult, edit: item.edit })) }, null, 2));
