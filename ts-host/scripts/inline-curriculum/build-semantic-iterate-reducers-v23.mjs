#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { makeGuidedSoftIterateCase } from './semantic-iterate-worlds-v15-soft-guided-builder.mjs';
import { reducerVariants, sourceDomains } from './semantic-iterate-reducers-v23-data.mjs';

export const REVISION = 'authored-semantic-iterate-reducers-v23/1-explicit-selection-cardinality';
const args = process.argv.slice(2);
if (args.length !== 2 || args[0] !== '--out' || !args[1] || args[1].startsWith('--'))
  throw new Error('usage: node build-semantic-iterate-reducers-v23.mjs --out FRESH_DIRECTORY');
const output = resolve(args[1]);
if (reducerVariants.length !== 16 || sourceDomains.length !== 8)
  throw new Error('V23 requires 16 exact-cardinality variants across eight inherited domain groups');

const rows = reducerVariants.map(({ spec, row: world }, index) => {
  const row = makeGuidedSoftIterateCase(world, index, { revision: REVISION, shapeVersion: 'v16' });
  row.id = row.id.replaceAll('authored_semantic_iterate_worlds_v15', 'authored_semantic_iterate_reducers_v23')
    .replaceAll(`:v15-${world.slug}-`, `:v23-${world.slug}-`);
  row.family = 'authored_semantic_iterate_reducers_v23'; row.family_version = 23;
  row.curriculum.family = row.family; row.curriculum.family_version = 23;
  row.curriculum.shape = `v23-${world.slug}-four-pass-evidence-reducer`;
  row.curriculum.variant = 'inherited-domain-explicit-selection-cardinality/1';
  row.semantics.shape = row.curriculum.shape;
  row.source_ids = [spec.sourceGroup]; row.source_groups = [spec.sourceGroup]; row.split_group = spec.sourceGroup;
  row.source_revisions = [REVISION]; row.split = spec.domainIndex % 2 === 0 ? 'train' : 'test';
  row.generation.generator = REVISION;
  row.generation.independent_world = spec.sourceGroup;
  row.generation.independent_world_credit = false;
  row.generation.source_quality = 'Counterfactual task-contract variant inheriting a reviewed factual domain group; it does not count as an additional independent world.';
  row.generation.source_domain = spec.domainIndex;
  row.generation.selection_cardinality = spec.selectionCardinality;
  row.generation.capture_contract.saved_with_examples = 'The parent selects the current path from the folder snapshot and passes only that read-only FileHandle plus prior notes to the child; the child receives no folder or snapshot handle and cannot select other paths.';

  const task = JSON.parse(row.semantics.folder_files['task.json']);
  const expected = row.semantics.expected;
  if (!task.output_contract.fields || !task.output_contract.decision_rule.includes('Selection cardinality:'))
    throw new Error(`${world.slug}: missing exact cardinality rule or field contract`);
  const wantedFormat = spec.selectionCardinality === 1 ? 'Exactly one' : `Up to ${spec.selectionCardinality}`;
  if (!task.output_contract.fields.selectedItems.includes(wantedFormat))
    throw new Error(`${world.slug}: selectedItems format disagrees with declared cardinality`);
  if (!task.instruction.includes(world.source_summary.requestId))
    throw new Error(`${world.slug}: exact request ID omitted from task instruction`);

  const selectedIds = expected.selectedItems === 'none' ? [] : expected.selectedItems.split('; ');
  const independentlySelected = spec.select(spec.candidates.filter(spec.eligible).sort(spec.compare));
  const independentlyExpected = {
    caseId: world.source_summary.requestId,
    selectedItems: spec.format(independentlySelected),
    measure: spec.measure(independentlySelected),
    decision: independentlySelected.length === 0 ? spec.noAction : (spec.authorized ? spec.approvedAction : 'hold'),
  };
  if (JSON.stringify(independentlyExpected) !== JSON.stringify(expected))
    throw new Error(`${world.slug}: derived gold differs from candidate facts and authority`);
  if (selectedIds.length !== independentlySelected.length)
    throw new Error(`${world.slug}: selected item count differs from fact-derived cardinality`);
  if (independentlySelected.length && independentlySelected.length !== spec.selectionCardinality)
    throw new Error(`${world.slug}: scenario lacks the requested number of eligible items`);
  const authority = world.evidence['pass-04-authority.md'];
  const status = [...authority.matchAll(/AUTHORITY STATUS: (approved|not approved)\./g)].map(m => m[1]);
  if (status.length !== 1 || (status[0] === 'approved') !== spec.authorized)
    throw new Error(`${world.slug}: authority marker disagrees with rendered source evidence`);
  if (/\bundefined\b|\bnull\b/.test(JSON.stringify({ contract: task.output_contract, authority, expected })))
    throw new Error(`${world.slug}: unresolved placeholder in generated contract/evidence/gold`);
  for (const pass of task.passes) {
    const content = world.evidence[pass.evidence_path];
    if (!content || !row.curriculum.reference.children.some(child => child.match === pass.evidence_path && child.soft_output.text.includes(content)))
      throw new Error(`${world.slug}: reference notes omit complete source evidence ${pass.evidence_path}`);
  }
  return row;
});

if (new Set(rows.map(row => row.source_groups[0])).size !== 8 ||
    rows.filter(row => row.split === 'train').length !== 8 || rows.filter(row => row.split === 'test').length !== 8)
  throw new Error('V23 must preserve eight inherited groups with an 8/8 train/test distribution');
for (const domain of sourceDomains) {
  const grouped = rows.filter(row => row.source_groups[0] === domain.inherited_group);
  if (grouped.length !== 2 || grouped.some(row => row.split !== domain.inherited_split))
    throw new Error(`${domain.slug}: variants must inherit one factual group and split`);
  const sizes = grouped.map(row => {
    const result = JSON.parse(row.semantics.expected_files['decision.json']);
    return result.selectedItems === 'none' ? 0 : result.selectedItems.split('; ').length;
  }).sort();
  if (JSON.stringify(sizes) !== JSON.stringify([1, 2]))
    throw new Error(`${domain.slug}: expected singleton and two-item gold, found ${sizes}`);
}

const sourceText = rows.map(row => JSON.stringify(row)).join('\n') + '\n';
const sourceSha = createHash('sha256').update(sourceText).digest('hex');
const dataFiles = ['semantic-iterate-reducers-v23-data.mjs', 'semantic-iterate-reducers-v22-novel-data.mjs', 'semantic-iterate-reducer-selection-contract.mjs'];
const dataHashes = Object.fromEntries(await Promise.all(dataFiles.map(async name =>
  [name, createHash('sha256').update(await readFile(new URL(`./${name}`, import.meta.url))).digest('hex')])));
const builderFiles = ['build-semantic-iterate-reducers-v23.mjs', 'semantic-iterate-worlds-v15-soft-guided-builder.mjs'];
const builderHashes = Object.fromEntries(await Promise.all(builderFiles.map(async name =>
  [name, createHash('sha256').update(await readFile(new URL(`./${name}`, import.meta.url))).digest('hex')])));
await mkdir(output, { recursive: false });
await writeFile(resolve(output, 'source.cases.jsonl'), sourceText);
const counts = { task_variants: rows.length, inherited_factual_groups: 8, independent_factual_world_credit: 0,
  domains: 8, train: 8, test: 8, one_item_variants: 8, two_item_variants: 8,
  evidence_passes: rows.length * 4, providers: 0, teacher_calls: 0 };
const proof = {
  schema: 'natlang.semantic-reducer-source-proof/3', revision: REVISION,
  source_cases_sha256: sourceSha, authored_data_hashes: dataHashes, counts, builder_hashes: builderHashes,
  provider_calls: 0, teacher_calls: 0, training_admission: false, trace_admission: false,
  domains: sourceDomains,
  case_derivations: rows.map((row, index) => ({ id: row.id, group: row.source_groups[0], split: row.split,
    selection_cardinality: row.generation.selection_cardinality,
    candidates: reducerVariants[index].spec.candidates,
    evidence: reducerVariants[index].row.evidence,
    expected: row.semantics.expected,
    decision_rule: JSON.parse(row.semantics.folder_files['task.json']).output_contract.decision_rule })),
  claims: { exact_cardinality_is_authored_for_each_variant: true,
    singleton_ties_choose_one_complete_id_and_never_all_tied_items: true,
    multi_item_measure_values_follow_selected_id_order_and_are_not_summed: true,
    gold_is_derived_from_candidate_facts_and_authority: true,
    reference_children_receive_only_the_current_file_handle_and_prior_notes: true,
    source_groups_and_splits_are_inherited_from_v22: true },
  scope: 'Authored source structure and direct deterministic facts proof only; variants inherit eight V22 factual groups and claim zero new independent worlds. No provider execution, teacher observation, semantic admission, or training admission.'
};
await writeFile(resolve(output, 'source-proof.json'), JSON.stringify(proof, null, 2) + '\n');
const manifest = { schema: 'natlang.neuralese-semantic-iterate-reducers-v23/1', revision: REVISION,
  source_cases: 'source.cases.jsonl', source_cases_sha256: sourceSha,
  authored_data_hashes: dataHashes, builder_hashes: builderHashes,
  source_proof: 'source-proof.json', counts,
  parent_factual_source: 'runs/neuralese-semantic-iterate-reducers-v22-20261008-v5/source.cases.jsonl',
  parent_source_sha256: '50f26edd42c9af8a86be0108ee0fb7aa7452246881aab73a40778058f1c06057',
  independent_factual_worlds: 0, teacher_observations: 0, provider_calls: 0,
  semantic_admission: false, training_admission: false,
  proof_scope: 'Exact authored one-item/two-item contracts and gold derivation from cited facts; no teacher/provider execution.' };
await writeFile(resolve(output, 'source-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
await writeFile(resolve(output, 'README.md'), `# V23 explicit selection-cardinality successor\n\nSixteen task variants cover the eight reviewed V22 domains. Each pair inherits the same factual source group and split: one variant requires exactly one selected item and the other exactly two ranked items. The pair is a task-contract comparison, not two independent factual worlds.\n\nThe singleton contract explicitly selects one item on ties by complete ID and forbids including additional tied items. The two-item contract specifies ranked order and records each measure in selected-item order without summing. Both contracts state the empty-selection behavior. The root task reads the output contract; each note child receives only its current FileHandle and prior notes through the reviewed one-file read-only snapshot boundary.\n\nThe CPU source proof verifies facts, authority, exact field formatting, selected cardinalities, and inherited train/test groups. It uses no provider or teacher calls and grants no source, action, trace, DPO, or training admission.\n\nSource SHA-256: ${sourceSha}\n`);
console.log(JSON.stringify({ revision: REVISION, source_sha256: sourceSha, authored_data_hashes: dataHashes, counts }, null, 2));
