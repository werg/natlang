#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { makeGuidedSoftIterateCase } from './semantic-iterate-worlds-v15-soft-guided-builder.mjs';
import { worlds } from './semantic-iterate-reducers-v21-counterfactual-data.mjs';
import { worldSpecs as v18Specs } from './semantic-iterate-reducers-v18-novel-data.mjs';

export const REVISION = 'authored-semantic-iterate-reducers-v21/1-fact-derived-counterfactuals';
const canonical = value => JSON.stringify(value);
const args = process.argv.slice(2);
if (args.length !== 2 || args[0] !== '--out' || !args[1] || args[1].startsWith('--'))
  throw new Error('usage: node build-semantic-iterate-reducers-v21.mjs --out FRESH_DIRECTORY');
const output = resolve(args[1]);
const parentBySlug = new Map(v18Specs.map((spec, index) => [spec.slug, { spec, index,
  group: `v18:${spec.slug}:world`, split: index % 2 === 0 ? 'train' : 'test' }]));
if (worlds.length !== 16) throw new Error(`V21 requires 16 task variants, got ${worlds.length}`);

const rows = worlds.map((world, index) => {
  const parentSlug = world.slug.match(/^(.*)-v21-/)?.[1];
  const parent = parentBySlug.get(parentSlug);
  if (!parent) throw new Error(`${world.slug}: no inherited V18 parent group`);
  if (world.group !== parent.group) throw new Error(`${world.slug}: source group changed`);
  const row = makeGuidedSoftIterateCase(world, index, { revision: REVISION, shapeVersion: 'v16' });
  row.id = row.id.replaceAll('authored_semantic_iterate_worlds_v15', 'authored_semantic_iterate_reducers_v21')
    .replaceAll(`:v15-${world.slug}-`, `:v21-${world.slug}-`);
  row.family = 'authored_semantic_iterate_reducers_v21'; row.family_version = 21;
  row.curriculum.family = row.family; row.curriculum.family_version = 21;
  row.curriculum.shape = `v21-${world.slug}-four-pass-counterfactual-reducer`;
  row.curriculum.variant = 'inherited-v18-group-fact-derived-counterfactual/1';
  row.semantics.shape = row.curriculum.shape; row.generation.generator = REVISION;
  row.generation.independent_world = null; row.generation.independent_world_credit = false;
  row.generation.counterfactual_parent_group = parent.group;
  row.generation.source_quality = 'Fact-derived counterfactual task variant under an inherited V18 source group and split. It adds no independent-world credit.';
  row.generation.capture_contract.saved_with_examples = 'Typed four-pass nl.with<Neuralese<string>> step children carry current evidence into cumulative notes; final typed Draft emits the derived selection, measure, and separate authorization decision.';
  row.split = parent.split; row.split_group = parent.group; row.source_groups = [parent.group];
  row.curriculum.source_group_lineage = { parent_family: 'authored_semantic_iterate_reducers_v18',
    parent_group: parent.group, inherited_split: parent.split };

  const task = JSON.parse(row.semantics.folder_files['task.json']);
  if (JSON.stringify(task).includes(parent.spec.requestId))
    throw new Error(`${world.slug}: a stale V18 request identifier remains in task context`);
  if (!task.instruction.includes(world.source_summary.requestId) || !task.output_contract.decision_rule ||
      !task.output_contract.decision_rule.includes(world.decision_rule))
    throw new Error(`${world.slug}: request identity is missing from the task context`);
  if (task.output_contract.final_field_enums.selectedItems || task.output_contract.final_field_enums.measure)
    throw new Error(`${world.slug}: data-dependent outputs must not be encoded as finite enums`);
  const notes = row.curriculum.reference.children.slice(1, 5).map(child => child.soft_output?.text ?? '');
  const passes = task.passes;
  if (notes.length !== 4) throw new Error(`${world.slug}: expected four cumulative reference notes`);
  for (let i = 0; i < 4; i++) {
    for (let observed = 0; observed <= i; observed++) {
      const text = world.evidence[passes[observed].evidence_path];
      if (!notes[i].includes(text)) throw new Error(`${world.slug}: reference pass ${i + 1} omits observed evidence ${passes[observed].evidence_path}`);
    }
    for (let future = i + 1; future < 4; future++) {
      const path = passes[future].evidence_path;
      if (notes[i].includes(world.evidence[path]) || notes[i].includes(path))
        throw new Error(`${world.slug}: future evidence leaks into reference pass ${i + 1}`);
    }
  }
  return row;
});

const groups = new Map();
for (const row of rows) groups.set(row.generation.counterfactual_parent_group,
  [...(groups.get(row.generation.counterfactual_parent_group) ?? []), row]);
if (groups.size !== 8 || [...groups.values()].some(cases => cases.length !== 2))
  throw new Error('V21 must have exactly two fact-derived variants for each of eight inherited V18 groups');
for (const [group, cases] of groups) {
  const parentSlug = group.slice('v18:'.length, -':world'.length);
  const parent = parentBySlug.get(parentSlug);
  if (cases.some(row => row.split !== parent.split || row.source_groups[0] !== group))
    throw new Error(`${group}: inherited split or source group changed`);
}

const sourceText = rows.map(canonical).join('\n') + '\n';
const sourceSha = createHash('sha256').update(sourceText).digest('hex');
const dataFiles = ['semantic-iterate-reducers-v18-novel-data.mjs', 'semantic-iterate-reducers-v21-counterfactual-data.mjs'];
const authoredData = Object.fromEntries(await Promise.all(dataFiles.map(async name => [name,
  createHash('sha256').update(await readFile(new URL(`./${name}`, import.meta.url))).digest('hex')])));
await mkdir(output, { recursive: false });
await writeFile(resolve(output, 'source.cases.jsonl'), sourceText);
const counts = { task_variants: rows.length, task_variants_per_group: 2, inherited_parent_groups: groups.size,
  independent_new_worlds: 0, train: rows.filter(row => row.split === 'train').length,
  test: rows.filter(row => row.split === 'test').length, passes: rows.length * 4 };
const proof = {
  schema: 'natlang.semantic-reducer-source-proof/2', revision: REVISION,
  source_cases_sha256: sourceSha, authored_data_sha256: authoredData, counts,
  provider_calls: 0, teacher_calls: 0, training_admission: false, trace_admission: false,
  claims: {
    each_variant_has_unique_request_and_fact_set: true,
    source_groups_and_splits_inherited_from_v18: true,
    final_selection_and_measure_computed_from_candidates: true,
    authority_changes_only_final_disposition: true,
    reference_notes_accumulate_only_current_and_prior_evidence: true,
    data_dependent_fields_not_encoded_as_enums: true,
    independent_new_worlds: 0,
  },
  case_derivations: rows.map((row, index) => ({ id: row.id, parent_group: row.generation.counterfactual_parent_group,
    split: row.split, request_id: row.semantics.expected.caseId,
    candidates: worlds[index].source_summary.candidates,
    selection: row.semantics.expected.selectedItems, measure: row.semantics.expected.measure,
    decision: row.semantics.expected.decision })),
  scope: 'Authored fact-derived counterfactual source validation only. The expected outputs are derived by the domain rules in the source builder; this static proof is not teacher evidence, semantic admission, independent-world credit, or training admission.'
};
await writeFile(resolve(output, 'source-proof.json'), JSON.stringify(proof, null, 2) + '\n');
const manifest = {
  schema: 'natlang.neuralese-semantic-iterate-reducers-v21/1', revision: REVISION,
  source_cases: 'source.cases.jsonl', source_cases_sha256: sourceSha,
  authored_data_hashes: authoredData, counts,
  parent_source: 'V18 authored operational reducer groups',
  parent_group_split: Object.fromEntries([...parentBySlug].map(([slug, parent]) =>
    [parent.group, { split: parent.split, parent_slug: slug }])),
  source_proof: 'source-proof.json', provider_calls: 0, teacher_observations: 0,
  independent_new_worlds: 0, training_admission: false, trace_admission: false,
};
await writeFile(resolve(output, 'source-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
await writeFile(resolve(output, 'README.md'), `# V21 fact-derived semantic reducer variants\n\nThis source contains sixteen counterfactual task variants, two under each of the eight V18 parent groups. It preserves each parent group and train/test split and adds zero independent-world credit. Candidate facts exercise exact-ID exceptions, cutoff ties, eligibility correction, scoped waivers, purpose-specific consent, same-person role constraints, linked retests, and authorization separate from selection.\n\nEach task retains the four-pass typed soft-note scaffold. The reference note chain includes the complete evidence read so far and excludes future pass files. The final literal-union Draft is derived from supported facts and the output contract; no per-world answer is placed in its prompt.\n\nThe included source proof checks authored data invariants only. It records zero teacher/provider calls and does not grant semantic or training admission. A separate frozen runtime reference proof is required before generation.\n\nSource SHA-256: ${sourceSha}\n`);
console.log(JSON.stringify({ revision: REVISION, source_sha256: sourceSha, counts, authored_data_sha256: authoredData }, null, 2));
