#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { makeGuidedSoftIterateCase } from './semantic-iterate-worlds-v15-soft-guided-builder.mjs';
import { worlds as v19Worlds } from './semantic-iterate-reducers-v19-paired-data.mjs';
import { worlds as v18Worlds } from './semantic-iterate-reducers-v18-novel-data.mjs';

export const REVISION = 'authored-semantic-iterate-reducers-v19/1-paired-counterfactuals-explicit-disposition-map';
const canonical = value => JSON.stringify(value);
const args = process.argv.slice(2);
if (args.length !== 2 || args[0] !== '--out' || !args[1] || args[1].startsWith('--'))
  throw new Error('usage: node build-semantic-iterate-reducers-v19.mjs --out FRESH_DIRECTORY');
const output = resolve(args[1]);
const parentIndex = new Map(v18Worlds.map((world, index) => [world.slug, index]));
const parentByGroup = new Map(v18Worlds.map((world, index) => [`v18:${world.slug}:world`, { world, index, split:index % 2 === 0 ? 'train' : 'test' }]));
const savedWith = new Set(['research-instrument-booking', 'privacy-export-minimization', 'volunteer-shift-coverage']);

if (v19Worlds.length !== 9) throw new Error('V19 expects eight paired facts variants and one RT-227 decision-contract revision');
const rows = v19Worlds.map((world, index) => {
  const parent = parentByGroup.get(world.group);
  if (!parent) throw new Error(`${world.slug}: missing V18 lineage group ${world.group}`);
  const parentSlug = parent.world.slug;
  const row = makeGuidedSoftIterateCase(world, parent.index, {
    revision: REVISION,
    shapeVersion: 'v16',
    savedWith: savedWith.has(parentSlug),
  });
  row.id = row.id
    .replaceAll('authored_semantic_iterate_worlds_v15', 'authored_semantic_iterate_reducers_v19')
    .replaceAll(`:v15-${world.slug}-`, `:v19-${world.slug}-`);
  row.family = 'authored_semantic_iterate_reducers_v19';
  row.family_version = 19;
  row.curriculum.family = row.family;
  row.curriculum.family_version = 19;
  row.curriculum.shape = `v19-${world.slug}-four-pass-paired-counterfactual-reducer`;
  row.curriculum.variant = 'paired-factual-counterfactual-with-complete-selection-authority-decision-map/1';
  row.semantics.shape = row.curriculum.shape;
  row.generation.generator = REVISION;
  row.generation.independent_world = null;
  row.generation.independent_world_credit = false;
  row.generation.counterfactual_parent_group = world.group;
  row.generation.source_quality = 'Paired factual counterfactual of a V18 authored reducer group. Expected output derives from this variant’s staged facts and rule; no new independent-world credit.';
  row.split = parent.split;
  row.split_group = world.group;
  row.source_groups = [world.group];
  row.curriculum.source_group_lineage = { parent_family:'authored_semantic_iterate_reducers_v18', parent_group:world.group, inherited_split:parent.split };
  row.generation.capture_contract.saved_with_examples = savedWith.has(parentSlug)
    ? 'The current pass child uses the saved nl<T> tag rebound through .with(captures), alongside four typed iterateOn steps.'
    : 'The current pass child uses direct nl.with(captures), alongside four typed iterateOn steps.';

  const task = JSON.parse(row.semantics.folder_files['task.json']);
  const mapping = world.decision_rule.match(/Decision mapping:.*$/)?.[0];
  if (!mapping || !task.output_contract.decision_rule.includes(mapping) || !task.output_contract.fields.decision.includes(mapping))
    throw new Error(`${world.slug}: complete disposition mapping is not present in the captured rule and decision format`);
  if (!task.instruction.includes(mapping)) throw new Error(`${world.slug}: task instruction omits the explicit decision mapping`);
  const final = row.semantics.expected;
  const decisionAction = world.decision_rule.match(/If the selection is nonempty and every stated authorization condition is met, use ([^;]+);/)?.[1];
  const expectedDecision = final.selectedItems === 'none' ? 'no_action' : world.source_summary.authorized ? decisionAction : 'hold';
  if (final.decision !== expectedDecision) throw new Error(`${world.slug}: decision mapping disagrees with source-derived expected state`);
  if (task.output_contract.final_field_enums.selectedItems || task.output_contract.final_field_enums.measure)
    throw new Error(`${world.slug}: data-dependent output leaked into finite enum metadata`);
  if (parent.world.group !== world.group || row.split !== parent.split)
    throw new Error(`${world.slug}: source group or split lineage changed`);

  const causalNotes = row.curriculum.reference.children.slice(1, 5).map(c => c.soft_output?.text ?? '');
  for (let pass=0; pass<causalNotes.length; pass++) {
    const note = causalNotes[pass];
    if (!note.includes(mapping)) throw new Error(`${world.slug}: pass ${pass + 1} reference note dropped the complete decision mapping`);
    for (let prior=0; prior<=pass; prior++) {
      const p=world.passes[prior];
      if (!note.includes(world.evidence[p.evidence_path])) throw new Error(`${world.slug}: pass ${pass + 1} note omits observed source ${p.evidence_path}`);
    }
    for (let future=pass+1; future<world.passes.length; future++) {
      const p=world.passes[future];
      if (note.includes(world.evidence[p.evidence_path]) || note.includes(p.evidence_path))
        throw new Error(`${world.slug}: pass ${pass + 1} note contains future evidence ${p.evidence_path}`);
    }
  }
  return row;
});

const byParent = new Map();
for (const row of rows) byParent.set(row.generation.counterfactual_parent_group, (byParent.get(row.generation.counterfactual_parent_group) ?? 0) + 1);
if (byParent.get('v18:records-retention-disposition:world') !== 2 || [...byParent.values()].filter(n=>n===1).length !== 7)
  throw new Error('V19 must contain seven one-to-one variants plus the RT-227 correction and RT-228 counterfactual pair');

const sourceText = rows.map(canonical).join('\n') + '\n';
const sourceSha = createHash('sha256').update(sourceText).digest('hex');
const dataFiles = ['semantic-iterate-reducers-v18-novel-data.mjs','semantic-iterate-reducers-v19-paired-data.mjs'];
const dataHashes = Object.fromEntries(await Promise.all(dataFiles.map(async name => [name, createHash('sha256').update(await readFile(new URL(`./${name}`, import.meta.url))).digest('hex')])));
await mkdir(output, { recursive:false });
await writeFile(resolve(output,'source.cases.jsonl'),sourceText);
const counts = {
  task_variants:rows.length,
  factual_counterfactual_variants:8,
  same_world_contract_revisions:1,
  independent_new_worlds:0,
  parent_groups:new Set(rows.map(r=>r.generation.counterfactual_parent_group)).size,
  train:rows.filter(r=>r.split==='train').length,
  test:rows.filter(r=>r.split==='test').length,
  passes:rows.length*4,
  saved_with_examples:rows.filter(r=>savedWith.has(r.curriculum.source_group_lineage.parent_group.slice(4,-6))).length,
};
const proof = {
  schema:'natlang.semantic-reducer-source-proof/1', revision:REVISION, source_cases_sha256:sourceSha, authored_data_sha256:dataHashes,
  counts, provider_calls:0, teacher_calls:0, training_admission:false, trace_admission:false,
  claims:{paired_variants_inherit_parent_group_and_split:true,all_expected_values_derive_from_variant_facts:true,empty_selection_maps_to_no_action_even_if_authority_absent:true,nonempty_unauthorized_selection_maps_to_hold:true,ranked_selection_not_enum_encoded:true,cumulative_notes_include_only_observed_prefix:true},
  case_derivations:rows.map(row=>({id:row.id,variant_group:row.source_groups[0],parent_group:row.generation.counterfactual_parent_group,split:row.split,request_id:row.semantics.expected.caseId,facts:JSON.parse(row.semantics.folder_files['task.json']).passes.map(p=>({pass:p.name,evidence_path:p.evidence_path})),expected:row.semantics.expected})),
  scope:'Static authored counterfactual derivation and reference scaffold checks only. No teacher/provider observations, independent-world credit, semantic admission, or training admission.',
};
await writeFile(resolve(output,'source-proof.json'),JSON.stringify(proof,null,2)+'\n');
const manifest={schema:'natlang.neuralese-semantic-iterate-reducers-v19/1',revision:REVISION,source_cases:'source.cases.jsonl',source_cases_sha256:sourceSha,authored_data_hashes:dataHashes,counts,parent_source:'V18 authored operational reducer source group lineage',parent_group_split:Object.fromEntries([...parentByGroup].map(([group,p])=>[group,{split:p.split,parent_slug:p.world.slug}])),scripted_proof:'source-proof.json',provider_calls:0,teacher_observations:0,independent_new_worlds:0,training_admission:false,trace_admission:false};
await writeFile(resolve(output,'source-manifest.json'),JSON.stringify(manifest,null,2)+'\n');
await writeFile(resolve(output,'source-quality-review.json'),JSON.stringify({schema:'natlang.neuralese-source-quality-review/1',revision:REVISION,source_cases_sha256:sourceSha,review_type:'paired V18-lineage counterfactual task variants; root semantic/source-contract review required',admission_granted:false,model_calls:0,provider_calls:0,teacher_trajectories:0,counts,worlds:rows.map(r=>({id:r.id,group:r.source_groups[0],split:r.split,parent_group:r.generation.counterfactual_parent_group,expected:r.semantics.expected}))},null,2)+'\n');
await writeFile(resolve(output,'README.md'),`# V19 paired semantic reducer source draft\n\nNine runnable source variants: eight factual counterfactuals across the V18 reducer groups plus one same-world RT-227 decision-contract revision. Every case inherits the V18 parent source group and split; these add zero independent-world credit. RT-227 and RT-228 remain in the same test group.\n\nThe explicit decision rule now maps an empty eligible selection to no_action even when authority is absent; a nonempty selection maps to the domain action only when required authority is present, and otherwise to hold. This rule appears in task instructions, output decision format, captured decision_rule, and the request-pass reference notes. Authority evidence is phrased as a fact and does not prescribe a final disposition.\n\nThe reference scaffolds use four current-file-only iterateOn passes, cumulative observed evidence, and a typed final interpreter. No provider calls or admission occurred. Source proof is static and scaffold-assisted; root semantic review and final capture-visibility review are pending.\n\nSource SHA-256: ${sourceSha}\n`);
console.log(JSON.stringify(manifest,null,2));
