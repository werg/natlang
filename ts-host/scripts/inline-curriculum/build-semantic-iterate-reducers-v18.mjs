#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { makeGuidedSoftIterateCase } from './semantic-iterate-worlds-v15-soft-guided-builder.mjs';
import { worlds } from './semantic-iterate-reducers-v18-novel-data.mjs';

export const REVISION = 'authored-semantic-iterate-reducers-v18/5-explicit-evidence-file-boundary';
const canonical = value => JSON.stringify(value);
const args = process.argv.slice(2);
if (args.length !== 2 || args[0] !== '--out' || !args[1] || args[1].startsWith('--'))
  throw new Error('usage: node build-semantic-iterate-reducers-v18.mjs --out FRESH_DIRECTORY');
const output = resolve(args[1]);
const savedWithWorlds = new Set(['research-instrument-booking', 'privacy-export-minimization', 'volunteer-shift-coverage']);

function validateAuthoredWorlds(items) {
  if (items.length < 8 || items.length > 16 || new Set(items.map(w => w.slug)).size !== items.length || new Set(items.map(w => w.group)).size !== items.length)
    throw new Error('V18 requires 8–16 unique authored worlds and source groups');
  if (new Set(items.map(w => w.domain)).size !== items.length)
    throw new Error('each V18 world must cover a distinct domain');
  for (const w of items) {
    const fields = Object.keys(w.fields);
    if (fields.length !== 4 || Object.keys(w.initial).sort().join() !== [...fields].sort().join())
      throw new Error(`${w.slug}: output and initial draft must have the same four fields`);
    if (w.passes.length !== 4 || w.passStates.length !== 4) throw new Error(`${w.slug}: exactly four passes required`);
    let previous = w.initial;
    const scoped = new Set();
    for (let i=0;i<4;i++) {
      const pass=w.passes[i], state=w.passStates[i];
      if (!w.evidence[pass.evidence_path] || !pass.allowed_fields.length) throw new Error(`${w.slug}: missing pass source or field scope`);
      for (const field of pass.allowed_fields) {
        if (!fields.includes(field)) throw new Error(`${w.slug}: pass ${i+1} uses unknown field ${field}`);
        scoped.add(field);
      }
      for (const field of fields) {
        if (typeof state[field] !== 'string') throw new Error(`${w.slug}: draft field ${field} must be a string`);
        if (!pass.allowed_fields.includes(field) && state[field] !== previous[field]) throw new Error(`${w.slug}: pass ${i+1} changed out-of-scope field ${field}`);
      }
      if (canonical(state)===canonical(previous)) throw new Error(`${w.slug}: pass ${i+1} has no state change`);
      previous=state;
    }
    if (scoped.size!==fields.length) throw new Error(`${w.slug}: not every output field has an evidence-scoped pass`);
    const metadata=w.field_enums?.decision;
    if (!metadata?.intermediate?.includes(w.initial.decision) || !metadata.final?.includes(w.passStates.at(-1).decision))
      throw new Error(`${w.slug}: decision enum does not distinguish placeholder and final contract`);
    if (w.field_enums?.selectedItems || w.field_enums?.measure)
      throw new Error(`${w.slug}: data-dependent selection or measurement must not be enumerated in a way that leaks hidden rank`);
    if (w.justified_revision.pass!==3 || w.passStates[1][w.justified_revision.field]===w.passStates[2][w.justified_revision.field])
      throw new Error(`${w.slug}: pass-three correction is not evidenced by a changed field`);
    const final=w.passStates.at(-1), expected=w.source_summary;
    if (final.caseId!==expected.requestId || final.selectedItems!==expected.selectedItems || final.measure!==expected.measure || final.decision!==expected.decision)
      throw new Error(`${w.slug}: final gold disagrees with deterministic source-fact derivation`);
    for (const candidate of expected.candidates)
      if (!w.evidence['pass-02-register.md'].includes(candidate.id) || !w.evidence['pass-03-conditions.md'].includes(candidate.id))
        throw new Error(`${w.slug}: a derived candidate is absent from staged source facts`);
  }
  const actionValues=new Set(items.map(w=>w.passStates.at(-1).decision));
  if (!actionValues.has('hold') || !items.some(w => w.passStates.at(-1).selectedItems!=='none' && w.passStates.at(-1).decision==='hold'))
    throw new Error('V18 must include selected work held for lack of authority');
  if (!items.some(w => w.passStates.at(-1).selectedItems==='none' && w.passStates.at(-1).decision==='no_action'))
    throw new Error('V18 must include a genuinely empty selection');
}

validateAuthoredWorlds(worlds);
const rows=worlds.map((world,index)=>{
  const row=makeGuidedSoftIterateCase(world,index,{revision:REVISION,shapeVersion:'v16',savedWith:savedWithWorlds.has(world.slug)});
  row.id=row.id.replaceAll('authored_semantic_iterate_worlds_v15','authored_semantic_iterate_reducers_v18')
    .replace(`:v15-${world.slug}-`,`:v18-${world.slug}-`);
  row.family='authored_semantic_iterate_reducers_v18';
  row.family_version=18;
  row.curriculum.family=row.family;
  row.curriculum.family_version=18;
  row.curriculum.shape=`v18-${world.slug}-four-pass-guided-soft-reducer`;
  row.curriculum.variant='evidence-scoped-guided-soft-reducer-with-separate-selection-and-authority/1';
  row.semantics.shape=`v18-${world.slug}-four-pass-guided-soft-reducer`;
  row.generation.generator=REVISION;
  row.generation.independent_world=world.group;
  row.generation.source_quality='New authored fictional operational reducer world with a source-derived provisional correction, explicit tie/exception rules, and separate selection and authority states. Static source derivation only; no model/provider calls.';
  row.generation.capture_contract.saved_with_examples=savedWithWorlds.has(world.slug)
    ? 'The current pass child is authored as a saved nl<T> tag rebound through .with(captures); these cases exercise the saved-tag capture surface.'
    : 'The current pass child uses the established direct nl.with(captures) form.';
  const task=JSON.parse(row.semantics.folder_files['task.json']);
  if (task.output_contract.final_field_enums.selectedItems || task.output_contract.final_field_enums.measure)
    throw new Error(`${world.slug}: ranked selection leaked into enum metadata`);
  const children=row.curriculum.reference.children;
  for (let pass=0;pass<world.passes.length;pass++) {
    const note=children[pass+1]?.soft_output?.text;
    if (typeof note!=='string') throw new Error(`${world.slug}: missing causal pass ${pass+1} note`);
    for (let prior=0;prior<=pass;prior++) {
      const p=world.passes[prior];
      if (!note.includes(p.evidence_path) || !note.includes(world.evidence[p.evidence_path])) throw new Error(`${world.slug}: pass ${pass+1} note omits observed source ${p.evidence_path}`);
    }
    for (let future=pass+1;future<world.passes.length;future++) {
      const p=world.passes[future];
      if (note.includes(p.evidence_path) || note.includes(world.evidence[p.evidence_path])) throw new Error(`${world.slug}: pass ${pass+1} note contains future evidence ${p.evidence_path}`);
    }
  }
  if (savedWithWorlds.has(world.slug) && !row.curriculum.reference.root[0][1].code.includes('stepTemplate.with('))
    throw new Error(`${world.slug}: saved tag .with form is absent from the visible scaffold`);
  return row;
});

const sourceText=rows.map(canonical).join('\n')+'\n';
const sourceSha=createHash('sha256').update(sourceText).digest('hex');
const dataPath=new URL('./semantic-iterate-reducers-v18-novel-data.mjs',import.meta.url);
const dataSha=createHash('sha256').update(await readFile(dataPath)).digest('hex');
await mkdir(output,{recursive:false});
await writeFile(resolve(output,'source.cases.jsonl'),sourceText);
const proof={
  schema:'natlang.semantic-reducer-source-proof/1',revision:REVISION,source_cases_sha256:sourceSha,authored_data_sha256:dataSha,
  counts:{task_variants:rows.length,independent_factual_worlds:rows.length,source_groups:rows.length,train:rows.filter(r=>r.split==='train').length,test:rows.filter(r=>r.split==='test').length,domains:new Set(worlds.map(w=>w.domain)).size,passes:rows.length*4,saved_with_examples:savedWithWorlds.size},
  provider_calls:0,teacher_calls:0,training_admission:false,trace_admission:false,
  claims:{final_values_derive_from_authored_facts:true,cumulative_notes_include_only_observed_prefix:true,ranked_selection_not_enum_encoded:true,selection_distinct_from_execution_authority:true},
  case_derivations:rows.map((row,i)=>({id:row.id,group:row.source_groups[0],split:row.split,domain:worlds[i].domain,request_id:worlds[i].source_summary.requestId,source_facts:worlds[i].source_summary.candidates,selected_items:worlds[i].source_summary.selectedItems,measure:worlds[i].source_summary.measure,authorized:worlds[i].source_summary.authorized,decision:worlds[i].source_summary.decision,expected:row.semantics.expected})),
  scope:'Authored static/reference proof only. It checks causal evidence completeness and exact gold derivation from authored fixture rules. It is neither teacher observation, independent semantic approval, nor training admission.',
};
await writeFile(resolve(output,'source-proof.json'),JSON.stringify(proof,null,2)+'\n');
const review={schema:'natlang.neuralese-source-quality-review/1',revision:REVISION,source_cases_sha256:sourceSha,review_type:'new authored V18 operational reducer task pool; independent root semantic review required',admission_granted:false,model_calls:0,provider_calls:0,teacher_trajectories:0,counts:proof.counts,worlds:proof.case_derivations.map(({id,group,split,domain,request_id,selected_items,measure,decision,expected})=>({id,group,split,domain,request_id,selected_items,measure,decision,expected}))};
await writeFile(resolve(output,'source-quality-review.json'),JSON.stringify(review,null,2)+'\n');
const manifest={schema:'natlang.neuralese-semantic-iterate-reducers-v18/1',revision:REVISION,source_cases:'source.cases.jsonl',source_cases_sha256:sourceSha,authored_data:'ts-host/scripts/inline-curriculum/semantic-iterate-reducers-v18-novel-data.mjs',authored_data_sha256:dataSha,world_count:rows.length,task_variant_count:rows.length,independent_factual_worlds:rows.length,source_groups_unique:true,train_count:proof.counts.train,test_count:proof.counts.test,domain_count:proof.counts.domains,pass_count:proof.counts.passes,saved_with_tag_worlds:[...savedWithWorlds],scripted_proof:'source-proof.json',proof_scope:'static authored facts and reference scaffold; no teacher calls or admission',teacher_observations:0,provider_calls:0,training_admission:false,trace_admission:false};
await writeFile(resolve(output,'source-manifest.json'),JSON.stringify(manifest,null,2)+'\n');
await writeFile(resolve(output,'README.md'),`# V18 semantic reducer source proposal\n\nEight new fictional operational reducer worlds cover software release gating, dependency upgrade batching, research instrument booking, privacy-preserving export, incident mitigation, sample batch release, volunteer shift coverage, and records retention. Each has its own factual source group, four pass-local evidence records, and deterministic results derived from the staged facts. The rules exercise exact-build exceptions, conflicts, tie-breaking, purpose-limited consent, linked retest identity, one-person assignment constraints, legal holds, and separate selection versus execution authority.\n\nThree scaffolds demonstrate a saved typed nl<T> tag rebound with .with(captures); the other cases retain the direct form. Each causal note includes the complete text of files observed through the current pass and excludes later files. Ranked result fields remain strings with explicit order/delimiter rules, never score-ordered enum unions.\n\nStatic authored proof: source-proof.json. This is a source proposal, not a teacher result or training admission. No provider calls, teacher observations, or admission occurred.\n\nSource SHA-256: ${sourceSha}\nAuthored data SHA-256: ${dataSha}\n`);
console.log(JSON.stringify(manifest,null,2));
