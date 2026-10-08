#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { makeGuidedSoftIterateCase } from './semantic-iterate-worlds-v15-soft-guided-builder.mjs';
import { worlds as v20Worlds } from './semantic-iterate-reducers-v20-counterfactual-data.mjs';
import { worlds as v18Worlds } from './semantic-iterate-reducers-v18-novel-data.mjs';

export const REVISION = 'authored-semantic-iterate-reducers-v20/4-note-only-child-context';
const canonical = value => JSON.stringify(value);
const args = process.argv.slice(2);
if (args.length !== 2 || args[0] !== '--out' || !args[1] || args[1].startsWith('--'))
  throw new Error('usage: node build-semantic-iterate-reducers-v20.mjs --out FRESH_DIRECTORY');
const output = resolve(args[1]);
const parentByGroup = new Map(v18Worlds.map((world, index) => [`v18:${world.slug}:world`, { world, index, split:index % 2 === 0 ? 'train' : 'test' }]));
const savedWith = new Set(['research-instrument-booking','privacy-export-minimization','volunteer-shift-coverage']);
if (v20Worlds.length !== 24) throw new Error(`V20 requires 24 cases, received ${v20Worlds.length}`);
const rows = v20Worlds.map((world, index) => {
  const parent = parentByGroup.get(world.group);
  if (!parent) throw new Error(`${world.slug}: missing V18 source group`);
  const row = makeGuidedSoftIterateCase(world, parent.index, { revision:REVISION, shapeVersion:'v16', savedWith:savedWith.has(parent.world.slug) });
  if (JSON.stringify(row).includes(parent.world.requestId)) throw new Error(`${world.slug}: a captured row still refers to its parent's old request ID`);
  row.id = row.id.replaceAll('authored_semantic_iterate_worlds_v15','authored_semantic_iterate_reducers_v20')
    .replaceAll(`:v15-${world.slug}-`,`:v20-${world.slug}-`);
  row.family='authored_semantic_iterate_reducers_v20'; row.family_version=20;
  row.curriculum.family=row.family; row.curriculum.family_version=20;
  row.curriculum.shape=`v20-${world.slug}-four-pass-counterfactual-reducer`;
  row.curriculum.variant='fact-derived-eligible-authority-empty-counterfactual/1';
  row.semantics.shape=row.curriculum.shape; row.generation.generator=REVISION;
  row.generation.independent_world=null; row.generation.independent_world_credit=false;
  row.generation.counterfactual_parent_group=world.group;
  row.generation.source_quality='Fact-derived counterfactual task variant under an inherited V18 source group and split. It adds no independent-world credit.';
  row.generation.capture_contract.saved_with_examples=savedWith.has(parent.world.slug)
    ? 'Current-pass child uses the saved nl<T> tag rebound through .with(captures), with four typed iterateOn steps.'
    : 'Current-pass child uses direct nl.with(captures), with four typed iterateOn steps.';
  row.split=parent.split; row.split_group=world.group; row.source_groups=[world.group];
  row.curriculum.source_group_lineage={parent_family:'authored_semantic_iterate_reducers_v18',parent_group:world.group,inherited_split:parent.split};
  const task=JSON.parse(row.semantics.folder_files['task.json']);
  if (JSON.stringify(task).includes(parent.world.requestId)) throw new Error(`${world.slug}: inherited parent request ID remains in task context`);
  const decisionMap=world.decision_rule.match(/Decision mapping:.*$/)?.[0];
  if (!decisionMap || !task.instruction.includes(decisionMap) || !task.output_contract.decision_rule.includes(decisionMap) || !task.output_contract.fields.decision.includes(decisionMap))
    throw new Error(`${world.slug}: complete empty/nonempty/authority mapping is not captured in all final task contexts`);
  if (task.output_contract.final_field_enums.selectedItems || task.output_contract.final_field_enums.measure)
    throw new Error(`${world.slug}: data-dependent selection or metric must not be encoded as enum values`);
  if (parent.world.slug==='research-instrument-booking') {
    if (!task.instruction.includes('Facility staffed hours begin at 09:00')) throw new Error(`${world.slug}: staffed-hours policy is missing`);
    const register=world.evidence['pass-02-register.md'];
    const audit=world.evidence['pass-03-conditions.md'];
    for (const match of register.matchAll(/(SLOT-[A-Z]): start (\d\d:\d\d); duration (\d+) minutes; end (\d\d:\d\d); instrument (MIC-\d+)\./g)) {
      const [,id,start,duration,end,instrument]=match;
      const auditLine=audit.split('\n').find(line=>line.startsWith(`${id}:`));
      if (duration!=='90' || end>='11:00' || start>='09:00' || !auditLine?.includes('after-hours=true') || !auditLine.includes(`safety permit names=${instrument}/${id}.`))
        throw new Error(`${world.slug}: every pre-09:00 candidate must be 90 minutes, end before 11:00, and carry its exact after-hours permit`);
    }
  }
  if (row.split!==parent.split || world.group!==`v18:${parent.world.slug}:world`) throw new Error(`${world.slug}: inherited group/split changed`);
  const notes=row.curriculum.reference.children.slice(1,5).map(c=>c.soft_output?.text??'');
  if (notes.length!==4) throw new Error(`${world.slug}: expected four reference notes`);
  const passes=JSON.parse(row.semantics.folder_files['task.json']).passes;
  for(let i=0;i<4;i++) {
    if(!notes[i].includes(decisionMap)) throw new Error(`${world.slug}: pass ${i+1} lost complete decision mapping`);
    for(let prior=0;prior<=i;prior++) if(!notes[i].includes(world.evidence[passes[prior].evidence_path])) throw new Error(`${world.slug}: pass ${i+1} omitted observed evidence`);
    for(let future=i+1;future<4;future++) if(notes[i].includes(world.evidence[passes[future].evidence_path])||notes[i].includes(passes[future].evidence_path)) throw new Error(`${world.slug}: future evidence leaked into pass ${i+1}`);
  }
  return row;
});
const grouped=new Map(); for(const r of rows) grouped.set(r.generation.counterfactual_parent_group,(grouped.get(r.generation.counterfactual_parent_group)??0)+1);
if(grouped.size!==8 || [...grouped.values()].some(n=>n!==3)) throw new Error('V20 must contain exactly three task variants for each of eight inherited parent groups');
for(const [group,variants] of new Map([...grouped.keys()].map(g=>[g,rows.filter(r=>r.generation.counterfactual_parent_group===g)]))) {
  const outcomes=variants.map(r=>r.semantics.expected);
  if(!outcomes.some(x=>x.selectedItems==='none'&&x.decision==='no_action')) throw new Error(`${group}: missing empty-selection/no_action branch`);
  if(!outcomes.some(x=>x.selectedItems!=='none'&&x.decision==='hold')) throw new Error(`${group}: missing eligible but unauthorized branch`);
  if(!outcomes.some(x=>x.selectedItems!=='none'&&x.decision!=='hold')) throw new Error(`${group}: missing eligible and authorized branch`);
}
const sourceText=rows.map(canonical).join('\n')+'\n';
const sourceSha=createHash('sha256').update(sourceText).digest('hex');
const files=['semantic-iterate-reducers-v18-novel-data.mjs','semantic-iterate-reducers-v20-counterfactual-data.mjs'];
const authoredData=Object.fromEntries(await Promise.all(files.map(async name=>[name,createHash('sha256').update(await readFile(new URL(`./${name}`,import.meta.url))).digest('hex')])));
await mkdir(output,{recursive:false});
await writeFile(resolve(output,'source.cases.jsonl'),sourceText);
const counts={task_variants:rows.length,counterfactuals:rows.length,parent_groups:grouped.size,independent_new_worlds:0,train:rows.filter(r=>r.split==='train').length,test:rows.filter(r=>r.split==='test').length,passes:rows.length*4,saved_with_examples:rows.filter(r=>savedWith.has(r.generation.counterfactual_parent_group.slice(4,-6))).length};
const proof={schema:'natlang.semantic-reducer-source-proof/1',revision:REVISION,source_cases_sha256:sourceSha,authored_data_sha256:authoredData,counts,provider_calls:0,teacher_calls:0,training_admission:false,trace_admission:false,claims:{each_parent_group_has_eligible_authorized_eligible_unauthorized_and_empty_branches:true,expected_values_derive_from_variant_candidate_facts:true,rank_selection_not_disclosed_by_field_enum:true,source_group_and_split_inherited_from_v18:true,cumulative_reference_notes_contain_only_observed_evidence_prefix:true},case_derivations:rows.map(r=>({id:r.id,parent_group:r.generation.counterfactual_parent_group,split:r.split,case_id:r.semantics.expected.caseId,candidate_facts:JSON.parse(r.semantics.folder_files['task.json']).passes.map(p=>({name:p.name,evidence_path:p.evidence_path})),expected:r.semantics.expected})),scope:'Static authored counterfactual and source-reference checks only; no teacher/provider observations, independent-world credit, semantic admission or training admission.'};
await writeFile(resolve(output,'source-proof.json'),JSON.stringify(proof,null,2)+'\n');
const manifest={schema:'natlang.neuralese-semantic-iterate-reducers-v20/1',revision:REVISION,source_cases:'source.cases.jsonl',source_cases_sha256:sourceSha,authored_data_hashes:authoredData,counts,parent_source:'V18 authored operational reducer groups',parent_group_split:Object.fromEntries([...parentByGroup].map(([group,p])=>[group,{split:p.split,parent_slug:p.world.slug}])),scripted_proof:'source-proof.json',provider_calls:0,teacher_observations:0,independent_new_worlds:0,training_admission:false,trace_admission:false};
await writeFile(resolve(output,'source-manifest.json'),JSON.stringify(manifest,null,2)+'\n');
await writeFile(resolve(output,'README.md'),`# V20 semantic reducer counterfactual source draft\n\nThis source contains 24 fact-derived task variants across the eight V18 parent source groups, three per group. Each parent group includes an eligible/authorized case, an eligible/unauthorized case, and an empty-selection case. The cases inherit the parent split and group and add zero independent-world credit. They vary candidate facts, exact-ID exceptions, score or time ties, ordering, and authority evidence.\n\nThe generated reference scaffold uses a typed seed, four current-file-only typed soft-note passes, and a typed final interpreter. Its cumulative notes quote only observed source prefixes and are checked for future-file leakage. Selection and authority are separate; empty eligible selection maps to no_action, and authorization cannot rewrite the evidence-derived selection or measure.\n\nThe proof is static source/reference validation, not teacher evidence or training admission. Use the next frozen runtime receipt for any runtime reference proof.\n\nSource SHA-256: ${sourceSha}\n`);
console.log(JSON.stringify({revision:REVISION,source_sha256:sourceSha,counts,authored_data_sha256:authoredData},null,2));
