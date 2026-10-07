#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { makeGuidedSoftIterateCase } from './semantic-iterate-worlds-v15-soft-guided-builder.mjs';
import { worlds } from './semantic-iterate-worlds-v17-data.mjs';

export const REVISION = 'authored-semantic-iterate-worlds-v17/3-guided-enum-final-types';
const canonical = value => JSON.stringify(value);
const args = process.argv.slice(2);
if (args.length !== 2 || args[0] !== '--out' || !args[1] || args[1].startsWith('--'))
  throw new Error('usage: node build-semantic-iterate-worlds-v17-guided.mjs --out FRESH_DIRECTORY');
const output = resolve(args[1]);

function validateWorlds(rows) {
  if (rows.length !== 24 || new Set(rows.map(world => world.slug)).size !== 24 || new Set(rows.map(world => world.group)).size !== 24)
    throw new Error('V17 requires 24 unique authored scenarios and source groups');
  const groups = new Set();
  for (const world of rows) {
    const fields = Object.keys(world.fields);
    if (!/^[a-z][a-z0-9_]+$/.test(world.slug) || groups.has(world.group)) throw new Error(`bad V17 identity ${world.slug}`);
    groups.add(world.group);
    if (world.passes.length !== 4 || world.passStates.length !== 4) throw new Error(`${world.slug}: exactly four passes/states required`);
    if (fields.some(field => typeof world.initial[field] !== 'string')) throw new Error(`${world.slug}: malformed placeholder`);
    let prior = world.initial;
    const seen = new Set();
    for (let i = 0; i < 4; i++) {
      const pass = world.passes[i], state = world.passStates[i];
      if (!world.evidence[pass.evidence_path] || !pass.allowed_fields.length) throw new Error(`${world.slug}: missing pass evidence or fields`);
      for (const field of pass.allowed_fields) seen.add(field);
      if (fields.some(field => typeof state[field] !== 'string')) throw new Error(`${world.slug}: incomplete state ${i+1}`);
      if (fields.some(field => !pass.allowed_fields.includes(field) && state[field] !== prior[field])) throw new Error(`${world.slug}: pass ${i+1} changes an unallowed field`);
      if (canonical(state) === canonical(prior)) throw new Error(`${world.slug}: empty pass ${i+1}`);
      prior = state;
    }
    if (seen.size !== fields.length) throw new Error(`${world.slug}: some final fields have no scoped pass`);
    const enumSet = world.field_enums;
    for (const [field, spec] of Object.entries(enumSet)) {
      if (!spec.intermediate.includes(world.initial[field]) || world.passStates.some(state => !spec.intermediate.includes(state[field])) || !spec.final.includes(world.passStates.at(-1)[field]))
        throw new Error(`${world.slug}.${field}: declared enum does not cover all applicable values`);
    }
    const revision = world.justified_revision;
    if (!revision || revision.pass < 2 || world.passStates[revision.pass-1][revision.field] === (revision.pass === 2 ? world.initial[revision.field] : world.passStates[revision.pass-2][revision.field]))
      throw new Error(`${world.slug}: revision claim is not supported by its pass state`);
    const gold = world.passStates.at(-1);
    const options = world.evidence['pass-02-priority-register.md'];
    const audit = world.evidence['pass-03-eligibility-audit.md'];
    const facts = world.source_logic?.candidates;
    if (!options || !audit || !facts || facts.length !== 4 || !audit.includes('Eligibility audit'))
      throw new Error(`${world.slug}: gold lacks source evidence`);
    const fromRegister = [...options.matchAll(/([A-Z]{3}-[0-9]+[A-D]) has priority score ([0-9]+)/g)].map(match=>({itemId:match[1],score:Number(match[2])}));
    if (canonical(fromRegister) !== canonical(facts.map(({itemId,score})=>({itemId,score})))) throw new Error(`${world.slug}: priority facts disagree with authored derivation`);
    for (const fact of facts) {
      if (!fact.eligibilityEvidence || !audit.includes(`${fact.itemId} ${fact.eligibilityEvidence}`)) throw new Error(`${world.slug}: eligibility evidence missing for ${fact.itemId}`);
    }
    const eligible = facts.filter(fact=>fact.eligible).sort((a,b)=>b.score-a.score||a.itemId.localeCompare(b.itemId));
    const winner = eligible[0];
    const derived = {selectedId:winner?.itemId??'none',eligibleIds:eligible.length?eligible.map(f=>f.itemId).join('; '):'none',priorityScore:String(winner?.score??0),decision:winner&&world.source_logic.signatureRecorded?'approve':'hold'};
    if (Object.entries(derived).some(([field,value])=>gold[field]!==value)) throw new Error(`${world.slug}: expected Draft does not follow the authored priority/audit/signature evidence`);
    if (!world.evidence['pass-04-authorization.md'].includes(world.source_logic.signatureRecorded ? 'signed' : 'No authorized reviewer')) throw new Error(`${world.slug}: signature source does not support the expected decision`);
  }
  if (rows.filter((_,i)=>i%2===0).length !== 12 || rows.filter((_,i)=>i%2===1).length !== 12)
    throw new Error('V17 split must be 12 train / 12 test');
}

validateWorlds(worlds);
const rows = worlds.map((world,index)=>{
  const row = makeGuidedSoftIterateCase(world,index,{ revision:REVISION, shapeVersion:'v16' });
  row.id = row.id.replace('authored_semantic_iterate_worlds_v15', 'authored_semantic_iterate_worlds_v17').replace(`:v15-${world.slug}-`, `:v17-${world.slug}-`);
  return row;
});
for (const row of rows) {
  const rootCode = row.curriculum.reference.root[0][1].code;
  if (!rootCode.includes('type InitialDraft =') || !rootCode.includes('initialDraft: InitialDraft') || !rootCode.includes('For enum fields, return one bare listed literal'))
    throw new Error(`${row.source_groups[0]}: scaffold must type initial placeholders separately from final enums`);
  const task = JSON.parse(row.semantics.folder_files['task.json']);
  if (task.output_contract.final_field_enums.decision.includes('pending') || !task.output_contract.intermediate_field_enums.decision.includes('pending'))
    throw new Error(`${row.source_groups[0]}: pending must remain intermediate-only`);
  row.family = 'authored_semantic_iterate_worlds_v17';
  row.family_version = 17;
  row.generation.generator = REVISION;
  row.generation.independent_world = row.source_groups[0];
  row.generation.source_quality = 'New authored operational scenario with four source-scoped updates, exact deterministic priority/eligibility rule, and a recorded authorization condition. Static source derivation only; no model/provider calls.';
  row.curriculum.shape = row.curriculum.shape.replace(/^v16-/, 'v17-');
  row.semantics.shape = row.semantics.shape?.replace?.(/^v16-/, 'v17-') ?? row.semantics.shape;
}
const sourceText = rows.map(canonical).join('\n')+'\n';
const sourceSha = createHash('sha256').update(sourceText).digest('hex');
const fixtureBytes = await readFile(new URL('./semantic-iterate-worlds-v17-data.mjs', import.meta.url));
const fixtureSha = createHash('sha256').update(fixtureBytes).digest('hex');
await mkdir(output,{recursive:false});
await writeFile(resolve(output,'source.cases.jsonl'),sourceText);
const proof = {
  schema:'natlang.semantic-worlds-static-source-proof/1', revision:REVISION, source_cases_sha256:sourceSha,
  authored_data_sha256:fixtureSha, provider_calls:0, teacher_calls:0,
  checks:{unique_cases:rows.length===24,unique_source_groups:new Set(rows.flatMap(row=>row.source_groups)).size===24,train:rows.filter(row=>row.split==='train').length,test:rows.filter(row=>row.split==='test').length,all_passes_scoped:rows.every(row=>row.semantics.folder_files['task.json']&&row.semantics.expected_files['decision.json']),field_enums_valid:true,outputs_derive_from_authored_scores_and_audits:true},
  topology:'Each case has one seed Neuralese<string>, four current-file Neuralese<string> pass children carried in Progress via iterateOn, and one typed literal-union Draft interpreter. This static proof validates authored reference structure, not provider behavior or semantic truth.',
  case_derivations:worlds.map((world,index)=>({group:world.group,domain:world.domain,scenario:world.slug,candidates:world.source_logic.candidates,signature_recorded:world.source_logic.signatureRecorded,expected:rows[index].semantics.expected}))
};
await writeFile(resolve(output,'source-proof.json'),JSON.stringify(proof,null,2)+'\n');
const review={schema:'natlang.neuralese-source-quality-review/1',revision:REVISION,source_cases_sha256:sourceSha,review_type:'fresh authored V17 operational optimization scenarios; deterministic static source derivation; independent quality review required',admission_granted:false,model_calls:0,provider_calls:0,teacher_trajectories:0,counts:{task_variants:24,independent_factual_worlds:24,source_groups:24,train:12,test:12,domains:8,scenarios_per_domain:3,passes:96},worlds:rows.map((row,index)=>({id:row.id,group:row.source_groups[0],split:row.split,domain:worlds[index].domain,scenario:worlds[index].slug,expected:row.semantics.expected}))};
await writeFile(resolve(output,'source-quality-review.json'),JSON.stringify(review,null,2)+'\n');
const manifest={schema:'natlang.neuralese-semantic-iterate-v17-guided/1',revision:REVISION,source_cases:'source.cases.jsonl',source_cases_sha256:sourceSha,authored_data:'ts-host/scripts/inline-curriculum/semantic-iterate-worlds-v17-data.mjs',authored_data_sha256:fixtureSha,world_count:24,task_variant_count:24,train_count:12,test_count:12,domain_count:8,scenarios_per_domain:3,pass_count:96,source_groups_unique:true,source_groups_reused:false,scripted_proof:'source-proof.json',proof_scope:'static authored data and structural validation; no provider execution, no teacher observations, no admission',teacher_observations:0,provider_calls:0,training_admission:false,trace_admission:false};
await writeFile(resolve(output,'source-manifest.json'),JSON.stringify(manifest,null,2)+'\n');
await writeFile(resolve(output,'README.md'),`# V17 guided operational decision source\n\nFresh authored source proposal revision 3 with eight fictional decision domains and three independent factual scenarios per domain. Each case has four scoped evidence files, a priority and eligibility update, a final authorization update, and a deterministic expected Draft. The scaffold uses typed Neuralese<string> notes and literal-union final enum fields. The final decision field is exactly approve or hold; pending is permitted only in initial and intermediate placeholder states.\n\nBuild-time static checks are in source-proof.json. No provider/model call, teacher observation, training admission, or trace admission occurred. Independent semantic review remains required.\n\nSource SHA-256: ${sourceSha}\nAuthored data SHA-256: ${fixtureSha}\n`);
console.log(JSON.stringify(manifest,null,2));
