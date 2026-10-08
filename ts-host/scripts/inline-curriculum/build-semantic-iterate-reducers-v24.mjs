#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { makeGuidedSoftIterateCase } from './semantic-iterate-worlds-v15-soft-guided-builder.mjs';
import { deriveWorldResult, sourceDomains, worlds } from './semantic-iterate-reducers-v24-data.mjs';

export const REVISION = 'authored-semantic-iterate-reducers-v24/1-additional-factual-domains';
const args = process.argv.slice(2);
if (args.length !== 2 || args[0] !== '--out' || !args[1] || args[1].startsWith('--'))
  throw new Error('usage: node build-semantic-iterate-reducers-v24.mjs --out FRESH_DIRECTORY');
const output = resolve(args[1]);
if (worlds.length !== 16 || sourceDomains.length !== 8)
  throw new Error('V24 requires 16 authored factual worlds across eight additional decision domains');

const savedWith = new Set(['TRN-520', 'ARC-531', 'GRT-550', 'BLD-581']);
const rows = worlds.map((world, index) => {
  const row = makeGuidedSoftIterateCase(world, index, {
    revision: REVISION, shapeVersion: 'v16', savedWith: savedWith.has(world.source_summary.requestId),
  });
  row.id = row.id.replaceAll('authored_semantic_iterate_worlds_v15', 'authored_semantic_iterate_reducers_v24')
    .replace(`:v15-${world.slug}-`, `:v24-${world.slug}-`);
  row.family = 'authored_semantic_iterate_reducers_v24';
  row.family_version = 24;
  row.curriculum.family = row.family;
  row.curriculum.family_version = 24;
  row.curriculum.shape = `v24-${world.slug}-four-pass-evidence-reducer`;
  row.curriculum.variant = 'additional-authored-factual-domain/1';
  row.semantics.shape = row.curriculum.shape;
  row.source_ids = [world.group];
  row.source_groups = [world.group];
  row.split_group = world.group;
  row.source_revisions = [REVISION];
  row.split = world.split;
  row.generation.generator = REVISION;
  row.generation.independent_world = world.group;
  row.generation.independent_world_credit = true;
  row.generation.source_quality = 'New authored fictional factual world with readable evidence, explicit eligibility and ranking rules, an authority branch, and a result mechanically derived from staged source facts. Static source derivation only; no teacher/provider calls.';
  row.generation.source_domain = world.domainIndex;
  row.generation.selection_cardinality = world.selectionCardinality;
  row.generation.capture_contract.saved_with_examples = savedWith.has(world.source_summary.requestId)
    ? 'The current pass child uses a saved typed nl<T> tag rebound through .with(captures); only the current snapshot FileHandle and prior notes enter the child.'
    : 'The current pass child uses the direct nl.with(captures) form with only the current snapshot FileHandle and prior notes.';

  const expected = row.semantics.expected;
  const derived = deriveWorldResult(world);
  if (JSON.stringify(derived) !== JSON.stringify(expected))
    throw new Error(`${world.slug}: independent result derivation differs from emitted gold`);
  const task = JSON.parse(row.semantics.folder_files['task.json']);
  if (task.passes.length !== 4 || task.output_contract.final_field_enums?.selectedItems ||
      task.output_contract.final_field_enums?.measure)
    throw new Error(`${world.slug}: expected four evidence passes and plain ranked-result fields`);
  if (!task.output_contract.decision_rule.includes('Selection cardinality:') ||
      !task.output_contract.decision_rule.includes('Decision mapping:'))
    throw new Error(`${world.slug}: task lacks explicit selection and authority branches`);
  const allSourceText = Object.values(world.evidence).join('\n');
  if (!allSourceText.includes(world.source_summary.requestId))
    throw new Error(`${world.slug}: request ID is missing from authored evidence`);
  if (savedWith.has(world.source_summary.requestId) &&
      !row.curriculum.reference.root[0][1].code.includes('stepTemplate.with('))
    throw new Error(`${world.slug}: saved tag .with form is missing from generated scaffold`);
  if (JSON.stringify({ fields:task.output_contract.fields, rule:task.output_contract.decision_rule,
      evidence:world.evidence, expected }).match(/\bundefined\b|\bnull\b/))
    throw new Error(`${world.slug}: generated source has an unresolved contract or fact placeholder`);
  for (let pass = 0; pass < task.passes.length; pass++) {
    const note = row.curriculum.reference.children[pass + 1]?.soft_output?.text;
    if (typeof note !== 'string') throw new Error(`${world.slug}: missing note for evidence pass ${pass + 1}`);
    for (let seen = 0; seen <= pass; seen++) {
      const source = task.passes[seen].evidence_path;
      if (!note.includes(source) || !note.includes(world.evidence[source]))
        throw new Error(`${world.slug}: cumulative note omits observed evidence ${source}`);
    }
    for (let future = pass + 1; future < task.passes.length; future++) {
      const source = task.passes[future].evidence_path;
      if (note.includes(source) || note.includes(world.evidence[source]))
        throw new Error(`${world.slug}: pass ${pass + 1} note contains future source ${source}`);
    }
  }
  return row;
});

if (new Set(rows.map(r => r.source_groups[0])).size !== 16 ||
    rows.filter(r => r.split === 'train').length !== 8 || rows.filter(r => r.split === 'test').length !== 8)
  throw new Error('V24 must have 16 distinct authored groups and a balanced 8/8 domain split');
for (const domain of sourceDomains) {
  const sameDomain = rows.filter(r => r.generation.source_domain === sourceDomains.indexOf(domain));
  if (sameDomain.length !== 2 || sameDomain.some(r => r.split !== domain.split))
    throw new Error(`${domain.slug}: two same-domain worlds must remain in one split`);
}

const canonical = value => `${JSON.stringify(value)}\n`;
const sourceText = rows.map(row => JSON.stringify(row)).join('\n') + '\n';
const sourceSha = createHash('sha256').update(sourceText).digest('hex');
const dataPath = new URL('./semantic-iterate-reducers-v24-data.mjs', import.meta.url);
const dataSha = createHash('sha256').update(await readFile(dataPath)).digest('hex');
const builderFiles = ['build-semantic-iterate-reducers-v24.mjs', 'semantic-iterate-worlds-v15-soft-guided-builder.mjs'];
const builderHashes = Object.fromEntries(await Promise.all(builderFiles.map(async file =>
  [file, createHash('sha256').update(await readFile(new URL(`./${file}`, import.meta.url))).digest('hex')])));
await mkdir(output, { recursive: false });
await writeFile(resolve(output, 'source.cases.jsonl'), sourceText);
const facts = rows.map((row, index) => ({
  id: row.id, request_id: worlds[index].source_summary.requestId,
  group: row.source_groups[0], split: row.split, domain: worlds[index].domain,
  source_facts: worlds[index].scenarioFacts,
  authority: worlds[index].scenarioAuthority,
  expected: deriveWorldResult(worlds[index]),
}));
const proof = {
  schema: 'natlang.semantic-reducer-source-proof/4', revision: REVISION,
  source_cases_sha256: sourceSha, authored_data_sha256: dataSha, builder_hashes: builderHashes,
  counts: { authored_factual_worlds: 16, domains: 8, source_groups: 16, train: 8, test: 8,
    evidence_records: 64, saved_tag_examples: savedWith.size, provider_calls: 0, teacher_calls: 0 },
  domain_groups: sourceDomains, case_derivations: facts,
  claims: { gold_recomputed_from_authored_candidate_facts: true,
    evidence_is_readable_prose_with_explicit_fact_registers: true,
    all_four_stages_are_used_and_cumulative_notes_exclude_future_files: true,
    empty_eligibility_branch_covered: rows.some(row => row.semantics.expected.selectedItems === 'none'),
    authority_is_independent_of_selection_and_explicit_in_each_case: true,
    saved_nl_with_capture_surface_exercised: savedWith.size > 0 },
  provider_calls: 0, teacher_calls: 0, semantic_admission: false, training_admission: false,
  scope: 'Static source structure and directly computed authored-fact proof only; no teacher/provider execution or admission.',
};
await writeFile(resolve(output, 'source-proof.json'), canonical(proof));
const manifest = { schema: 'natlang.neuralese-semantic-iterate-reducers-v24/1', revision: REVISION,
  source_cases: 'source.cases.jsonl', source_cases_sha256: sourceSha,
  authored_data: 'ts-host/scripts/inline-curriculum/semantic-iterate-reducers-v24-data.mjs',
  authored_data_sha256: dataSha, builder_hashes: builderHashes, source_proof: 'source-proof.json',
  counts: proof.counts, independent_factual_worlds: 16, independent_source_groups: 16,
  domains: 8, teacher_observations: 0, provider_calls: 0, semantic_admission: false, training_admission: false };
await writeFile(resolve(output, 'source-manifest.json'), canonical(manifest));
await writeFile(resolve(output, 'source-quality-review.json'), canonical({
  schema: 'natlang.neuralese-source-quality-review/1', revision: REVISION,
  source_cases_sha256: sourceSha, review_type: 'authored factual-domain pool; root semantic review pending',
  admission_granted: false, worlds: facts.map(({ id, request_id, group, split, domain, expected }) =>
    ({ id, request_id, group, split, domain, expected })),
}));
await writeFile(resolve(output, 'README.md'), `# V24 authored factual decision pool\n\nSixteen individually authored fictional requests cover eight additional decision domains: flood-response pumps, accessible transit detours, archival digitization, rural broadband sites, grant milestone reimbursement, manufacturing lot release, habitat restoration, and permit inspection scheduling. Each pair belongs to one domain and one split; the pool has sixteen distinct source groups, eight train and eight test cases.\n\nEach case has four readable evidence files, an explicit eligibility and ranking rule, an explicit empty-selection branch, and authority kept separate from selection. Cases include one-item and two-item outputs, an empty eligible result, and authorized and held dispositions. Four scaffolds exercise saved typed tags rebound through .with(captures). Reference notes preserve complete observed evidence and do not include future pass files.\n\nThe source proof recomputes the result from authored candidate facts. It makes no provider calls and grants no semantic or training admission.\n\nSource SHA-256: ${sourceSha}\nAuthored-data SHA-256: ${dataSha}\n`);
console.log(JSON.stringify({ revision: REVISION, source_sha256: sourceSha, data_sha256: dataSha,
  counts: proof.counts, domains: sourceDomains.map(d=>d.slug) }, null, 2));
