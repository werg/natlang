#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { makeGuidedSoftIterateCase } from './semantic-iterate-worlds-v15-soft-guided-builder.mjs';
import { deriveWorldResult, sourceDomains, worlds } from './semantic-iterate-reducers-v26-data.mjs';

export const REVISION = 'authored-semantic-iterate-reducers-v26/2-authority-scoped-evidence-ledger-worlds';
const args = process.argv.slice(2);
if (args.length !== 2 || args[0] !== '--out' || !args[1] || args[1].startsWith('--'))
  throw new Error('usage: node build-semantic-iterate-reducers-v26.mjs --out FRESH_DIRECTORY');
const output = resolve(args[1]);
if (worlds.length !== 16 || sourceDomains.length !== 8)
  throw new Error('V26 requires 16 new factual worlds across eight domains');

const savedWith = new Set(['EDIT-811', 'ARCH-820', 'RECORD-841', 'POLICY-870']);
const rows = worlds.map((world, index) => {
  const row = makeGuidedSoftIterateCase(world, index, {
    revision: REVISION, shapeVersion: 'v16', savedWith: savedWith.has(world.source_summary.requestId),
  });
  row.id = row.id.replaceAll('authored_semantic_iterate_worlds_v15', 'authored_semantic_iterate_reducers_v26')
    .replace(`:v15-${world.slug}-`, `:v26-${world.slug}-`);
  row.family = 'authored_semantic_iterate_reducers_v26';
  row.family_version = 26;
  row.curriculum.family = row.family;
  row.curriculum.family_version = 26;
  row.curriculum.shape = `v26-${world.slug}-four-pass-derived-decision`;
  row.curriculum.variant = 'new-authored-domain-world/1';
  row.semantics.shape = row.curriculum.shape;
  row.source_ids = [world.group];
  row.source_groups = [world.group];
  row.split_group = world.group;
  row.source_revisions = [REVISION];
  row.split = world.split;
  row.generation.generator = REVISION;
  row.generation.independent_world = world.group;
  row.generation.independent_world_credit = true;
  row.generation.source_quality = 'New authored fictional factual world with exhaustively rendered candidate conditions, readable evidence, explicit ranking and request-scoped authority metadata, and a gold directly recomputed from authored facts. Static source derivation only; no teacher/provider calls.';
  row.generation.source_domain = world.domainIndex;
  row.generation.source_domain_group = world.domainGroup;
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
  if (!task.output_contract.fields.measure.includes('digits'))
    throw new Error(`${world.slug}: numeric measure field lacks explicit digit formatting`);
  if (!task.output_contract.fields.measure.includes(world.metricUnit))
    throw new Error(`${world.slug}: measure contract omits the exact metric unit`);
  if (!task.output_contract.carry_forward.includes('complete running evidence record in readable prose') ||
      !task.output_contract.carry_forward.includes('not a Draft-shaped JSON summary'))
    throw new Error(`${world.slug}: note contract does not preserve the evidence record for final interpretation`);
  const allSourceText = Object.values(world.evidence).join('\n');
  if (!allSourceText.includes(world.source_summary.requestId))
    throw new Error(`${world.slug}: request ID is missing from authored evidence`);
  if (savedWith.has(world.source_summary.requestId) &&
      !row.curriculum.reference.root[0][1].code.includes('stepTemplate.with('))
    throw new Error(`${world.slug}: saved tag .with form is missing from generated scaffold`);
  if (!/folder\.snapshot\(\)\.file\(current\.(?:pass\.)?evidence_path\)/.test(row.curriculum.reference.root[0][1].code))
    throw new Error(`${world.slug}: scaffold does not select only the current pass evidence at runtime`);
  if (JSON.stringify({ fields: task.output_contract.fields, rule: task.output_contract.decision_rule,
      evidence: world.evidence, expected }).match(/\bundefined\b|\bnull\b/))
    throw new Error(`${world.slug}: generated source has an unresolved contract or fact placeholder`);
  for (let pass = 0; pass < task.passes.length; pass++) {
    const note = row.curriculum.reference.children[pass + 1]?.soft_output?.text;
    if (typeof note !== 'string') throw new Error(`${world.slug}: missing note for evidence pass ${pass + 1}`);
    for (let seen = 0; seen <= pass; seen++) {
      const path = task.passes[seen].evidence_path;
      if (!note.includes(path) || !note.includes(world.evidence[path]))
        throw new Error(`${world.slug}: cumulative note omits observed evidence ${path}`);
    }
    for (let future = pass + 1; future < task.passes.length; future++) {
      const path = task.passes[future].evidence_path;
      if (note.includes(path) || note.includes(world.evidence[path]))
        throw new Error(`${world.slug}: pass ${pass + 1} note contains future source ${path}`);
    }
  }
  return row;
});

if (new Set(rows.map(row => row.source_groups[0])).size !== 16 ||
    rows.filter(row => row.split === 'train').length !== 8 || rows.filter(row => row.split === 'test').length !== 8)
  throw new Error('V26 requires 16 distinct factual groups and a balanced 8/8 train/test split');
for (const domain of sourceDomains) {
  const domainRows = rows.filter(row => row.generation.source_domain === domain.domain_index);
  if (domainRows.length !== 2 || domainRows.some(row => row.split !== domain.split))
    throw new Error(`${domain.slug}: both factual worlds must remain in one explicit split`);
}
if (!rows.some(row => row.semantics.expected.selectedItems === 'none'))
  throw new Error('V26 must include a source-derived empty selection');
if (!rows.some(row => row.semantics.expected.selectedItems.includes('; ')))
  throw new Error('V26 must include a source-derived multi-item selection');

const canonical = value => `${JSON.stringify(value)}\n`;
const sourceText = rows.map(row => JSON.stringify(row)).join('\n') + '\n';
const sourceSha = createHash('sha256').update(sourceText).digest('hex');
const dataFiles = ['semantic-iterate-reducers-v26-data.mjs', 'semantic-iterate-reducer-selection-contract.mjs',
  'semantic-iterate-reducers-v18-novel-data.mjs'];
const dataHashes = Object.fromEntries(await Promise.all(dataFiles.map(async file =>
  [file, createHash('sha256').update(await readFile(new URL(`./${file}`, import.meta.url))).digest('hex')])));
const builderFiles = ['build-semantic-iterate-reducers-v26.mjs', 'semantic-iterate-worlds-v15-soft-guided-builder.mjs'];
const builderHashes = Object.fromEntries(await Promise.all(builderFiles.map(async file =>
  [file, createHash('sha256').update(await readFile(new URL(`./${file}`, import.meta.url))).digest('hex')])));
await mkdir(output, { recursive: false });
await writeFile(resolve(output, 'source.cases.jsonl'), sourceText);
const facts = rows.map((row, index) => ({
  id: row.id, request_id: worlds[index].source_summary.requestId, group: row.source_groups[0],
  domain_group: worlds[index].domainGroup, split: row.split, domain: worlds[index].domain,
  candidate_facts: worlds[index].scenarioFacts, conditions: worlds[index].conditionSchema,
  authority: worlds[index].scenarioAuthority,
  authority_evidence: worlds[index].authorityEvidence,
  authority_requirement: worlds[index].authorityRequirement, authority_scope: 'request',
  expected: deriveWorldResult(worlds[index]),
}));
const proof = {
  schema: 'natlang.semantic-reducer-source-proof/4', revision: REVISION,
  source_cases_sha256: sourceSha, authored_data_hashes: dataHashes, builder_hashes: builderHashes,
  counts: { independently_authored_factual_worlds: 16, domains: 8, source_groups: 16, train: 8, test: 8,
    evidence_records: 64, saved_tag_examples: savedWith.size, empty_selection_cases: rows.filter(row => row.semantics.expected.selectedItems === 'none').length,
    multi_item_cases: rows.filter(row => row.semantics.expected.selectedItems.includes('; ')).length, provider_calls: 0, teacher_calls: 0 },
  domain_groups: sourceDomains, case_derivations: facts,
  claims: { gold_recomputed_from_authored_candidate_facts: true,
    every_candidate_condition_assertion_is_rendered_and_checked: true,
    numeric_register_values_include_exact_digits_and_units: true,
    evidence_is_readable_prose_with_explicit_fact_registers: true,
    all_four_runtime_selected_stages_are_used_and_notes_exclude_future_files: true,
    empty_eligibility_and_multi_item_branches_covered: true,
    authority_is_independent_of_selection_and_declares_named_request_scope_in_each_case: true,
    saved_nl_with_capture_surface_exercised: savedWith.size > 0 },
  lineage: { all_16_groups_are_new_authored_worlds: true, inherited_worlds: 0, derived_task_variants: 0,
    grouping: 'Each request has a unique factual source group. Each pair shares only its explicit domain group and split.' },
  provider_calls: 0, teacher_calls: 0, semantic_admission: false, training_admission: false,
  scope: 'Static source structure and directly computed authored-fact proof only; no teacher/provider execution or admission.',
};
await writeFile(resolve(output, 'source-proof.json'), canonical(proof));
const manifest = { schema: 'natlang.neuralese-semantic-iterate-reducers-v26/1', revision: REVISION,
  source_cases: 'source.cases.jsonl', source_cases_sha256: sourceSha,
  authored_data: 'ts-host/scripts/inline-curriculum/semantic-iterate-reducers-v26-data.mjs',
  authored_data_hashes: dataHashes, builder_hashes: builderHashes, source_proof: 'source-proof.json',
  counts: proof.counts, independently_authored_factual_worlds: 16, independent_source_groups: 16,
  domains: 8, teacher_observations: 0, provider_calls: 0, semantic_admission: false, training_admission: false };
await writeFile(resolve(output, 'source-manifest.json'), canonical(manifest));
await writeFile(resolve(output, 'source-quality-review.json'), canonical({
  schema: 'natlang.neuralese-source-quality-review/1', revision: REVISION,
  source_cases_sha256: sourceSha, review_type: 'new authored factual-domain pool; root semantic review pending',
  admission_granted: false, worlds: facts.map(({ id, request_id, group, domain_group, split, domain, expected }) =>
    ({ id, request_id, group, domain_group, split, domain, expected })),
}));
const readable = facts.map(fact => `## ${fact.request_id} — ${fact.domain}\n\n` +
  `Group: ${fact.group} (domain group ${fact.domain_group}); split: ${fact.split}.\n\n` +
  `${fact.candidate_facts.map(candidate => `${candidate.id}: ${candidate.metric} ${worlds.find(world => world.source_summary.requestId === fact.request_id).metricUnit}; ${candidate.facts}`).join('\n')}\n\n` +
  `Authority rule: ${fact.authority_requirement} scoped to ${fact.authority_scope} ${fact.request_id}.\n` +
  `Authority evidence: ${fact.authority_evidence}\n\nExpected: ${JSON.stringify(fact.expected)}\n`).join('\n');
await writeFile(resolve(output, 'readable-facts-and-golds.md'), `# V26 CPU-authored facts and golds\n\n${readable}`);
await writeFile(resolve(output, 'README.md'), `# V26 authored factual decision pool\n\nSixteen newly authored fictional requests span eight decision domains: editorial corrections, archive provenance, grant dossier review, public-record redaction, translation quality routing, museum accession cataloging, policy citation repair, and procurement specification clarification. Every request has its own factual source group; each domain pair shares only its domain group and split. The pool has sixteen unique worlds, eight train and eight test cases, with no inherited task variants.\n\nEach case has four readable evidence records, exhaustive candidate conditions, an explicit ranking/cardinality rule, and request-scoped named authority kept separate from eligibility. Numeric register fields state exact values and units; typed measure fields require digits. The pool covers empty and multiple-item outcomes, including four saved-tag rebound examples. The shared scaffold opens each pass's evidence path at runtime and passes only that read-only snapshot handle and prior notes to the child. Note children retain an accumulated prose evidence record for final interpretation.\n\n` +
  `The CPU source proof recomputes each result from authored facts and verifies every condition against its evidence statement. It makes no provider calls and grants no semantic or training admission.\n\nSource SHA-256: ${sourceSha}\n`);
console.log(JSON.stringify({ revision: REVISION, source_sha256: sourceSha, data_hashes: dataHashes,
  counts: proof.counts, domains: sourceDomains.map(domain => domain.slug) }, null, 2));
