#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { makeGuidedSoftIterateCase } from './semantic-iterate-worlds-v15-soft-guided-builder.mjs';
import { worlds, sourceDomains } from './semantic-iterate-reducers-v22-novel-data.mjs';

export const REVISION = 'authored-semantic-iterate-reducers-v22/5-single-winner-readonly-evidence-snapshot';
const args = process.argv.slice(2);
if (args.length !== 2 || args[0] !== '--out' || !args[1] || args[1].startsWith('--'))
  throw new Error('usage: node build-semantic-iterate-reducers-v22.mjs --out FRESH_DIRECTORY');
const output = resolve(args[1]);
if (worlds.length !== 16 || sourceDomains.length !== 8) throw new Error('V22 requires 16 cases across eight new operational domains');

const rows = worlds.map((world, index) => {
  const row = makeGuidedSoftIterateCase(world, index, { revision: REVISION, shapeVersion: 'v16' });
  row.id = row.id.replaceAll('authored_semantic_iterate_worlds_v15', 'authored_semantic_iterate_reducers_v22')
    .replaceAll(`:v15-${world.slug}-`, `:v22-${world.slug}-`);
  row.family = 'authored_semantic_iterate_reducers_v22'; row.family_version = 22;
  row.curriculum.family = row.family; row.curriculum.family_version = 22;
  row.curriculum.shape = `v22-${world.slug}-four-pass-evidence-reducer`;
  row.curriculum.variant = 'new-domain-fact-derived-operational-decision/1';
  row.semantics.shape = row.curriculum.shape;
  row.source_ids = [world.group]; row.source_groups = [world.group]; row.split_group = world.group;
  row.source_revisions = [REVISION]; row.split = world.source_summary.domain_index % 2 === 0 ? 'train' : 'test';
  row.generation.generator = REVISION;
  row.generation.independent_world = world.group;
  row.generation.independent_world_credit = world.source_summary.scenario === 'A';
  row.generation.source_quality = 'New self-contained fictional operational fact world with a separately authored domain rule, four scoped evidence records, and deterministic fact-derived expected output.';
  row.generation.source_domain = world.source_summary.domain_index;
  row.generation.capture_contract.saved_with_examples = 'Current pass notes preserve the complete observed file and decision-relevant facts; final Draft derives from accumulated evidence and exact field formats.';
  row.curriculum.source_group_lineage = { source_revision: REVISION, factual_group: world.group };

  const task = JSON.parse(row.semantics.folder_files['task.json']);
  const sourceProgram = row.semantics.files['reconcile_scoped_evidence.nl'];
  if (!sourceProgram.includes('folder.snapshot().file(current.evidence_path)') ||
      sourceProgram.includes('folder.file(current.evidence_path)') ||
      !sourceProgram.includes('folder.file(task.output_path).writeText'))
    throw new Error(`${world.slug}: evidence must be passed from an immutable FolderSnapshot while only the declared output remains writable`);
  if (!task.instruction.includes('Preserve their complete contents exactly') ||
      !task.instruction.includes('write only decision.json'))
    throw new Error(`${world.slug}: read-only evidence/output boundary missing from task instruction`);
  if (JSON.stringify(task).includes('undefined')) throw new Error(`${world.slug}: undefined task contract field`);
  if (!task.output_contract.fields || !task.output_contract.decision_rule.includes('Decision mapping:'))
    throw new Error(`${world.slug}: missing field formats or full decision mapping`);
  if (!task.instruction.includes(world.source_summary.requestId)) throw new Error(`${world.slug}: request identity omitted`);
  if (row.semantics.expected.caseId !== world.source_summary.requestId)
    throw new Error(`${world.slug}: expected request identifier mismatch`);
  if (new Set(row.semantics.expected_files['decision.json'] && Object.keys(row.semantics.expected)).size !== 4)
    throw new Error(`${world.slug}: output must contain exactly four declared fields`);
  const eligible = world.source_summary.candidates.filter(world.eligible).sort(world.compare);
  const selected = world.select(eligible);
  const authorityRecord = world.evidence['pass-04-authority.md'];
  const authorityStates = [...authorityRecord.matchAll(/AUTHORITY STATUS: (approved|not approved)\./g)].map(match => match[1]);
  if (authorityStates.length !== 1)
    throw new Error(`${world.slug}: authority evidence must contain one explicit status: ${authorityRecord}`);
  const evidenceAuthorized = authorityStates[0] === 'approved';
  if (evidenceAuthorized !== world.authorized)
    throw new Error(`${world.slug}: declared authority does not match rendered authority record`);
  const independentlyDerived = {
    caseId: world.source_summary.requestId,
    selectedItems: world.format(selected),
    measure: world.measure(selected),
    decision: selected.length === 0 ? world.noAction : (evidenceAuthorized ? world.approvedAction : 'hold'),
  };
  if (JSON.stringify(independentlyDerived) !== JSON.stringify(row.semantics.expected))
    throw new Error(`${world.slug}: gold does not match direct eligibility, ranking, measurement, and authority derivation`);
  for (const pass of task.passes) {
    const content = world.evidence[pass.evidence_path];
    if (!content || !row.curriculum.reference.children.some(child => child.match === pass.evidence_path && child.soft_output.text.includes(content)))
      throw new Error(`${world.slug}: cumulative reference does not retain complete pass evidence ${pass.evidence_path}`);
  }
  const contractText = JSON.stringify({ fields: task.output_contract.fields,
    decision_rule: task.output_contract.decision_rule, authorityRecord,
    initialDraft: task.initialDraft, expected: row.semantics.expected });
  if (/\bundefined\b|\bnull\b/.test(contractText))
    throw new Error(`${world.slug}: unresolved placeholder in the authored output contract, authority record, or expected result`);
  const finalEnumText = JSON.stringify(task.output_contract.final_field_enums ?? {});
  if (finalEnumText.includes('selectedItems') || finalEnumText.includes('measure'))
    throw new Error(`${world.slug}: data-dependent values must not appear as finite enums`);
  return row;
});
if (new Set(rows.map(row => row.source_groups[0])).size !== 8 ||
    rows.filter(row => row.split === 'train').length !== 8 || rows.filter(row => row.split === 'test').length !== 8)
  throw new Error('V22 must have eight factual groups and an 8/8 train/test split');
for (const domain of sourceDomains) {
  const group = rows.filter(row => row.source_groups[0].startsWith(`v22:${domain.slug}:`));
  if (group.length !== 2 || new Set(group.map(row => row.split)).size !== 1)
    throw new Error(`${domain.slug}: expected two fact scenarios in one inherited-safe split group`);
  const cases = worlds.filter(world => world.source_summary.domain_index === sourceDomains.indexOf(domain));
  if (cases[0].source_summary.candidates.map(x => `${x.id}:${x.metric}`).join('|') ===
      cases[1].source_summary.candidates.map(x => `${x.id}:${x.metric}`).join('|'))
    throw new Error(`${domain.slug}: two scenarios must have distinct candidate facts`);
}

const sourceText = rows.map(row => JSON.stringify(row)).join('\n') + '\n';
const sourceSha = createHash('sha256').update(sourceText).digest('hex');
const dataSha = createHash('sha256').update(await readFile(new URL('./semantic-iterate-reducers-v22-novel-data.mjs', import.meta.url))).digest('hex');
const builderFiles = ['build-semantic-iterate-reducers-v22.mjs', 'semantic-iterate-worlds-v15-soft-guided-builder.mjs'];
const builderHashes = Object.fromEntries(await Promise.all(builderFiles.map(async name =>
  [name, createHash('sha256').update(await readFile(new URL(`./${name}`, import.meta.url))).digest('hex')])));
await mkdir(output, { recursive: false });
await writeFile(resolve(output, 'source.cases.jsonl'), sourceText);
const counts = { task_variants: rows.length, new_factual_source_groups: new Set(rows.map(row => row.source_groups[0])).size,
  independent_factual_worlds: 8,
  domains: sourceDomains.length, train: rows.filter(row => row.split === 'train').length,
  test: rows.filter(row => row.split === 'test').length, passes: rows.length * 4,
  variants_per_domain: 2, providers: 0, teacher_calls: 0 };
const proof = {
  schema: 'natlang.semantic-reducer-source-proof/3', revision: REVISION,
  source_cases_sha256: sourceSha, authored_data_sha256: dataSha, counts,
  builder_hashes: builderHashes,
  provider_calls: 0, teacher_calls: 0, training_admission: false, trace_admission: false,
  domains: sourceDomains,
  case_derivations: rows.map((row, index) => ({ id: row.id, group: row.source_groups[0], split: row.split,
    domain: sourceDomains[row.generation.source_domain].domain,
    decision_rule: JSON.parse(row.semantics.folder_files['task.json']).output_contract.decision_rule,
    candidates: worlds[index].source_summary.candidates,
    evidence: worlds[index].evidence,
    expected: row.semantics.expected })),
  claims: { unique_factual_groups: true, new_operational_domain_classes: 8,
    all_four_evidence_files_are_complete_and_cumulative_in_reference: true,
    gold_fields_derive_from_authored_candidate_facts_and_authority: true,
    data_dependent_selection_fields_are_plain_strings: true,
  source_family_groups_are_not_renamed_v18_groups: true,
  each_independent_group_has_two_counterfactual_task_variants_in_one_split: true },
  scope: 'Authored source structure and deterministic data-derived expected values only. No teacher/provider execution, semantic admission, or training admission.'
};
await writeFile(resolve(output, 'source-proof.json'), JSON.stringify(proof, null, 2) + '\n');
const manifest = { schema:'natlang.neuralese-semantic-iterate-reducers-v22/1', revision:REVISION,
  source_cases:'source.cases.jsonl', source_cases_sha256:sourceSha,
  authored_data:'ts-host/scripts/inline-curriculum/semantic-iterate-reducers-v22-novel-data.mjs',
  authored_data_sha256:dataSha, counts, source_proof:'source-proof.json',
  builder_hashes:builderHashes,
  proof_scope:'Authored fact derivation and structural checks; no teacher observations or admission.',
  independent_factual_worlds:8, teacher_observations:0, provider_calls:0, semantic_admission:false, training_admission:false };
await writeFile(resolve(output, 'source-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
await writeFile(resolve(output, 'README.md'), `# V22 new operational semantic reducer domains\n\nSixteen authored task variants cover eight new operational decision classes: accessible procurement, vendor API migration, research cohort classification, municipal maintenance, claim substantiation, community service eligibility, records retention, and incident mitigation. Each class has one factual source group with two counterfactual task variants, kept together in the same train or test split. The pool claims eight independent factual worlds, not sixteen. The four-pass typed evidence scaffold is shared; candidate metrics, qualifications, exceptions, and authority are staged in current-pass files.\n\nThis pool is not a set of relabeled V18 cases and has no inherited-group relationship. Static proof checks source structure and expected-value derivation only. A separate frozen-runtime proof and root review are required before any teacher launch. No teacher calls, semantic admission, or training admission occurred.\n\nSource SHA-256: ${sourceSha}\nAuthored data SHA-256: ${dataSha}\n`);
console.log(JSON.stringify({revision:REVISION,source_sha256:sourceSha,authored_data_sha256:dataSha,counts},null,2));
