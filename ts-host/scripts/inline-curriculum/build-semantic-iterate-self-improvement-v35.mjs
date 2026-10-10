#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { deriveWorldResult, sourceDomains, worlds } from './semantic-iterate-self-improvement-v35-data.mjs';
import { makeGuidedSoftIterateCase } from './semantic-iterate-worlds-v15-soft-guided-builder.mjs';

export const REVISION = 'authored-semantic-iterate-self-improvement-v35/4-cardinality-consistent-rules';
const args = process.argv.slice(2);
if (args.length !== 2 || args[0] !== '--out' || !args[1] || args[1].startsWith('--'))
  throw new Error('usage: node build-semantic-iterate-self-improvement-v35.mjs --out FRESH_DIRECTORY');
const output = resolve(args[1]);
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const canonical = value => `${JSON.stringify(value)}\n`;

if (worlds.length !== 32 || sourceDomains.length !== 8) throw new Error('V35 requires 32 worlds in eight domains');
const ids = new Set(), groups = new Set();
const rows = worlds.map((world, index) => {
  const row = makeGuidedSoftIterateCase(world, index, { revision: REVISION, shapeVersion: 'v35' });
  row.id = row.id.replaceAll('authored_semantic_iterate_worlds_v15', 'authored_semantic_iterate_self_improvement_v35')
    .replace(`:v15-${world.slug}-`, `:v35-${world.slug}-`);
  row.family = 'authored_semantic_iterate_self_improvement_v35';
  row.family_version = 35;
  row.curriculum.family = row.family;
  row.curriculum.family_version = 35;
  row.curriculum.shape = `v35-${world.slug}-evidence-ledger-decision`;
  row.curriculum.variant = 'new-authored-semantic-decision-world/1';
  row.semantics.shape = row.curriculum.shape;
  row.source_ids = [world.group];
  row.source_groups = [world.group, world.domainGroup];
  row.split_group = world.domainGroup;
  row.source_revisions = [REVISION];
  row.split = world.split;
  row.generation.generator = REVISION;
  row.generation.independent_world = world.domainGroup;
  row.generation.independent_world_credit = false;
  row.generation.source_domain = world.domain;
  row.generation.source_domain_group = world.domainGroup;
  row.generation.selection_cardinality = world.selectionCardinality;
  row.generation.source_quality = 'One of 32 newly authored fictional requests in eight related task families. Multiple scoped evidence records render all candidate conditions, measured ranking/cardinality and separately scoped authority. Gold is recomputed from authored visible facts; no task-specific answer is embedded in prompt or evidence. No external dataset grounding or independent-world qualification is claimed. Static source derivation only; no teacher/provider calls; quality review pending.';
  const expected = row.semantics.expected;
  const derived = deriveWorldResult(world);
  if (JSON.stringify(derived) !== JSON.stringify(expected)) throw new Error(`${world.slug}: independently derived result differs from authored gold`);
  const task = JSON.parse(row.semantics.folder_files['task.json']);
  const decisionRule = task.output_contract.decision_rule;
  if (task.passes.length !== 4 || task.output_contract.decision_rule.includes('undefined') ||
      !decisionRule.includes('Selection cardinality:') ||
      !decisionRule.includes('Decision mapping:'))
    throw new Error(`${world.slug}: missing four-stage or complete decision rule`);
  const baseRule = decisionRule.split(' Selection cardinality:', 1)[0];
  if (world.domainDefaultCardinality === 2 && world.selectionCardinality === 1 &&
      !/select exactly one qualifying/i.test(baseRule))
    throw new Error(`${world.slug}: one-item request is missing its matching ranking clause`);
  if (world.domainDefaultCardinality === 2 && world.selectionCardinality === 2 &&
      !/select up to two qualifying/i.test(baseRule))
    throw new Error(`${world.slug}: two-item request is missing its matching ranking clause`);
  if (!task.output_contract.fields.measure.includes('digits') || !task.output_contract.fields.measure.includes(world.metricUnit))
    throw new Error(`${world.slug}: metric contract must name exact digit values and unit`);
  const texts = Object.values(world.evidence).join('\n');
  for (const candidate of world.scenarioFacts) {
    if (!texts.includes(candidate.id) || !texts.includes(candidate.facts) || !texts.includes(String(candidate.metric)))
      throw new Error(`${world.slug}/${candidate.id}: evidence does not show the metric and all eligibility facts`);
  }
  if (!texts.includes(world.authorityEvidence)) throw new Error(`${world.slug}: authority evidence not rendered`);
  if (row.semantics.folder_files['task.json'].includes(JSON.stringify(expected)) ||
      JSON.stringify(JSON.parse(row.semantics.folder_files['decision.json'])) === JSON.stringify(expected))
    throw new Error(`${world.slug}: visible task or decision file exposes exact expected output`);
  if (ids.has(row.id) || groups.has(row.source_groups[0])) throw new Error(`${world.slug}: duplicate ID or source group`);
  ids.add(row.id); groups.add(row.source_groups[0]);
  row.generation.proof_derivation = {
    rule_conditions: world.conditionSchema.map(condition => condition.label),
    candidates: world.scenarioFacts.map(candidate => ({ id: candidate.id, metric: candidate.metric, flags: candidate.flags })),
    authority_record: world.authorityEvidence,
    recomputed_expected: derived,
  };
  return row;
});

if (rows.filter(row => row.split === 'train').length !== 16 || rows.filter(row => row.split === 'test').length !== 16)
  throw new Error('V35 requires explicit 16/16 train/test split');
for (const [domainIndex, domain] of sourceDomains.entries()) {
  const subset = rows.filter(row => row.generation.source_domain === domain.label);
  const expectedSplit = domainIndex < 4 ? 'train' : 'test';
  if (subset.length !== 4 || subset.some(row => row.split !== expectedSplit ||
      !row.source_groups.includes(`v35:${domain.slug}:domain`) || row.split_group !== `v35:${domain.slug}:domain`))
    throw new Error(`${domain.slug}: all four requests must inherit one domain-group split (${expectedSplit})`);
}
const bytes = Buffer.from(rows.map(row => JSON.stringify(row)).join('\n') + '\n');
const sourceSha = sha(bytes);
const dataPath = fileURLToPath(new URL('./semantic-iterate-self-improvement-v35-data.mjs', import.meta.url));
const builderPath = fileURLToPath(import.meta.url);
const guidedPath = fileURLToPath(new URL('./semantic-iterate-worlds-v15-soft-guided-builder.mjs', import.meta.url));
const pins = Object.fromEntries(await Promise.all([dataPath, builderPath, guidedPath].map(async path => [path, sha(await readFile(path))])));
await mkdir(output, { recursive: false });
await writeFile(resolve(output, 'source.cases.jsonl'), bytes, { flag: 'wx' });
const derivations = rows.map((row, index) => ({ index, id: row.id, request_id: row.id, split: row.split, case_group: worlds[index].group,
  domain_group: row.generation.source_domain_group, domain: worlds[index].domain,
  candidates: worlds[index].scenarioFacts, authority: worlds[index].authorityEvidence,
  expected_recomputed: deriveWorldResult(worlds[index]), independent_world_credit: false }));
const proof = {
  schema: 'natlang.semantic-self-improvement-source-proof/1', revision: REVISION,
  source_cases_sha256: sourceSha, source_builder_pins: pins,
  counts: { authored_requests: 32, related_domain_families: 8, case_groups: 32, total_lineage_groups: 40, train: 16, test: 16,
    evidence_records: 128, provider_calls: 0, teacher_calls: 0, training_admission: false },
  domains: sourceDomains.map(domain => ({ slug: domain.slug, label: domain.label, source_group: `v35:${domain.slug}:domain`, ids: domain.scenarios.map(item => item.id) })),
  derivations,
  claims: { all_candidate_conditions_visible_in_evidence: true, measured_values_and_units_visible: true,
    authority_separate_from_selection: true, source_and_gold_derived_without_hidden_answer_in_prompt: true,
    shared_four_pass_inline_iterate_scaffold: true, no_teacher_or_provider_calls: true,
    semantic_review_pending: true, externally_grounded_independent_worlds: 0, training_admission: false },
};
await writeFile(resolve(output, 'source-proof.json'), canonical(proof));
await writeFile(resolve(output, 'source-manifest.json'), canonical({ schema: 'natlang.semantic-self-improvement-source/1',
  revision: REVISION, source_cases: 'source.cases.jsonl', source_cases_sha256: sourceSha,
  source_proof: 'source-proof.json', source_builder_pins: pins, counts: proof.counts,
  root_review: 'pending', authored_requests: 32, related_domain_families: 8,
  externally_grounded_independent_worlds: 0, semantic_admission: false, training_admission: false }));
await writeFile(resolve(output, 'root-review-plan.json'), canonical({ schema: 'natlang.source-review-plan/1',
  source_cases_sha256: sourceSha, review_scope: 'All 32 visible requests, evidence records, eligibility conditions, ranking/cardinality, authority scope, domain-group split/source lineage, and independently recomputed golds.',
  admission: false, generation_authorized: false, reviewers: ['root'],
  limitations: ['32 authored requests are organized into eight related fictional task families; they are not 32 independently grounded domains.', 'No external dataset grounding.', 'Static derivation is not teacher performance or admission.'] }));
await writeFile(resolve(output, 'readable-facts-and-golds.md'), `# V35 source review material\n\n` + derivations.map(item =>
  `## ${item.request_id} — ${item.domain}\n\nCase group: ${item.case_group}; domain group: ${item.domain_group}; split: ${item.split}.\n\n` +
  `Candidates:\n${item.candidates.map(candidate => `- ${candidate.id}: ${candidate.metric}; ${candidate.facts}`).join('\n')}\n\n` +
  `Authority evidence: ${item.authority}\n\nRecomputed gold: ${JSON.stringify(item.expected_recomputed)}\n`).join('\n'));
await writeFile(resolve(output, 'README.md'), `# V35 semantic self-improvement source pool — V4\n\n` +
  `This proposal contains 32 newly authored fictional requests organized into eight related task families, with four cases per family. The eight complete domain groups are split as four train groups and four test groups, so no family crosses the split. Cases cover safety corrective-action selection, provenance reconciliation, experimental-plan revision, constrained resource scheduling, compliance document revision, incident response, evidence-backed claim triage, and maintenance work-order sequencing. These are authored fictional families, not 32 independently grounded domains.\n\n` +
  `Each task has four evidence files and uses the shared iterateOn/inline Neuralese scaffold. Decisions require carrying candidate-specific findings forward, applying conditions before ranking, and applying request-scoped authority separately. The CPU proof recomputes the gold from authored facts only. No model/provider calls were made. Root semantic review is pending; no generation or training admission is authorized.\n\nSHA-256: ${sourceSha}\n`);
console.log(JSON.stringify({ revision: REVISION, source_cases_sha256: sourceSha, counts: proof.counts, domains: proof.domains.map(domain => domain.slug) }, null, 2));
