#!/usr/bin/env node
import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { worlds } from './semantic-iterate-worlds-v15-enum-data.mjs';
import { writeIterateCandidate } from './authored-iterate-source-builder.mjs';

const args = process.argv.slice(2);
if (args.length !== 2 || args[0] !== '--out' || !args[1] || args[1].startsWith('--'))
  throw new Error('usage: node build-semantic-iterate-worlds-v15-enums.mjs --out FRESH_DIRECTORY');
const revision = 'authored-semantic-iterate-worlds-v15/10-enum-contracts-canonical-boundaries';
const out = resolve(args[1]);
const qualityReview = worlds.map(world => ({
  slug: world.slug, group: world.group, domain: world.domain, evidence: world.evidence,
  ordered_passes: world.passes.map((pass, index) => ({ pass: index + 1, name: pass.name, evidence_path: pass.evidence_path, allowed_fields: pass.allowed_fields, constraint: pass.constraint, resulting_draft: world.passStates[index] })),
  field_enums: world.field_enums,
  justified_revision: world.justified_revision,
  calculation_or_decision_derivation: world.derivation,
  final_expected: world.passStates.at(-1),
}));
const built = await writeIterateCandidate({ out, worlds, revision, qualityReview });
const lineage = {
  schema: 'natlang.authored-semantic-world-lineage/1', source_revision: revision,
  source_sha256: built.sourceSha,
  parent_sources: [
    { path: 'runs/neuralese-semantic-iterate-v15-source-candidate-20261007-v8/source.cases.jsonl', sha256: '13a1da835ac738f1beefecd120a94c2731109e0eca1fc511a4c1a11eb2445b36' },
    { path: 'runs/neuralese-semantic-iterate-v15-soft-state-20261007-v10/source.cases.jsonl', sha256: '4a8f95f6ba6703437651c6c827b4042d89f536ab59cb2b7cc9496ed8199b7084' },
  ],
  world_count: worlds.length, task_variant_count: built.rows.length,
  source_groups: built.rows.map(row => row.source_groups[0]),
  splits: Object.fromEntries(built.rows.map(row => [row.id, row.split])),
  factual_delta: 'Retains all 12 V15 source groups, splits, evidence facts, and final decision labels; adds explicit intermediate and final enum metadata from the authored field contracts. The course addendum boundary and vaccine decision-only write boundary remain as established in V8/V10.',
  schema_delta: 'Categorical finite fields now have explicit intermediate value sets that include placeholders and separate final value sets. Free-form identifiers, measurements, and calculations remain strings.',
  preservation: 'Fresh source variant; prior candidates and artifacts are unchanged. No provider calls, teacher observations, trace admission, or training admission.',
};
await writeFile(resolve(out, 'lineage.json'), JSON.stringify(lineage, null, 2) + '\n');
await writeFile(resolve(out, 'README.md'), `# Authored V15 iterative worlds with enum contracts\n\nFresh source-only variant of the 12 V15 worlds. It preserves the source groups, splits, evidence facts, and gold drafts while declaring intermediate and final categorical value sets from the authored field contracts. Intermediate sets explicitly include placeholders such as pending and unknown; final sets state the terminal contract.\n\nNo provider calls or teacher observations. Training and trace admission are false.\n\nSource SHA-256: ${built.sourceSha}.\n`);
console.log(JSON.stringify({ out, source_sha256: built.sourceSha, worlds: worlds.length, task_variants: built.rows.length, train: 6, test: 6, admission_granted: false }, null, 2));
