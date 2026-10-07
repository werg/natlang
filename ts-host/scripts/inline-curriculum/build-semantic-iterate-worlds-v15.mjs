#!/usr/bin/env node
import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { worlds } from './semantic-iterate-worlds-v15-data.mjs';
import { writeIterateCandidate } from './authored-iterate-source-builder.mjs';

const args = process.argv.slice(2);
if (args.length !== 2 || args[0] !== '--out' || !args[1] || args[1].startsWith('--'))
  throw new Error('usage: node build-semantic-iterate-worlds-v15.mjs --out FRESH_DIRECTORY');
const revision = 'authored-semantic-iterate-worlds-v15/6';
const out = resolve(args[1]);
const qualityReview = worlds.map(world => ({
  slug: world.slug,
  group: world.group,
  domain: world.domain,
  evidence: world.evidence,
  ordered_passes: world.passes.map((pass, index) => ({
    pass: index + 1,
    name: pass.name,
    evidence_path: pass.evidence_path,
    allowed_fields: pass.allowed_fields,
    constraint: pass.constraint,
    resulting_draft: world.passStates[index],
  })),
  justified_revision: world.justified_revision,
  calculation_or_decision_derivation: world.derivation,
  final_expected: world.passStates.at(-1),
}));
const built = await writeIterateCandidate({ out, worlds, revision, qualityReview });
const lineage = {
  schema: 'natlang.authored-semantic-world-lineage/1',
  source_revision: revision,
  source_sha256: built.sourceSha,
  parent_source_sha256: '3870eeadf2729cbcc46766f0f6d4c87fc344658603db84429301d20fd2f09ce0',
  world_count: worlds.length,
  task_variant_count: built.rows.length,
  source_groups: built.rows.map(row => row.source_groups[0]),
  splits: Object.fromEntries(built.rows.map(row => [row.id, row.split])),
  factual_delta: 'Retains the 12 V15 factual groups, splits, source facts, evidence text, pass states, and final expected decisions from parent source candidate v6.',
  schema_delta: 'Clarifies protected-species decision logic as a conjunction of listing, same-site/date closure, and absence of a matching signed exception. This avoids denying listed species when no seasonal closure applies. The current world satisfies all three conditions; no pass values or labels changed.',
  label_contract: 'Pass outputs are progressive drafts derived only from the current evidence and carried prior draft. Each world declares one evidence-grounded intermediate revision. Exact outputs no longer depend on matching unconstrained explanatory prose.',
  preservation: 'No existing source or campaign artifact changed. This candidate contains no provider calls, no teacher observations, no trace admission, and no training admission.',
};
await writeFile(resolve(out, 'lineage.json'), JSON.stringify(lineage, null, 2) + '\n');
await writeFile(resolve(out, 'README.md'), `# Authored V15 iterative semantic worlds\n\n` +
  `This source-only candidate contains 12 factual worlds, each represented by one source group and four separately scoped evidence files. Every world has four ordered iterateOn passes, a carried complete draft, and a documented evidence-based field revision. Several final decisions require calculation or conjunction evaluation; signed records supply facts or authority, not a copied answer label.\n\n` +
  `The scripted reference proof is a deterministic CPU reference, not a teacher trajectory. This v7 candidate preserves v6 world groups, splits, source facts, pass states, and final expected decisions. It clarifies the protected-species rule's listing, closure, and exception conjunction. Independent source review is pending. Training admission and trace admission are false.\n\n` +
  `Source SHA-256: ${built.sourceSha}.\n`);
console.log(JSON.stringify({ out, source_sha256: built.sourceSha, worlds: worlds.length, task_variants: built.rows.length, groups: new Set(built.rows.map(row => row.source_groups[0])).size, train: built.rows.filter(row => row.split === 'train').length, test: built.rows.filter(row => row.split === 'test').length, passes: worlds.reduce((n, world) => n + world.passes.length, 0), admission_granted: false }, null, 2));
