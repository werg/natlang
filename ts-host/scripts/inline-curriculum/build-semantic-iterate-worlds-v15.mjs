#!/usr/bin/env node
import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { worlds } from './semantic-iterate-worlds-v15-data.mjs';
import { writeIterateCandidate } from './authored-iterate-source-builder.mjs';

const args = process.argv.slice(2);
if (args.length !== 2 || args[0] !== '--out' || !args[1] || args[1].startsWith('--'))
  throw new Error('usage: node build-semantic-iterate-worlds-v15.mjs --out FRESH_DIRECTORY');
const revision = 'authored-semantic-iterate-worlds-v15/7';
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
  parent_source_sha256: 'd060012c25e93d88f7df86f579eece7e1c49027aea9589590bb55a8d3411b7a5',
  world_count: worlds.length,
  task_variant_count: built.rows.length,
  source_groups: built.rows.map(row => row.source_groups[0]),
  splits: Object.fromEntries(built.rows.map(row => [row.id, row.split])),
  factual_delta: 'Retains the 12 V15 factual groups, splits, source facts, evidence text, and final decision labels from parent source candidate v7.',
  schema_delta: 'Carries the grant fundraising line amount from pass 1 as an unclassified source fact, then computes allowable costs in pass 2 after reading the current allowability rule. Specifies an archive channel enum and normalization from “reading-room” to “reading room”. These are task/output-contract clarifications; no factual group or final decision label changed.',
  label_contract: 'Pass outputs are progressive drafts derived only from the current evidence and carried prior draft. Each world declares one evidence-grounded intermediate revision. Exact outputs no longer depend on matching unconstrained explanatory prose.',
  preservation: 'No existing source or campaign artifact changed. This candidate contains no provider calls, no teacher observations, no trace admission, and no training admission.',
};
await writeFile(resolve(out, 'lineage.json'), JSON.stringify(lineage, null, 2) + '\n');
await writeFile(resolve(out, 'README.md'), `# Authored V15 iterative semantic worlds\n\n` +
  `This source-only candidate contains 12 factual worlds, each represented by one source group and four separately scoped evidence files. Every world has four ordered iterateOn passes, a carried complete draft, and a documented evidence-based field revision. Several final decisions require calculation or conjunction evaluation; signed records supply facts or authority, not a copied answer label.\n\n` +
  `The scripted reference proof is a deterministic CPU reference, not a teacher trajectory. This v8 candidate preserves v7 world groups, splits, source facts, and final decision labels. It carries the grant fundraising line amount before the later rule pass and defines canonical archive channel labels. Independent source review is pending. Training admission and trace admission are false.\n\n` +
  `Source SHA-256: ${built.sourceSha}.\n`);
console.log(JSON.stringify({ out, source_sha256: built.sourceSha, worlds: worlds.length, task_variants: built.rows.length, groups: new Set(built.rows.map(row => row.source_groups[0])).size, train: built.rows.filter(row => row.split === 'train').length, test: built.rows.filter(row => row.split === 'test').length, passes: worlds.reduce((n, world) => n + world.passes.length, 0), admission_granted: false }, null, 2));
