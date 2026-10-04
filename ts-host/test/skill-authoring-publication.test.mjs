import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { stagePublication } from '../scripts/skills/stage-training-publication.mjs';

const sha = value => createHash('sha256').update(value).digest('hex');

async function fixture(repo, { rows = 1 } = {}) {
  const candidate = join(repo, 'candidate'); await mkdir(candidate);
  const row = { task: { program_ir: { family: 'skill-authoring', split: 'train' } } };
  const turns = rows ? Buffer.from(JSON.stringify(row) + '\n') : Buffer.alloc(0);
  const negatives = Buffer.alloc(0);
  const manifest = {
    schema: 'natlang.skill-authoring-training-candidate/1', lane: 'skill-authoring',
    rows, rows_sha256: sha(turns), negative_sha256: sha(negatives), provider_calls: 0, dpo_pairs: 0,
    cases: rows ? [
      { disposition: 'verified-support-sft', paired_replay: true, providerCalls: 0, turns: rows },
      { disposition: 'quarantined', reason: 'interrupted' },
    ] : [],
  };
  const manifestBytes = Buffer.from(JSON.stringify(manifest) + '\n');
  await writeFile(join(candidate, 'manifest.json'), manifestBytes);
  await writeFile(join(candidate, 'verified-turns.jsonl'), turns);
  await writeFile(join(candidate, 'negative-evidence.jsonl'), negatives);
  const evidence = join(repo, 'source-policy-evidence.txt'); await writeFile(evidence, 'pinned source decision');
  const review = join(repo, 'review.json'); await writeFile(review, JSON.stringify({
    schema: 'natlang.skill-authoring-source-review/1', decision: 'approve', reviewer: 'reviewer-1',
    candidate_manifest_sha256: sha(manifestBytes),
    source_policy: { decision: 'approved', evidence: [{ path: 'source-policy-evidence.txt', sha256: sha(await readFile(evidence)) }] },
  }));
  return { candidate, review };
}

test('publication staging requires positive replay and explicit source review and does not activate registry', async () => {
  const repo = await mkdtemp(join(tmpdir(), 'natlang-skill-publication-'));
  try {
    const { candidate, review } = await fixture(repo);
    const proposalPath = join(repo, 'proposal.json');
    const proposal = await stagePublication({ repo, exportDirectory: candidate, reviewPath: review, outputPath: proposalPath });
    assert.equal(proposal.status, 'staged_for_independent_review');
    assert.equal(proposal.proposed_registry_entry.rows, 1);
    assert.equal(proposal.proposed_registry_entry.verification_sha256, proposal.candidate_manifest_sha256);
    await assert.rejects(readFile(join(repo, 'data/teacher/self-improvement/current-manifest.json')));
    await assert.rejects(() => stagePublication({ repo, exportDirectory: candidate, reviewPath: review, outputPath: proposalPath }), /already exists/);
  } finally { await rm(repo, { recursive: true, force: true }); }
});

test('publication staging refuses zero-row negative exports', async () => {
  const repo = await mkdtemp(join(tmpdir(), 'natlang-skill-publication-negative-'));
  try {
    const { candidate, review } = await fixture(repo, { rows: 0 });
    await assert.rejects(() => stagePublication({ repo, exportDirectory: candidate, reviewPath: review,
      outputPath: join(repo, 'proposal.json') }), /positive provider-free SFT rows/);
  } finally { await rm(repo, { recursive: true, force: true }); }
});
