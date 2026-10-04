#!/usr/bin/env node
/** Stage a reviewed positive skill-authoring export as a proposal; never edits current-manifest.json. */
import { readFile, writeFile, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, relative, sep } from 'node:path';

const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const inside = (root, path) => {
  const rel = relative(root, path);
  return rel !== '' && rel !== '..' && !rel.startsWith(`..${sep}`);
};
const readJson = async path => JSON.parse(await readFile(path, 'utf8'));

export async function stagePublication({ repo, exportDirectory, reviewPath, outputPath }) {
  repo = resolve(repo); exportDirectory = resolve(exportDirectory); reviewPath = resolve(reviewPath);
  outputPath = resolve(outputPath);
  if (![exportDirectory, reviewPath, outputPath].every(path => inside(repo, path)))
    throw new Error('export, review receipt, and proposal output must be inside the repository');

  const manifestPath = resolve(exportDirectory, 'manifest.json');
  const turnsPath = resolve(exportDirectory, 'verified-turns.jsonl');
  const negativesPath = resolve(exportDirectory, 'negative-evidence.jsonl');
  const [manifestBytes, turnsBytes, negativesBytes, reviewBytes] = await Promise.all([
    readFile(manifestPath), readFile(turnsPath), readFile(negativesPath), readFile(reviewPath),
  ]);
  const manifest = JSON.parse(manifestBytes);
  const review = JSON.parse(reviewBytes);
  if (manifest.schema !== 'natlang.skill-authoring-training-candidate/1' || manifest.lane !== 'skill-authoring')
    throw new Error('unsupported candidate manifest');
  if (!Number.isSafeInteger(manifest.rows) || manifest.rows <= 0 || manifest.provider_calls !== 0 || manifest.dpo_pairs !== 0)
    throw new Error('candidate must contain positive provider-free SFT rows and no DPO pairs');
  if (sha(turnsBytes) !== manifest.rows_sha256 || sha(negativesBytes) !== manifest.negative_sha256)
    throw new Error('candidate data digest mismatch');
  const rows = turnsBytes.toString('utf8').split('\n').filter(Boolean).map(JSON.parse);
  if (rows.length !== manifest.rows) throw new Error('candidate row count mismatch');
  if (!Array.isArray(manifest.cases) || !manifest.cases.length || manifest.cases.some(item =>
      !['verified-support-sft', 'quarantined', 'missing-result'].includes(item.disposition)))
    throw new Error('candidate case dispositions are missing or unrecognized');
  const acceptedCases = manifest.cases.filter(item => item.disposition === 'verified-support-sft');
  if (!acceptedCases.length || acceptedCases.some(item => item.paired_replay !== true || item.providerCalls !== 0) ||
      acceptedCases.reduce((total, item) => total + (Number.isSafeInteger(item.turns) ? item.turns : 0), 0) !== rows.length)
    throw new Error('published rows must map to exact paired replay cases with verified support-only SFT');
  if (rows.some(row => row?.task?.program_ir?.family !== 'skill-authoring' || row?.task?.program_ir?.split !== 'train'))
    throw new Error('candidate rows must remain skill-authoring support train rows');
  if (review.schema !== 'natlang.skill-authoring-source-review/1' || review.decision !== 'approve' ||
      typeof review.reviewer !== 'string' || !review.reviewer.trim() ||
      review.candidate_manifest_sha256 !== sha(manifestBytes) ||
      review.source_policy?.decision !== 'approved' || !Array.isArray(review.source_policy.evidence) ||
      review.source_policy.evidence.length === 0)
    throw new Error('explicit source-policy review receipt is missing, negative, or not bound to this export');
  for (const evidence of review.source_policy.evidence) {
    if (typeof evidence.path !== 'string' || !/^[a-f0-9]{64}$/.test(evidence.sha256))
      throw new Error('source-policy evidence needs a repository path and SHA-256');
    const evidencePath = resolve(repo, evidence.path);
    if (!inside(repo, evidencePath) || sha(await readFile(evidencePath)) !== evidence.sha256)
      throw new Error(`source-policy evidence digest mismatch: ${evidence.path}`);
  }
  await stat(outputPath).then(() => { throw new Error('proposal output already exists'); }, error => {
    if (error.code !== 'ENOENT') throw error;
  });
  const entry = {
    path: relative(repo, turnsPath), sha256: manifest.rows_sha256, rows: manifest.rows,
    lane: 'skill-authoring', runtime_api: 'native-ordinary-runtime',
    verification_artifact: relative(repo, manifestPath), verification_sha256: sha(manifestBytes),
  };
  const proposal = {
    schema: 'natlang.skill-authoring-publication-proposal/1',
    status: 'staged_for_independent_review',
    candidate_manifest: relative(repo, manifestPath), candidate_manifest_sha256: sha(manifestBytes),
    turns: entry.path, turns_sha256: entry.sha256, rows: entry.rows,
    negative_evidence: relative(repo, negativesPath), negative_sha256: sha(negativesBytes),
    source_review: relative(repo, reviewPath), source_review_sha256: sha(reviewBytes),
    proposed_registry_entry: entry,
    activation: 'not_performed; merge this entry into current-manifest.json only after independent review',
  };
  await writeFile(outputPath, JSON.stringify(proposal, null, 2) + '\n', { flag: 'wx' });
  return proposal;
}

if (process.argv[1] === new URL(import.meta.url).pathname) {
  const args = process.argv.slice(2);
  if (args.length !== 8 || args[0] !== '--repo' || args[2] !== '--export' || args[4] !== '--review' || args[6] !== '--out')
    throw Error('usage: node stage-training-publication.mjs --repo REPO --export DIR --review RECEIPT.json --out PROPOSAL.json');
  console.log(JSON.stringify(await stagePublication({ repo: args[1], exportDirectory: args[3], reviewPath: args[5], outputPath: args[7] }), null, 2));
}
