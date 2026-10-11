#!/usr/bin/env node
import { createHash, randomUUID } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { link, mkdir, open, rename, unlink, readFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { dirname, isAbsolute, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { materializeNativeRows } from '../dist/teacher/native-materializer.js';
import { sourceQualityClearanceProblems } from '../dist/teacher/source-quality-clearance.js';
import { canonical } from '../dist/adaptation/identity.js';
import { hexDigest } from '../dist/native/hash.js';
import { fileDigest, jsonlRows } from './jsonl-stream.mjs';

const {values, positionals: positional} = parseArgs({allowPositionals:true, options:{replace:{type:'boolean'}, 'direct-answers':{type:'boolean'}, 'decision-review':{type:'string'}, 'source-quality-review':{type:'string'}}});
const [inputPath, outputPath] = positional;
if (!inputPath || !outputPath || positional.length !== 2) {
  console.error('usage: node scripts/materialize-native-teacher.mjs INPUT.jsonl OUTPUT.jsonl [--replace] [--direct-answers] [--decision-review FILE] [--source-quality-review FILE]');
  process.exit(2);
}
const input = resolve(inputPath), output = resolve(outputPath);
const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const repoPath = path => isAbsolute(path) ? resolve(path) : resolve(repoRoot, path);
const review = values['decision-review'] ? JSON.parse(await readFile(values['decision-review'], 'utf8')) : null;
if (review && (review.schema !== 'natlang.native-decision-review/1' ||
    (!Array.isArray(review.holds) && !Array.isArray(review.approvals)) ||
    (review.holds !== undefined && !Array.isArray(review.holds)) ||
    (review.approvals !== undefined && !Array.isArray(review.approvals))))
  throw new Error('invalid native decision review schema');
const qualityReviewPath = values['source-quality-review'] ? resolve(values['source-quality-review']) : null;
const qualityReviewBytes = qualityReviewPath ? await readFile(qualityReviewPath) : null;
const qualityReview = qualityReviewBytes ? JSON.parse(qualityReviewBytes.toString('utf8')) : null;
let verifiedQualityReview = null;
if (qualityReview) {
  const problems = sourceQualityClearanceProblems(qualityReview);
  if (problems.length) throw new Error(`invalid source quality clearance: ${problems.join(', ')}`);
  const policy = qualityReview.source_policy;
  const policyPath = repoPath(policy.path);
  if (policyPath !== resolve(repoRoot, 'training/decision_source_quality_holds.json'))
    throw new Error('source quality policy must bind the current canonical repository policy');
  if (await fileDigest(policyPath) !== policy.sha256) throw new Error('source quality policy byte hash mismatch');
  const policyJson = JSON.parse(await readFile(policyPath, 'utf8'));
  if (policyJson.schema !== 'natlang.decision-source-quality-holds/1' || !Array.isArray(policyJson.excluded_items) ||
      !Array.isArray(policyJson.state_truncation_rules) || policyJson.state_truncation_rules.some(rule =>
        !rule || typeof rule.rule_id !== 'string' || !rule.rule_id || !Number.isSafeInteger(rule.state_codepoints) ||
        rule.state_codepoints <= 0 || typeof rule.suffix !== 'string' || !rule.suffix))
    throw new Error('unsupported source quality policy schema');
  const policyIds = [...new Set(policyJson.excluded_items.map(item => item.item_id))].sort();
  const policyGroups = [...new Set(policyJson.excluded_items.map(item => item.original_source_group))].sort();
  if (JSON.stringify(policyIds) !== JSON.stringify([...policy.held_source_ids].sort()) ||
      JSON.stringify(policyGroups) !== JSON.stringify([...policy.held_source_groups].sort()))
    throw new Error('source quality policy held ID/group census mismatch');
  const verifiedFileDigests = new Map();
  const verifyFileDigest = async (path, expected, label) => {
    const resolved = repoPath(path);
    const actual = verifiedFileDigests.has(resolved) ? verifiedFileDigests.get(resolved) : await fileDigest(resolved);
    verifiedFileDigests.set(resolved, actual);
    if (actual !== expected) throw new Error(`${label} byte hash mismatch: ${path}`);
  };
  for (const evidence of qualityReview.evidence) {
    await verifyFileDigest(evidence.path, evidence.sha256, `source quality evidence ${evidence.role}`);
  }
  const sourceRowsByPath = new Map();
  for (const entry of qualityReview.programs) {
    if (entry.source_ids.some(id => policyIds.includes(id)) || entry.source_groups.some(group => policyGroups.includes(group)))
      throw new Error(`source quality clearance overlaps canonical holds: ${entry.program_id}`);
    await verifyFileDigest(entry.source_path, entry.source_snapshot_sha256, `source snapshot ${entry.program_id}`);
    await verifyFileDigest(entry.source_manifest_path, entry.source_manifest_sha256, `source manifest ${entry.program_id}`);
    const sourcePath = repoPath(entry.source_path);
    const rows = sourceRowsByPath.get(sourcePath) ?? new Map();
    for (const row of entry.source_rows) {
      const previous = rows.get(row.row_index);
      if (previous && (previous.row_sha256 !== row.row_sha256 || previous.id !== row.id || previous.group !== row.group))
        throw new Error(`conflicting source row binding: ${entry.program_id}/${row.row_index}`);
      rows.set(row.row_index, row);
    }
    sourceRowsByPath.set(sourcePath, rows);
  }
  for (const [sourcePath, rows] of sourceRowsByPath) {
    const lineHash = createHash('sha256');
    let lineIndex = 0, pending = Buffer.alloc(0);
    for await (const chunk of createReadStream(sourcePath)) {
      lineHash.update(chunk);
      pending = pending.length ? Buffer.concat([pending, chunk]) : chunk;
      let newline;
      while ((newline = pending.indexOf(10)) >= 0) {
        let line = pending.subarray(0, newline); pending = pending.subarray(newline + 1);
        if (line.at(-1) === 13) line = line.subarray(0, line.length - 1);
        const expected = rows.get(lineIndex);
        if (expected) verifySourceLine(line, expected, sourcePath, policyJson.state_truncation_rules);
        lineIndex++;
      }
    }
    if (pending.length) {
      const expected = rows.get(lineIndex);
      if (expected) verifySourceLine(pending, expected, sourcePath, policyJson.state_truncation_rules);
      lineIndex++;
    }
    if ([...rows.keys()].some(index => index >= lineIndex)) throw new Error(`source row index absent: ${sourcePath}`);
    const expectedSnapshots = new Set(qualityReview.programs.filter(entry => repoPath(entry.source_path) === sourcePath)
      .map(entry => entry.source_snapshot_sha256));
    if (expectedSnapshots.size !== 1 || lineHash.digest('hex') !== [...expectedSnapshots][0])
      throw new Error(`source changed or snapshot disagreement: ${sourcePath}`);
  }
  verifiedQualityReview = { schema: 'natlang.verified-source-quality-clearance/1',
    review_sha256: createHash('sha256').update(qualityReviewBytes).digest('hex'),
    review_canonical_sha256: hexDigest(canonical(qualityReview)), evidence_files_verified: true, review: qualityReview };
}
function verifySourceLine(line, expected, sourcePath, truncationRules) {
  if (createHash('sha256').update(line).digest('hex') !== expected.row_sha256)
    throw new Error(`source row byte hash mismatch: ${sourcePath}/${expected.id}`);
  const row = JSON.parse(line.toString('utf8'));
  if (row.id !== expected.id || row.group !== expected.group || expected.split !== 'train' || row.role !== 'train' ||
      (row.split !== undefined && row.split !== 'train'))
    throw new Error(`source row identity/group/split mismatch: ${sourcePath}/${expected.id}`);
  if (truncationRules.length && typeof row.state !== 'string')
    throw new Error(`canonical source state truncation rules cannot be checked (state missing): ${expected.id}`);
  if (typeof row.state === 'string') {
    const codepoints = Array.from(row.state).length;
    const matchedRule = truncationRules.find(rule => codepoints === rule.state_codepoints && row.state.endsWith(rule.suffix));
    if (matchedRule) throw new Error(`canonical source state truncation hold applies (${matchedRule.rule_id}): ${expected.id}`);
  }
}
const seen = new Set();
const seenQualityPrograms = new Set();
// --direct-answers: train answers given without reasoning towards them, for a student that answers directly.
await mkdir(dirname(output), { recursive: true });
const staged = `${output}.building-${process.pid}-${randomUUID()}`;
const authoredOutput = `${output}.authored-actions.jsonl`;
const stagedAuthored = `${authoredOutput}.building-${process.pid}-${randomUUID()}`;
const handle = await open(staged, 'wx');
let authoredHandle, accepted = 0, rejected = 0, turns = 0, authoredActions = 0;
try { for await (const row of jsonlRows(input)) {
  seen.add(row.id);
  if (verifiedQualityReview) {
    const program = row.task?.program_ir;
    const entry = verifiedQualityReview.review.programs.find(item => item.program_id === program?.id);
    if (entry) {
      if (hexDigest(canonical(program)) !== entry.program_ir_sha256) throw new Error(`program IR SHA mismatch: ${program.id}`);
      const external = program.external_source;
      if (hexDigest(canonical(external?.quality)) !== entry.source_quality_sha256 || external?.source_path !== entry.source_path ||
          external?.snapshot_sha256 !== entry.source_snapshot_sha256 || external?.source_manifest_sha256 !== entry.source_manifest_sha256 ||
          canonical(external?.source_rows) !== canonical(entry.source_rows) ||
          canonical(program.source_ids) !== canonical(entry.source_ids) || canonical(program.source_groups) !== canonical(entry.source_groups))
        throw new Error(`program source binding mismatch: ${program.id}`);
      seenQualityPrograms.add(program.id);
    }
  }
  const result = materializeNativeRows([row], { directAnswers: values['direct-answers'], decisionHolds: review?.holds, decisionApprovals: review?.approvals,
    sourceQualityClearance: verifiedQualityReview });
  accepted += result.acceptedRows; rejected += result.rejectedRows;
  for (const missed of result.unlinked)
    console.error(`  ${missed.id}: not used, ${missed.outcomes} action outcomes could not be linked to their decisions`);
  for (const turn of result.turns) { await handle.writeFile(JSON.stringify(turn) + '\n'); turns++; }
  if (result.authored_actions.length) {
    authoredHandle ??= await open(stagedAuthored, 'wx');
    for (const action of result.authored_actions) {
      await authoredHandle.writeFile(JSON.stringify(action) + '\n'); authoredActions++;
    }
  }
} } catch (error) {
  await handle.close(); await authoredHandle?.close(); await unlink(staged);
  if (authoredHandle) await unlink(stagedAuthored);
  throw error;
}
if (verifiedQualityReview && verifiedQualityReview.review.programs.some(entry => !seenQualityPrograms.has(entry.program_id))) {
  await handle.close(); await authoredHandle?.close(); await unlink(staged);
  if (authoredHandle) await unlink(stagedAuthored);
  throw new Error('source quality clearance references a program absent from input');
}
if ([...(review?.holds ?? []), ...(review?.approvals ?? [])].some(item => !seen.has(item.trajectory_id))) {
  await handle.close(); await authoredHandle?.close(); await unlink(staged);
  if (authoredHandle) await unlink(stagedAuthored);
  throw new Error('semantic review references absent input trajectory');
}
await handle.close(); await authoredHandle?.close();
console.error(`${accepted} rows -> ${turns} turns (${rejected} rows not used); ${authoredActions} held authored actions`);
if (values.replace) {
  await rename(staged, output);
  if (authoredHandle) await rename(stagedAuthored, authoredOutput);
  else await unlink(authoredOutput).catch(error => { if (error?.code !== 'ENOENT') throw error; });
} else {
  let linkedOutput = false;
  try {
    await link(staged, output); linkedOutput = true;
    if (authoredHandle) await link(stagedAuthored, authoredOutput);
  } catch (error) {
    await unlink(staged);
    if (linkedOutput) await unlink(output);
    if (authoredHandle) await unlink(stagedAuthored);
    if (error && typeof error === 'object' && 'code' in error && error.code === 'EEXIST')
      throw new Error(`refusing to overwrite ${output}`);
    throw error;
  }
  await unlink(staged);
  if (authoredHandle) await unlink(stagedAuthored);
}
console.log(JSON.stringify({ output, ...(authoredActions ? { authored_actions_output: authoredOutput } : {}),
  accepted_rows: accepted, rejected_rows: rejected, training_decisions: turns, authored_actions_held: authoredActions }));
