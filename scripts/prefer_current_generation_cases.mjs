#!/usr/bin/env node
/**
 * Resolve historical generation inputs to their exact current canonical case.
 * This is a selection/provenance guard, not a trajectory migration or admission
 * bypass. A caller must still run normal native admission on selected rows.
 */
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { applyEvidenceScaleContract, applyTatqaNumericContract, applyReviewedTatqaUnitContract } from '../ts-host/scripts/inline-curriculum/directory-sources.mjs';
import { applyTatqaProportionDisplay } from '../ts-host/scripts/inline-curriculum/tatqa-proportion-display-reviewed.mjs';
import { sourceReviewReason } from '../ts-host/dist/teacher/source-review.js';
import { generationHoldReason } from '../ts-host/dist/teacher/curriculum-policy.js';

const sha256 = value => createHash('sha256').update(value).digest('hex');
const canonical = value => {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(',')}}`;
  return JSON.stringify(value);
};
const recordDigest = record => sha256(canonical(record));
const same = (a, b) => canonical(a) === canonical(b);

function soleIdentity(record) {
  const ext = record?.external_source;
  if (!record || !Array.isArray(record.source_ids) || record.source_ids.length !== 1 ||
      !Array.isArray(record.source_groups) || record.source_groups.length < 1 ||
      !Array.isArray(record.source_revisions) || record.source_revisions.length < 1 ||
      typeof ext?.source_id !== 'string' || ext.source_id !== record.source_ids[0]) return null;
  return record.source_ids[0];
}

function preservedSourceFields(record) {
  return {
    source: record.source,
    split: record.split,
    source_ids: record.source_ids,
    source_groups: record.source_groups,
    source_revisions: record.source_revisions,
    license: record.license,
    gold_sources: record.gold_sources,
    external_source: {
      source: record.external_source?.source,
      source_id: record.external_source?.source_id,
      original_split: record.external_source?.original_split,
      revision: record.external_source?.revision,
      snapshot_sha256: record.external_source?.snapshot_sha256,
      license: record.external_source?.license,
      files: record.external_source?.files,
    },
    source_files: record.semantics?.folder_files,
    expected: record.semantics?.expected,
    expected_files: record.semantics?.expected_files,
    inputs: record.semantics?.inputs,
  };
}

function deriveApprovedCurrentVariant(historical) {
  const derived = structuredClone(historical);
  const operations = [];
  if (derived.source === 'tatqa') {
    const originalId = derived.id;
    applyEvidenceScaleContract(derived);
    applyTatqaNumericContract(derived);
    applyReviewedTatqaUnitContract(derived);
    if (derived.id !== originalId) operations.push('tatqa-evidence-scale-and-numeric-contracts');
    // This call is intentionally delegated to the exact source-pinned registry.
    // Unknown prompt variants throw and cannot be accepted by a generic diff.
    const beforeDisplayId = derived.id;
    try {
      const reviewed = applyTatqaProportionDisplay(derived);
      if (reviewed.id !== beforeDisplayId) operations.push('tatqa-reviewed-proportion-display');
      return { record: reviewed, operations };
    } catch (error) {
      if (!String(error?.message ?? error).includes('tatqa_proportion_display_unregistered')) throw error;
      return { record: derived, operations };
    }
  }
  return { record: derived, operations };
}

/**
 * Resolve one historical row against records from a pinned canonical corpus.
 * `sourceHeld` / `generationHeld` are injected in tests and default to current
 * compiled policy in the CLI.
 */
export function preferCurrentGenerationCase(historical, canonicalRows, {
  sourceHeld = sourceReviewReason,
  generationHeld = generationHoldReason,
} = {}) {
  const sourceId = soleIdentity(historical);
  if (!sourceId) return { status: 'blocked', reason: 'historical_source_identity_incomplete' };
  const candidates = canonicalRows.filter(row => soleIdentity(row) === sourceId);
  if (candidates.length === 0) return { status: 'unresolved', reason: 'no_canonical_source_candidate', sourceId };
  if (candidates.length !== 1) return { status: 'blocked', reason: 'ambiguous_canonical_source_candidates', sourceId,
    candidateProgramIds: candidates.map(row => row.id) };

  const current = candidates[0];
  if (!same(preservedSourceFields(historical), preservedSourceFields(current)))
    return { status: 'blocked', reason: 'source_group_revision_files_or_gold_changed', sourceId,
      historicalProgramId: historical.id, candidateProgramId: current.id };

  let derived;
  try { derived = deriveApprovedCurrentVariant(historical); }
  catch (error) { return { status: 'blocked', reason: 'canonical_transform_rejected', sourceId,
    historicalProgramId: historical.id, candidateProgramId: current.id, detail: String(error?.message ?? error) }; }
  if (!same(derived.record, current)) return { status: 'blocked', reason: 'unregistered_or_unexpected_semantic_change', sourceId,
    historicalProgramId: historical.id, candidateProgramId: current.id };

  const holdReasons = [sourceHeld(current), generationHeld(current)].filter(Boolean);
  const lineage = {
    sourceId,
    sourceGroups: historical.source_groups,
    historicalProgramId: historical.id,
    currentProgramId: current.id,
    historicalProgramSha256: recordDigest(historical),
    currentProgramSha256: recordDigest(current),
    transformations: derived.operations,
    sameSourceFilesAndGold: true,
  };
  if (holdReasons.length) return { status: 'held', reason: holdReasons.join(','), lineage };
  return { status: 'preferred_current', reason: 'exact_approved_current_variant', lineage, record: current };
}

function args(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i++) {
    if (!argv[i].startsWith('--') || !argv[i + 1]) throw Error(`invalid_argument:${argv[i]}`);
    out[argv[i].slice(2)] = argv[++i];
  }
  for (const key of ['input', 'canonical', 'canonical-sha256', 'output', 'decisions'])
    if (!out[key]) throw Error(`missing_argument:--${key}`);
  return out;
}
const parseJsonl = path => readFile(path, 'utf8').then(text => text.split(/\r?\n/).filter(line => line.trim()).map(line => JSON.parse(line)));

async function main() {
  const options = args(process.argv);
  const canonicalText = await readFile(options.canonical, 'utf8');
  const canonicalHash = sha256(canonicalText);
  if (canonicalHash !== options['canonical-sha256']) throw Error(`canonical_corpus_hash_mismatch:${canonicalHash}`);
  // Select from the exact bytes whose hash was checked, even if a canonical
  // pointer is replaced by another publisher while this invocation is running.
  const canonicalRows = canonicalText.split(/\r?\n/).filter(line => line.trim()).map(line => JSON.parse(line));
  const historicalRows = await parseJsonl(options.input);
  const used = new Set();
  const selected = [], decisions = [];
  for (const historical of historicalRows) {
    const decision = preferCurrentGenerationCase(historical, canonicalRows);
    if (decision.status === 'preferred_current') {
      if (used.has(decision.record.id)) throw Error(`duplicate_selected_program_id:${decision.record.id}`);
      used.add(decision.record.id);
      selected.push(decision.record);
    }
    decisions.push({ historicalProgramId: historical.id, sourceId: soleIdentity(historical), ...decision,
      ...(decision.record ? { record: undefined } : {}) });
    if (decisions.at(-1).record === undefined) delete decisions.at(-1).record;
  }
  await writeFile(options.output, `${selected.map(row => JSON.stringify(row)).join('\n')}${selected.length ? '\n' : ''}`, { flag: 'wx' });
  await writeFile(options.decisions, `${decisions.map(row => JSON.stringify(row)).join('\n')}${decisions.length ? '\n' : ''}`, { flag: 'wx' });
  const counts = Object.fromEntries(['preferred_current', 'held', 'unresolved', 'blocked']
    .map(status => [status, decisions.filter(row => row.status === status).length]));
  const receipt = { version: 'preferred-current-generation-cases/1', input: options.input,
    canonicalCorpus: options.canonical, canonicalCorpusSha256: canonicalHash,
    helperSha256: sha256(await readFile(new URL(import.meta.url))),
    sourceAdapterSha256: sha256(await readFile(new URL('../ts-host/scripts/inline-curriculum/directory-sources.mjs', import.meta.url))),
    proportionRegistrySha256: sha256(await readFile(new URL('../ts-host/scripts/inline-curriculum/tatqa-proportion-display-reviewed.mjs', import.meta.url))),
    sourceReviewModuleSha256: sha256(await readFile(new URL('../ts-host/dist/teacher/source-review.js', import.meta.url))),
    curriculumPolicyModuleSha256: sha256(await readFile(new URL('../ts-host/dist/teacher/curriculum-policy.js', import.meta.url))),
    selectedPath: options.output, selectedSha256: sha256(await readFile(options.output)),
    decisionsPath: options.decisions, decisionsSha256: sha256(await readFile(options.decisions)),
    inputCount: historicalRows.length, selectedCount: selected.length, counts };
  const receiptPath = `${options.decisions}.receipt.json`;
  await writeFile(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, { flag: 'wx' });
  console.log(JSON.stringify({ ...receipt, receiptPath }));
  if (counts.blocked || counts.unresolved) process.exitCode = 2;
}

if (process.argv[1] && import.meta.url === new URL(`file://${process.argv[1]}`).href) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
