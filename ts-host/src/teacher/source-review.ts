import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import '../benchmarks/builtin.js';
import { sourceContractsFor } from '../benchmarks/registry.js';

/**
 * Source annotation disputes stay out of generation and training until adjudicated.
 *
 * The verdicts are data: `training/source-reviews/holds.jsonl` (one SourceReview per line, in match order),
 * `family-holds.json` (holds of a whole task family) and `datasets.json` (what the code infers about datasets).
 * This module is the exact matcher over them; recommendations from the natlang reviewers never change the data
 * (see source-review-nl.ts and plans/SOURCE_REVIEW_PROGRAM.md).
 */
export type SourceReview = {
  dataset: string;
  id: string;
  aliases: readonly string[];
  text: string;
  annotatedLabel: string;
  status: 'pending' | 'resolved';
  reason: string;
  sourceRevision?: string;
  sourceSnapshotSha256?: string;
  sourcePrompt?: string;
};

/** A hold of a whole task family; `primarySourceId` null holds every task of the family. */
export type FamilyHold = { dataset: string; family: string; primarySourceId: string | null; reason: string };

/** What the code infers about datasets, as data. */
export type SourceReviewDatasets = {
  familyDataset: Readonly<Record<string, string>>;
  displayNames: Readonly<Record<string, string>>;
  identityFromSourceGroups: readonly string[];
  identityFromCurriculumShape: readonly string[];
};

const REVIEW_KEYS = new Set(['dataset', 'id', 'aliases', 'text', 'annotatedLabel', 'status', 'reason',
  'sourceRevision', 'sourceSnapshotSha256', 'sourcePrompt']);

/** The folder of the registry: `NATLANG_SOURCE_REVIEWS`, else `training/source-reviews` found above this module. */
export function sourceReviewsDirectory(): string {
  const configured = process.env.NATLANG_SOURCE_REVIEWS;
  if (configured) return configured;
  let directory = dirname(fileURLToPath(import.meta.url));
  for (let depth = 0; depth < 8; depth++, directory = dirname(directory)) {
    const candidate = join(directory, 'training', 'source-reviews');
    if (existsSync(join(candidate, 'holds.jsonl'))) return candidate;
  }
  throw new Error('training/source-reviews/holds.jsonl was not found above ' + dirname(fileURLToPath(import.meta.url)) +
    '; set NATLANG_SOURCE_REVIEWS to the folder that holds holds.jsonl, family-holds.json and datasets.json');
}

function fail(file: string, where: string, problem: string): never {
  throw new Error(`${file} ${where}: ${problem}`);
}

/** Parse and validate the lines of holds.jsonl. Any problem is an error: a registry that loads is a registry that holds. */
export function parseSourceReviews(text: string, file = 'holds.jsonl'): SourceReview[] {
  const reviews: SourceReview[] = [];
  const identities = new Set<string>();
  text.split('\n').forEach((line, index) => {
    if (!line.trim()) return;
    const where = `line ${index + 1}`;
    let value: Record<string, unknown>;
    try { value = JSON.parse(line); } catch (error) { return fail(file, where, `is not JSON (${(error as Error).message})`); }
    if (!value || typeof value !== 'object' || Array.isArray(value)) return fail(file, where, 'is not a JSON object');
    for (const key of Object.keys(value)) if (!REVIEW_KEYS.has(key)) fail(file, where, `has the unknown key ${JSON.stringify(key)}; the keys are ${[...REVIEW_KEYS].join(', ')}`);
    for (const key of ['dataset', 'id', 'text', 'annotatedLabel', 'reason'])
      if (typeof value[key] !== 'string') fail(file, where, `${key} must be a string`);
    for (const key of ['sourceRevision', 'sourceSnapshotSha256', 'sourcePrompt'])
      if (value[key] !== undefined && typeof value[key] !== 'string') fail(file, where, `${key} must be a string when present`);
    if (value.status !== 'pending' && value.status !== 'resolved') fail(file, where, 'status must be "pending" or "resolved"');
    if (!Array.isArray(value.aliases) || value.aliases.some(alias => typeof alias !== 'string')) fail(file, where, 'aliases must be an array of strings');
    for (const identity of [value.id as string, ...(value.aliases as string[])]) {
      const key = `${value.dataset}\0${identity}`;
      if (identities.has(key)) fail(file, where, `repeats the identity ${JSON.stringify(identity)} of dataset ${JSON.stringify(value.dataset)}; one entry holds each identity`);
      identities.add(key);
    }
    reviews.push(value as unknown as SourceReview);
  });
  return reviews;
}

function readJson(file: string): Record<string, unknown> {
  const value = JSON.parse(readFileSync(file, 'utf8'));
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${file} is not a JSON object`);
  return value as Record<string, unknown>;
}

/** Parse and validate family-holds.json. */
export function parseFamilyHolds(value: Record<string, unknown>, file = 'family-holds.json'): FamilyHold[] {
  if (!Array.isArray(value.holds)) return fail(file, 'holds', 'must be an array');
  return value.holds.map((hold: Record<string, unknown>, index: number) => {
    const where = `holds[${index}]`;
    for (const key of ['dataset', 'family', 'reason']) if (typeof hold?.[key] !== 'string') fail(file, where, `${key} must be a string`);
    if (hold.primarySourceId !== null && typeof hold.primarySourceId !== 'string') fail(file, where, 'primarySourceId must be a string or null');
    return { dataset: hold.dataset, family: hold.family, primarySourceId: hold.primarySourceId, reason: hold.reason } as FamilyHold;
  });
}

/** Parse and validate datasets.json. */
export function parseSourceReviewDatasets(value: Record<string, unknown>, file = 'datasets.json'): SourceReviewDatasets {
  const table = (key: string): Record<string, string> => {
    const entries = value[key] as Record<string, unknown> | undefined;
    if (!entries || typeof entries !== 'object' || Object.values(entries).some(item => typeof item !== 'string')) fail(file, key, 'must map names to dataset names');
    return entries as Record<string, string>;
  };
  const list = (key: string): string[] => {
    const items = value[key];
    if (!Array.isArray(items) || items.some(item => typeof item !== 'string')) fail(file, key, 'must be an array of dataset names');
    return items as string[];
  };
  return { familyDataset: table('familyDataset'), displayNames: table('displayNames'),
    identityFromSourceGroups: list('identityFromSourceGroups'), identityFromCurriculumShape: list('identityFromCurriculumShape') };
}

const directory = sourceReviewsDirectory();
const FAMILY_HOLDS = parseFamilyHolds(readJson(join(directory, 'family-holds.json')));
const DATASETS = parseSourceReviewDatasets(readJson(join(directory, 'datasets.json')));

/** Correct source answers can still have an invalid family-specific oracle. */
export const SOURCE_CONTRACT_REVIEWS: readonly { dataset: string; family: string; primarySourceId: string; reason: string }[] =
  FAMILY_HOLDS.filter((hold): hold is FamilyHold & { primarySourceId: string } => hold.primarySourceId !== null);

/** IDs use model-visible text; aliases preserve the earlier text+label identity. */
export const SOURCE_REVIEWS: readonly SourceReview[] = parseSourceReviews(readFileSync(join(directory, 'holds.jsonl'), 'utf8'));

/** Stable visible identity also catches legacy batches that omitted dataset_records. */
export function anliReviewText(story: Record<string, unknown>): string | undefined {
  const fields = ['beginning', 'ending', 'a', 'b'].map(key => story[key]);
  return fields.every(value => typeof value === 'string') ? fields.join('\n') : undefined;
}

export function pendingSourceReview(dataset: string, id: string, record?: Record<string, unknown>): SourceReview | undefined {
  return SOURCE_REVIEWS.find(review => {
    if (review.status !== 'pending' || review.dataset !== dataset || (review.id !== id && !review.aliases.includes(id))) return false;
    if (!record) return true; // ID-only lookups enumerate the pending review; task checks below enforce exact pins.
    if (review.sourceRevision && !(Array.isArray(record.source_revisions) && record.source_revisions.includes(review.sourceRevision))) return false;
    if (review.sourceSnapshotSha256 && (record.external_source as Record<string, unknown> | undefined)?.snapshot_sha256 !== review.sourceSnapshotSha256) return false;
    if (review.sourcePrompt) {
      const semantics = record?.semantics as { root?: string; files?: Record<string, unknown>; expected?: unknown } | undefined;
      if (semantics?.files?.[semantics.root ?? ''] !== review.sourcePrompt || semantics.expected !== review.annotatedLabel) return false;
    }
    return true;
  });
}

const own = (table: Readonly<Record<string, string>>, key: string | undefined): string | undefined =>
  key !== undefined && Object.hasOwn(table, key) ? table[key] : undefined;

/** A whole task is held when any of its original source records needs review. */
export function sourceReviewReason(record: Record<string, unknown>):
    string | undefined {
  const curriculum = record.curriculum as { family?: string; shape?: string } | undefined;
  const inferred = own(DATASETS.familyDataset, curriculum?.family);
  const declaredDataset = record.dataset ?? inferred ?? record.source;
  // The first source-card adapter recorded this display name; keep the same
  // factual hold effective for that immutable snapshot and native Hotpot cases.
  const dataset = typeof declaredDataset === 'string' ? own(DATASETS.displayNames, declaredDataset) ?? declaredDataset : declaredDataset;
  if (typeof dataset !== 'string') return undefined;
  // Family holds: the banking77 "unexpected extra charges" contract (every task of the family) and the Hotpot
  // contracts whose requested question is the primary source id. These families declare the requested question
  // first; later identities are distractor questions. Do not quarantine unrelated QA about the same source.
  const primarySourceId = Array.isArray(record.dataset_records) ? record.dataset_records[0] : undefined;
  if (FAMILY_HOLDS.some(hold => hold.dataset === dataset && hold.family === curriculum?.family &&
      (hold.primarySourceId === null || hold.primarySourceId === primarySourceId)))
    return 'source_review_pending';
  if (dataset === 'anli') {
    const semantics = record.semantics as { inputs?: { stories?: unknown[] } } | undefined;
    const texts = semantics?.inputs?.stories?.flatMap(story => story && typeof story === 'object' ?
      [anliReviewText(story as Record<string, unknown>)] : []) ?? [];
    if (SOURCE_REVIEWS.some(review => review.dataset === dataset && review.status === 'pending' && texts.includes(review.text)))
      return 'source_review_pending';
  }
  const ids = [record.dataset_records, record.source_ids, DATASETS.identityFromSourceGroups.includes(dataset) ? record.source_groups : []]
    .flatMap(value => Array.isArray(value) ? value : []);
  if (DATASETS.identityFromCurriculumShape.includes(dataset) && curriculum?.shape) ids.push(curriculum.shape);
  const reviewedIds = new Set(sourceContractsFor(dataset ?? '').filter(contract => contract.isReviewedVariant(record))
    .map(contract => contract.sourceId));
  return ids.some(id => typeof id === 'string' && pendingSourceReview(dataset, id, record) && !reviewedIds.has(id)) ?
    'source_review_pending' : undefined;
}
