/**
 * A frozen copy of the source-review hold functions as they were when the verdicts were TypeScript literals
 * (ts-host/src/teacher/source-review.ts at sha256 abb6f5594086b3672b6826a81729ff916b8152a290a29ffd6359c217e3da9a74),
 * run over the frozen records in source-review-legacy-records.json. The migration test compares the data-driven module
 * with this one. Do not edit: a change in behaviour belongs in the module and the registry, not here.
 */
import { readFileSync } from 'node:fs';
import '../../dist/benchmarks/builtin.js';
import { sourceContractsFor } from '../../dist/benchmarks/registry.js';

export const LEGACY_RECORDS = JSON.parse(readFileSync(new URL('./source-review-legacy-records.json', import.meta.url), 'utf8'));

export const LEGACY_CONTRACT_REVIEWS = [
  { dataset: 'hotpotqa', family: 'knowledge_evidence', primarySourceId: '9e84e5af371d9280c1b4935d10dc8e02c230618c2d168ee445999dc499f597b4' },
  { dataset: 'hotpotqa', family: 'knowledge_evidence', primarySourceId: 'c5b955678123f51525b7e3c83d317961b1c2093262aa317b9ceb60a3a26ccc42' },
];

export function legacyAnliReviewText(story) {
  const fields = ['beginning', 'ending', 'a', 'b'].map(key => story[key]);
  return fields.every(value => typeof value === 'string') ? fields.join('\n') : undefined;
}

export function legacyPendingSourceReview(dataset, id, record) {
  return LEGACY_RECORDS.find(review => {
    if (review.status !== 'pending' || review.dataset !== dataset || (review.id !== id && !review.aliases.includes(id))) return false;
    if (!record) return true;
    if (review.sourceRevision && !(Array.isArray(record.source_revisions) && record.source_revisions.includes(review.sourceRevision))) return false;
    if (review.sourceSnapshotSha256 && record.external_source?.snapshot_sha256 !== review.sourceSnapshotSha256) return false;
    if (review.sourcePrompt) {
      const semantics = record?.semantics;
      if (semantics?.files?.[semantics.root ?? ''] !== review.sourcePrompt || semantics.expected !== review.annotatedLabel) return false;
    }
    return true;
  });
}

export function legacySourceReviewReason(record) {
  const curriculum = record.curriculum;
  const inferred = curriculum?.family === 'entailment_premises' ? 'entailmentbank' : curriculum?.family === 'kqapro_question' ? 'kqapro' : curriculum?.family === 'anli_batch' ? 'anli' :
    ['folio_batch', 'folio_entailment'].includes(curriculum?.family ?? '') ? 'folio' : undefined;
  const declaredDataset = record.dataset ?? inferred ?? record.source;
  const dataset = declaredDataset === 'HotpotQA distractor' ? 'hotpotqa' : declaredDataset;
  if (typeof dataset !== 'string') return undefined;
  if (dataset === 'banking77' && curriculum?.family === 'cross_source_folders')
    return 'source_review_pending';
  const primarySourceId = Array.isArray(record.dataset_records) ? record.dataset_records[0] : undefined;
  if (LEGACY_CONTRACT_REVIEWS.some(review => review.dataset === dataset &&
      review.family === curriculum?.family && review.primarySourceId === primarySourceId))
    return 'source_review_pending';
  if (dataset === 'anli') {
    const semantics = record.semantics;
    const texts = semantics?.inputs?.stories?.flatMap(story => story && typeof story === 'object' ?
      [legacyAnliReviewText(story)] : []) ?? [];
    if (LEGACY_RECORDS.some(review => review.dataset === dataset && review.status === 'pending' && texts.includes(review.text)))
      return 'source_review_pending';
  }
  const ids = [record.dataset_records, record.source_ids, ['folio', 'entailmentbank'].includes(dataset) ? record.source_groups : []]
    .flatMap(value => Array.isArray(value) ? value : []);
  if (['folio', 'kqapro', 'entailmentbank'].includes(dataset) && curriculum?.shape) ids.push(curriculum.shape);
  const reviewedIds = new Set(sourceContractsFor(dataset ?? '').filter(contract => contract.isReviewedVariant(record))
    .map(contract => contract.sourceId));
  return ids.some(id => typeof id === 'string' && legacyPendingSourceReview(dataset, id, record) && !reviewedIds.has(id)) ?
    'source_review_pending' : undefined;
}
