import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { SOURCE_REVIEWS, SOURCE_CONTRACT_REVIEWS, anliReviewText, pendingSourceReview, sourceReviewReason,
  parseSourceReviews, parseFamilyHolds, parseSourceReviewDatasets, sourceReviewsDirectory } from '../dist/teacher/source-review.js';
import { LEGACY_RECORDS, legacyAnliReviewText, legacyPendingSourceReview, legacySourceReviewReason } from './fixtures/source-review-legacy.mjs';

const sha = text => createHash('sha256').update(text).digest('hex');
const KEYS = ['dataset', 'id', 'aliases', 'text', 'annotatedLabel', 'status', 'reason', 'sourceRevision', 'sourceSnapshotSha256', 'sourcePrompt'];
const canonical = review => Object.fromEntries(KEYS.filter(key => review[key] !== undefined).map(key => [key, review[key]]));

test('the registry has the fingerprints of the pre-migration literals', () => {
  const records = SOURCE_REVIEWS.map(canonical);
  assert.equal(records.length, 184);
  assert.equal(sha(JSON.stringify(records)), '006941286fc79f951bfde45d87edcbd490b88549bd77215172fca9b31743f4e2');
  const identities = records.map(review => [review.dataset, review.id, ...review.aliases].join('\0'));
  assert.equal(identities.join('\n').split('\n').length, 184);
  assert.equal(records.reduce((count, review) => count + 1 + review.aliases.length, 0), 255);
  assert.equal(sha(identities.join('\n')), '0d1cd22acda8eeb2c1c1202aa9b6dafaac7772202ba84682358809768ec3af31');
});

test('the registry equals the frozen export of the old module, in order and byte for byte', () => {
  assert.deepEqual(SOURCE_REVIEWS.map(canonical), LEGACY_RECORDS);
  SOURCE_REVIEWS.forEach((review, index) => {
    const old = LEGACY_RECORDS[index];
    for (const identity of [[review.id, old.id], ...review.aliases.map((alias, at) => [alias, old.aliases[at]])])
      assert.ok(Buffer.from(identity[0], 'utf8').equals(Buffer.from(identity[1], 'utf8')), `identity bytes of entry ${index}`);
  });
  assert.equal(SOURCE_REVIEWS.filter(review => review.status !== 'pending').length, 0);
  assert.deepEqual(SOURCE_CONTRACT_REVIEWS.map(({ dataset, family, primarySourceId }) => ({ dataset, family, primarySourceId })), [
    { dataset: 'hotpotqa', family: 'knowledge_evidence', primarySourceId: '9e84e5af371d9280c1b4935d10dc8e02c230618c2d168ee445999dc499f597b4' },
    { dataset: 'hotpotqa', family: 'knowledge_evidence', primarySourceId: 'c5b955678123f51525b7e3c83d317961b1c2093262aa317b9ceb60a3a26ccc42' }]);
});

/** The record shapes of source-review.test.mjs and folder-families.test.mjs, for one identity of one review. */
function shapes(review, id) {
  const pinned = {
    ...(review.sourceRevision ? { source_revisions: [review.sourceRevision] } : {}),
    ...(review.sourceSnapshotSha256 ? { external_source: { snapshot_sha256: review.sourceSnapshotSha256 } } : {}),
    semantics: review.sourcePrompt ? { root: 'root', files: { root: review.sourcePrompt }, expected: review.annotatedLabel } : {},
  };
  const out = [
    { dataset: review.dataset },
    { dataset: review.dataset, dataset_records: [id], ...pinned },
    { dataset: review.dataset, source_ids: [id], ...pinned },
    { dataset: review.dataset, dataset_records: ['other', id], ...pinned },
    { dataset: review.dataset, dataset_records: [id] },
    { dataset: review.dataset, dataset_records: [id], semantics: {} },
    { dataset: review.dataset === 'sms_spam' ? 'sst2' : 'sms_spam', dataset_records: [id] },
    { dataset_records: [id] },
    { source: review.dataset, source_ids: [id], ...pinned },
    { dataset: review.dataset, source_groups: [id], ...pinned },
    { dataset: review.dataset, source_groups: [`${review.dataset}:${id}`] },
    { dataset: review.dataset, curriculum: { shape: id }, ...pinned },
    { curriculum: { family: 'entailment_premises', shape: id } },
    { curriculum: { family: 'kqapro_question', shape: id } },
    { curriculum: { family: 'folio_batch', shape: id } },
    { curriculum: { family: 'folio_entailment' }, source_groups: [id] },
    { curriculum: { family: 'anli_batch' }, source_groups: [id], dataset_records: [id] },
    { dataset: 'HotpotQA distractor', dataset_records: [id], ...pinned },
    { dataset: 'HotpotQA distractor', curriculum: { family: 'knowledge_evidence' }, dataset_records: [id, 'x'] },
    { dataset: 'banking77', curriculum: { family: 'cross_source_folders' }, dataset_records: [id] },
    { dataset: 'banking77', curriculum: { family: 'folder_triage' }, dataset_records: [id] },
    { dataset: review.dataset, curriculum: { family: 'cross_source_folders' }, dataset_records: [id] },
  ];
  // Each pin broken in turn.
  if (review.sourceRevision) out.push({ ...out[1], source_revisions: ['other-revision'] }, { ...out[1], source_revisions: 'musique-v1.0' });
  if (review.sourceSnapshotSha256) out.push({ ...out[1], external_source: { snapshot_sha256: 'x' } }, { ...out[1], external_source: undefined });
  if (review.sourcePrompt) out.push({ ...out[1], semantics: { root: 'root', files: { root: review.sourcePrompt }, expected: 'different' } },
    { ...out[1], semantics: { root: 'root', files: { root: `${review.sourcePrompt}.` }, expected: review.annotatedLabel } },
    { ...out[1], semantics: undefined });
  if (review.dataset === 'anli') {
    const [beginning, ending, a, b] = review.text.split('\n');
    const story = { id: 'N1', beginning, ending, a, b };
    out.push({ curriculum: { family: 'anli_batch' }, semantics: { inputs: { stories: [story] } } },
      { dataset: 'anli', semantics: { inputs: { stories: [{ ...story, b: 'changed' }, null, 4] } } },
      { dataset: 'anli', semantics: { inputs: { stories: [{ beginning, ending, a }] } } });
  }
  return out;
}

test('differential: the data-driven functions equal the frozen old functions on every identity and record shape', () => {
  let compared = 0;
  for (const review of SOURCE_REVIEWS) for (const id of [review.id, ...review.aliases, `${review.id}-unknown`]) {
    assert.equal(pendingSourceReview(review.dataset, id)?.id, legacyPendingSourceReview(review.dataset, id)?.id);
    assert.equal(pendingSourceReview('sms_spam', id)?.id, legacyPendingSourceReview('sms_spam', id)?.id);
    for (const record of shapes(review, id)) {
      assert.equal(sourceReviewReason(record), legacySourceReviewReason(record), JSON.stringify(record).slice(0, 200));
      const recordDataset = record.dataset ?? review.dataset;
      assert.equal(pendingSourceReview(recordDataset, id, record)?.id, legacyPendingSourceReview(recordDataset, id, record)?.id);
      compared++;
    }
  }
  assert.ok(compared > 5000, `compared ${compared} shapes`);
});

test('differential: the family holds, the display name, the family inference and the odd inputs', () => {
  const cases = [];
  for (const hold of [...SOURCE_CONTRACT_REVIEWS, { dataset: 'hotpotqa', family: 'knowledge_evidence', primarySourceId: 'unlisted' }])
    for (const dataset of ['hotpotqa', 'HotpotQA distractor', 'musique', undefined])
      for (const family of ['knowledge_evidence', 'knowledge_research', undefined])
        for (const records of [[hold.primarySourceId], ['q', hold.primarySourceId], [], undefined, 'text', [3]])
          cases.push({ ...(dataset ? { dataset } : { source: 'hotpotqa' }), curriculum: family ? { family } : undefined, dataset_records: records });
  for (const family of ['cross_source_folders', 'folder_triage', 'anli_batch', 'folio_batch', 'folio_entailment', 'entailment_premises',
    'kqapro_question', 'constructor', '__proto__', 'toString', 5, null])
    for (const dataset of ['banking77', 'sst2', 'constructor', 'toString', '__proto__', undefined, 5, null, 'HotpotQA distractor'])
      for (const shape of [undefined, 'story337', 'train_33143', 'LEAP_7_10338', 'story999'])
        cases.push({ dataset, curriculum: { family, shape }, dataset_records: ['x'], source_groups: ['folio:story:337'], source_ids: ['story337'] });
  cases.push({}, { curriculum: null }, { source: null }, { dataset: 'folio', curriculum: 'text' });
  for (const record of cases) assert.equal(sourceReviewReason(record), legacySourceReviewReason(record), JSON.stringify(record));
  assert.equal(anliReviewText({ beginning: 'a', ending: 'b', a: 'c', b: 'd' }), legacyAnliReviewText({ beginning: 'a', ending: 'b', a: 'c', b: 'd' }));
  assert.equal(anliReviewText({ beginning: 'a' }), undefined);
  assert.equal(cases.filter(record => sourceReviewReason(record) === 'source_review_pending').length > 10, true);
});

test('the registry files are what the loader reads, and the loader refuses a registry that does not hold', () => {
  const directory = sourceReviewsDirectory();
  assert.match(directory, /training[\\/]source-reviews$/);
  const text = readFileSync(`${directory}/holds.jsonl`, 'utf8');
  assert.deepEqual(parseSourceReviews(text), [...SOURCE_REVIEWS]);
  const line = JSON.stringify(canonical(SOURCE_REVIEWS[0]));
  assert.equal(parseSourceReviews(line).length, 1);
  assert.throws(() => parseSourceReviews(`${line}\n${line}`), /repeats the identity/);
  assert.throws(() => parseSourceReviews(JSON.stringify({ ...canonical(SOURCE_REVIEWS[0]), extra: 1 })), /unknown key "extra"/);
  assert.throws(() => parseSourceReviews(JSON.stringify({ ...canonical(SOURCE_REVIEWS[0]), status: 'open' })), /status must be/);
  assert.throws(() => parseSourceReviews(JSON.stringify({ ...canonical(SOURCE_REVIEWS[0]), aliases: 'x' })), /aliases must be/);
  assert.throws(() => parseSourceReviews(JSON.stringify({ ...canonical(SOURCE_REVIEWS[0]), reason: 3 })), /reason must be a string/);
  assert.throws(() => parseSourceReviews('{broken'), /is not JSON/);
  assert.throws(() => parseFamilyHolds({ holds: [{ dataset: 'x', family: 'y', reason: 'z' }] }), /primarySourceId must be/);
  assert.throws(() => parseSourceReviewDatasets({ familyDataset: {}, displayNames: {} }), /identityFromSourceGroups/);
  const family = JSON.parse(readFileSync(`${directory}/family-holds.json`, 'utf8'));
  assert.deepEqual(family.holds.map(hold => [hold.dataset, hold.family, hold.primarySourceId]), [
    ['hotpotqa', 'knowledge_evidence', '9e84e5af371d9280c1b4935d10dc8e02c230618c2d168ee445999dc499f597b4'],
    ['hotpotqa', 'knowledge_evidence', 'c5b955678123f51525b7e3c83d317961b1c2093262aa317b9ceb60a3a26ccc42'],
    ['banking77', 'cross_source_folders', null]]);
});
