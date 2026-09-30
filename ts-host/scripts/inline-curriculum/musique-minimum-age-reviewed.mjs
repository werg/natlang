/** Exact evidence-scoped production adapter; intentionally not wired into a builder yet. */
import { createHash } from 'node:crypto';

const sha256 = value => createHash('sha256').update(value).digest('hex');
const canonical = value => Array.isArray(value) ? `[${value.map(canonical).join(',')}]` :
  value && typeof value === 'object' ? `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(',')}}` : JSON.stringify(value);
const digest = value => sha256(canonical(value));

export const MINIMUM_AGE_SOURCE_ID = '2hop__130422_69489';
export const MINIMUM_AGE_CONTRACT_SUFFIX = ':minimum-age-answer-v1';
export const MINIMUM_AGE_CONTRACT_REVISION = 'musique-minimum-age-equivalence/2026-09-30-v1';
export const MINIMUM_AGE_ALIASES = Object.freeze(['21', '21 years old', '21 years']);
export const MINIMUM_AGE_TASK_PROMPT = `---
args: {}
returns: "string"
kind: directory-reducer
---
What is the minimum age requirement to buy a handgun in the state where Wartburg College is located?
Read the articles to answer. Return only the answer text. Preserve the workspace.
`;

const BASE_REMOVED_CANONICAL_ID = 'inline-curriculum:source_musique:e403cfa783ea6d1932b0:v1:reviewed-oracle-alias-removal-v2';
const BASE_REMOVED_TITLED_ID = `${BASE_REMOVED_CANONICAL_ID}:title-path-v2-r2`;
const PINNED_BASE_RECORD_SHA256 = Object.freeze({
  [BASE_REMOVED_CANONICAL_ID]: 'b3cff9c4c8c459229ffdd25d9198674e3db57cf05a413fca5c683c2ce25893d6',
  [BASE_REMOVED_TITLED_ID]: '1e52f8417e5500e98eef131c7c70d32e589dabd189d68a9057c0f1fc120bdcaa',
});
const SNAPSHOT_SHA256 = 'cb4c55dfe2de2daa36a567771e862d8575ce4fa64e68631d7055b09351de28e5';
const PRIMARY = '21 or older.';
const ARTICLE_EVIDENCE = Object.freeze([
  Object.freeze({ source_path: 'articles/0.md', titled_path: 'articles/000-wartburg-college.md', sha256: 'ffef5cea04be0accc780bc4e6375594d840249eea982748fbc3736a5cfb34e09', text: 'located in Waverly, Iowa' }),
  Object.freeze({ source_path: 'articles/9.md', titled_path: 'articles/009-gun-laws-in-iowa.md', sha256: 'b0b07c0d37d4d3d700d8f1eaea4605a96b18cfa61c52bf5720f5b2b70148194e', text: 'shall be issued to qualified applicants aged 21 or older' }),
]);

function fail(reason) { throw new Error(`musique_minimum_age_review_${reason}:${MINIMUM_AGE_SOURCE_ID}`); }

function exactOracle(oracle, alternates) {
  return oracle && typeof oracle === 'object' && !Array.isArray(oracle) &&
    Object.keys(oracle).sort().join(',') === 'alternates,level' && oracle.level === 'normalized' &&
    Array.isArray(oracle.alternates) && JSON.stringify(oracle.alternates) === JSON.stringify(alternates);
}

function placementForId(id) {
  if (id === BASE_REMOVED_CANONICAL_ID) return 'canonical-before-title-remap';
  if (id === BASE_REMOVED_TITLED_ID) return 'historical-titled-migration';
  return null;
}

function evidencePath(record, evidence) {
  const files = record.semantics?.folder_files;
  const titleReview = record.generation?.article_path_review;
  const sourceBody = files?.[evidence.source_path];
  if (typeof sourceBody === 'string') return evidence.source_path;
  if (titleReview?.registry === 'musique-title-paths/2026-09-30-v2-r2' &&
      titleReview.source_snapshot_sha256 === SNAPSHOT_SHA256 &&
      titleReview.path_map?.[evidence.source_path] === evidence.titled_path &&
      typeof files?.[evidence.titled_path] === 'string') return evidence.titled_path;
  return null;
}

function expectedReview(record, { baseId, baseRecordSha256, placement }) {
  return {
    registry: MINIMUM_AGE_CONTRACT_REVISION,
    review: 'source-pinned-minimum-age-equivalence',
    source_id: MINIMUM_AGE_SOURCE_ID,
    base_ir_id: baseId,
    variant_ir_id: `${baseId}${MINIMUM_AGE_CONTRACT_SUFFIX}`,
    source_snapshot_sha256: SNAPSHOT_SHA256,
    base_record_sha256: baseRecordSha256,
    placement,
    primary: PRIMARY,
    accepted: [...MINIMUM_AGE_ALIASES],
    rejected_prior_alias: 'Gun laws in Iowa',
    task_prompt: MINIMUM_AGE_TASK_PROMPT,
    rationale: 'The exact prompt requests the minimum age; Wartburg is in Iowa, and the linked Iowa text states the permit threshold is 21 or older. These three explicit strings state the same minimum age.',
    evidence: ARTICLE_EVIDENCE.map(item => ({
      path: evidencePath(record, item), source_path: item.source_path, sha256: item.sha256, text: item.text,
    })),
  };
}

function validateBase(record, expectedBaseId = record.id) {
  const placement = placementForId(expectedBaseId);
  const sourceId = record?.external_source?.source_id ?? record?.source_ids?.[0];
  const prompt = record?.semantics?.files?.[record?.semantics?.root];
  const review = record?.generation?.oracle_review;
  if (!placement || sourceId !== MINIMUM_AGE_SOURCE_ID || record.id !== expectedBaseId ||
      digest(record) !== PINNED_BASE_RECORD_SHA256[expectedBaseId] ||
      record.source !== 'musique' || JSON.stringify(record.source_ids) !== JSON.stringify([MINIMUM_AGE_SOURCE_ID]) ||
      record.external_source?.snapshot_sha256 !== SNAPSHOT_SHA256 || record.semantics?.expected !== PRIMARY ||
      prompt !== MINIMUM_AGE_TASK_PROMPT || !exactOracle(record.semantics?.oracle, []) ||
      record.generation?.minimum_age_equivalence_review !== undefined ||
      review?.registry !== 'musique-reviewed-alias-removals/2026-09-30-v2' ||
      review?.source_id !== MINIMUM_AGE_SOURCE_ID || review?.source_snapshot_sha256 !== SNAPSHOT_SHA256 ||
      review?.primary !== PRIMARY || review?.rejected !== 'Gun laws in Iowa' ||
      review?.task_prompt !== MINIMUM_AGE_TASK_PROMPT ||
      record.generation?.article_path_review?.registry !== (placement === 'historical-titled-migration' ?
        'musique-title-paths/2026-09-30-v2-r2' : undefined)) return false;
  return ARTICLE_EVIDENCE.every(evidence => {
    const currentPath = evidencePath(record, evidence);
    const body = currentPath && record.semantics.folder_files[currentPath];
    const sourceEvidence = review.evidence?.find(item => item.path === evidence.source_path);
    return typeof body === 'string' && sha256(body) === evidence.sha256 && body.includes(evidence.text) &&
      sourceEvidence?.sha256 === evidence.sha256 && body.includes(sourceEvidence.text);
  });
}

/** Revalidate every application; original IR remains untouched and variants are idempotent. */
export function validateMinimumAgeReviewedVariant(record) {
  const audit = record?.generation?.minimum_age_equivalence_review;
  if (!audit || audit.registry !== MINIMUM_AGE_CONTRACT_REVISION ||
      audit.source_id !== MINIMUM_AGE_SOURCE_ID || ![BASE_REMOVED_CANONICAL_ID, BASE_REMOVED_TITLED_ID].includes(audit.base_ir_id))
    return false;
  const baseId = audit.base_ir_id;
  const placement = placementForId(baseId);
  const targetId = `${baseId}${MINIMUM_AGE_CONTRACT_SUFFIX}`;
  if (record.id !== targetId || audit.variant_ir_id !== targetId || audit.placement !== placement ||
      record.source !== 'musique' || JSON.stringify(record.source_ids) !== JSON.stringify([MINIMUM_AGE_SOURCE_ID]) ||
      record.external_source?.snapshot_sha256 !== SNAPSHOT_SHA256 || record.semantics?.expected !== PRIMARY ||
      record.semantics?.files?.[record.semantics?.root] !== MINIMUM_AGE_TASK_PROMPT ||
      !exactOracle(record.semantics?.oracle, [...MINIMUM_AGE_ALIASES]) ||
      record.generation?.oracle_review?.rejected !== 'Gun laws in Iowa' ||
      record.generation?.oracle_review?.source_snapshot_sha256 !== SNAPSHOT_SHA256 ||
      record.generation?.article_path_review?.registry !== (placement === 'historical-titled-migration' ?
        'musique-title-paths/2026-09-30-v2-r2' : undefined)) return false;
  for (const evidence of ARTICLE_EVIDENCE) {
    const currentPath = evidencePath(record, evidence);
    const body = currentPath && record.semantics?.folder_files?.[currentPath];
    const evidenceAudit = audit.evidence?.find(item => item.source_path === evidence.source_path);
    if (currentPath !== evidenceAudit?.path || typeof body !== 'string' || sha256(body) !== evidence.sha256 ||
        !body.includes(evidence.text) || evidenceAudit.sha256 !== evidence.sha256 || evidenceAudit.text !== evidence.text)
      return false;
  }
  if (audit.base_record_sha256 !== PINNED_BASE_RECORD_SHA256[baseId]) return false;
  const expected = expectedReview(record, { baseId, baseRecordSha256: audit.base_record_sha256, placement });
  if (JSON.stringify(audit) !== JSON.stringify(expected)) return false;

  const restored = structuredClone(record);
  restored.id = baseId;
  restored.semantics.oracle = { level: 'normalized', alternates: [] };
  delete restored.generation.minimum_age_equivalence_review;
  return validateBase(restored, baseId) && digest(restored) === PINNED_BASE_RECORD_SHA256[baseId];
}

/** Applies only to the removed-alias source record, before title remapping, or to its pinned historical title IR. */
export function applyMinimumAgeReviewedContract(record) {
  const sourceId = record?.external_source?.source_id ?? record?.source_ids?.[0];
  if (sourceId !== MINIMUM_AGE_SOURCE_ID) return record;
  if (record.generation?.minimum_age_equivalence_review !== undefined) {
    if (!validateMinimumAgeReviewedVariant(record)) fail('variant_drift');
    return record;
  }
  const placement = placementForId(record.id);
  if (!placement || !validateBase(record, record.id)) fail('source_or_prompt_drift');
  const result = structuredClone(record);
  const baseRecordSha256 = PINNED_BASE_RECORD_SHA256[record.id];
  result.id = `${record.id}${MINIMUM_AGE_CONTRACT_SUFFIX}`;
  result.semantics.oracle = { level: 'normalized', alternates: [...MINIMUM_AGE_ALIASES] };
  result.generation = { ...result.generation, minimum_age_equivalence_review:
    expectedReview(result, { baseId: record.id, baseRecordSha256, placement }) };
  if (!validateMinimumAgeReviewedVariant(result)) fail('variant_invalid');
  return result;
}
