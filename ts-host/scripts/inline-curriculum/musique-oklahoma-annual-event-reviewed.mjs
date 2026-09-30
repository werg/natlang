/** Production, exact-record-scoped oracle contract applied after source-group assembly. */
import { createHash } from 'node:crypto';

const sha256 = value => createHash('sha256').update(value).digest('hex');
const canonical = value => Array.isArray(value) ? `[${value.map(canonical).join(',')}]` :
  value && typeof value === 'object' ? `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(',')}}` : JSON.stringify(value);
const digest = value => sha256(canonical(value));

export const OKLAHOMA_EVENT_SOURCE_ID = '2hop__5371_5400';
export const OKLAHOMA_EVENT_BASE_IR_ID = 'inline-curriculum:source_musique:4467549fe691097dee77:v1';
export const OKLAHOMA_EVENT_VARIANT_SUFFIX = ':one-annual-event-answers-reviewed-v1';
export const OKLAHOMA_EVENT_VARIANT_IR_ID = `${OKLAHOMA_EVENT_BASE_IR_ID}${OKLAHOMA_EVENT_VARIANT_SUFFIX}`;
export const OKLAHOMA_EVENT_REVIEW_REVISION = 'musique-one-annual-event-answers/2026-09-30-reviewed-v1';
export const OKLAHOMA_EVENT_SNAPSHOT_SHA256 = 'cb4c55dfe2de2daa36a567771e862d8575ce4fa64e68631d7055b09351de28e5';
export const OKLAHOMA_EVENT_PRIMARY = 'Big 12 Baseball Tournament';
export const OKLAHOMA_EVENT_ALTERNATES = Object.freeze([
  'World Cup of Softball', "NCAA Women's College World Series",
]);
export const OKLAHOMA_EVENT_ACCEPTED = Object.freeze([
  OKLAHOMA_EVENT_PRIMARY, ...OKLAHOMA_EVENT_ALTERNATES,
]);
export const OKLAHOMA_EVENT_PROMPT = `---
args: {}
returns: "string"
kind: directory-reducer
---
What is one annual event that Oklahoma's largest urbanized area hosts?
Read the articles to answer. Return only the answer text. Preserve the workspace.
`;
const BASE_RECORD_SHA256 = 'b7398fac19602b559a54b3d095712e8d14623ade925a63f7d942e4ba3817f7e5';
const VARIANT_RECORD_SHA256 = '80068b92af8bdda7fd9ae3bc65346c046e375eac587f4c5ebbf738b5276c355f';
const SOURCE_ARCHIVE_SHA256 = '98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd';
const SOURCE_ARCHIVE_URL = 'https://drive.usercontent.google.com/download?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&export=download&confirm=t';
const ARTICLE_EVIDENCE = Object.freeze([
  Object.freeze({ path: 'articles/10.md', sha256: 'ca1746699f236dc4311b4b5f7133fb8e7c2651ccf25b34ad907ebff24d974a09',
    excerpt: "Oklahoma City is the annual host of the Big 12 Baseball Tournament, the World Cup of Softball, and the annual NCAA Women's College World Series." }),
  Object.freeze({ path: 'articles/16.md', sha256: 'e2bd9f3e68fbc2ccaacb714103a569c21f1a32ce45911f3c6638159fc0d60bc2',
    excerpt: "Oklahoma City is the principal city of the eight-county Oklahoma City Metropolitan Statistical Area in Central Oklahoma and is the state's largest urbanized area." }),
]);

function exactOracle(oracle, alternates) {
  return oracle && typeof oracle === 'object' && !Array.isArray(oracle) &&
    Object.keys(oracle).sort().join(',') === 'alternates,level' && oracle.level === 'normalized' &&
    JSON.stringify(oracle.alternates) === JSON.stringify(alternates);
}
function fail(reason) { throw new Error(`musique_oklahoma_event_review_${reason}:${OKLAHOMA_EVENT_SOURCE_ID}`); }
function sourceId(record) { return record?.external_source?.source_id ?? record?.source_ids?.[0]; }

function validateBase(record) {
  const sourceFile = record?.external_source?.files;
  return record?.id === OKLAHOMA_EVENT_BASE_IR_ID && digest(record) === BASE_RECORD_SHA256 &&
    record.source === 'musique' && JSON.stringify(record.source_ids) === JSON.stringify([OKLAHOMA_EVENT_SOURCE_ID]) &&
    record.external_source?.source === 'musique' && sourceId(record) === OKLAHOMA_EVENT_SOURCE_ID &&
    record.external_source?.revision === 'musique-v1.0' && record.external_source?.original_split === 'train' &&
    record.external_source?.snapshot_sha256 === OKLAHOMA_EVENT_SNAPSHOT_SHA256 &&
    record.external_source?.license === 'CC-BY-4.0' && record.license === 'CC-BY-4.0' &&
    JSON.stringify(sourceFile) === JSON.stringify([{ url: SOURCE_ARCHIVE_URL, sha256: SOURCE_ARCHIVE_SHA256, member: 'data/musique_full_v1.0_train.jsonl' }]) &&
    record.semantics?.expected === OKLAHOMA_EVENT_PRIMARY &&
    record.semantics?.files?.[record.semantics?.root] === OKLAHOMA_EVENT_PROMPT &&
    exactOracle(record.semantics?.oracle, []) &&
    record.generation?.oklahoma_annual_event_review === undefined &&
    record.external_source?.oklahoma_annual_event_review === undefined &&
    ARTICLE_EVIDENCE.every(item => typeof record.semantics?.folder_files?.[item.path] === 'string' &&
      sha256(record.semantics.folder_files[item.path]) === item.sha256 &&
      record.semantics.folder_files[item.path].includes(item.excerpt));
}

function reviewMetadata(baseId, placement = 'after-oracle-alias-and-source-group-assembly') {
  return {
    registry: OKLAHOMA_EVENT_REVIEW_REVISION,
    source_id: OKLAHOMA_EVENT_SOURCE_ID,
    base_ir_id: baseId,
    variant_ir_id: OKLAHOMA_EVENT_VARIANT_IR_ID,
    source_snapshot_sha256: OKLAHOMA_EVENT_SNAPSHOT_SHA256,
    base_record_sha256: BASE_RECORD_SHA256,
    placement,
    task_prompt: OKLAHOMA_EVENT_PROMPT,
    primary: OKLAHOMA_EVENT_PRIMARY,
    accepted: [...OKLAHOMA_EVENT_ACCEPTED],
    rationale: 'The prompt asks for one annual event. The exact Oklahoma City source sentence lists the Big 12 Baseball Tournament, the World Cup of Softball, and the NCAA Women’s College World Series as annual events. Preserve the source gold and accept these three source-backed answers only.',
    evidence: ARTICLE_EVIDENCE.map(item => ({ ...item })),
  };
}

/** True only for the fully pinned reviewed IR; the original source row stays held. */
export function isReviewedOklahomaAnnualEventVariant(record) {
  if (sourceId(record) !== OKLAHOMA_EVENT_SOURCE_ID || record?.id !== OKLAHOMA_EVENT_VARIANT_IR_ID ||
      digest(record) !== VARIANT_RECORD_SHA256 || record.source !== 'musique' ||
      JSON.stringify(record.source_ids) !== JSON.stringify([OKLAHOMA_EVENT_SOURCE_ID]) ||
      record.external_source?.source !== 'musique' || record.external_source?.source_id !== OKLAHOMA_EVENT_SOURCE_ID ||
      record.external_source?.snapshot_sha256 !== OKLAHOMA_EVENT_SNAPSHOT_SHA256 ||
      record.semantics?.expected !== OKLAHOMA_EVENT_PRIMARY ||
      record.semantics?.files?.[record.semantics?.root] !== OKLAHOMA_EVENT_PROMPT ||
      !exactOracle(record.semantics?.oracle, [...OKLAHOMA_EVENT_ALTERNATES])) return false;
  const audit = record.generation?.oklahoma_annual_event_review;
  const externalAudit = record.external_source?.oklahoma_annual_event_review;
  const expected = reviewMetadata(OKLAHOMA_EVENT_BASE_IR_ID);
  if (JSON.stringify(audit) !== JSON.stringify(expected) || JSON.stringify(externalAudit) !== JSON.stringify(expected)) return false;
  return ARTICLE_EVIDENCE.every(item => typeof record.semantics?.folder_files?.[item.path] === 'string' &&
    sha256(record.semantics.folder_files[item.path]) === item.sha256 &&
    record.semantics.folder_files[item.path].includes(item.excerpt));
}

/** Transform only the exact original IR, after its groups and any reviewed alias adapters. */
export function applyReviewedOklahomaAnnualEventContract(record) {
  if (sourceId(record) !== OKLAHOMA_EVENT_SOURCE_ID) return record;
  if (record?.id === OKLAHOMA_EVENT_VARIANT_IR_ID ||
      record?.generation?.oklahoma_annual_event_review !== undefined ||
      record?.external_source?.oklahoma_annual_event_review !== undefined) {
    if (!isReviewedOklahomaAnnualEventVariant(record)) fail('variant_drift');
    return record;
  }
  if (!validateBase(record)) fail('base_source_or_record_drift');
  const result = structuredClone(record);
  const audit = reviewMetadata(record.id);
  result.id = OKLAHOMA_EVENT_VARIANT_IR_ID;
  result.semantics.oracle = { level: 'normalized', alternates: [...OKLAHOMA_EVENT_ALTERNATES] };
  result.generation = { ...result.generation, oklahoma_annual_event_review: audit };
  result.external_source = { ...result.external_source, oklahoma_annual_event_review: audit };
  if (digest(result) !== VARIANT_RECORD_SHA256 || !isReviewedOklahomaAnnualEventVariant(result))
    fail('variant_invalid');
  return result;
}
