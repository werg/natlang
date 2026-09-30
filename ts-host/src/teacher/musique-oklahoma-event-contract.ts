import { createHash } from 'node:crypto';

/** Exact full-record validator for the single reviewed three-answer MuSiQue variant. */
export const OKLAHOMA_EVENT_SOURCE_ID = '2hop__5371_5400';
export const OKLAHOMA_EVENT_VARIANT_IR_ID =
  'inline-curriculum:source_musique:4467549fe691097dee77:v1:one-annual-event-answers-reviewed-v1';
export const OKLAHOMA_EVENT_VARIANT_RECORD_SHA256 =
  '80068b92af8bdda7fd9ae3bc65346c046e375eac587f4c5ebbf738b5276c355f';
export const OKLAHOMA_EVENT_REVIEW_REVISION = 'musique-one-annual-event-answers/2026-09-30-reviewed-v1';
export const OKLAHOMA_EVENT_SNAPSHOT_SHA256 = 'cb4c55dfe2de2daa36a567771e862d8575ce4fa64e68631d7055b09351de28e5';
export const OKLAHOMA_EVENT_PRIMARY = 'Big 12 Baseball Tournament';
export const OKLAHOMA_EVENT_ACCEPTED = Object.freeze([
  'Big 12 Baseball Tournament',
  'World Cup of Softball',
  "NCAA Women's College World Series",
]);
export const OKLAHOMA_EVENT_PROMPT = `---
args: {}
returns: "string"
kind: directory-reducer
---
What is one annual event that Oklahoma's largest urbanized area hosts?
Read the articles to answer. Return only the answer text. Preserve the workspace.
`;

type JsonRecord = Record<string, any>;
const isObject = (value: unknown): value is JsonRecord => !!value && typeof value === 'object' && !Array.isArray(value);
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value as JsonRecord).sort(([a], [b]) =>
    a.localeCompare(b)).map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(',')}}`;
  return JSON.stringify(value) ?? 'null';
}
const digest = (value: unknown) => createHash('sha256').update(canonical(value)).digest('hex');
const textSha = (value: string) => createHash('sha256').update(value).digest('hex');

const evidence = [
  { path: 'articles/10.md', sha256: 'ca1746699f236dc4311b4b5f7133fb8e7c2651ccf25b34ad907ebff24d974a09',
    excerpt: "Oklahoma City is the annual host of the Big 12 Baseball Tournament, the World Cup of Softball, and the annual NCAA Women's College World Series." },
  { path: 'articles/16.md', sha256: 'e2bd9f3e68fbc2ccaacb714103a569c21f1a32ce45911f3c6638159fc0d60bc2',
    excerpt: "Oklahoma City is the principal city of the eight-county Oklahoma City Metropolitan Statistical Area in Central Oklahoma and is the state's largest urbanized area." },
];

/** Original source review remains pending; only this complete revised record is exempted. */
export function isReviewedOklahomaAnnualEventVariant(record: unknown): record is JsonRecord {
  if (!isObject(record) || digest(record) !== OKLAHOMA_EVENT_VARIANT_RECORD_SHA256 ||
      record.id !== OKLAHOMA_EVENT_VARIANT_IR_ID || record.source !== 'musique' ||
      JSON.stringify(record.source_ids) !== JSON.stringify([OKLAHOMA_EVENT_SOURCE_ID])) return false;
  const external = record.external_source;
  const semantics = record.semantics;
  if (!isObject(external) || external.source !== 'musique' || external.source_id !== OKLAHOMA_EVENT_SOURCE_ID ||
      external.revision !== 'musique-v1.0' || external.original_split !== 'train' ||
      external.snapshot_sha256 !== OKLAHOMA_EVENT_SNAPSHOT_SHA256 ||
      semantics?.root !== 'process_workspace.nl' || semantics.expected !== OKLAHOMA_EVENT_PRIMARY ||
      semantics.files?.[semantics.root] !== OKLAHOMA_EVENT_PROMPT) return false;
  const oracle = semantics.oracle;
  if (!isObject(oracle) || Object.keys(oracle).sort().join(',') !== 'alternates,level' ||
      oracle.level !== 'normalized' || !Array.isArray(oracle.alternates) ||
      JSON.stringify([OKLAHOMA_EVENT_PRIMARY, ...oracle.alternates]) !== JSON.stringify(OKLAHOMA_EVENT_ACCEPTED) ||
      JSON.stringify(oracle.alternates) !== JSON.stringify(OKLAHOMA_EVENT_ACCEPTED.slice(1))) return false;
  const generationAudit = record.generation?.oklahoma_annual_event_review;
  const externalAudit = external.oklahoma_annual_event_review;
  if (!isObject(generationAudit) || JSON.stringify(generationAudit) !== JSON.stringify(externalAudit) ||
      generationAudit.registry !== OKLAHOMA_EVENT_REVIEW_REVISION ||
      generationAudit.source_id !== OKLAHOMA_EVENT_SOURCE_ID ||
      generationAudit.variant_ir_id !== OKLAHOMA_EVENT_VARIANT_IR_ID ||
      generationAudit.source_snapshot_sha256 !== OKLAHOMA_EVENT_SNAPSHOT_SHA256 ||
      JSON.stringify(generationAudit.accepted) !== JSON.stringify(OKLAHOMA_EVENT_ACCEPTED)) return false;
  return evidence.every(item => typeof semantics.folder_files?.[item.path] === 'string' &&
    textSha(semantics.folder_files[item.path]) === item.sha256 &&
    semantics.folder_files[item.path].includes(item.excerpt));
}
