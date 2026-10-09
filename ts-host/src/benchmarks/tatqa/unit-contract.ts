import { createHash } from 'node:crypto';

/** Reviewed, single-row representation for the source's explicit lakh unit. */
export const TATQA_LAKH_SOURCE_ID = 'd3a5439d-5041-4856-8c2c-434203b8acaa';
export const TATQA_LAKH_CONTRACT_REVISION = 'tatqa-source-unit-lakh-v1';
export const TATQA_LAKH_VARIANT_SUFFIX = ':source-unit-contract-v1';
export const TATQA_LAKH_BASE_ID = 'inline-curriculum:source_tatqa:0d49906c1c87b05df0d5:v1:evidence-scale-v2:numeric-answer-v1';
export const TATQA_LAKH_VARIANT_ID = `${TATQA_LAKH_BASE_ID}${TATQA_LAKH_VARIANT_SUFFIX}`;
export const TATQA_LAKH_EXPECTED = '{"answer":"242.5","scale":""}';
export const TATQA_LAKH_SOURCE_REVISION = '870accc41953dcde885aabeb963d94aabdc0fbc3';
export const TATQA_LAKH_SNAPSHOT_SHA256 = '9404a53bb8f0088e58e9113ed41f6e42cd9d07dd5d0e757bd2dd650fd07bbdfa';
export const TATQA_LAKH_SOURCE_FILE_SHA256 = '2df6e722cdbaaa37efcbfb280f5c9a15be29a6ec18f618ef936fe63cc6d07c69';
export const TATQA_LAKH_SOURCE_FILE_URL = 'https://raw.githubusercontent.com/NExTplusplus/TAT-QA/870accc41953dcde885aabeb963d94aabdc0fbc3/dataset_raw/tatqa_dataset_train.json';
export const TATQA_LAKH_FOLDER_FILES_SHA256 = '11a6fdec0e9817a3a2f2d6311d7474d9a97fa7d4aa86f82c7752f4e69158c6ad';
export const TATQA_LAKH_REFERENCE_ROOT_SHA256 = '5ac616dbf8eb9e17b11aa0cbed9b7658c0e52450907dbefd4083a6eee31144e0';
export const TATQA_LAKH_SOURCE_GROUP = 'tatqa:context:b74eaf04-3737-4bdc-ad03-c33cac13ebee';
export const TATQA_LAKH_QUESTION = 'What is the total gross salary of the CEO and MD?';
export const TATQA_LAKH_BASE_PROMPT = `---
args: {}
returns: "string"
kind: directory-reducer
---
${TATQA_LAKH_QUESTION}
Use table.json and notes/. Write answer.json and return the same JSON string with exactly answer (a string; multiple spans separated by "; ") and scale ("", "percent", "thousand", "million" or "billion"). Use a nonempty scale only when the question or source evidence establishes it for the requested quantity. Use empty scale for dimensionless quantities or when no scale is stated. Do not infer scale from financial-report conventions or unrelated table rows.
For numeric answers, put only the numeric value in answer (no currency symbol, percent sign, units, or explanatory words); put the source-supported unit only in scale. You may round numeric answers to two decimal places when needed. Preserve the sign and use the scale established by the question or source evidence.
Preserve source files.
`;
export const TATQA_LAKH_REPLACEMENT_PROMPT = `---
args: {}
returns: "string"
kind: directory-reducer
---
${TATQA_LAKH_QUESTION}
Use table.json and notes/. Write answer.json and return the same JSON string with exactly answer (a string; multiple spans separated by "; ") and scale ("", "percent", "thousand", "million" or "billion").
The table's monetary values are stated in lakh. Preserve that source unit as displayed; do not convert it or put it in scale. Use a numeric-only answer and leave scale empty.
Preserve source files.
`;

type JsonRecord = Record<string, any>;
const isObject = (value: unknown): value is JsonRecord => !!value && typeof value === 'object' && !Array.isArray(value);
const hash = (value: string) => createHash('sha256').update(value).digest('hex');
const folderDigest = (files: unknown): string | null => {
  if (!isObject(files) || Object.values(files).some(value => typeof value !== 'string')) return null;
  const canonical = Object.entries(files).sort(([a], [b]) => a.localeCompare(b));
  return hash(JSON.stringify(canonical));
};
function expectedFilesMatch(semantics: JsonRecord): boolean {
  const expected = semantics.expected_files;
  const source = semantics.folder_files;
  if (!isObject(expected) || !isObject(source)) return false;
  const entries = Object.entries(source);
  if (Object.keys(expected).length !== entries.length + 1) return false;
  return entries.every(([path, contents]) => expected[path] === contents) &&
    expected['answer.json'] === `${TATQA_LAKH_EXPECTED}\n`;
}

function sharedIdentityMatches(record: unknown): record is JsonRecord {
  if (!isObject(record)) return false;
  const semantics = record.semantics;
  const external = record.external_source;
  const expectedFiles = semantics?.expected_files;
  const sourceFile = Array.isArray(external?.files) ? external.files : [];
  return record.source === 'tatqa' && Array.isArray(record.source_ids) &&
    record.source_ids.length === 1 && record.source_ids[0] === TATQA_LAKH_SOURCE_ID &&
    Array.isArray(record.source_groups) && record.source_groups.length === 1 && record.source_groups[0] === TATQA_LAKH_SOURCE_GROUP &&
    Array.isArray(record.source_revisions) && record.source_revisions.length === 1 && record.source_revisions[0] === TATQA_LAKH_SOURCE_REVISION &&
    record.split === 'train' && record.license === 'CC-BY-4.0' &&
    record.task_modality === 'directory-reducer' &&
    Array.isArray(record.gold_sources) && record.gold_sources.length === 2 &&
    record.gold_sources[0] === 'tatqa:original-gold' && record.gold_sources[1] === 'native-file-replay' &&
    hash(JSON.stringify(record.curriculum?.reference?.root)) === TATQA_LAKH_REFERENCE_ROOT_SHA256 &&
    semantics?.root === 'process_workspace.nl' && semantics?.expected === TATQA_LAKH_EXPECTED &&
    isObject(semantics.files) && Object.keys(semantics.files).length === 1 &&
    typeof semantics.files['process_workspace.nl'] === 'string' &&
    folderDigest(semantics?.folder_files) === TATQA_LAKH_FOLDER_FILES_SHA256 &&
    isObject(expectedFiles) && expectedFilesMatch(semantics) &&
    external?.source === 'tatqa' && external?.source_id === TATQA_LAKH_SOURCE_ID &&
    external?.original_split === 'train' && external?.license === 'CC-BY-4.0' &&
    external?.revision === TATQA_LAKH_SOURCE_REVISION && external?.snapshot_sha256 === TATQA_LAKH_SNAPSHOT_SHA256 &&
    sourceFile.length === 1 && sourceFile[0]?.url === TATQA_LAKH_SOURCE_FILE_URL && sourceFile[0]?.sha256 === TATQA_LAKH_SOURCE_FILE_SHA256 &&
    isObject(semantics?.oracle) && semantics.oracle.level === 'normalized' &&
    semantics.oracle.normalization === 'tatqa-answer-record' && semantics.oracle.alternates === undefined &&
    isObject(semantics?.files_oracle) && semantics.files_oracle.compare === 'tatqa-answer-record' &&
    semantics.files_oracle.threshold === 1;
}

/** Exact precondition for adding the reviewed variant to the original numeric row. */
export function validateTatqaLakhBase(record: unknown): record is JsonRecord {
  if (!sharedIdentityMatches(record) || record.id !== TATQA_LAKH_BASE_ID) return false;
  const prompt = record.semantics.files?.[record.semantics.root];
  return prompt === TATQA_LAKH_BASE_PROMPT &&
    record.generation?.task_contract_revision === 'tatqa-evidence-scale-v2' &&
    record.external_source?.task_contract_revision === 'tatqa-evidence-scale-v2' &&
    record.generation?.numeric_answer_contract_revision === 'tatqa-numeric-answer-v1' &&
    record.external_source?.numeric_answer_contract_revision === 'tatqa-numeric-answer-v1';
}

/** True only for the immutable, approved visible-prompt variant (for a narrow hold exception). */
export function isReviewedTatqaLakhVariant(record: unknown): record is JsonRecord {
  if (!sharedIdentityMatches(record) || record.id !== TATQA_LAKH_VARIANT_ID) return false;
  const prompt = record.semantics.files?.[record.semantics.root];
  return prompt === TATQA_LAKH_REPLACEMENT_PROMPT &&
    record.generation?.task_contract_revision === 'tatqa-evidence-scale-v2' &&
    record.external_source?.task_contract_revision === 'tatqa-evidence-scale-v2' &&
    record.generation?.numeric_answer_contract_revision === 'tatqa-numeric-answer-v1' &&
    record.external_source?.numeric_answer_contract_revision === 'tatqa-numeric-answer-v1' &&
    record.generation?.source_unit_contract_revision === TATQA_LAKH_CONTRACT_REVISION &&
    record.external_source?.source_unit_contract_revision === TATQA_LAKH_CONTRACT_REVISION &&
    record.generation?.source_unit_contract_source_id === TATQA_LAKH_SOURCE_ID &&
    record.external_source?.source_unit_contract_source_id === TATQA_LAKH_SOURCE_ID;
}
