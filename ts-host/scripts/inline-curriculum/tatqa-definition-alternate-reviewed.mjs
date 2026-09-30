import { createHash } from 'node:crypto';

/** Exact, source-pinned record alternate for one reviewed TATQA definition. */
export const TATQA_DEFINITION_SOURCE_ID = '1252eeab-a1d0-443d-8776-78e254e34945';
export const TATQA_DEFINITION_CONTRACT_REVISION = 'tatqa-definition-source-alternate-v1';
export const TATQA_DEFINITION_SUFFIX = ':source-answer-alternate-v1';
export const TATQA_DEFINITION_BASE_ID = 'inline-curriculum:source_tatqa:82cf956e2bd1b48afbff:v1:evidence-scale-v2';
export const TATQA_DEFINITION_VARIANT_ID = `${TATQA_DEFINITION_BASE_ID}${TATQA_DEFINITION_SUFFIX}`;
export const TATQA_DEFINITION_BASE_ROW_SHA256 = '4320c5a05dea7d46ed0e36bc6839732b88e35976c58cf9ab3d499e0485a85846';
export const TATQA_DEFINITION_IR_SHA256 = '8de81e144764a8afd7761d48223aca489721113d6efc9234700b3ac35898456a';
export const TATQA_DEFINITION_RESULT_SHA256 = 'a23ff219d1d6232a4a68a57cd59061c0e11d02cb6573e1390530f37e941af3d0';
export const TATQA_DEFINITION_PRIMARY = '{"answer":"new store marketing allowance of $1,000 for each store added to our distribution network, as well as the non-capitalized freight costs associated with Freshpet Fridge replacements","scale":""}';
export const TATQA_DEFINITION_SOURCE_ANSWER = '{"answer":"Represents new store marketing allowance of $1,000 for each store added to our distribution network, as well as the non-capitalized freight costs associated with Freshpet Fridge replacements.","scale":""}';
const sourceRevision = '870accc41953dcde885aabeb963d94aabdc0fbc3';
const sourceSnapshot = '9404a53bb8f0088e58e9113ed41f6e42cd9d07dd5d0e757bd2dd650fd07bbdfa';
const sourceFile = '2df6e722cdbaaa37efcbfb280f5c9a15be29a6ec18f618ef936fe63cc6d07c69';
const sourceGroup = 'tatqa:context:d4dce5be-4bb8-46cb-9ca2-99dea4ba8f5c';
const question = 'What does launch expense represent?';

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
    .map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(',')}}`;
  return JSON.stringify(value);
}
const hash = value => createHash('sha256').update(value).digest('hex');
const isObject = value => !!value && typeof value === 'object' && !Array.isArray(value);
const clone = value => structuredClone(value);

function baseIdentityMatches(record) {
  return isObject(record) && record.id === TATQA_DEFINITION_BASE_ID &&
    record.source === 'tatqa' && record.split === 'train' && record.license === 'CC-BY-4.0' &&
    Array.isArray(record.source_ids) && record.source_ids.length === 1 && record.source_ids[0] === TATQA_DEFINITION_SOURCE_ID &&
    Array.isArray(record.source_groups) && record.source_groups.length === 1 && record.source_groups[0] === sourceGroup &&
    Array.isArray(record.source_revisions) && record.source_revisions.length === 1 && record.source_revisions[0] === sourceRevision &&
    record.semantics?.files?.['process_workspace.nl']?.includes(`${question}\n`) &&
    record.external_source?.snapshot_sha256 === sourceSnapshot &&
    record.external_source?.files?.length === 1 && record.external_source.files[0]?.sha256 === sourceFile;
}

function baseRecordHash(record) {
  return hash(canonical(record));
}

function revertVariant(record) {
  const base = clone(record);
  base.id = TATQA_DEFINITION_BASE_ID;
  for (const metadata of [base.generation, base.external_source]) {
    if (!isObject(metadata)) return null;
    delete metadata.source_answer_alternate_contract_revision;
    delete metadata.source_answer_alternate_source_id;
  }
  if (!isObject(base.semantics?.oracle) || !isObject(base.semantics?.files_oracle)) return null;
  delete base.semantics.oracle.alternates;
  delete base.semantics.files_oracle.alternates;
  return base;
}

/** Exact base check; the full canonical row digest pins question, files, gold, reference, and provenance. */
export function validateTatqaDefinitionAlternateBase(record) {
  return baseIdentityMatches(record) && baseRecordHash(record) === TATQA_DEFINITION_BASE_ROW_SHA256;
}

/** True only for the reviewed variant and the exact one-record alternate, with all other fields unchanged. */
export function isReviewedTatqaDefinitionAlternate(record) {
  if (!isObject(record) || record.id !== TATQA_DEFINITION_VARIANT_ID ||
      record.generation?.source_answer_alternate_contract_revision !== TATQA_DEFINITION_CONTRACT_REVISION ||
      record.external_source?.source_answer_alternate_contract_revision !== TATQA_DEFINITION_CONTRACT_REVISION ||
      record.generation?.source_answer_alternate_source_id !== TATQA_DEFINITION_SOURCE_ID ||
      record.external_source?.source_answer_alternate_source_id !== TATQA_DEFINITION_SOURCE_ID ||
      canonical(record.semantics?.oracle?.alternates) !== canonical([TATQA_DEFINITION_SOURCE_ANSWER]) ||
      canonical(record.semantics?.files_oracle?.alternates) !== canonical({ 'answer.json': [TATQA_DEFINITION_SOURCE_ANSWER] })) return false;
  const base = revertVariant(record);
  return !!base && baseIdentityMatches(base) && baseRecordHash(base) === TATQA_DEFINITION_BASE_ROW_SHA256;
}

/** Produce the reviewed exact-string variant without editing its primary gold or visible inputs. */
export function applyReviewedTatqaDefinitionAlternate(record) {
  if (record?.id === TATQA_DEFINITION_VARIANT_ID) {
    if (!isReviewedTatqaDefinitionAlternate(record))
      throw new Error(`tatqa_definition_alternate_variant_mismatch:${TATQA_DEFINITION_SOURCE_ID}`);
    return record;
  }
  if (!validateTatqaDefinitionAlternateBase(record))
    throw new Error(`tatqa_definition_alternate_base_mismatch:${record?.source_ids?.[0] ?? 'unknown'}`);
  const variant = clone(record);
  variant.id = TATQA_DEFINITION_VARIANT_ID;
  variant.semantics.oracle = { ...variant.semantics.oracle, alternates: [TATQA_DEFINITION_SOURCE_ANSWER] };
  variant.semantics.files_oracle = { ...variant.semantics.files_oracle,
    alternates: { 'answer.json': [TATQA_DEFINITION_SOURCE_ANSWER] } };
  variant.generation = { ...variant.generation,
    source_answer_alternate_contract_revision: TATQA_DEFINITION_CONTRACT_REVISION,
    source_answer_alternate_source_id: TATQA_DEFINITION_SOURCE_ID };
  variant.external_source = { ...variant.external_source,
    source_answer_alternate_contract_revision: TATQA_DEFINITION_CONTRACT_REVISION,
    source_answer_alternate_source_id: TATQA_DEFINITION_SOURCE_ID };
  if (!isReviewedTatqaDefinitionAlternate(variant))
    throw new Error(`tatqa_definition_alternate_output_mismatch:${TATQA_DEFINITION_SOURCE_ID}`);
  return variant;
}
