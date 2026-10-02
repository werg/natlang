import { createHash } from 'node:crypto';

/** Narrow prompt-only formatting clarification for two fully pinned TATQA proportion cases. */
export const PROPORTION_DISPLAY_REVISION = 'tatqa-proportion-display-v1';
export const PROPORTION_DISPLAY_SUFFIX = ':proportion-display-v1';
const pins = [
  {
    baseId: 'inline-curriculum:source_tatqa:ef6d7e96029e26385e63:v1:evidence-scale-v2:numeric-answer-v1',
    sourceId: '64a6c46c-a5cd-4065-bbc2-3156a4d50eda',
    sourceRevision: '870accc41953dcde885aabeb963d94aabdc0fbc3',
    sourceSnapshotSha256: '9404a53bb8f0088e58e9113ed41f6e42cd9d07dd5d0e757bd2dd650fd07bbdfa',
    sourceFileSha256: '2df6e722cdbaaa37efcbfb280f5c9a15be29a6ec18f618ef936fe63cc6d07c69',
    baseRecordSha256: '5360a16b7b0ce3c13c4bacc066557a7d1d81f47dae5fff7cb6f7281db5aba829',
    referenceValue: '{"answer":"0.72","scale":""}',
  },
  {
    baseId: 'inline-curriculum:source_tatqa:5cdfa3aba293babea2c6:v1:evidence-scale-v2:numeric-answer-v1',
    sourceId: '537a28dd-eee9-4fa9-8990-c18422160bed',
    sourceRevision: '870accc41953dcde885aabeb963d94aabdc0fbc3',
    sourceSnapshotSha256: '9404a53bb8f0088e58e9113ed41f6e42cd9d07dd5d0e757bd2dd650fd07bbdfa',
    sourceFileSha256: '2df6e722cdbaaa37efcbfb280f5c9a15be29a6ec18f618ef936fe63cc6d07c69',
    baseRecordSha256: '332db94d9961e9aa9eec04ae7cbf91c68132abc638dd5e18a7078dfb6276edea',
    referenceValue: '{"answer":"0.14","scale":""}',
  },
];

const instruction = 'For this requested share or proportion, express the dimensionless fraction as a decimal rounded to two decimal places; put only the numeric value in answer and leave scale empty. This formatting instruction applies only to the requested share or proportion in this task.';
const canonical = value => {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => `${JSON.stringify(k)}:${canonical(v)}`).join(',')}}`;
  return JSON.stringify(value);
};
const sha = value => createHash('sha256').update(value).digest('hex');
const clone = value => structuredClone(value);
const pinFor = record => pins.find(pin => record?.id === pin.baseId || record?.id === `${pin.baseId}${PROPORTION_DISPLAY_SUFFIX}`);

function validBase(record, pin) {
  const expectedPrompt = record?.semantics?.files?.[record?.semantics?.root];
  return !!record && record.id === pin.baseId && sha(canonical(record)) === pin.baseRecordSha256 &&
    record.source === 'tatqa' && record.split === 'train' && record.license === 'CC-BY-4.0' &&
    canonical(record.source_ids) === canonical([pin.sourceId]) && record.source_revisions?.length === 1 && record.source_revisions[0] === pin.sourceRevision &&
    record.external_source?.source_id === pin.sourceId && record.external_source?.revision === pin.sourceRevision &&
    record.external_source?.snapshot_sha256 === pin.sourceSnapshotSha256 && record.external_source?.files?.length === 1 &&
    record.external_source.files[0].sha256 === pin.sourceFileSha256 &&
    record.semantics?.expected === pin.referenceValue &&
    record.semantics?.expected_files?.['answer.json'] === `${pin.referenceValue}\n` &&
    typeof expectedPrompt === 'string' && expectedPrompt.endsWith('Preserve source files.\n');
}

function baseFromVariant(record, pin) {
  const base = clone(record);
  base.id = pin.baseId;
  for (const key of ['generation', 'external_source']) {
    if (!base[key] || typeof base[key] !== 'object') return null;
    delete base[key].proportion_display_contract_revision;
  }
  const prompt = base.semantics?.files?.[base.semantics?.root];
  if (typeof prompt !== 'string' || !prompt.endsWith(`\n${instruction}\n`)) return null;
  base.semantics.files[base.semantics.root] = prompt.slice(0, -`\n${instruction}\n`.length);
  return base;
}

export function validateTatqaProportionDisplayBase(record) {
  const pin = pinFor(record);
  return !!pin && validBase(record, pin);
}

export function isReviewedTatqaProportionDisplay(record) {
  const pin = pinFor(record);
  if (!pin || record.id !== `${pin.baseId}${PROPORTION_DISPLAY_SUFFIX}` ||
      record.generation?.proportion_display_contract_revision !== PROPORTION_DISPLAY_REVISION ||
      record.external_source?.proportion_display_contract_revision !== PROPORTION_DISPLAY_REVISION ||
      record.semantics?.files?.[record.semantics?.root]?.split(`\n${instruction}\n`).length !== 2) return false;
  const base = baseFromVariant(record, pin);
  if (!base || !validBase(base, pin)) return false;
  const expected = clone(base);
  expected.id = record.id;
  expected.semantics.files[expected.semantics.root] += `\n${instruction}\n`;
  expected.generation.proportion_display_contract_revision = PROPORTION_DISPLAY_REVISION;
  expected.external_source.proportion_display_contract_revision = PROPORTION_DISPLAY_REVISION;
  return canonical(expected) === canonical(record);
}

export function applyTatqaProportionDisplay(record) {
  const pin = pinFor(record);
  if (!pin) throw new Error(`tatqa_proportion_display_unregistered:${record?.id ?? 'unknown'}`);
  if (record.id === `${pin.baseId}${PROPORTION_DISPLAY_SUFFIX}`) {
    if (!isReviewedTatqaProportionDisplay(record)) throw new Error(`tatqa_proportion_display_variant_mismatch:${pin.sourceId}`);
    return record;
  }
  if (!validBase(record, pin)) throw new Error(`tatqa_proportion_display_base_mismatch:${pin.sourceId}`);
  const variant = clone(record);
  variant.id = `${pin.baseId}${PROPORTION_DISPLAY_SUFFIX}`;
  variant.semantics.files[variant.semantics.root] += `\n${instruction}\n`;
  variant.generation = { ...variant.generation, proportion_display_contract_revision: PROPORTION_DISPLAY_REVISION };
  variant.external_source = { ...variant.external_source, proportion_display_contract_revision: PROPORTION_DISPLAY_REVISION };
  if (!isReviewedTatqaProportionDisplay(variant)) throw new Error(`tatqa_proportion_display_output_mismatch:${pin.sourceId}`);
  return variant;
}

export function tatqaProportionDisplayPins() {
  return pins.map(({ baseId, sourceId, sourceRevision, sourceSnapshotSha256, sourceFileSha256, baseRecordSha256, referenceValue }) =>
    ({ baseId, sourceId, sourceRevision, sourceSnapshotSha256, sourceFileSha256, baseRecordSha256, referenceValue,
      variantId: `${baseId}${PROPORTION_DISPLAY_SUFFIX}` }));
}
