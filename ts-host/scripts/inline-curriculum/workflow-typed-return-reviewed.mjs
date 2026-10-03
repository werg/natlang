/** Reviewed migration for legacy WorkflowEvals reducer cases with unknown-valued records. */
import { createHash } from 'node:crypto';

export const WORKFLOW_TYPED_RETURN_VERSION = 'workflow-typed-returns-v1';
const ALLOWED_FAMILIES = new Set([
  'workflow_customer-service',
  'workflow_agent-trace-observability',
]);

const sha = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const tsString = value => JSON.stringify(String(value));

function typeFor(question) {
  const criteria = question?.criteria;
  if (question.type === 'noul') return 'boolean';
  if (!criteria || typeof criteria !== 'object') throw new Error('missing_question_criteria');
  const keys = Array.isArray(criteria) ? criteria.map((_, index) => String(index)) : Object.keys(criteria);
  if (!keys.length) throw new Error('empty_question_criteria');
  if (question.type === 'choice') {
    if (Array.isArray(criteria)) throw new Error('choice_criteria_must_be_keyed');
    return [...new Set(keys)].map(tsString).join(' | ');
  }
  if (question.type === 'score') return keys.map((_, index) => tsString(String(index))).join(' | ');
  throw new Error(`unsupported_question_type:${question.type}`);
}

/**
 * Version a legacy reducer row to a question-derived result record. Only the
 * declared return type, task wording and lineage metadata change; visible files,
 * expected values, source identities, groups, references and licensing remain.
 */
export function applyReviewedWorkflowTypedReturns(record) {
  const nested = record?.task?.kind === 'whole_program' && record.task.program_ir?.kind === 'lambda_source';
  const ir = nested ? record.task.program_ir : record;
  if (ir?.kind !== 'lambda_source') return record;
  if (!ALLOWED_FAMILIES.has(ir.curriculum?.family) || ir.curriculum?.slice !== 'folder_failure') return record;
  const semantics = ir.semantics;
  const rootPath = semantics?.root;
  const source = semantics?.files?.[rootPath];
  if (typeof source !== 'string' || !source.startsWith('---\n') || !semantics.folder_files || typeof semantics.folder_files !== 'object')
    throw new Error(`workflow_typed_return_missing_visible_inputs:${ir.id}`);
  if (!source.includes('returns: "Record<string, unknown>"')) return record;

  const fields = Object.entries(semantics.folder_files).map(([path, content]) => {
    const match = /^jobs\/([^/]+)\.json$/.exec(path);
    if (!match || typeof content !== 'string') throw new Error(`workflow_typed_return_unexpected_path:${path}`);
    const item = JSON.parse(content);
    if (typeof item.question?.type !== 'string') throw new Error(`workflow_typed_return_question_missing:${path}`);
    return { key: match[1], type: typeFor(item.question), questionType: item.question.type };
  }).sort((a, b) => a.key.localeCompare(b.key));
  if (!fields.length) throw new Error(`workflow_typed_return_empty_batch:${ir.id}`);
  if (!ir.semantics.expected || fields.some(field => !Object.hasOwn(ir.semantics.expected, field.key)))
    throw new Error(`workflow_typed_return_expected_keys_mismatch:${ir.id}`);
  if (Object.keys(ir.semantics.expected).length !== fields.length) throw new Error(`workflow_typed_return_expected_extra_keys:${ir.id}`);

  const returnType = `{ ${fields.map(field => `${tsString(field.key)}: ${field.type}`).join('; ')} }`;
  const anchor = 'Preserve all files.';
  if (!source.includes(anchor)) throw new Error(`workflow_typed_return_preserve_anchor_missing:${ir.id}`);
  const returnMarker = 'returns: "Record<string, unknown>"';
  if (source.split(returnMarker).length !== 2) throw new Error(`workflow_typed_return_signature_not_unique:${ir.id}`);
  const contract = `\n\nReturn a record with these exact keys and declared value types. A noul answer is a native boolean: write true or false without quotation marks. A choice or score answer is a string exactly matching its declared alternatives. Do not stringify booleans. Use eval to construct and return the record with native values; when a field needs semantic judgment, a typed await nl<boolean> call is one option, not a requirement. The runtime checks the complete typed record before it is accepted.\n`;
  const nextSource = source.replace('returns: "Record<string, unknown>"', `returns: ${JSON.stringify(returnType)}`)
    .replace(/(Preserve all files\.[\s\S]*?)(\n)/, `$1${contract}$2`);
  if (nextSource === source || nextSource.includes(returnMarker)) throw new Error(`workflow_typed_return_replacement_failed:${ir.id}`);

  const baseId = ir.id;
  const baseDigest = sha(ir);
  const next = structuredClone(record);
  const nextIr = nested ? next.task.program_ir : next;
  nextIr.id = `${baseId}:${WORKFLOW_TYPED_RETURN_VERSION}`;
  nextIr.curriculum.variant = `${nextIr.curriculum.variant ?? 'directory-v1'}:${WORKFLOW_TYPED_RETURN_VERSION}`;
  nextIr.semantics.files[rootPath] = nextSource;
  nextIr.generation ??= {};
  nextIr.generation.workflow_typed_return_review = {
    version: WORKFLOW_TYPED_RETURN_VERSION,
    base_program_id: baseId,
    base_ir_sha256: baseDigest,
    field_types_from_visible_question_schemas: fields,
    source_gold_reference_files_preserved: true,
  };
  return next;
}
