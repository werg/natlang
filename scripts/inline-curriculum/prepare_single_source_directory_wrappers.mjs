#!/usr/bin/env node
/** Build balanced, one-question source-backed directory tasks from standalone workflow IR. */
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { standaloneComponentProblem, buildQuestionBatchPayload } from '../workflow_directory_composite_factory.mjs';

const [sourcePath, splitPath, protectedPath, outArg, limitArg] = process.argv.slice(2);
if (!sourcePath || !splitPath || !protectedPath || !outArg)
  throw Error('usage: prepare_single_source_directory_wrappers.mjs SOURCE.ir.jsonl SPLITS.json PROTECTED_ALIAS_SCAN.json OUT_DIR [per_family_limit=256]');
const limit = limitArg === undefined ? 256 : Number(limitArg);
if (!Number.isInteger(limit) || limit < 1) throw Error('per_family_limit must be a positive integer');
const out = path.resolve(outArg);
const families = [
  ['workflow_invoice-processing', 'invoice'],
  ['workflow_security-incidents', 'security'],
  ['workflow_agent-trace-observability', 'agent'],
  ['workflow_customer-service', 'customer_service'],
];
const sha = value => createHash('sha256').update(value).digest('hex');
const sourceBytes = await readFile(path.resolve(sourcePath));
const splitBytes = await readFile(path.resolve(splitPath));
const protectedBytes = await readFile(path.resolve(protectedPath));
const rows = sourceBytes.toString('utf8').split('\n').filter(Boolean).map(JSON.parse);
const split = JSON.parse(splitBytes.toString('utf8'));
const protectedReceipt = JSON.parse(protectedBytes.toString('utf8'));
if (protectedReceipt.status !== 'passed' || !Array.isArray(protectedReceipt.protected_identity_alias_values) ||
    !Array.isArray(protectedReceipt.protected_source_alias_values) || !Array.isArray(protectedReceipt.protected_program_alias_values) ||
    !Array.isArray(protectedReceipt.protected_group_alias_values))
  throw Error('protected alias input must be a passed full-v13-test-alias-closure receipt with explicit alias arrays');
const protectedAliases = new Set([
  ...protectedReceipt.protected_identity_alias_values,
  ...protectedReceipt.protected_source_alias_values,
  ...protectedReceipt.protected_program_alias_values,
  ...protectedReceipt.protected_group_alias_values,
]);
const trainGroups = new Set(Object.entries(split.groups ?? {}).filter(([, value]) => value === 'train').map(([group]) => group));
const exclusions = { non_target_family: 0, nested_or_nonstandalone: 0, missing_group_or_split: 0, protected_alias_hit: 0, unsupported_return_type: 0 };
const grouped = new Map(families.map(([family]) => [family, new Map()]));

function decodeYamlScalar(value) {
  const s = value.trim();
  if (s.startsWith('"')) return JSON.parse(s);
  if (s.startsWith("'")) {
    if (!s.endsWith("'")) throw Error(`unterminated YAML single-quoted scalar: ${s}`);
    return s.slice(1, -1).replaceAll("''", "'");
  }
  return s;
}

function getReturnType(ir) {
  const file = ir.semantics.files[ir.semantics.root];
  const match = file.match(/^returns:\s*(.*)$/m);
  if (!match) return null;
  const expr = decodeYamlScalar(match[1]);
  // Restrict the wrapper frontmatter to safe scalar/enum type expressions. In
  // particular, don't guess the fields of Record<string, unknown> or a record.
  const atom = '(?:boolean|string|number|unknown|"[^"\\n]+"|\'[^\'\\n]+\')';
  if (!new RegExp(`^\\s*${atom}(?:\\s*\\|\\s*${atom})*\\s*$`).test(expr)) return null;
  return expr;
}

function lineageAliases(ir) {
  return [ir.id, ...(ir.source_ids ?? []), ...(ir.source_groups ?? []), ...(ir.generation?.composite_component_ids ?? [])]
    .filter(value => typeof value === 'string');
}

for (const ir of rows) {
  const family = ir.curriculum?.family;
  if (!grouped.has(family)) { exclusions.non_target_family++; continue; }
  if (standaloneComponentProblem(ir)) { exclusions.nested_or_nonstandalone++; continue; }
  const componentGroups = [...new Set(ir.source_groups ?? [])];
  if (!componentGroups.length || !componentGroups.every(g => trainGroups.has(g))) { exclusions.missing_group_or_split++; continue; }
  const group = componentGroups[0];
  if (lineageAliases(ir).some(alias => protectedAliases.has(alias))) { exclusions.protected_alias_hit++; continue; }
  const returnType = getReturnType(ir);
  if (!returnType) { exclusions.unsupported_return_type++; continue; }
  const bucket = grouped.get(family);
  if (!bucket.has(group)) bucket.set(group, []);
  bucket.get(group).push({ ir, returnType });
}

function roundRobin(groups) {
  const ordered = [...groups.entries()].sort(([a], [b]) => a.localeCompare(b));
  for (const [, items] of ordered) items.sort((a, b) => a.ir.id.localeCompare(b.ir.id));
  const output = [];
  for (let offset = 0; ; offset++) {
    let found = false;
    for (const [group, items] of ordered) {
      if (items[offset]) { output.push({ ...items[offset], group }); found = true; }
    }
    if (!found) return output;
  }
}

const availableByFamily = Object.fromEntries(families.map(([family, short]) => [short, roundRobin(grouped.get(family))]));
const balancedCount = Math.min(limit, ...Object.values(availableByFamily).map(items => items.length));
if (balancedCount < 1) throw Error('no balanced source-safe cases available');
const picked = [];
for (const [family, short] of families) {
  const items = availableByFamily[short].slice(0, balancedCount);
  for (const { ir, returnType, group } of items) {
    const sourceId = ir.source_ids?.[0] ?? ir.id;
    const key = sourceId.split(':').at(-1).replace(/[^A-Za-z0-9_-]/g, '_').slice(-72);
    const payload = buildQuestionBatchPayload([ir], [key]);
    const instruction = `Read jobs/${key}.json. Answer the source question in that file using its own state and criteria. Return one direct record with exactly the filename-stem key ${key}, using its declared return type (${returnType}). Preserve the source file.`;
    const recordType = `{ ${JSON.stringify(key)}: ${returnType} }`.replaceAll("'", "''");
    const rootText = `---\nargs: {}\nreturns: '${recordType}'\nkind: directory-reducer\n---\n${instruction}\n`;
    const expected = payload.expected;
    const expectedValue = Object.values(expected)[0];
    if (JSON.stringify(expectedValue) !== JSON.stringify(ir.semantics.expected)) throw Error(`gold changed for ${ir.id}`);
    const exactQuestion = JSON.parse(payload.folder_files[`jobs/${key}.json`]).question === ir.semantics.files[ir.semantics.root];
    const exactState = JSON.stringify(JSON.parse(payload.folder_files[`jobs/${key}.json`]).state) === JSON.stringify(ir.semantics.inputs.state);
    if (!exactQuestion || !exactState) throw Error(`source question/state changed for ${ir.id}`);
    const allowedSlices = new Set(['inline_placement','observation_followup','nested_scoped','iterate','folder_failure']);
    if (!allowedSlices.has(ir.curriculum?.slice)) throw Error(`source curriculum has unsupported slice ${ir.curriculum?.slice}: ${ir.id}`);
    const id = `inline-curriculum:${family}:${sha(`${group}\0${ir.id}\0single-source-v4`).slice(0, 24)}:directory-one-source-v4`;
    const ext = structuredClone(ir.external_source ?? {});
    ext.adaptation = 'one-source workflow task wrapper v4; exact source question/state and gold copied from immutable quality-v4 standalone component; primitive and enum return type preserved';
    const wrapper = {
      version: ir.version, id, kind: 'lambda_source', family: ir.family, source: ir.source, split: 'train',
      source_ids: structuredClone(ir.source_ids ?? []), source_groups: structuredClone(ir.source_groups ?? []),
      source_revisions: structuredClone(ir.source_revisions ?? []), license: ir.license ?? null,
      gold_sources: structuredClone(ir.gold_sources ?? []),
      generation: { ...structuredClone(ir.generation ?? {}), adapter_revision: 'source-scenario-directory-one-source-v4', parent_program_id: ir.id },
      curriculum: { ...structuredClone(ir.curriculum ?? {}), shape: `one-source-${short}`, variant: 'directory-one-source-v4', split_group: group,
        mode: 'single_call', reference: { root: [
          ...Object.keys(payload.folder_files).map(file => ['read_file', { path: file }]),
          ['return_result', { status: 'success', value: expected }],
        ] } },
      semantics: { root: 'review_jobs.nl', files: { 'review_jobs.nl': rootText }, inputs: {}, expected,
        folder_files: payload.folder_files, expected_files: structuredClone(payload.folder_files) },
      external_source: ext,
    };
    picked.push({ wrapper, ir, group, returnType, key, short });
  }
}
const ids = new Set(picked.map(x => x.wrapper.id));
if (ids.size !== picked.length) throw Error('duplicate generated program IDs');
const fileText = picked.map(x => JSON.stringify(x.wrapper)).join('\n') + '\n';
const sourceProof = picked.map((x, index) => JSON.stringify({ index, id: x.wrapper.id, parent_program_id: x.ir.id,
  program_sha256: sha(JSON.stringify(x.wrapper)), source_ids: x.wrapper.source_ids, source_groups: x.wrapper.source_groups,
  source_revisions: x.wrapper.source_revisions, license: x.wrapper.license, family: x.short, return_type: x.returnType,
  question_file: `jobs/${x.key}.json`, question_file_sha256: sha(x.wrapper.semantics.folder_files[`jobs/${x.key}.json`]),
  source_question_exact: true, source_state_exact: true, gold_exact: true, source_group_v13_train: trainGroups.has(x.group),
  fresh_source_claim: false, alternate_single_source_wrapper: true })).join('\n') + '\n';
const familyCounts = Object.fromEntries(families.map(([, short]) => [short, picked.filter(x => x.short === short).length]));
const manifest = { schema: 'workflow.one-source-directory-wrapper-selection/2', status: 'prepared_for_review_no_provider_launch',
  source_ir_path: path.resolve(sourcePath), source_ir_sha256: sha(sourceBytes), source_rows: rows.length,
  split_path: path.resolve(splitPath), split_sha256: sha(splitBytes), protected_alias_path: path.resolve(protectedPath),
  protected_alias_sha256: sha(protectedBytes), protected_test_rows: protectedReceipt.protected_test_rows_scanned,
  protected_identity_alias_count: protectedAliases.size, requested_per_family: limit, balanced_per_family_selected: balancedCount,
  available_standalone_source_counts: Object.fromEntries(Object.entries(availableByFamily).map(([k, v]) => [k, v.length])),
  selected_count: picked.length, selected_family_counts: familyCounts, selected_distinct_source_ids: new Set(picked.map(x => x.ir.source_ids?.[0] ?? x.ir.id)).size,
  selected_distinct_source_groups: new Set(picked.map(x => x.group)).size, all_selected_groups_train: picked.every(x => trainGroups.has(x.group)),
  selected_protected_alias_overlap_count: picked.flatMap(x => lineageAliases(x.wrapper)).filter(a => protectedAliases.has(a)).length,
  exclusions, output_ir_path: path.join(out, 'cases.ir.jsonl'), output_ir_sha256: sha(fileText), output_ir_bytes: Buffer.byteLength(fileText),
  selection_proof_path: path.join(out, 'selection-proof.jsonl'), selection_proof_sha256: sha(sourceProof),
  provider_calls: 0, model_inference_calls: 0, launch_authorized: false,
  note: 'Source task components may have prior Qwen exposure. This recipe preserves every component question/state/gold and creates new one-source directory task payloads preserving the source curriculum slice; it makes no fresh-source claim.' };
await mkdir(out, { recursive: true });
await writeFile(path.join(out, 'cases.ir.jsonl'), fileText, { flag: 'wx' });
await writeFile(path.join(out, 'selection-proof.jsonl'), sourceProof, { flag: 'wx' });
await writeFile(path.join(out, 'selection-manifest.json'), JSON.stringify(manifest, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify({ out, selected: picked.length, familyCounts, available: manifest.available_standalone_source_counts,
  groups: manifest.selected_distinct_source_groups, protectedOverlap: manifest.selected_protected_alias_overlap_count,
  irSha256: manifest.output_ir_sha256 }, null, 2));
