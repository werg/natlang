#!/usr/bin/env node
// Regrade stored edit_stream and contract_diagnosis rows with the repaired oracles (2026-10-28). The input is one or
// more `by-corpus.json` files of scripts/neuralese_data/scan_task_family_rows.py (files that mention the families). Every
// row carrying a program record of either family gets a verdict:
//   episode rows (a collected trajectory with outcome.value): regraded.
//     edit_stream: the final value against the row's per-step constraints plus editFidelityFromRecord (the superseded
//       include_words/word_count constraints are dropped).
//     contract_diagnosis: line_total as the run left it is rebuilt from the recorded edit_code actions (each step's
//       patchedHash must match exactly), then checkFunctionEffects with the case's hidden lines (lineTotalChecks).
//     Verdicts: pass | fail | unverifiable (the final function cannot be rebuilt exactly) | already_rejected.
//   case rows (a program record without an outcome, e.g. an RL task or a packet): oracle_current | oracle_outdated.
//   derived rows (training turns, rendered/graph rows naming a teacher trajectory): inherit their parent episode's
//     verdict by trajectory id (resolved in a second pass), else unresolved.
// Writes OUT/verdicts.jsonl (one line per row) and OUT/summary.json (counts per corpus, file and verdict).
// node scripts/inline-curriculum/regrade-family-rows.mjs --out DIR BY-CORPUS.json...
import { createHash } from 'node:crypto';
import { createReadStream, mkdirSync, writeFileSync, createWriteStream, readFileSync, statSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { createZstdDecompress } from 'node:zlib';
import { parseArgs } from 'node:util';
import { checkConstraints, editFidelityProblems } from '../../dist/evaluation/constraints.js';
import { checkFunctionEffects } from '../../dist/teacher/function-effects.js';
import { editTextContent } from '../../dist/native/scoped-fs.js';
import { hexDigest } from '../../dist/native/hash.js';
import { editFidelityFromRecord } from './edit-stream.mjs';
import { lineTotalChecks } from './applications.mjs';

const FAMILIES = new Set(['edit_stream', 'contract_diagnosis']);
const PRESERVATION = new Set(['include_words', 'word_count']);
const TRAJECTORY = /teacher-program:[0-9a-f]{16,}/;

/** The program record a row carries (task.program_ir, program_ir, or the row itself), if it is of a regraded family. */
export function programRecord(row) {
  for (const candidate of [row?.task?.program_ir, row?.program_ir, row?.task, row]) {
    const family = candidate?.curriculum?.family;
    if (FAMILIES.has(family) && candidate?.semantics && typeof candidate.semantics === 'object') return candidate;
  }
  return undefined;
}

/** line_total as the run left it, rebuilt from the recorded edit_code actions with exact hash checks. */
export function finalLineTotal(record, outcome) {
  const path = Object.keys(record.semantics.files).find(p => p.endsWith('/line_total.ts') || p === 'line_total.ts');
  const original = path ? record.semantics.files[path] : undefined;
  if (original === undefined) return { error: 'no line_total source in the record' };
  const actions = Array.isArray(outcome?.action_ledger) ? outcome.action_ledger : null;
  if (!actions) return { error: 'no action ledger' };
  let text = original;
  for (const action of actions) {
    const args = typeof action.arguments === 'string' ? safeJson(action.arguments) : action.arguments;
    if (action.name === 'eval' && /\bedit_code\s*\(/.test(String(args?.code ?? ''))) return { error: 'edit_code called inside eval' };
    if (action.name !== 'edit_code' || action.outcome !== 'ok') continue;
    if (args?.name !== undefined && !/(^|\.)line_total$/.test(String(args.name))) return { error: `edit of another function ${args.name}` };
    const result = safeJson(action.result_text);
    try { text = editTextContent(text, String(args.find ?? ''), String(args.replace_with ?? ''), args.fuzzy === true); }
    catch (error) { return { error: `recorded edit does not apply: ${error.message}` }; }
    if (!result?.patchedHash || result.patchedHash !== hexDigest(text)) return { error: 'rebuilt source differs from the recorded patchedHash' };
  }
  return { original, final: text };
}
const safeJson = text => { try { return JSON.parse(text); } catch { return undefined; } };

export function regradeEpisode(record, outcome) {
  const family = record.curriculum.family;
  if (outcome.accepted !== true) return { verdict: 'already_rejected', problems: [] };
  if (family === 'edit_stream') {
    let fidelity = record.semantics.expected.find(c => c?.kind === 'edit_fidelity');
    try { fidelity ??= editFidelityFromRecord(record); }
    catch (error) { return { verdict: 'unverifiable', problems: [`no edit_fidelity oracle: ${error.message}`] }; }
    const constraints = [...record.semantics.expected.filter(c => !PRESERVATION.has(c?.kind) && c?.kind !== 'edit_fidelity'), fidelity];
    const verdict = checkConstraints(outcome.value, constraints);
    return { verdict: verdict.passed ? 'pass' : 'fail',
      problems: verdict.passed ? [] : [...verdict.failed, ...(typeof outcome.value === 'string' ? editFidelityProblems(outcome.value, fidelity) : [])] };
  }
  const rebuilt = finalLineTotal(record, outcome);
  if (rebuilt.error) return { verdict: 'unverifiable', problems: [rebuilt.error] };
  const spec = record.semantics.function_checks ?? lineTotalChecks(record.semantics.inputs.lines,
    record.curriculum.edits === 'required' ? 'required' : 'forbidden');
  const result = checkFunctionEffects(spec, rebuilt.original, rebuilt.final, outcome.value, record.semantics.inputs ?? {});
  return { verdict: result.ok ? 'pass' : 'fail', problems: result.problems, final_sha256: result.final_sha256, changed: result.changed };
}

function oracleEra(record) {
  if (record.curriculum.family === 'edit_stream') {
    const kinds = (record.semantics.expected ?? []).map(c => c?.kind);
    return kinds.includes('edit_fidelity') ? 'edit_fidelity' : kinds.some(k => PRESERVATION.has(k)) ? 'preservation' : 'exclusion_only';
  }
  return record.semantics.function_checks ? 'function_checks' : 'value_only';
}

async function* lines(path) {
  let stream = createReadStream(path);
  if (path.endsWith('.zst')) stream = stream.pipe(createZstdDecompress());
  let number = 0;
  for await (const line of createInterface({ input: stream, crlfDelay: Infinity })) yield [++number, line];
}

async function main() {
  const { values, positionals } = parseArgs({ allowPositionals: true, options: { out: { type: 'string' } } });
  if (!values.out || !positionals.length) throw new Error('usage: regrade-family-rows.mjs --out DIR BY-CORPUS.json...');
  mkdirSync(values.out, { recursive: true });
  const owners = new Map();
  for (const input of positionals) for (const corpus of JSON.parse(readFileSync(input, 'utf8')).corpora)
    for (const file of Object.keys(corpus.files)) (owners.get(file) ?? owners.set(file, new Set()).get(file)).add(corpus.corpus);
  const verdicts = [];
  for (const [file, corpora] of owners) {
    let size = 0; try { size = statSync(file).size; } catch { verdicts.push({ file, corpora: [...corpora], kind: 'missing_file', verdict: 'unreadable' }); continue; }
    if (file.endsWith('.json')) {
      // A whole-document JSON (manifest, review, audit record) is not a row source; record it once.
      verdicts.push({ file, corpora: [...corpora], kind: 'json_document', verdict: 'not_rows', bytes: size });
      continue;
    }
    for await (const [line, text] of lines(file)) {
      if (!text.includes('edit_stream') && !text.includes('contract_diagnosis')) continue;
      const row = safeJson(text);
      const base = { file, line, row_sha256: createHash('sha256').update(text).digest('hex'), corpora: [...corpora] };
      if (!row || typeof row !== 'object') { verdicts.push({ ...base, kind: 'unparsed', verdict: 'unresolved' }); continue; }
      const record = programRecord(row);
      const rowId = row.id ?? row.row_id ?? null;
      const trajectory = (typeof row.source_ref?.trajectory_id === 'string' && row.source_ref.trajectory_id) ||
        (TRAJECTORY.exec(String(rowId ?? ''))?.[0]) || (TRAJECTORY.exec(String(row.conversion?.source_row_id ?? ''))?.[0]) || null;
      const isEpisode = record && row.outcome && Object.hasOwn(row.outcome, 'value') && Array.isArray(row.trajectory);
      if (isEpisode) {
        const graded = regradeEpisode(record, row.outcome);
        verdicts.push({ ...base, kind: 'episode', row_id: rowId, row_version: row.version ?? null, program_id: record.id,
          family: record.curriculum.family, variant: record.curriculum.variant, split: row.split ?? record.split ?? null,
          oracle_era: oracleEra(record), old_accepted: row.outcome.accepted, ...graded });
      } else if (record && !row.outcome && !trajectory) {
        verdicts.push({ ...base, kind: 'case', row_id: rowId, program_id: record.id, family: record.curriculum.family,
          variant: record.curriculum.variant, split: row.split ?? record.split ?? null, oracle_era: oracleEra(record),
          verdict: ['edit_fidelity', 'function_checks'].includes(oracleEra(record)) ? 'oracle_current' : 'oracle_outdated' });
      } else {
        const family = record?.curriculum?.family ?? (text.includes('edit_stream') ? 'edit_stream' : 'contract_diagnosis');
        const programId = record?.id ?? /inline-curriculum:[^"\s]*(?:edit_stream|contract_diagnosis)[^"\s]*/.exec(text)?.[0] ?? null;
        verdicts.push({ ...base, kind: 'derived', row_id: rowId, row_version: row.version ?? row.schema ?? null, program_id: programId,
          family, split: row.split ?? null, trajectory_id: trajectory, verdict: 'unresolved' });
      }
    }
  }
  // Second pass: derived rows inherit their parent episode's verdict by trajectory id.
  const byTrajectory = new Map();
  for (const v of verdicts) if (v.kind === 'episode' && v.row_id) byTrajectory.set(v.row_id, v);
  for (const v of verdicts) if (v.kind === 'derived') {
    const parent = v.trajectory_id ? byTrajectory.get(v.trajectory_id) : undefined;
    if (parent) Object.assign(v, { verdict: parent.verdict, inherited_from: { file: parent.file, line: parent.line, row_id: parent.row_id } });
    else v.verdict = 'unresolved';
  }
  const out = createWriteStream(`${values.out}/verdicts.jsonl`);
  for (const v of verdicts) out.write(JSON.stringify(v) + '\n');
  await new Promise(resolve => out.end(resolve));
  const summary = { schema: 'natlang.oracle-regrade-summary/1', inputs: positionals, rows: verdicts.length, by_verdict: {}, by_corpus: {} };
  for (const v of verdicts) {
    const key = `${v.kind}:${v.family ?? '-'}:${v.verdict}`;
    summary.by_verdict[key] = (summary.by_verdict[key] ?? 0) + 1;
    for (const corpus of v.corpora) {
      const c = summary.by_corpus[corpus] ??= {};
      c[key] = (c[key] ?? 0) + 1;
    }
  }
  writeFileSync(`${values.out}/summary.json`, JSON.stringify(summary, null, 1) + '\n');
  console.log(JSON.stringify({ rows: verdicts.length, by_verdict: summary.by_verdict }, null, 1));
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
