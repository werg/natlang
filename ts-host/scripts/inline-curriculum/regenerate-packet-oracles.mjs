#!/usr/bin/env node
// Regenerate ONLY the oracle side of selected families' rows of a task packet, from the same draws (as packet v3 did
// for v2's edit_stream rows). Each selected row is rebuilt by its family's generator at the packet's seed and shape
// index; every field except the oracle fields (semantics.expected, semantics.function_checks) must equal the packet
// row exactly, or the script fails. The packet row keeps its bytes elsewhere: only the oracle fields are replaced.
// Rows of other families are copied byte for byte. Writes the new cases.jsonl and a JSON change report.
//
// node scripts/inline-curriculum/regenerate-packet-oracles.mjs --in CASES.jsonl --out NEW.jsonl --report REPORT.json \
//   --seed 71028 --split test --families edit_stream,contract_diagnosis [--draw-split edit_stream=train]
// --draw-split names the source partition a family's packet rows were drawn from when it differs from --split.
// --from-record edit_stream derives that family's oracle from each row's own stored document, steps and reference
// states (edit-stream.mjs editFidelityFromRecord) instead of regenerating the row: held v3's edit_stream rows predate
// later ag_news pool filtering, so the current generator no longer draws those documents. The row's per-step
// constraints are kept and only the superseded preservation constraints (include_words, word_count) are replaced. In
// generator mode the same derivation is checked against the generated constraint for every edit_stream row.
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { isDeepStrictEqual, parseArgs } from 'node:util';
import { buildRecords } from './families.mjs';
import { editFidelityFromRecord } from './edit-stream.mjs';

const ORACLE_FIELDS = ['expected', 'function_checks'];
const PACKET_FIELDS = ['heldout_packet'];
const { values } = parseArgs({ options: { in: { type: 'string' }, out: { type: 'string' }, report: { type: 'string' },
  seed: { type: 'string' }, split: { type: 'string', default: 'test' }, families: { type: 'string' },
  'draw-split': { type: 'string', multiple: true, default: [] }, 'from-record': { type: 'string', multiple: true, default: [] } } });
if (!values.in || !values.out || !values.report || !values.seed || !values.families)
  throw new Error('usage: --in CASES --out NEW --report REPORT --seed N --families a,b [--split test] [--draw-split family=split]');
const seed = Number(values.seed), families = new Set(values.families.split(','));
const drawSplit = Object.fromEntries(values['draw-split'].map(item => item.split('=')));
const fromRecord = new Set(values['from-record']);
for (const family of fromRecord) if (family !== 'edit_stream') throw new Error(`--from-record supports edit_stream only, not ${family}`);
const PRESERVATION = new Set(['include_words', 'word_count']);
const sha = text => createHash('sha256').update(text).digest('hex');
const lines = readFileSync(values.in, 'utf8').split('\n').filter(Boolean);
const withoutOracle = record => {
  const copy = structuredClone(record);
  for (const field of ORACLE_FIELDS) delete copy.semantics[field];
  // Annotations a packet assembler added to its rows (not generator output).
  for (const field of PACKET_FIELDS) delete copy[field];
  return copy;
};
const changes = [];
const out = lines.map((line, row) => {
  const record = JSON.parse(line);
  const family = record.curriculum.family;
  if (!families.has(family)) return line;
  const index = Number(/(\d+)$/.exec(record.curriculum.shape)?.[1]);
  if (!Number.isInteger(index)) throw new Error(`row ${row}: no shape index in ${record.curriculum.shape}`);
  let rebuilt, method;
  if (fromRecord.has(family)) {
    rebuilt = structuredClone(record);
    rebuilt.semantics.expected = [...record.semantics.expected.filter(c => !PRESERVATION.has(c.kind)), editFidelityFromRecord(record)];
    method = 'from-record';
  } else {
    rebuilt = buildRecords({ seed, shapes: 1, start: index, families: [family], split: values.split, hints: false,
      drawSplit: drawSplit[family] }).find(candidate => candidate.id === record.id);
    if (!rebuilt) throw new Error(`row ${row}: ${record.id} is not generated at seed ${seed} index ${index}`);
    if (!isDeepStrictEqual(withoutOracle(rebuilt), withoutOracle(record)))
      throw new Error(`row ${row}: ${record.id} regenerates with different non-oracle fields; the draws differ`);
    if (family === 'edit_stream' && !isDeepStrictEqual(editFidelityFromRecord(record), rebuilt.semantics.expected.at(-1)))
      throw new Error(`row ${row}: ${record.id}: the from-record edit_fidelity differs from the generated one`);
    method = 'generator';
  }
  const before = Object.fromEntries(ORACLE_FIELDS.map(field => [field, record.semantics[field] ?? null]));
  for (const field of ORACLE_FIELDS) {
    if (rebuilt.semantics[field] === undefined) delete record.semantics[field];
    else record.semantics[field] = rebuilt.semantics[field];
  }
  const replaced = JSON.stringify(record);
  changes.push({ row, id: record.id, family, method, shape: record.curriculum.shape, variant: record.curriculum.variant,
    old_line_sha256: sha(line), new_line_sha256: sha(replaced),
    old_oracle_sha256: sha(JSON.stringify(before)),
    new_oracle_sha256: sha(JSON.stringify(Object.fromEntries(ORACLE_FIELDS.map(field => [field, record.semantics[field] ?? null])))),
    non_oracle_fields_identical: true,
    old_expected_kinds: Array.isArray(before.expected) ? before.expected.map(c => c?.kind ?? typeof c) : typeof before.expected,
    new_expected_kinds: Array.isArray(record.semantics.expected) ? record.semantics.expected.map(c => c?.kind ?? typeof c) : typeof record.semantics.expected,
    function_checks: record.semantics.function_checks ? { change: record.semantics.function_checks.change, runs: record.semantics.function_checks.runs.length } : null });
  return replaced;
});
const text = out.join('\n') + '\n';
writeFileSync(values.out, text);
const report = { schema: 'natlang.packet-oracle-regeneration/1', input: { path: values.in, sha256: sha(readFileSync(values.in)), lines: lines.length },
  output: { path: values.out, sha256: sha(text), lines: out.length }, seed, split: values.split, families: [...families],
  draw_split: drawSplit, oracle_fields: ORACLE_FIELDS, rows_changed: changes.length,
  rows_byte_identical: out.filter((line, i) => line === lines[i]).length, changes };
writeFileSync(values.report, JSON.stringify(report, null, 1) + '\n');
console.log(JSON.stringify({ input: report.input, output: report.output, rows_changed: changes.length, rows_byte_identical: report.rows_byte_identical }));
