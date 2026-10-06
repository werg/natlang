/** Reproduce explicit held-out reservations without storing source text. */
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { normalizeSourceText, sourceRecordId } from './source-split.mjs';

const source = process.argv[2] ?? 'data/external_pilot/jeff-test-tasks.jsonl';
const output = process.argv[3] ?? 'training/source-split-reservations.json';
const bytes = readFileSync(source);
const compatible = new Set(['sms_spam', 'sst2', 'ag_news', 'emotion']);
const records = new Map();
for (const line of bytes.toString('utf8').split(/\r?\n/).filter(Boolean)) {
  const row = JSON.parse(line), dataset = row.source_meta?.dataset;
  if (!compatible.has(dataset)) continue;
  if (row.split !== 'test' || typeof row.state !== 'string' || !row.id) {
    throw new Error(`invalid explicit test reservation: ${row.id}`);
  }
  const id = sourceRecordId(dataset, normalizeSourceText(row.state), '');
  const entry = records.get(id) ?? { id, dataset, split: 'test', source_ids: [] };
  if (!entry.source_ids.includes(row.id)) entry.source_ids.push(row.id);
  entry.source_ids.sort();
  records.set(id, entry);
}
if (!records.size) throw new Error('no compatible held-out records');
const value = {
  schema: 'natlang.source-split-reservations/1',
  policy: 'explicit-test-reservations-override-content-hash-partition; no-retrospective-train-admission',
  normalization: 'model-visible-text: replace-backslashes-with-spaces; collapse-whitespace; trim; preserve-case',
  source: { path: source, sha256: createHash('sha256').update(bytes).digest('hex') },
  records: [...records.values()].sort((a, b) => a.id.localeCompare(b.id)),
};
writeFileSync(output, JSON.stringify(value, null, 2) + '\n');
console.log(JSON.stringify({ output, reservations: records.size, source_sha256: value.source.sha256 }));
