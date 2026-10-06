/** One content partition for labeled leaves and composed/folder cases. */
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

const reservations = JSON.parse(readFileSync(new URL('../../../training/source-split-reservations.json', import.meta.url), 'utf8'));
if (reservations.schema !== 'natlang.source-split-reservations/1') throw new Error('unsupported source split reservations');
const held = new Set(reservations.records.map(row => row.id));
export const normalizeSourceText = text => String(text).replace(/\\/g, ' ').replace(/\s+/g, ' ').trim();
export function sourceRecordId(dataset, text, label) {
  return createHash('sha256').update(`${dataset}\0${text}\0${label}`).digest('hex');
}
export function sourceRecordSplit(id) {
  // An established test reservation takes priority over the generated hash split.
  return held.has(id) || Number.parseInt(id.slice(0, 8), 16) % 10 === 0 ? 'test' : 'train';
}
