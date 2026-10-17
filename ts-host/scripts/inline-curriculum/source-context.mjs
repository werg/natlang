/** Exact source-format full-sentence membership, never inferred from punctuation. */
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { SOURCES, cachePath } from './acquire.mjs';
import { normalizeSourceText } from './source-split.mjs';

export const sstContextText = text => normalizeSourceText(text).toLowerCase();
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

export function sstFullSentenceIndex(sentenceBytes, splitBytes, authority) {
  for (const [bytes, binding] of [[sentenceBytes, authority.sentences], [splitBytes, authority.splits]])
    if (sha256(bytes) !== binding.sha256 || bytes.length !== binding.bytes)
      throw new Error('SST full-sentence source differs from its pinned context authority');
  const splitLines = splitBytes.toString('utf8').trimEnd().split(/\r?\n/);
  if (splitLines.shift() !== 'sentence_index,splitset_label') throw new Error('invalid SST sentence split header');
  const splits = new Map();
  for (const line of splitLines) {
    const match = /^(\d+),([123])$/.exec(line);
    if (!match || Number(match[1]) < 1 || splits.has(Number(match[1]))) throw new Error('invalid or duplicate SST sentence split');
    splits.set(Number(match[1]), Number(match[2]));
  }
  const lines = sentenceBytes.toString('utf8').trimEnd().split(/\r?\n/);
  if (lines.shift() !== 'sentence_index\tsentence') throw new Error('invalid SST sentence header');
  const seen = new Set(), index = new Map();
  for (const line of lines) {
    const separator = line.indexOf('\t'), idText = line.slice(0, separator), text = line.slice(separator + 1);
    const id = Number(idText);
    if (separator < 1 || !/^[1-9]\d*$/.test(idText) || !text || seen.has(id) || !splits.has(id))
      throw new Error('invalid, duplicate or unbound SST full sentence');
    seen.add(id);
    if (splits.get(id) !== 1) continue;
    const key = sstContextText(text);
    const members = index.get(key) ?? [];
    members.push({ sentence_index: id, original_sentence: text, original_split: 'train' });
    index.set(key, members);
  }
  if (seen.size !== splits.size) throw new Error('SST sentence/split inventories differ');
  return index;
}

export function sstRecordContext(row, index, authority) {
  const members = index.get(sstContextText(row.text));
  if (!members) return null;
  return { schema: 'natlang.sst-full-sentence-context/1', matching: 'source-whitespace-and-case-fold-v1',
    record_id: row.id, record_label: row.label, ...authority, members };
}

export function loadSstFullSentenceContext(cache) {
  const source = SOURCES.sst2, file = source.files.find(file => file.role === 'full-sentence-membership');
  const directory = cachePath(cache, 'sst2', source.revision, file.path.replace(/\.zip$/, ''));
  const authority = { source_url: file.url, archive_sha256: file.sha256, sentences: file.members.sentences, splits: file.members.splits };
  let sentences, splits;
  try { sentences = readFileSync(`${directory}/${file.members.sentences.path}`); splits = readFileSync(`${directory}/${file.members.splits.path}`); }
  catch { throw new Error('sst2 full-sentence context is not cached; run node scripts/inline-curriculum/acquire.mjs --source sst2'); }
  return { index: sstFullSentenceIndex(sentences, splits, authority), authority };
}
