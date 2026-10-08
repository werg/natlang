/**
 * The coding tools' crisp helpers equal pi-durable's own, and the edit matching policy as applyEdits.nl states it,
 * composed from those helpers step by step, equals pi's applyEditsToNormalizedContent, errors included.
 * Run: node --test applications/pi/test/tools-crisp.test.mjs
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as pi from '../vendor/durable/src/tools/edit-diff.ts';
import { createEditTool } from '../vendor/durable/src/tools/edit.ts';
import { detectSupportedImageMimeType as piDetect, detectSupportedImageMimeTypeOf as piDetectOf } from '../vendor/durable/src/tools/image.ts';
import * as text from '../extensions/coding-tools/edit/applyEdits/text.ts';
import * as lines from '../extensions/coding-tools/edit/text.ts';
import prepare from '../extensions/coding-tools/edit/prepare.ts';
import { detectSupportedImageMimeType, detectSupportedImageMimeTypeOf } from '../extensions/coding-tools/read/image.ts';

let seed = 11;
const random = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
const pick = items => items[Math.floor(random() * items.length)];
const pieces = ['a', 'b', 'foo', 'bar()', ' ', '  ', '\t', '\n', '\r\n', '\r', '’', '“', '–', ' ', 'é', 'ﬁ', '😀', 'x = 1;', '}'];
const str = (max = 30) => Array.from({ length: Math.floor(random() * max) }, () => pick(pieces)).join('');

/** applyEdits.nl, step by step, over the crisp helpers. */
function applyEdits(content, edits, path) {
  const n = edits.length;
  const normalized = edits.map(edit => ({ oldText: text.normalizeToLF(edit.oldText), newText: text.normalizeToLF(edit.newText) }));
  for (let i = 0; i < n; i++) if (normalized[i].oldText === '')
    return { error: n === 1 ? `oldText must not be empty in ${path}.` : `edits[${i}].oldText must not be empty in ${path}.` };
  const fuzzy = normalized.some(edit => text.fuzzyFindText(content, edit.oldText).usedFuzzyMatch);
  const base = fuzzy ? text.normalizeForFuzzyMatch(content) : content;
  const matches = [];
  for (let i = 0; i < n; i++) {
    const m = text.fuzzyFindText(base, normalized[i].oldText);
    if (!m.found) return { error: n === 1 ? `Could not find the exact text in ${path}. The old text must match exactly including all whitespace and newlines.`
      : `Could not find edits[${i}] in ${path}. The oldText must match exactly including all whitespace and newlines.` };
    const k = text.countOccurrences(base, normalized[i].oldText);
    if (k > 1) return { error: n === 1 ? `Found ${k} occurrences of the text in ${path}. The text must be unique. Please provide more context to make it unique.`
      : `Found ${k} occurrences of edits[${i}] in ${path}. Each oldText must be unique. Please provide more context to make it unique.` };
    matches.push({ editIndex: i, matchIndex: m.index, matchLength: m.matchLength, newText: normalized[i].newText });
  }
  matches.sort((a, b) => a.matchIndex - b.matchIndex);
  for (let i = 1; i < matches.length; i++) if (matches[i - 1].matchIndex + matches[i - 1].matchLength > matches[i].matchIndex)
    return { error: `edits[${matches[i - 1].editIndex}] and edits[${matches[i].editIndex}] overlap in ${path}. Merge them into one edit or target disjoint regions.` };
  let next;
  try { next = fuzzy ? text.applyReplacementsPreservingUnchangedLines(content, base, matches) : text.applyReplacements(base, matches); }
  catch (error) { return { error: error.message }; }
  if (next === content) return { error: n === 1 ? `No changes made to ${path}. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected.`
    : `No changes made to ${path}. The replacements produced identical content.` };
  return { base: content, content: next };
}

function piApply(content, edits, path) {
  try { const { baseContent, newContent } = pi.applyEditsToNormalizedContent(content, edits, path); return { base: baseContent, content: newContent }; }
  catch (error) { return { error: error.message }; }
}

test('line endings, byte-order mark and tolerant normalization equal pi', () => {
  for (let i = 0; i < 3000; i++) {
    const t = (random() < 0.2 ? '﻿' : '') + str();
    assert.deepEqual(lines.stripBom(t), pi.stripBom(t));
    assert.equal(lines.detectLineEnding(t), pi.detectLineEnding(t));
    assert.equal(lines.normalizeToLF(t), pi.normalizeToLF(t));
    assert.equal(text.normalizeToLF(t), pi.normalizeToLF(t));
    assert.equal(lines.restoreLineEndings(pi.normalizeToLF(t), '\r\n'), pi.restoreLineEndings(pi.normalizeToLF(t), '\r\n'));
    assert.equal(text.normalizeForFuzzyMatch(t), pi.normalizeForFuzzyMatch(t));
    const needle = str(4);
    const ours = text.fuzzyFindText(t, needle), theirs = pi.fuzzyFindText(t, needle);
    assert.deepEqual(ours, { found: theirs.found, index: theirs.index, matchLength: theirs.matchLength, usedFuzzyMatch: theirs.usedFuzzyMatch });
  }
});

test('the matching policy of applyEdits.nl equals pi, errors and messages included', () => {
  let changed = 0, failed = new Set();
  for (let i = 0; i < 4000; i++) {
    const content = pi.normalizeToLF(str(40));
    const count = 1 + Math.floor(random() * 3);
    const edits = Array.from({ length: count }, () => {
      const start = Math.floor(random() * (content.length + 1)), length = Math.floor(random() * 8);
      let oldText = random() < 0.85 ? content.slice(start, start + length) : str(3);
      if (random() < 0.2) oldText = oldText.replace(/'/g, '’').replace(/ /g, ' ') + (random() < 0.5 ? '  ' : '');
      return { oldText, newText: random() < 0.1 ? oldText : str(5) };
    });
    const ours = applyEdits(content, edits, 'f.txt'), theirs = piApply(content, edits, 'f.txt');
    assert.deepEqual(ours, theirs, JSON.stringify({ content, edits }));
    if (ours.error) failed.add(ours.error.replace(/\d+/g, 'N')); else changed++;
  }
  assert.ok(changed > 200, `only ${changed} successful edits exercised`);
  assert.ok(failed.size >= 6, `only ${failed.size} kinds of error exercised: ${[...failed].join(' | ')}`);
});

test('the argument repair equals pi prepareEditArguments', () => {
  const piPrepare = createEditTool().prepareArguments;
  const cases = [null, 3, [], {}, { path: 'a', edits: [] }, { path: 'a', edits: '[{"oldText":"x","newText":"y"}]' },
    { path: 'a', edits: '{"oldText":"x","newText":"y"}' }, { path: 'a', edits: 'not json' }, { path: 'a', edits: { oldText: 'x', newText: 'y' } },
    { path: 'a', oldText: 'x', newText: 'y' }, { path: 'a', edits: [{ oldText: 'p', newText: 'q' }], oldText: 'x', newText: 'y' },
    { path: 'a', oldText: 'x' }, { path: 'a', edits: '"string"' }];
  for (const input of cases) assert.deepEqual(prepare(structuredClone(input)), piPrepare(structuredClone(input)), JSON.stringify(input));
});

test('image sniffing equals pi', async () => {
  const png = (chunks) => {
    const bytes = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
    for (const [type, length] of chunks) {
      bytes.push((length >>> 24) & 255, (length >>> 16) & 255, (length >>> 8) & 255, length & 255, ...[...type].map(c => c.charCodeAt(0)));
      for (let i = 0; i < length + 4; i++) bytes.push(0);
    }
    return new Uint8Array(bytes);
  };
  const samples = [new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 1, 2]), new Uint8Array([0xff, 0xd8, 0xff, 0xf7]), png([['IHDR', 13], ['IDAT', 4]]),
    png([['IHDR', 13], ['acTL', 8], ['IDAT', 4]]), png([['IHDR', 13], ['tEXt', 70000], ['acTL', 8]]), png([['IHDR', 12]]),
    new TextEncoder().encode('GIF89a......'), new TextEncoder().encode('RIFF....WEBPVP8 '), new TextEncoder().encode('hello world, plain text'),
    new Uint8Array([0x42, 0x4d, 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 24, 0, 0, 0])];
  for (let i = 0; i < 300; i++) samples.push(new Uint8Array(Array.from({ length: Math.floor(random() * 40) }, () => Math.floor(random() * 256))));
  for (const bytes of samples) {
    assert.equal(detectSupportedImageMimeType(bytes), piDetect(bytes));
    const source = { size: bytes.length, read: async (offset, length) => bytes.subarray(offset, offset + length) };
    assert.equal(await detectSupportedImageMimeTypeOf(source), (await piDetectOf(source)) ?? null);
  }
});
