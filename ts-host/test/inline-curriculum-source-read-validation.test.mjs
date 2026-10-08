import test from 'node:test';
import assert from 'node:assert/strict';
import { expectedSourcePageCount, matchDeclaredSourceRead } from '../scripts/inline-curriculum/source-read-validation.mjs';

const packet = 'First complete source sentence.\nSecond sentence with exact punctuation.';
const folderFiles = { 'packet.md': packet };

test('accepts a complete read from any path declared by the source reference', () => {
  assert.deepEqual(matchDeclaredSourceRead({ readPath: 'packet.md', expectedPath: 'packet.md',
    resultText: packet, folderFiles }), { source_path: 'packet.md', source_text: packet,
    exact_complete_source_read: true, page_count: 0 });
});

test('rejects a path different from the declared source read', () => {
  assert.equal(matchDeclaredSourceRead({ readPath: 'other.md', expectedPath: 'packet.md',
    resultText: packet, folderFiles: { ...folderFiles, 'other.md': packet } }), null);
});

test('rejects partial or altered source content even at the declared path', () => {
  assert.equal(matchDeclaredSourceRead({ readPath: 'packet.md', expectedPath: 'packet.md',
    resultText: 'First complete source sentence.', folderFiles }), null);
  assert.equal(matchDeclaredSourceRead({ readPath: 'packet.md', expectedPath: 'packet.md',
    resultText: `${packet} `, folderFiles }), null);
});

test('accepts ordered pages only when they reassemble the complete original source', () => {
  const source = `${'first line\n'.repeat(140)}${'middle text\n'.repeat(170)}${'last line\n'.repeat(160)}`;
  const pageCount = expectedSourcePageCount(source);
  const headLength = (() => {
    let head = Math.floor(2000 * 0.75);
    const breakAt = source.lastIndexOf('\n', head);
    if (breakAt > head - 200) head = breakAt;
    return head;
  })();
  const pages = [source.slice(0, headLength)];
  for (let start = headLength; start < source.length; start += 2000)
    pages.push(source.slice(start, start + 2000));
  assert.equal(pages.length, pageCount);
  const pageEvents = pages.map((text, index) => ({
    arguments: { id: 'amber', page: index + 1 },
    result_text: text + (index + 1 < pageCount
      ? `\n<<page ${index + 1} of ${pageCount} shown; read_page("amber", ${index + 2}) shows the next part>>`
      : `\n<<page ${pageCount} of ${pageCount}, the last>>`),
  }));
  const resultText = `<<cut off; read_page("amber", 2) shows the next part>>`;
  const args = { readPath: 'packet.md', expectedPath: 'packet.md', resultText,
    folderFiles: { 'packet.md': source }, pageEvents };
  assert.equal(matchDeclaredSourceRead(args)?.exact_complete_source_read, true);
  assert.equal(matchDeclaredSourceRead({ ...args, pageEvents: pageEvents.slice(0, -1) }), null);
  assert.equal(matchDeclaredSourceRead({ ...args, pageEvents: [pageEvents[0], pageEvents[0], ...pageEvents.slice(2)] }), null);
  const changed = pageEvents.map(event => ({ ...event, arguments: { ...event.arguments } }));
  changed[1].result_text = changed[1].result_text.replace('middle text', 'altered text');
  assert.equal(matchDeclaredSourceRead({ ...args, pageEvents: changed }), null);
});
