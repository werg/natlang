import test from 'node:test';
import assert from 'node:assert/strict';
import { matchDeclaredSourceRead } from '../scripts/inline-curriculum/source-read-validation.mjs';

const packet = 'First complete source sentence.\nSecond sentence with exact punctuation.';
const folderFiles = { 'packet.md': packet };

test('accepts a complete read from any path declared by the source reference', () => {
  assert.deepEqual(matchDeclaredSourceRead({ readPath: 'packet.md', expectedPath: 'packet.md',
    resultText: packet, folderFiles }), { source_path: 'packet.md', source_text: packet });
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
