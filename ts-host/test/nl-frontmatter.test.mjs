import assert from 'node:assert/strict';
import { test } from 'node:test';
import { loadVirtualNatlang } from '../dist/index.js';
import { readNatlangFrontmatter } from '../dist/runtime/loader.js';

test('frontmatter types are TypeScript text, unquoted, in block or inline form', () => {
  const block = readNatlangFrontmatter([
    'description: "Rank paragraphs: most useful first"',
    'args:',
    '  question: string',
    '  paragraphs: { title: string, text: string }[]',
    '  score?: (x: string) => number',
    '  # a comment line',
    '  verdict: "knight" | "knave"',
    '  tags: "string[]"',
    '  long: {',
    '      a: string;',
    '      b: number }',
    'types:',
    '  Row: Record<string, "a" | "b">',
    'returns: { a: string; b: number }[]',
  ].join('\n'));
  assert.deepEqual(block.args, { question: 'string', paragraphs: '{ title: string, text: string }[]', 'score?': '(x: string) => number',
    verdict: '"knight" | "knave"', tags: 'string[]', long: '{ a: string; b: number }' });
  assert.deepEqual(block.types, { Row: 'Record<string, "a" | "b">' });
  assert.equal(block.returns, '{ a: string; b: number }[]');
  assert.equal(block.description, 'Rank paragraphs: most useful first');
  // Inline maps read as an object type; quoted members keep their YAML meaning.
  const inline = readNatlangFrontmatter('args: { names: string[], reviews: "string[]", m?: Record<string, number> }\nreturns: "string[]"');
  assert.deepEqual(inline.args, { names: 'string[]', reviews: 'string[]', 'm?': 'Record<string, number>' });
  assert.equal(inline.returns, 'string[]');
  assert.deepEqual(readNatlangFrontmatter('args: {}\nreturns: string').args, {});
});

test('frontmatter mistakes get errors that say what to write', () => {
  assert.throws(() => readNatlangFrontmatter('args: "{ q: string }"\nreturns: string'), /must map names to types, not be one string/);
  assert.throws(() => readNatlangFrontmatter('args: string[]\nreturns: string'), /must map names to types/);
  assert.throws(() => readNatlangFrontmatter('args:\n  q string\nreturns: string'), /expected `name: type`/);
  assert.throws(() => readNatlangFrontmatter('args:\n  q:\nreturns: string'), /q needs a type/);
  assert.throws(() => readNatlangFrontmatter('returns:\n'), /returns needs a type/);
});

test('a function with unquoted structured types loads and checks its signature', () => {
  const fn = loadVirtualNatlang({ 'rank.nl': '---\nargs:\n  paragraphs: { title: string }[]\n  pick: (titles: string[]) => string\nreturns: string[]\n---\nRank them.\n' }, 'rank.nl');
  assert.equal(typeof fn, 'function');
  assert.throws(() => loadVirtualNatlang({ 'bad.nl': '---\nargs:\n  x: Missing\nreturns: string\n---\nx\n' }, 'bad.nl'), /Missing/);
});

test('an inline map may start on the next, indented line', () => {
  assert.deepEqual(readNatlangFrontmatter('args:\n  {}\nreturns: number').args, {});
  assert.deepEqual(readNatlangFrontmatter('args:\n  { q: string,\n    n: number[] }\nreturns: number').args, { q: 'string', n: 'number[]' });
});
