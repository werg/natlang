import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { loadNatlang, checkProject } from '../dist/index.js';

const fn = (uses, body = 'x') => `---\nargs:\n  n: number\nreturns: number\n${uses ? `uses: [${uses}]\n` : ''}---\n${body}\n`;
function layout(files) {
  const root = mkdtempSync(join(tmpdir(), 'uses-sib-'));
  for (const [path, text] of Object.entries({ 'natlang.json': '{}', ...files })) {
    mkdirSync(join(root, path, '..'), { recursive: true }); writeFileSync(join(root, path), text);
  }
  return root;
}

test('uses naming a sibling in the same callable folder is not a cycle', () => {
  const root = layout({
    'merge.nl': fn(), 'merge/block.nl': fn(),
    'merge/block/refine.nl': fn('merge/block/prose'), 'merge/block/prose.nl': fn(),
  });
  const refine = loadNatlang(join(root, 'merge/block/refine.nl'));
  assert.equal(typeof refine, 'function');
  assert.equal(checkProject(root).ok, true);
});

test('uses naming a child of the function\'s own owner is not a cycle', () => {
  const root = layout({
    'migrate.nl': fn(), 'migrate/repair.nl': fn('migrate/locate'), 'migrate/locate.nl': fn(),
  });
  assert.equal(typeof loadNatlang(join(root, 'migrate/repair.nl')), 'function');
  assert.equal(typeof loadNatlang(join(root, 'migrate.nl')), 'function');
});

test('a true cycle among siblings names its path', () => {
  const root = layout({
    'top.nl': fn(), 'top/a.nl': fn('top/b'), 'top/b.nl': fn('top/a'),
  });
  assert.throws(() => loadNatlang(join(root, 'top.nl')), /top\/a\.nl -> top\/b\.nl -> top\/a\.nl|top\/b\.nl -> top\/a\.nl -> top\/b\.nl/);
  assert.throws(() => loadNatlang(join(root, 'top/a.nl')), /cannot reach itself/);
});
