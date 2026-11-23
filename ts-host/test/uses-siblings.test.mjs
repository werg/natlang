import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { loadNatlang, checkProject, loadNamedFunction, virtualSourceFiles } from '../dist/index.js';
import { nodeSourceFiles } from '../dist/runtime/node-files.js';

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
  assert.throws(() => loadNatlang(join(root, 'top/a.nl')), /leads back to this function \(b\.nl -> a\.nl -> b\.nl\)/);
});

test('uses inside a mounted package keeps that package as its lexical root', () => {
  const root = layout({
    'top.nl': fn(),
    'top/domains/team/natlang.json': '{}',
    'top/domains/team/inspect.nl': fn('helpers/explain'),
    'top/domains/team/helpers/explain.nl': fn(),
  });
  const top = loadNatlang(join(root, 'top.nl'));
  assert.equal(typeof top.domains.team.inspect.explain, 'function');
  assert.equal(checkProject(root).ok, true);
});

// A relative `uses:` into a cousin folder of the same loading owner binds the very record of that tree (no reload, no
// second copy), whether the target's folder loads before or after the user's folder.
const cousinTree = layout;
const records = (root, path) => loadNamedFunction(join(root, path), nodeSourceFiles(root)).codebase;

test('uses naming a function in a cousin folder loaded earlier binds that record', () => {
  const root = cousinTree({
    'ask.nl': fn(), 'ask/billing.nl': fn(), 'ask/billing/price_line.nl': fn(),
    'ask/catalog/list_items.nl': fn('../billing/price_line'),
  });
  const ask = records(root, 'ask.nl');
  assert.equal(ask.catalog.codebase.list_items.codebase.price_line, ask.billing.codebase.price_line);
  assert.equal(typeof loadNatlang(join(root, 'ask.nl')), 'function');
  assert.equal(typeof loadNatlang(join(root, 'ask/catalog/list_items.nl'), root), 'function');
  assert.equal(checkProject(root).ok, true);
});

test('uses naming a function in a cousin folder loaded later binds that record', () => {
  const root = cousinTree({
    'hive_office.nl': fn(), 'hive_office/members/renew_membership.nl': fn('../zbilling/dues_rate'),
    'hive_office/zbilling/dues_rate.nl': fn(),
  });
  const office = records(root, 'hive_office.nl');
  assert.equal(office.members.codebase.renew_membership.codebase.dues_rate, office.zbilling.codebase.dues_rate);
  assert.equal(typeof loadNatlang(join(root, 'hive_office.nl')), 'function');
  assert.equal(typeof loadNatlang(join(root, 'hive_office/members/renew_membership.nl'), root), 'function');
  assert.equal(checkProject(root).ok, true);
});

test('package-root uses into a cousin folder works without natlang.json', () => {
  const root = mkdtempSync(join(tmpdir(), 'uses-sib-'));
  for (const [path, text] of Object.entries({
    'ask.nl': fn(), 'ask/billing.nl': fn(), 'ask/billing/price_line.nl': fn(),
    'ask/catalog/list_items.nl': fn('ask/billing/price_line'),
    'ask/alpha/early.nl': fn('ask/billing/price_line'),
  })) { mkdirSync(join(root, path, '..'), { recursive: true }); writeFileSync(join(root, path), text); }
  const ask = records(root, 'ask.nl');
  assert.equal(ask.catalog.codebase.list_items.codebase.price_line, ask.billing.codebase.price_line);
  assert.equal(ask.alpha.codebase.early.codebase.price_line, ask.billing.codebase.price_line);
  assert.equal(checkProject(root).ok, true);
});

test('package-root uses in an in-memory project without a manifest resolve from the project root', () => {
  const files = { 'ask.nl': fn(), 'ask/billing.nl': fn(), 'ask/billing/price_line.nl': fn(), 'ask/catalog/list_items.nl': fn('ask/billing/price_line') };
  const ask = loadNamedFunction('/project/ask.nl', virtualSourceFiles(files)).codebase;
  assert.equal(ask.catalog.codebase.list_items.codebase.price_line, ask.billing.codebase.price_line);
});

test('a measure naming a cousin bound by uses is checked after the bind', () => {
  const root = cousinTree({
    'ask.nl': fn(), 'ask/catalog/list_items.nl': `---\nargs:\n  n: number\nreturns: number\nuses: [../zeta/size]\nmeasure: size(n)\n---\nx\n`,
    'ask/zeta/size.nl': fn(),
  });
  const ask = records(root, 'ask.nl');
  assert.equal(ask.catalog.codebase.list_items.codebase.size, ask.zeta.codebase.size);
  assert.equal(checkProject(root).ok, true);
});

test('a real cycle across cousin folders still names its path', () => {
  const root = cousinTree({
    'ask.nl': fn(), 'ask/billing/price_line.nl': fn('../catalog/list_items'),
    'ask/catalog/list_items.nl': fn('../billing/price_line'),
  });
  assert.throws(() => loadNatlang(join(root, 'ask.nl')), /leads back to this function \(.*(price_line|list_items)\.nl -> .*\.nl -> .*(price_line|list_items)\.nl\)/);
});

test('a real cycle through a cousin and a sibling still names its path', () => {
  const root = cousinTree({
    'ask.nl': fn(), 'ask/billing/price_line.nl': fn('./tax'), 'ask/billing/tax.nl': fn('../catalog/list_items'),
    'ask/catalog/list_items.nl': fn('../billing/price_line'),
  });
  assert.throws(() => loadNatlang(join(root, 'ask.nl')), /leads back to this function/);
});

test('a real cycle from a cousin through a function outside the tree still names its path', () => {
  const root = cousinTree({
    'ask.nl': fn(), 'ask/alpha/a.nl': fn('../../lib/x'), 'lib/x.nl': fn('../ask/zeta/b'),
    'ask/zeta/b.nl': fn('../alpha/a'),
  });
  assert.throws(() => loadNatlang(join(root, 'ask.nl')), /leads back to this function/);
});
