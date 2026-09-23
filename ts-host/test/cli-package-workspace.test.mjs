import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { findPackageWorkspace } from '../dist/cli/main.js';

test('source-file CLI chooses the nearest owning package.json', async t => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-cli-package-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const project = join(root, 'project');
  const nested = join(project, 'src', 'nested');
  await mkdir(nested, { recursive: true });
  await writeFile(join(root, 'package.json'), '{}');
  await writeFile(join(project, 'package.json'), '{}');
  assert.equal(findPackageWorkspace(nested), project);
  assert.equal(findPackageWorkspace(root), root);
});
