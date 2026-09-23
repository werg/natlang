import assert from 'node:assert/strict';
import { execFile as execFileCallback } from 'node:child_process';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import test from 'node:test';
import { loadFunctionFile } from '../dist/index.js';

const execFile = promisify(execFileCallback);
const migration = new URL('../../scripts/migrate_companion_imports.mjs', import.meta.url);

test('migration turns cross-file imports into companion functions and is idempotent', async t => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-companion-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, 'main'));
  await writeFile(join(root, 'main.nl'), `import helper from "./helper";\nimport inner from "./main/inner";\n\n---\nargs: {}\nreturns: number\n---\nReturn helper plus inner.\n`);
  await writeFile(join(root, 'helper.ts'), 'export default function helper(): number { return 2; }\n');
  await writeFile(join(root, 'main/inner.ts'), 'export default function inner(): number { return 3; }\n');
  const first = await execFile(process.execPath, [migration.pathname, root]);
  assert.match(first.stdout, /Copied 1 companion functions/);
  assert.doesNotMatch(await readFile(join(root, 'main.nl'), 'utf8'), /import /);
  assert.equal(await readFile(join(root, 'main/helper.ts'), 'utf8'), await readFile(join(root, 'helper.ts'), 'utf8'));
  assert.deepEqual(Object.keys(loadFunctionFile(join(root, 'main.nl')).codebase).sort(), ['helper', 'inner']);
  const second = await execFile(process.execPath, [migration.pathname, root]);
  assert.match(second.stdout, /Copied 0 companion functions; removed local imports from 0 files/);
});
