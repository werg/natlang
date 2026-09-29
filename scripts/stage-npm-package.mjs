/** Stage the built runtime into an npm package directory under npm-packages/. */
import { cpSync, mkdirSync, readdirSync, rmSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const kind = process.argv[2];
if (!['node', 'browser'].includes(kind)) throw new Error('usage: stage-npm-package.mjs node|browser');
const source = join(root, 'ts-host', 'dist');
const destination = join(root, 'npm-packages', kind, 'dist');
rmSync(destination, { recursive: true, force: true }); mkdirSync(destination, { recursive: true });

/** Copy a tree, keeping the files `keep` accepts and skipping the directories in `skip`. */
function copyTree(from, to, keep, skip = []) {
  for (const name of readdirSync(from)) {
    const path = join(from, name), target = join(to, name);
    if (skip.includes(path) || name === '.natlang') continue;
    if (statSync(path).isDirectory()) { mkdirSync(target, { recursive: true }); copyTree(path, target, keep, skip); }
    else if (keep(path)) cpSync(path, target);
  }
}

if (kind === 'node') {
  const { DEFAULT_MODEL_RELEASE } = await import(new URL('../ts-host/dist/model-default.js', import.meta.url));
  if (!DEFAULT_MODEL_RELEASE.downloadUrl || new URL(DEFAULT_MODEL_RELEASE.downloadUrl).protocol !== 'https:')
    throw new Error('the published default model needs an HTTPS downloadUrl before @natlang/node can be packed');
  const adaptationExamples = join(root, 'npm-packages', 'node', 'examples', 'adaptation');
  mkdirSync(adaptationExamples, { recursive: true });
  copyTree(join(root, 'examples', 'adaptation'), adaptationExamples, () => true);
  // The whole Node build, without the browser bundle and the repository's teacher tooling.
  copyTree(source, destination, () => true, [join(source, 'browser'), join(source, 'teacher')]);
  const pythonAssets = join(root, 'npm-packages', 'node', 'vendor', 'pyodide');
  mkdirSync(pythonAssets, { recursive: true });
  for (const name of readdirSync(join(root, 'ts-host', 'vendor', 'pyodide')))
    cpSync(join(root, 'ts-host', 'vendor', 'pyodide', name), join(pythonAssets, name));
  cpSync(join(root, 'ts-host', 'prelude.js'), join(root, 'npm-packages', 'node', 'prelude.js'));
  const axVendorSource = join(root, 'vendor', 'ax-gepa');
  const axVendorTarget = join(root, 'npm-packages', 'node', 'vendor', 'ax-gepa');
  mkdirSync(axVendorTarget, { recursive: true });
  for (const name of ['LICENSE', 'CHANGES.md', 'UPSTREAM.json'])
    cpSync(join(axVendorSource, name), join(axVendorTarget, name));
  const modelAssets = join(root, 'npm-packages', 'node', 'model-assets');
  rmSync(modelAssets, { recursive: true, force: true }); mkdirSync(modelAssets, { recursive: true });
  cpSync(join(root, 'models', 'templates', DEFAULT_MODEL_RELEASE.template), join(modelAssets, 'default.jinja'));
} else {
  // The single-file bundle and its WASM assets at the package root; declarations keep the build's layout.
  for (const name of readdirSync(join(source, 'browser')))
    if (['natlang.js', 'wllama-compat.js'].includes(name) || name.endsWith('.wasm')) cpSync(join(source, 'browser', name), join(destination, name));
  cpSync(join(source, 'browser', 'pyodide'), join(destination, 'pyodide'), { recursive: true });
  cpSync(join(source, 'browser', 'chunks'), join(destination, 'chunks'), { recursive: true });
  const types = join(destination, 'types'); mkdirSync(types, { recursive: true });
  copyTree(source, types, path => path.endsWith('.d.ts') &&
    !path.includes('/evaluation/') && !path.includes('/optimization/'), [join(source, 'teacher')]);
}
