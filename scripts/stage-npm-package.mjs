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
    if (skip.includes(path)) continue;
    if (statSync(path).isDirectory()) { mkdirSync(target, { recursive: true }); copyTree(path, target, keep, skip); }
    else if (keep(path)) cpSync(path, target);
  }
}

if (kind === 'node') {
  const { DEFAULT_MODEL_RELEASE } = await import(new URL('../ts-host/dist/model-default.js', import.meta.url));
  if (!DEFAULT_MODEL_RELEASE.downloadUrl || new URL(DEFAULT_MODEL_RELEASE.downloadUrl).protocol !== 'https:')
    throw new Error('the published default model needs an HTTPS downloadUrl before @natlang/node can be packed');
  // The whole Node build, without the browser bundle and the repository's teacher tooling.
  copyTree(source, destination, () => true, [join(source, 'browser'), join(source, 'teacher')]);
  cpSync(join(root, 'ts-host', 'prelude.js'), join(root, 'npm-packages', 'node', 'prelude.js'));
  const modelAssets = join(root, 'npm-packages', 'node', 'model-assets');
  rmSync(modelAssets, { recursive: true, force: true }); mkdirSync(modelAssets, { recursive: true });
  cpSync(join(root, 'models', 'templates', DEFAULT_MODEL_RELEASE.template), join(modelAssets, 'default.jinja'));
} else {
  // The single-file bundle and its WASM assets at the package root; declarations keep the build's layout.
  for (const name of readdirSync(join(source, 'browser')))
    if (['natlang.js', 'wllama-compat.js'].includes(name) || name.endsWith('.wasm')) cpSync(join(source, 'browser', name), join(destination, name));
  const types = join(destination, 'types'); mkdirSync(types, { recursive: true });
  copyTree(source, types, path => path.endsWith('.d.ts'), [join(source, 'teacher')]);
}
