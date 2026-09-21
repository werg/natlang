import { cpSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const kind = process.argv[2];
if (!['core', 'node', 'browser'].includes(kind)) throw new Error('usage: stage-npm-package.mjs core|node|browser');
const destination = join(root, 'npm-packages', kind, 'dist');
rmSync(destination, { recursive: true, force: true }); mkdirSync(destination, { recursive: true });

if (kind === 'core') {
  const source = join(root, 'ts-host', 'dist');
  cpSync(join(source, 'contracts.d.ts'), join(destination, 'contracts.d.ts'));
  cpSync(join(source, 'contracts.js'), join(destination, 'contracts.js'));
  const nativeDestination = join(destination, 'native'); mkdirSync(nativeDestination, { recursive: true });
  for (const stem of ['agent', 'codebase', 'evaluator', 'hash', 'prompt', 'runtime', 'scenario',
    'source-core', 'trace', 'type-aliases', 'types', 'values'])
    for (const extension of ['.js', '.d.ts']) cpSync(join(source, 'native', stem + extension),
      join(nativeDestination, stem + extension));
} else if (kind === 'node') {
  const source = join(root, 'ts-host', 'dist');
  for (const name of ['cli', 'model', 'native', 'package', 'terminal'])
    cpSync(join(source, name), join(destination, name), { recursive: true });
  for (const name of ['contracts.d.ts', 'contracts.js', 'desktop.d.ts', 'desktop.js',
    'environment.d.ts', 'environment.js', 'index.d.ts', 'index.js', 'node-runtime.d.ts', 'node-runtime.js'])
    cpSync(join(source, name), join(destination, name));
  cpSync(join(root, 'ts-host', 'prelude.js'), join(root, 'npm-packages', 'node', 'prelude.js'));
} else {
  cpSync(join(root, 'ts-host', 'dist', 'browser'), destination, { recursive: true });
  for (const name of readdirSync(destination).filter(name => name.endsWith('.js') &&
      !['natlang.js', 'wllama-compat.js'].includes(name))) rmSync(join(destination, name));
  // The public declarations refer to shared interpreter declaration files.
  const nativeSource = join(root, 'ts-host', 'dist', 'native'), nativeDestination = join(destination, 'native');
  mkdirSync(nativeDestination, { recursive: true });
  for (const name of readdirSync(nativeSource).filter(name => name.endsWith('.d.ts')))
    cpSync(join(nativeSource, name), join(nativeDestination, name));
  for (const name of ['contracts.d.ts']) cpSync(join(root, 'ts-host', 'dist', name), join(destination, name));
  // Rebase declaration paths because browser files now live at package dist root.
  for (const name of readdirSync(destination).filter(name => name.endsWith('.d.ts'))) {
    const path = join(destination, name);
    writeFileSync(path, readFileSync(path, 'utf8').replaceAll("'../native/", "'./native/")
      .replaceAll("'../contracts.js'", "'./contracts.js'"));
  }
}
