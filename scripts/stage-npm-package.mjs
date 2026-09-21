import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const kind = process.argv[2];
if (!['node', 'browser'].includes(kind)) throw new Error('usage: stage-npm-package.mjs node|browser');
const destination = join(root, 'npm-packages', kind, 'dist');
rmSync(destination, { recursive: true, force: true }); mkdirSync(destination, { recursive: true });

if (kind === 'node') {
  const source = join(root, 'ts-host', 'dist');
  for (const name of ['cli', 'model', 'native', 'package', 'terminal'])
    cpSync(join(source, name), join(destination, name), { recursive: true });
  for (const name of ['contracts.d.ts', 'contracts.js', 'desktop.d.ts', 'desktop.js',
    'environment.d.ts', 'environment.js', 'index.d.ts', 'index.js'])
    cpSync(join(source, name), join(destination, name));
  cpSync(join(root, 'ts-host', 'prelude.js'), join(root, 'npm-packages', 'node', 'prelude.js'));
} else {
  cpSync(join(root, 'ts-host', 'dist', 'browser'), destination, { recursive: true });
  // The public declarations refer to shared interpreter declaration files.
  const nativeSource = join(root, 'ts-host', 'dist', 'native'), nativeDestination = join(destination, 'native');
  mkdirSync(nativeDestination, { recursive: true });
  for (const name of ['agent.d.ts', 'codebase.d.ts', 'evaluator.d.ts', 'runtime.d.ts', 'scenario.d.ts',
    'source.d.ts', 'types.d.ts', 'values.d.ts', 'workspace.d.ts'])
    cpSync(join(nativeSource, name), join(nativeDestination, name));
  for (const name of ['contracts.d.ts']) cpSync(join(root, 'ts-host', 'dist', name), join(destination, name));
  // Rebase declaration paths because browser files now live at package dist root.
  for (const name of ['index.d.ts', 'host.d.ts']) {
    const path = join(destination, name);
    writeFileSync(path, readFileSync(path, 'utf8').replaceAll("'../native/", "'./native/")
      .replaceAll("'../contracts.js'", "'./contracts.js'"));
  }
}
