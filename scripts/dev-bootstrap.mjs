import { existsSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

/** The newest modification time under `path` (a file or a directory tree). */
function newest(path) {
  if (!existsSync(path)) return 0;
  const status = statSync(path);
  if (!status.isDirectory()) return status.mtimeMs;
  return readdirSync(path, { withFileTypes: true }).reduce((latest, entry) =>
    Math.max(latest, entry.isDirectory() ? newest(join(path, entry.name)) : statSync(join(path, entry.name)).mtimeMs), status.mtimeMs);
}

/** Compile Node-facing TypeScript only when a checkout input is newer than the CLI (a few ms of stat calls otherwise). */
export async function ensureNodeBuild(root) {
  const host = join(root, 'ts-host'), output = join(host, 'dist', 'cli', 'main.js');
  const built = existsSync(output) ? statSync(output).mtimeMs : 0;
  const changed = Math.max(newest(join(host, 'src')), newest(join(host, 'prelude.js')),
    newest(join(host, 'tsconfig.json')), newest(join(host, 'package.json')));
  if (built >= changed) return false;
  const { createRequire } = await import('node:module');
  try { createRequire(join(host, 'package.json')).resolve('typescript'); }
  catch { throw new Error('TypeScript dependencies are missing; run scripts/setup_dev.sh --node-only first'); }
  process.stderr.write('natlang dev: TypeScript sources changed; rebuilding the Node host\n');
  const { execFileSync } = await import('node:child_process');
  execFileSync('npm', ['--prefix', host, 'run', 'build:node'], { stdio: 'inherit' });
  return true;
}
