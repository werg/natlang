import { existsSync, readdirSync, statSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { createRequire } from 'node:module';

function newest(path) {
  if (!existsSync(path)) return 0;
  const status = statSync(path);
  if (!status.isDirectory()) return status.mtimeMs;
  return readdirSync(path).reduce((latest, name) => Math.max(latest, newest(join(path, name))), status.mtimeMs);
}

/** Compile Node-facing TypeScript only when a checkout input is newer than the CLI. */
export function ensureNodeBuild(root) {
  const host = join(root, 'ts-host'), output = join(host, 'dist', 'cli', 'main.js');
  try { createRequire(join(host, 'package.json')).resolve('typescript'); }
  catch { throw new Error('TypeScript dependencies are missing; run scripts/setup_dev.sh --node-only first'); }
  const built = existsSync(output) ? statSync(output).mtimeMs : 0;
  const changed = Math.max(newest(join(host, 'src')), newest(join(host, 'prelude.js')),
    newest(join(host, 'tsconfig.json')), newest(join(host, 'package.json')));
  if (built >= changed) return false;
  process.stderr.write('natlang dev: TypeScript sources changed; rebuilding the Node host\n');
  execFileSync('npm', ['--prefix', host, 'run', 'build:node'], { stdio: 'inherit' });
  return true;
}
