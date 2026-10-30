/** Watch authored inputs and run one CLI child at a time. Generated build/declaration files never trigger reruns. */
import { watch, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { spawn, type ChildProcess } from 'node:child_process';
import { fileURLToPath } from 'node:url';

export async function develop(source: string, runArguments: string[]): Promise<number> {
  const path = resolve(source), root = statSync(path).isDirectory() ? path : dirname(path);
  let child: ChildProcess | undefined, timer: ReturnType<typeof setTimeout> | undefined, stopping = false, pending = false;
  const launch = () => {
    if (stopping) return;
    pending = false;
    process.stderr.write('natlang: running; edit a source or input file to rerun (Ctrl+C to stop)\n');
    child = spawn(process.execPath, [fileURLToPath(new URL('./main.js', import.meta.url)), 'run', ...runArguments], { stdio: 'inherit' });
    child.once('error', error => process.stderr.write(`natlang: ${error.message}\n`));
    child.once('close', () => { child = undefined; if (pending) launch(); });
  };
  return new Promise<number>((done, reject) => {
    const watcher = watch(root, { recursive: true }, (_event, filename) => {
      const name = String(filename ?? '');
      if (!name || name.split(/[\\/]/).some(part => ['.git', '.natlang', 'node_modules', 'dist'].includes(part)) ||
          /(?:\.d\.nl\.ts|\.tmp|~|\.swp)$/.test(name) || !/\.(?:nl|nz|ts|mts|js|json|md)$/.test(name)) return;
      clearTimeout(timer);
      timer = setTimeout(() => {
        pending = true;
        if (child) {
          const previous = child;
          previous.kill('SIGINT');
          const kill = setTimeout(() => { if (child === previous) previous.kill('SIGKILL'); }, 8000); kill.unref();
        } else launch();
      }, 250);
    });
    const stop = () => {
      if (stopping) return;
      stopping = true; watcher.close(); clearTimeout(timer);
      process.removeListener('SIGINT', stop); process.removeListener('SIGTERM', stop);
      if (!child) { done(0); return; }
      const previous = child;
      previous.once('close', () => { clearTimeout(kill); done(0); });
      previous.kill('SIGINT');
      const kill = setTimeout(() => previous.kill('SIGKILL'), 8000);
    };
    watcher.on('error', error => { stop(); reject(error); });
    process.on('SIGINT', stop); process.on('SIGTERM', stop);
    launch();
  });
}
