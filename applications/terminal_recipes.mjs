import { spawn } from 'node:child_process';
import { realpathSync } from 'node:fs';
import { isAbsolute, relative, resolve } from 'node:path';

function inside(root, path) {
  const rel = relative(root, path);
  return rel === '' || (!rel.startsWith('..') && !isAbsolute(rel));
}

/** Exact argv recipes for a trusted local workspace. No shell text is evaluated. */
export class CommandRecipeLibrary {
  constructor(root, definitions, { outputBytes = 64 * 1024, timeoutMs = 10 * 60_000 } = {}) {
    this.root = realpathSync(root);
    this.outputBytes = outputBytes;
    this.timeoutMs = timeoutMs;
    this.definitions = definitions.map(definition => {
      if (!definition.id || !definition.description || !Array.isArray(definition.argv) ||
          !definition.argv.length || definition.argv.some(value => typeof value !== 'string'))
        throw new Error('command recipe needs id, description, and string argv');
      const cwd = resolve(this.root, definition.cwd ?? '.');
      if (!inside(this.root, cwd)) throw new Error(`recipe cwd escapes workspace: ${definition.id}`);
      return { ...definition, cwd };
    });
    if (new Set(this.definitions.map(row => row.id)).size !== this.definitions.length)
      throw new Error('duplicate command recipe IDs');
  }

  recipes() {
    return this.definitions.map(definition => ({ id: definition.id,
      description: definition.description,
      run: context => this.run(definition, context) }));
  }

  run(definition, { signal } = {}) {
    return new Promise(resolveResult => {
      const child = spawn(definition.argv[0], definition.argv.slice(1), {
        cwd: definition.cwd, shell: false, stdio: ['ignore', 'pipe', 'pipe'],
        env: definition.env ? { ...process.env, ...definition.env } : process.env,
      });
      let stdout = Buffer.alloc(0), stderr = Buffer.alloc(0), settled = false, timedOut = false, cancelled = false;
      const capture = (field, chunk) => {
        const value = Buffer.concat([field === 'stdout' ? stdout : stderr, chunk]);
        const bounded = value.subarray(Math.max(0, value.length - this.outputBytes));
        if (field === 'stdout') stdout = bounded; else stderr = bounded;
      };
      child.stdout.on('data', chunk => capture('stdout', chunk));
      child.stderr.on('data', chunk => capture('stderr', chunk));
      let timer;
      const finish = (status, detail) => {
        if (settled) return; settled = true; clearTimeout(timer);
        signal?.removeEventListener('abort', abort);
        resolveResult({ status, detail });
      };
      child.on('error', error => finish('unknown', error.message));
      child.on('exit', (code, processSignal) => {
        const output = [stdout.toString('utf8').trim(), stderr.toString('utf8').trim()].filter(Boolean).join('\n');
        if (cancelled || timedOut) return finish('unknown',
          `${cancelled ? 'Cancellation' : 'Timeout'} requested; exit=${code}; signal=${processSignal}; ${output}`.trim());
        finish(code === 0 ? 'ok' : 'failed', `exit=${code}; ${output || '(no output)'}`);
      });
      const abort = () => {
        cancelled = true; child.kill('SIGTERM');
        const force = setTimeout(() => { if (!settled) child.kill('SIGKILL'); }, 5000);
        force.unref();
      };
      signal?.addEventListener('abort', abort, { once: true });
      if (signal?.aborted) abort();
      const duration = Math.max(1, Math.min(definition.timeoutMs ?? this.timeoutMs, 24 * 60 * 60_000));
      timer = setTimeout(() => { timedOut = true; child.kill('SIGTERM'); }, duration);
      timer.unref();
    });
  }
}

export function natlangWorkspaceRecipes(root) {
  return new CommandRecipeLibrary(root, [
    { id: 'repository-status', description: 'inspect concise Git working tree status', argv: ['git', 'status', '--short'] },
    { id: 'repository-diff', description: 'summarize current uncommitted Git changes', argv: ['git', 'diff', '--stat'] },
    { id: 'list-files', description: 'list workspace files tracked or visible to ripgrep', argv: ['rg', '--files'] },
    { id: 'test-python', description: 'run the natlang Python test suite',
      argv: [process.env.NATLANG_PYTHON ?? 'python3', '-m', 'pytest', '-q'], timeoutMs: 60 * 60_000 },
    { id: 'build-typescript-host', description: 'build the natlang TypeScript host and browser bundle',
      argv: ['npm', '--prefix', 'ts-host', 'run', 'build'], timeoutMs: 60 * 60_000 },
  ]);
}
