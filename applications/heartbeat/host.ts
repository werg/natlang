// The machine, as the heartbeat touches it. Every read and every write goes through a Host, so the collectors, the
// applier and the cycle run on fixtures in tests and on the real machine in production.
import { execFile } from 'node:child_process';
import { appendFile, mkdir, open, readFile, stat, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { hostname } from 'node:os';

export type ExecResult = { ok: boolean, code: number | null, stdout: string, stderr: string };
export type ExecOptions = { timeoutMs?: number, input?: string, cwd?: string, env?: Record<string, string> };
export type Host = {
  /** Run argv (no shell) and return its output; a failure to start or a timeout is `ok: false`, never a throw. */
  exec(argv: string[], options?: ExecOptions): Promise<ExecResult>;
  /** A small text file, or null when it does not exist. */
  readText(path: string): Promise<{ text: string, mtimeMs: number } | null>;
  /** The last `bytes` bytes of a file (a log), or null when it does not exist. */
  readTail(path: string, bytes: number): Promise<{ text: string, mtimeMs: number } | null>;
  appendText(path: string, text: string): Promise<void>;
  writeText(path: string, text: string): Promise<void>;
  fetchText(url: string, timeoutMs: number): Promise<string | null>;
  now(): Date;
  /** Which machine this is: "dgx" or "pop". */
  machine(): string;
};

const MACHINES: Record<string, string> = { mltick: 'dgx', 'pop-os': 'pop' };

export function nodeHost(): Host {
  return {
    exec: (argv, options = {}) => new Promise(done => {
      const child = execFile(argv[0]!, argv.slice(1), { timeout: options.timeoutMs ?? 60_000, cwd: options.cwd, maxBuffer: 32 << 20,
        env: { ...process.env, ...options.env } }, (error, stdout, stderr) => {
        const code = error ? (typeof (error as { code?: unknown }).code === 'number' ? (error as { code: number }).code : null) : 0;
        done({ ok: !error, code, stdout: String(stdout), stderr: String(stderr || (error ? error.message : '')) });
      });
      if (options.input !== undefined) child.stdin?.end(options.input);
    }),
    async readText(path) {
      try { return { text: await readFile(path, 'utf8'), mtimeMs: (await stat(path)).mtimeMs }; } catch { return null; }
    },
    async readTail(path, bytes) {
      let handle;
      try {
        handle = await open(path, 'r');
        const info = await handle.stat();
        const length = Math.min(bytes, info.size);
        const buffer = Buffer.alloc(length);
        await handle.read(buffer, 0, length, info.size - length);
        return { text: buffer.toString('utf8'), mtimeMs: info.mtimeMs };
      } catch { return null; } finally { await handle?.close(); }
    },
    async appendText(path, text) { await mkdir(dirname(path), { recursive: true }); await appendFile(path, text); },
    async writeText(path, text) { await mkdir(dirname(path), { recursive: true }); await writeFile(path, text); },
    async fetchText(url, timeoutMs) {
      try { return await (await fetch(url, { signal: AbortSignal.timeout(timeoutMs) })).text(); } catch { return null; }
    },
    now: () => new Date(),
    machine: () => process.env.HEARTBEAT_MACHINE ?? MACHINES[hostname()] ?? hostname(),
  };
}
