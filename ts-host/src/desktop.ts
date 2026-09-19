import { readFileSync, statSync, writeFileSync } from 'node:fs';
import { spawn, spawnSync, type ChildProcessWithoutNullStreams } from 'node:child_process';
import type { HostEvent } from './environment.js';

export type JobState = { id: string; status: 'running' | 'finished' | 'failed';
  exitCode: number | null; signal: string | null; stdout: string; stderr: string;
  error: string | null };
type Job = { child: ChildProcessWithoutNullStreams; state: JobState; timeout: NodeJS.Timeout };

/** Optional direct Node bindings for a trusted desktop application. */
export class DesktopBindings {
  private nextId = 1;
  private buffers = new Map<string, Buffer>();
  private bufferBytes = 0;
  private jobs = new Map<string, Job>();
  private events: HostEvent[] = [];
  private dropped = 0;
  private disposed = false;
  readonly limits: Readonly<{ buffers: number; bufferBytes: number; jobs: number; outputBytes: number }>;

  constructor(limits: Partial<{ buffers: number; bufferBytes: number; jobs: number; outputBytes: number }> = {}) {
    this.limits = Object.freeze({ buffers: limits.buffers ?? 32, bufferBytes: limits.bufferBytes ?? 64 * 1024 * 1024,
      jobs: limits.jobs ?? 32, outputBytes: limits.outputBytes ?? 1024 * 1024 });
    if (Object.values(this.limits).some(v => !Number.isInteger(v) || v < 1)) throw new RangeError('limits must be positive integers');
  }

  private active(): void { if (this.disposed) throw new Error('desktop bindings are disposed'); }
  private id(prefix: string): string { return `${prefix}-${this.nextId++}`; }
  private record(operation: string, detail: Record<string, unknown>): void {
    if (this.events.length < 1024) this.events.push({ operation, ...detail }); else this.dropped++;
  }
  drainEvents(): HostEvent[] {
    const events = this.events; this.events = [];
    if (this.dropped) events.unshift({ operation: 'observation.dropped', count: this.dropped });
    this.dropped = 0;
    return events;
  }

  readText(path: string): string {
    this.active();
    if (statSync(path).size > this.limits.bufferBytes) throw new Error('file exceeds read limit');
    const text = readFileSync(path, 'utf8');
    this.record('file.readText', { path, bytes: Buffer.byteLength(text) });
    return text;
  }
  writeText(path: string, text: string): void {
    this.active();
    if (Buffer.byteLength(text) > this.limits.bufferBytes) throw new Error('write exceeds limit');
    writeFileSync(path, text);
    this.record('file.writeText', { path, bytes: Buffer.byteLength(text) });
  }
  readBytes(path: string): string {
    this.active();
    const size = statSync(path).size;
    if (this.buffers.size >= this.limits.buffers || this.bufferBytes + size > this.limits.bufferBytes)
      throw new Error('retained buffer limit exceeded');
    const buffer = readFileSync(path);
    const id = this.id('buffer');
    this.buffers.set(id, buffer); this.bufferBytes += buffer.length;
    this.record('file.readBytes', { path, id, bytes: buffer.length });
    return id;
  }
  buffer(id: string): Buffer {
    this.active();
    const value = this.buffers.get(id);
    if (!value) throw new Error(`unknown or disposed buffer ${id}`);
    return value;
  }
  release(id: string): void {
    this.active();
    const buffer = this.buffers.get(id);
    if (buffer) {
      this.bufferBytes -= buffer.length; this.buffers.delete(id);
      this.record('buffer.release', { id }); return;
    }
    const job = this.jobs.get(id);
    if (!job) throw new Error(`unknown or disposed resource ${id}`);
    if (job.state.status === 'running') throw new Error('cancel or wait for the job before release');
    clearTimeout(job.timeout); this.jobs.delete(id);
    this.record('job.release', { id });
  }
  private argv(argv: string[]): void {
    if (!Array.isArray(argv) || argv.length === 0 || argv.some(x => typeof x !== 'string'))
      throw new TypeError('command requires nonempty string argv');
  }
  run(argv: string[], options: { cwd?: string; input?: string; timeoutMs?: number } = {}) {
    this.active(); this.argv(argv);
    const out = spawnSync(argv[0]!, argv.slice(1), { cwd: options.cwd, input: options.input,
      encoding: 'utf8', timeout: options.timeoutMs ?? 5000, maxBuffer: this.limits.outputBytes });
    this.record('process.run', { argv, status: out.status, signal: out.signal, error: out.error?.message ?? null });
    return { status: out.status, signal: out.signal, stdout: out.stdout ?? '', stderr: out.stderr ?? '',
      error: out.error?.message ?? null };
  }
  start(argv: string[], options: { cwd?: string; timeoutMs?: number } = {}): string {
    this.active(); this.argv(argv);
    if (this.jobs.size >= this.limits.jobs) throw new Error('retained job limit exceeded');
    const id = this.id('job');
    const child = spawn(argv[0]!, argv.slice(1), { cwd: options.cwd, stdio: ['pipe', 'pipe', 'pipe'] });
    child.stdin.end();
    const state: JobState = { id, status: 'running', exitCode: null, signal: null, stdout: '', stderr: '', error: null };
    const capture = (field: 'stdout' | 'stderr', chunk: Buffer) => {
      const combined = state[field] + chunk.toString('utf8');
      state[field] = combined.slice(0, this.limits.outputBytes);
      if (combined.length > this.limits.outputBytes) this.record('process.outputTruncated', { id, field });
    };
    child.stdout.on('data', chunk => capture('stdout', chunk));
    child.stderr.on('data', chunk => capture('stderr', chunk));
    child.on('error', error => { state.status = 'failed'; state.error = error.message;
      this.record('process.failed', { id, error: error.message }); });
    child.on('exit', (code, signal) => { state.status = state.status === 'failed' ? 'failed' : 'finished';
      state.exitCode = code; state.signal = signal; clearTimeout(timeout);
      this.record('process.completed', { id, exitCode: code, signal }); });
    const duration = Math.min(Math.max(options.timeoutMs ?? 60_000, 1), 600_000);
    const timeout = setTimeout(() => { if (state.status === 'running') {
      child.kill(); this.record('process.timeout', { id }); } }, duration);
    timeout.unref();
    this.jobs.set(id, { child, state, timeout });
    this.record('process.start', { id, argv });
    return id;
  }
  poll(id: string): JobState {
    this.active();
    const job = this.jobs.get(id);
    if (!job) throw new Error(`unknown or disposed job ${id}`);
    return { ...job.state };
  }
  cancel(id: string) {
    this.active();
    const job = this.jobs.get(id);
    if (!job) throw new Error(`unknown or disposed job ${id}`);
    const requested = job.state.status === 'running' && job.child.kill();
    this.record('process.cancel', { id, requested });
    return { id, requested, status: job.state.status };
  }
  close(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const job of this.jobs.values()) {
      clearTimeout(job.timeout);
      if (job.state.status === 'running') job.child.kill();
    }
    this.jobs.clear(); this.buffers.clear(); this.bufferBytes = 0;
  }
}
