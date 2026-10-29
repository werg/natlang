/**
 * The dashboard's jobs: natlang CLI commands it starts and supervises (runs, calls, asks, builds, setup, tuning,
 * specialization). Each job is a child process of this CLI with its own directory under the state home: job.json,
 * stdout.log, stderr.log and live.jsonl (the run's live events, NATLANG_LIVE_FILE). Jobs survive a dashboard restart as
 * history; a job still marked running whose process is gone is shown as lost.
 */
import { spawn, type ChildProcess } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { createWriteStream, existsSync, mkdirSync, openSync, readdirSync, readFileSync, readSync, closeSync, rmSync, statSync,
  writeFileSync, type WriteStream } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

export type JobStatus = 'running' | 'done' | 'failed' | 'stopped' | 'lost';
export type JobRecord = { id: string; kind: string; title: string; argv: string[]; cwd: string; status: JobStatus;
  exitCode: number | null; pid: number | null; startedAt: string; endedAt: string | null; interactive: boolean;
  meta?: Record<string, unknown> };
export type JobSpec = { kind: string; title: string; argv: string[]; cwd: string; stdin?: string; interactive?: boolean;
  env?: Record<string, string>; meta?: Record<string, unknown> };

const CLI = fileURLToPath(new URL('../cli/main.js', import.meta.url));
const ID = /^[0-9a-z-]{8,64}$/;

export class JobSupervisor {
  private readonly children = new Map<string, { child: ChildProcess; out: WriteStream; err: WriteStream }>();
  constructor(readonly root: string, private readonly environment: NodeJS.ProcessEnv = process.env) {
    mkdirSync(root, { recursive: true });
    for (const job of this.list()) if (job.status === 'running') this.save({ ...job, status: 'lost', endedAt: job.endedAt ?? new Date().toISOString() });
  }
  directory(id: string): string { if (!ID.test(id)) throw new Error('invalid job id'); return join(this.root, id); }
  private save(job: JobRecord): JobRecord { writeFileSync(join(this.directory(job.id), 'job.json'), JSON.stringify(job, null, 2) + '\n'); return job; }

  start(spec: JobSpec): JobRecord {
    const id = `${new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '')}-${randomUUID().slice(0, 8)}`.toLowerCase();
    const directory = this.directory(id); mkdirSync(directory, { recursive: true });
    const live = join(directory, 'live.jsonl'); writeFileSync(live, '');
    const child = spawn(process.execPath, [CLI, ...spec.argv], { cwd: spec.cwd, stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...this.environment, ...spec.env, NATLANG_LIVE_FILE: live, NATLANG_PROGRESS: 'line', NO_COLOR: '1', FORCE_COLOR: '0' } });
    const out = createWriteStream(join(directory, 'stdout.log')), err = createWriteStream(join(directory, 'stderr.log'));
    child.stdout!.pipe(out); child.stderr!.pipe(err);
    if (spec.stdin !== undefined) child.stdin!.end(spec.stdin);
    else if (!spec.interactive) child.stdin!.end();
    const job = this.save({ id, kind: spec.kind, title: spec.title, argv: spec.argv, cwd: spec.cwd, status: 'running', exitCode: null,
      pid: child.pid ?? null, startedAt: new Date().toISOString(), endedAt: null, interactive: !!spec.interactive, ...(spec.meta ? { meta: spec.meta } : {}) });
    this.children.set(id, { child, out, err });
    const finish = (code: number | null, failure?: Error) => {
      if (!this.children.has(id)) return;
      this.children.delete(id);
      if (failure) err.write(`\nnatlang dashboard: could not run the job: ${failure.message}\n`);
      const current = this.get(id)!;
      const stopped = current.status === 'stopped';
      this.save({ ...current, status: stopped ? 'stopped' : code === 0 ? 'done' : 'failed', exitCode: code, endedAt: new Date().toISOString() });
    };
    child.once('error', failure => finish(null, failure));
    child.once('close', code => finish(code));
    return job;
  }

  list(): JobRecord[] {
    if (!existsSync(this.root)) return [];
    return readdirSync(this.root).filter(name => ID.test(name) && existsSync(join(this.root, name, 'job.json')))
      .map(name => JSON.parse(readFileSync(join(this.root, name, 'job.json'), 'utf8')) as JobRecord)
      .sort((a, b) => b.startedAt.localeCompare(a.startedAt));
  }
  get(id: string): JobRecord | undefined {
    const path = join(this.directory(id), 'job.json');
    return existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) as JobRecord : undefined;
  }
  running(): number { return this.children.size; }

  /** Bytes of a job file from `offset` (at most 1 MiB at a time) and the offset to continue from. */
  read(id: string, file: 'stdout.log' | 'stderr.log' | 'live.jsonl', offset = 0): { text: string; offset: number } {
    const path = join(this.directory(id), file);
    if (!existsSync(path)) return { text: '', offset };
    const size = statSync(path).size;
    if (offset >= size) return { text: '', offset: size };
    const length = Math.min(size - offset, 2 ** 20), buffer = Buffer.alloc(length), fd = openSync(path, 'r');
    try { readSync(fd, buffer, 0, length, offset); } finally { closeSync(fd); }
    // Only whole lines of the event file, so a reader never parses half an event.
    let text = buffer.toString('utf8'), end = offset + length;
    if (file === 'live.jsonl') {
      const cut = text.lastIndexOf('\n') + 1;
      end = offset + Buffer.byteLength(text.slice(0, cut)); text = text.slice(0, cut);
    }
    return { text, offset: end };
  }

  input(id: string, text: string): void {
    const running = this.children.get(id);
    if (!running?.child.stdin?.writable) throw new Error('the job is not running or does not read input');
    running.child.stdin.write(text.endsWith('\n') ? text : text + '\n');
  }

  stop(id: string): JobRecord {
    const job = this.get(id);
    if (!job) throw new Error(`no job ${id}`);
    const running = this.children.get(id);
    if (!running) return job;
    this.save({ ...job, status: 'stopped' });
    running.child.kill('SIGINT');
    const timer = setTimeout(() => { if (this.children.has(id)) running.child.kill('SIGKILL'); }, 8000); timer.unref();
    return this.get(id)!;
  }

  remove(id: string): void {
    if (this.children.has(id)) throw new Error('stop the job before removing it');
    rmSync(this.directory(id), { recursive: true, force: true });
  }

  async close(): Promise<void> {
    for (const id of [...this.children.keys()]) this.stop(id);
    const deadline = Date.now() + 9000;
    while (this.children.size && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 100));
  }
}
