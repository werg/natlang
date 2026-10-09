/**
 * Build system in natural language. build.nl reads the declared tasks as a dependency graph, plans the goal's closure
 * and order, and runs one round at a time: pick a ready task, judge whether its recorded outputs are still valid,
 * run or reuse it, diagnose a failure, and report. This file is the outside world those stages act on, as the `build`
 * service: running a command without a shell, keeping paths inside the root, input and output digests, the cache of
 * exact built-in operations, and the ledger of last successful runs. Tasks are trusted argv commands with declared
 * inputs and outputs. The workspace refuses preexisting outputs unless the ledger recorded exactly those bytes, and
 * refuses a task whose declared input changed while it ran. A timeout is an unknown outcome.
 */
import { createHash, randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createReadStream } from 'node:fs';
import { copyFile, lstat, mkdir, readFile, realpath, rename, stat, unlink, writeFile } from 'node:fs/promises';
import { isAbsolute, join, relative, resolve } from 'node:path';
import { ONCE_EFFECTS, type FolderHandle, type NatlangRuntime } from '@natlang/node';
import build from './build.nl';
import type { BuildReport, Evidence, Task, TaskResult } from './types.js';

export type * from './types.js';

/** The pluggable policy points, and the implementation each runs when the host does not choose. */
export type PolicyPoint = 'ready' | 'choose' | 'validity';
export type Implementation = 'crisp' | 'natural-language';
export const DEFAULT_POLICY: Record<PolicyPoint, Implementation> = { ready: 'crisp', choose: 'natural-language', validity: 'natural-language' };

/** What the natural-language stages see of the `build` service. Types are those of types.ts. */
export const buildDeclaration = `/** The build workspace: the outside world of the build stages. */
/** Which implementation runs a pluggable policy point: 'crisp' or 'natural-language'. */
export function implementation(point: 'ready' | 'choose' | 'validity'): 'crisp' | 'natural-language';
/**
 * What the workspace sees and recorded for a task: its declaration digest, its inputs' and outputs' current digests
 * (sha256 is null for a missing file) and the ledger entry of its last successful run (recorded is null when none).
 */
export function inspect(task: Task): Promise<Evidence>;
/**
 * Run a task without a shell, inside the root. Declared inputs are digested before and after. Outputs that already
 * exist are refused unless the ledger recorded exactly those bytes. A task runs once per workspace. status unknown
 * means the process was interrupted and its effects may have happened.
 */
export function execute(task: Task): Promise<TaskResult>;
/**
 * Settle a task from its ledger entry without running it. It checks the declaration, inputs and outputs against the
 * entry again, and answers status failed with the first difference when they differ.
 */
export function reuse(task: Task): Promise<TaskResult>;`;

function within(root: string, path: string): boolean {
  const rel = relative(root, path);
  return rel !== '' && rel !== '..' && !rel.startsWith(`..${process.platform === 'win32' ? '\\' : '/'}`) && !isAbsolute(rel);
}

async function digestFile(path: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}
const errorCode = (error: unknown) => (error as NodeJS.ErrnoException).code;
const sha = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
type LedgerEntry = { fingerprint: string, inputs: [string, string][], outputs: [string, string][] };
const fingerprint = (task: Task) => sha({ argv: task.argv, inputs: task.inputs, outputs: task.outputs });

export class BuildWorkspace {
  private readonly rootPath: string;
  private readonly cacheDir: string;
  private readonly ledgerDir: string;
  private readonly timeoutMs: number;
  private readonly outputLimit: number;
  private readonly maxBuiltinBytes: number;
  private readonly policy: Record<PolicyPoint, Implementation>;
  private root: string | null = null;
  private ledger: Record<string, LedgerEntry> = {};
  private readonly events: Record<string, unknown>[] = [];
  /** Running a command or reusing a task happens once per workspace and task: an executor repeating a call gets the first result. */
  readonly [ONCE_EFFECTS] = ['execute', 'reuse'];

  constructor(root: string, { timeoutMs = 30_000, outputLimit = 4096, cacheDir = null as string | null, ledgerDir = null as string | null,
    maxBuiltinBytes = 64 * 1024 * 1024, policy = {} as Partial<Record<PolicyPoint, Implementation>> } = {}) {
    this.rootPath = resolve(root);
    this.cacheDir = resolve(cacheDir ?? join(this.rootPath, '.natlang-build-cache'));
    this.ledgerDir = resolve(ledgerDir ?? join(this.rootPath, '.natlang-build-ledger'));
    this.timeoutMs = timeoutMs;
    this.outputLimit = outputLimit;
    this.maxBuiltinBytes = maxBuiltinBytes;
    this.policy = { ...DEFAULT_POLICY, ...policy };
  }

  async open(): Promise<this> {
    this.root = await realpath(this.rootPath);
    try { this.ledger = JSON.parse(await readFile(join(this.ledgerDir, 'ledger.json'), 'utf8')); }
    catch (error) {
      if (errorCode(error) !== 'ENOENT') this.events.push({ operation: 'build.ledger', status: 'unreadable', detail: (error as Error).message });
      this.ledger = {};
    }
    return this;
  }

  implementation(point: PolicyPoint): Implementation {
    if (!Object.hasOwn(this.policy, point)) throw new Error(`unknown policy point: ${point}`);
    return this.policy[point];
  }

  private reserved(lexical: string): boolean {
    return [this.cacheDir, this.ledgerDir].some(dir => within(this.root!, dir) && (lexical === dir || within(dir, lexical)));
  }

  private async path(name: string, { existing, allowExisting = false }: { existing: boolean, allowExisting?: boolean }): Promise<string> {
    const root = this.root;
    if (!root || typeof name !== 'string' || !name || isAbsolute(name)) throw new Error(`invalid workspace path: ${name}`);
    const lexical = resolve(root, name);
    if (!within(root, lexical)) throw new Error(`path escapes workspace: ${name}`);
    if (this.reserved(lexical)) throw new Error(`task path enters host cache: ${name}`);
    if (existing) {
      const actual = await realpath(lexical);
      if (!within(root, actual)) throw new Error(`input escapes workspace: ${name}`);
      if (!(await lstat(actual)).isFile()) throw new Error(`input is not a file: ${name}`);
      return actual;
    }
    const parent = await realpath(resolve(lexical, '..'));
    if (parent !== root && !within(root, parent)) throw new Error(`output escapes workspace: ${name}`);
    if (!allowExisting) {
      try { await lstat(lexical); throw new Error(`output already exists: ${name}`); }
      catch (error) { if (errorCode(error) !== 'ENOENT') throw error; }
    }
    return lexical;
  }

  /** The digest of an existing file inside the root, or null when it is missing or unreadable. */
  private async digestOrNull(name: string): Promise<string | null> {
    try { return await digestFile(await this.path(name, { existing: true })); } catch { return null; }
  }

  private async saveLedger(): Promise<void> {
    try {
      await mkdir(this.ledgerDir, { recursive: true });
      const temporary = join(this.ledgerDir, `${randomUUID()}.json.tmp`);
      await writeFile(temporary, JSON.stringify(this.ledger));
      await rename(temporary, join(this.ledgerDir, 'ledger.json'));
    } catch (error) {
      this.events.push({ operation: 'build.ledger', status: 'write-failed', detail: error instanceof Error ? error.message : String(error) });
    }
  }

  async inspect(task: Task): Promise<Evidence> {
    const entry = this.ledger[task.id];
    const inputs = await Promise.all(task.inputs.map(async path => ({ path, sha256: await this.digestOrNull(path) })));
    const outputs = await Promise.all(task.outputs.map(async path => ({ path, sha256: await this.digestOrNull(path) })));
    return { task: task.id, fingerprint: fingerprint(task), inputs, outputs,
      recorded: entry ? { fingerprint: entry.fingerprint, inputs: entry.inputs.map(([path, sha256]) => ({ path, sha256 })),
        outputs: entry.outputs.map(([path, sha256]) => ({ path, sha256 })) } : null };
  }

  /** Whether the ledger entry still describes the files exactly; the first difference otherwise. */
  private async mismatch(task: Task): Promise<string | null> {
    const entry = this.ledger[task.id];
    if (!entry) return 'no record of an earlier run';
    if (entry.fingerprint !== fingerprint(task)) return 'declaration changed';
    for (const [name, before] of entry.inputs) if (await this.digestOrNull(name) !== before) return `input changed: ${name}`;
    for (const [name, before] of entry.outputs) {
      const now = await this.digestOrNull(name);
      if (now === null) return `output missing: ${name}`;
      if (now !== before) return `output modified: ${name}`;
    }
    return null;
  }

  async reuse(task: Task): Promise<TaskResult> {
    const id = String(task.id);
    const problem = await this.mismatch(task);
    if (problem) {
      this.events.push({ operation: 'build.reuse', task: id, status: 'refused', detail: problem });
      return { id, status: 'failed', exit_code: -1, input_sha256: '', output_sha256: '', detail: `reuse refused: ${problem}` };
    }
    const entry = this.ledger[id]!;
    const result: TaskResult = { id, status: 'ok', exit_code: 0, input_sha256: sha(entry.inputs), output_sha256: sha(entry.outputs), detail: 'up to date' };
    this.events.push({ operation: 'build.reuse', task: id, status: 'ok', input_sha256: result.input_sha256, output_sha256: result.output_sha256 });
    return result;
  }

  /** Forget a task's earlier run and remove its recorded outputs, but only those still holding the recorded bytes. */
  private async retire(task: Task): Promise<void> {
    const entry = this.ledger[task.id];
    if (!entry) return;
    for (const [name, before] of entry.outputs) {
      if (!task.outputs.includes(name) || await this.digestOrNull(name) !== before) continue;
      await unlink(await this.path(name, { existing: true }));
      this.events.push({ operation: 'build.retire', task: task.id, output: name });
    }
    delete this.ledger[task.id];
    await this.saveLedger();
  }

  private async remember(task: Task, inputs: [string, string][]): Promise<void> {
    const outputs: [string, string][] = [];
    for (const name of task.outputs) outputs.push([name, await digestFile(await this.path(name, { existing: true }))]);
    this.ledger[task.id] = { fingerprint: fingerprint(task), inputs, outputs };
    await this.saveLedger();
  }

  private async builtin(task: Task, inputs: [string, string][], input_sha256: string): Promise<TaskResult> {
    const op = task.argv[1];
    if (!['copy', 'concat', 'uppercase'].includes(op) || task.argv.length !== 2 ||
        task.outputs.length !== 1 || !task.inputs.length ||
        (op !== 'concat' && task.inputs.length !== 1))
      throw new Error('invalid built-in operation or arity');
    const outputName = task.outputs[0]!;
    const key = createHash('sha256').update(JSON.stringify({ version: 1, node: process.version,
      op, outputName, inputs })).digest('hex');
    const directory = join(this.cacheDir, key), data = join(directory, 'output.bin');
    const manifestPath = join(directory, 'manifest.json');
    let manifest: { key: string, sha256: string, output_sha256: string };
    try {
      manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
      if (manifest.key !== key || manifest.sha256 !== await digestFile(data))
        throw new Error('cache content mismatch');
      for (const [name, before] of inputs)
        if (await digestFile(await this.path(name, { existing: true })) !== before)
          throw new Error(`input changed during cache lookup: ${name}`);
      const target = await this.path(outputName, { existing: false, allowExisting: true });
      try {
        await lstat(target);
        if (await digestFile(target) !== manifest.sha256) throw new Error(`existing output differs from cache: ${outputName}`);
      } catch (error) {
        if (errorCode(error) !== 'ENOENT') throw error;
        await copyFile(data, target);
      }
      this.events.push({ operation: 'build.cache', task: task.id, status: 'hit', key });
      return { id: task.id, status: 'ok', exit_code: 0, input_sha256,
        output_sha256: manifest.output_sha256, detail: 'cache hit' };
    } catch (error) {
      if (errorCode(error) !== 'ENOENT') this.events.push({ operation: 'build.cache', task: task.id,
        status: 'invalid', key, detail: (error as Error).message });
    }
    const target = await this.path(outputName, { existing: false });
    const paths = await Promise.all(task.inputs.map(name => this.path(name, { existing: true })));
    const sizes = await Promise.all(paths.map(path => stat(path).then(info => info.size)));
    if (sizes.reduce((sum, size) => sum + size, 0) > this.maxBuiltinBytes)
      throw new Error('built-in input exceeds configured memory allowance');
    const buffers = await Promise.all(paths.map(path => readFile(path)));
    for (let index = 0; index < buffers.length; index++)
      if (createHash('sha256').update(buffers[index]!).digest('hex') !== inputs[index]![1])
        throw new Error(`input changed during built-in operation: ${task.inputs[index]}`);
    const value = op === 'copy' ? buffers[0]! : op === 'concat' ? Buffer.concat(buffers) :
      Buffer.from(buffers[0]!.toString('utf8').toUpperCase(), 'utf8');
    const temporary = `${target}.${randomUUID()}.tmp`;
    await writeFile(temporary, value);
    await rename(temporary, target);
    const sha256 = await digestFile(target);
    const output_sha256 = createHash('sha256').update(JSON.stringify([[outputName, sha256]])).digest('hex');
    try {
      await mkdir(directory, { recursive: true });
      const cacheTemp = join(directory, `${randomUUID()}.tmp`);
      await copyFile(target, cacheTemp); await rename(cacheTemp, data);
      const manifestTemp = join(directory, `${randomUUID()}.json.tmp`);
      await writeFile(manifestTemp, JSON.stringify({ key, sha256, output_sha256 }));
      await rename(manifestTemp, manifestPath);
      this.events.push({ operation: 'build.cache', task: task.id, status: 'stored', key });
    } catch (error) {
      this.events.push({ operation: 'build.cache', task: task.id, status: 'write-failed', key,
        detail: error instanceof Error ? error.message : String(error) });
    }
    return { id: task.id, status: 'ok', exit_code: 0, input_sha256, output_sha256,
      detail: `built-in ${op}` };
  }

  async execute(task: Task): Promise<TaskResult> {
    const id = String(task.id);
    let input_sha256 = '';
    const fail = (status: 'failed' | 'unknown', detail: string, exit_code = -1): TaskResult => {
      this.events.push({ operation: 'build.execute', task: id, status, detail });
      return { id, status, exit_code, input_sha256, output_sha256: '', detail };
    };
    try {
      if (!Array.isArray(task.argv) || !task.argv.length ||
          !Array.isArray(task.inputs) || !Array.isArray(task.outputs))
        return fail('failed', 'invalid task declaration');
      await this.retire(task);
      const inputs: [string, string][] = [];
      for (const input of task.inputs) {
        const path = await this.path(input, { existing: true });
        inputs.push([input, await digestFile(path)]);
      }
      input_sha256 = sha(inputs);
      if (task.argv[0] === '@builtin') {
        const result = await this.builtin(task, inputs, input_sha256);
        await this.remember(task, inputs);
        this.events.push({ operation: 'build.execute', task: id, status: 'ok',
          input_sha256, output_sha256: result.output_sha256, detail: result.detail });
        return result;
      }
      for (const output of task.outputs) await this.path(output, { existing: false });
      const child = spawn(task.argv[0]!, task.argv.slice(1), {
        cwd: this.root!, shell: false, stdio: ['ignore', 'pipe', 'pipe'],
        env: { PATH: process.env.PATH ?? '', LANG: 'C', LC_ALL: 'C' },
      });
      let output = '';
      for (const stream of [child.stdout, child.stderr]) stream.on('data', (chunk: Buffer) => {
        if (output.length < this.outputLimit) output += chunk.toString().slice(0, this.outputLimit - output.length);
      });
      let timedOut = false;
      const timer = setTimeout(() => { timedOut = true; child.kill('SIGKILL'); }, this.timeoutMs);
      const closed = await new Promise<{ error?: Error, code?: number | null, signal?: NodeJS.Signals | null }>(resolveClose => {
        child.once('error', error => resolveClose({ error }));
        child.once('close', (code, signal) => resolveClose({ code, signal }));
      });
      clearTimeout(timer);
      if (timedOut || closed.signal) return fail('unknown', `process interrupted; effects may have occurred: ${output}`);
      if (closed.error) return fail('failed', `${closed.error.message}: ${output}`);
      if (closed.code !== 0) return fail('failed', `exit ${closed.code}: ${output}`, closed.code ?? -1);
      for (const [name, before] of inputs) {
        const path = await this.path(name, { existing: true });
        if (await digestFile(path) !== before) return fail('failed', `declared input changed: ${name}`);
      }
      const hashes: [string, string][] = [];
      for (const name of task.outputs) {
        const path = await this.path(name, { existing: true });
        hashes.push([name, await digestFile(path)]);
      }
      const output_sha256 = sha(hashes);
      await this.remember(task, inputs);
      this.events.push({ operation: 'build.execute', task: id, status: 'ok', input_sha256, output_sha256 });
      return { id, status: 'ok', exit_code: 0, input_sha256, output_sha256, detail: output };
    } catch (error) {
      return fail('failed', error instanceof Error ? error.message : String(error));
    }
  }

  drainEvents(): Record<string, unknown>[] { return this.events.splice(0); }
}

/** The options that give natural-language stages the workspace as the `build` service. */
export function buildServices(workspace: BuildWorkspace) {
  return { services: { build: workspace }, serviceDeclarations: { build: buildDeclaration } };
}

/**
 * Build `goal` from `tasks` with the natural-language build system, one round per task at most. The workspace is its
 * outside world; `files` lets the scheduling policy read a task's named local input.
 */
export function buildGoal(runtime: NatlangRuntime, workspace: BuildWorkspace, goal: string, tasks: Task[], files?: FolderHandle): Promise<BuildReport> {
  return runtime.run(() => build(goal, tasks, files), buildServices(workspace));
}
