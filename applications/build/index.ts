/**
 * Build workbench: a declared dependency graph built one task at a time. `build` walks the goal's
 * dependency closure with `iterateOn`; natlang picks among ready tasks and `BuildWorkspace` runs the
 * pick. Tasks are trusted argv commands with declared inputs and outputs: the workspace runs them
 * without a shell, keeps paths inside the root, refuses preexisting outputs and changed inputs, and
 * records digests. A timeout is an unknown outcome. `@builtin` copy/concat/uppercase are exact
 * operations that may use the local cache; arbitrary commands are never cached.
 */
import { createHash, randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createReadStream } from 'node:fs';
import { copyFile, lstat, mkdir, readFile, realpath, rename, stat, writeFile } from 'node:fs/promises';
import { isAbsolute, join, relative, resolve } from 'node:path';
import { iterateOn, type FolderHandle } from '@natlang/node';
import choose from './choose.nl';
import type { Task } from './types.js';

export type * from './types.js';
export type TaskResult = { id: string, status: 'ok' | 'failed' | 'unknown', exit_code: number, input_sha256: string,
  output_sha256: string, detail: string };
export type BuildState = { goal: string, tasks: Task[], order: string[], results: TaskResult[], blocked: string[],
  status: 'running' | 'done' | 'failed' | 'unknown' | 'blocked' | 'invalid', detail: string };

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

export class BuildWorkspace {
  private readonly rootPath: string;
  private readonly cacheDir: string;
  private readonly timeoutMs: number;
  private readonly outputLimit: number;
  private readonly maxBuiltinBytes: number;
  private root: string | null = null;
  private readonly events: Record<string, unknown>[] = [];

  constructor(root: string, { timeoutMs = 30_000, outputLimit = 4096, cacheDir = null as string | null,
    maxBuiltinBytes = 64 * 1024 * 1024 } = {}) {
    this.rootPath = resolve(root);
    this.cacheDir = resolve(cacheDir ?? join(this.rootPath, '.natlang-build-cache'));
    this.timeoutMs = timeoutMs;
    this.outputLimit = outputLimit;
    this.maxBuiltinBytes = maxBuiltinBytes;
  }

  async open(): Promise<this> {
    this.root = await realpath(this.rootPath);
    return this;
  }

  private async path(name: string, { existing, allowExisting = false }: { existing: boolean, allowExisting?: boolean }): Promise<string> {
    const root = this.root;
    if (!root || typeof name !== 'string' || !name || isAbsolute(name)) throw new Error(`invalid workspace path: ${name}`);
    const lexical = resolve(root, name);
    if (!within(root, lexical)) throw new Error(`path escapes workspace: ${name}`);
    if (within(root, this.cacheDir) && (lexical === this.cacheDir || within(this.cacheDir, lexical)))
      throw new Error(`task path enters host cache: ${name}`);
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
      const inputs: [string, string][] = [];
      for (const input of task.inputs) {
        const path = await this.path(input, { existing: true });
        inputs.push([input, await digestFile(path)]);
      }
      input_sha256 = createHash('sha256').update(JSON.stringify(inputs)).digest('hex');
      if (task.argv[0] === '@builtin') {
        const result = await this.builtin(task, inputs, input_sha256);
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
      const output_sha256 = createHash('sha256').update(JSON.stringify(hashes)).digest('hex');
      this.events.push({ operation: 'build.execute', task: id, status: 'ok', input_sha256, output_sha256 });
      return { id, status: 'ok', exit_code: 0, input_sha256, output_sha256, detail: output };
    } catch (error) {
      return fail('failed', error instanceof Error ? error.message : String(error));
    }
  }

  drainEvents(): Record<string, unknown>[] { return this.events.splice(0); }
}

/** Check a graph before running it. */
function prepare(goal: string, tasks: Task[]): BuildState {
  const ids = new Set(tasks.map(task => task.id)), outputs = tasks.flatMap(task => task.outputs);
  let detail = '';
  if (!ids.has(goal)) detail = `unknown goal: ${goal}`;
  else if (ids.size !== tasks.length || tasks.some(task => !task.id)) detail = 'duplicate or empty task ID';
  else if (outputs.length !== new Set(outputs).size) detail = 'two tasks declare the same output';
  else if (tasks.some(task => !task.argv.length || task.outputs.length === 0)) detail = 'task lacks command or output';
  else if (tasks.some(task => task.needs.includes(task.id) || new Set(task.needs).size !== task.needs.length)) detail = 'self or duplicate dependency';
  else if (tasks.some(task => task.inputs.some(path => task.outputs.includes(path)))) detail = 'task reads its own output';
  return { goal, tasks, order: [], results: [], blocked: [], status: detail ? 'invalid' : 'running', detail };
}

function closure(state: BuildState): Set<string> {
  const byId = new Map(state.tasks.map(task => [task.id, task]));
  const needed = new Set<string>(), pending = [state.goal];
  for (const id of pending) if (!needed.has(id)) { needed.add(id); pending.push(...byId.get(id)?.needs ?? []); }
  return needed;
}

/** Run one ready task chosen by natlang, or report the graph blocked. */
async function advance(build: BuildWorkspace, state: BuildState, files?: FolderHandle): Promise<BuildState> {
  const done = new Set(state.order), needed = closure(state);
  const ready = state.tasks.filter(task => needed.has(task.id) && !done.has(task.id) && task.needs.every(need => done.has(need)));
  if (!ready.length) return { ...state, status: 'blocked', blocked: [...needed].filter(id => !done.has(id)).sort(),
    detail: 'No declared task is ready; dependencies may be missing or cyclic.' };
  const chosen = await choose(ready, state.goal, files);
  const task = ready.find(row => row.id === chosen);
  if (!task) return { ...state, status: 'invalid', detail: `chosen task is not ready: ${chosen}` };
  const result = await build.execute(task);
  const results = [...state.results, result];
  if (result.status !== 'ok') return { ...state, results, status: result.status, detail: `${task.id}: ${result.detail}` };
  return { ...state, order: [...state.order, task.id], results, status: task.id === state.goal ? 'done' : 'running', detail: '' };
}

/** Build `goal` from `tasks`, at most one round per declared task. */
export async function buildGoal(build: BuildWorkspace, goal: string, tasks: Task[], files?: FolderHandle): Promise<BuildState> {
  const initial = prepare(goal, tasks);
  if (initial.status !== 'running') return initial;
  return iterateOn((state: BuildState) => advance(build, state, files), initial)
    .withLimit({ maxSteps: tasks.length + 1 }).until(state => state.status !== 'running');
}
