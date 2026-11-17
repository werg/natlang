/**
 * Interactive worlds that run in their own process (ScienceWorld, ALFWorld), exposed to a program as a host service.
 * The bridge speaks JSON lines with `scripts/inline-curriculum/<kind>_bridge.py serve`; the service's
 * methods are asynchronous, so eval code awaits them.
 */
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { createInterface } from 'node:readline';
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

/**
 * A ScienceWorld task variation, an ALFWorld game (task: its path under ALFWORLD_DATA), or a memetic improvement
 * episode (task: its `natlang.memetic-episode/1` spec; the process is training/neuralese rl/memetic_workspace.py and the
 * program sees it as the service `workspace`).
 */
export type WorldSpec = { kind: 'scienceworld' | 'alfworld' | 'memetic'; task: string; variation?: number; simplifications?: string };

const script = (kind: 'scienceworld' | 'alfworld') => fileURLToPath(new URL(`../../scripts/inline-curriculum/${kind}_bridge.py`, import.meta.url));
const vendor = (path: string) => fileURLToPath(new URL(`../../../vendor/${path}`, import.meta.url));
const dataset = (path: string) => process.env.NATLANG_DATASETS
  ? resolve(process.env.NATLANG_DATASETS, path)
  : vendor(`datasets/${path}`);

export class WorldBridge {
  private next = 1;
  /** Name, declaration and offered methods of a world that declares itself (memetic); the others are `world`. */
  serviceName = 'world';
  declarationText?: string;
  private methods?: string[];
  private readonly pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
  private constructor(private readonly process: ChildProcessWithoutNullStreams) {
    createInterface({ input: process.stdout }).on('line', line => {
      let reply: { id: number; result?: unknown; error?: string };
      try { reply = JSON.parse(line); } catch { globalThis.process.stderr.write(`world process wrote a non-protocol line: ${line.slice(0, 200)}\n`); return; }
      const waiter = this.pending.get(reply.id);
      if (!waiter) return;
      this.pending.delete(reply.id);
      if (reply.error !== undefined) waiter.reject(new Error(reply.error)); else waiter.resolve(reply.result);
    });
    const fail = (error: Error) => {
      for (const waiter of this.pending.values()) waiter.reject(error);
      this.pending.clear();
    };
    process.on('error', fail);
    process.on('exit', () => fail(new Error('the world process exited')));
  }

  /**
   * Start a world process and load the task. SCIENCEWORLD_PYTHON / ALFWORLD_PYTHON name a Python with the package
   * installed; ALFWORLD_DATA points at ALFWorld's downloaded data.
   */
  static async open(spec: WorldSpec, runId?: string): Promise<WorldBridge> {
    if (spec.kind === 'memetic') {
      const repo = fileURLToPath(new URL('../../../', import.meta.url));
      const venv = resolve(repo, '.venv-neuralese/bin/python');
      const python = process.env.NATLANG_MEMETIC_PYTHON ?? (existsSync(venv) ? venv : 'python3');
      const pythonPath = [resolve(repo, 'training/neuralese'), repo, process.env.PYTHONPATH].filter(Boolean).join(':');
      const child = spawn(python, ['-m', 'natlang_neuralese.rl.memetic_workspace', 'serve'],
        { stdio: ['pipe', 'pipe', 'pipe'], env: { ...process.env, PYTHONPATH: pythonPath, NATLANG_NODE: process.execPath } });
      // Its nested collections log progress on stderr; drain it into the collector log so the pipe never fills.
      child.stderr.on('data', chunk => process.stderr.write(chunk));
      const bridge = new WorldBridge(child);
      // One workspace per rollout: the run id keys its changes, budget, commit and score.
      const loaded = await bridge.request('load', { task: spec.task, run: runId ?? null }) as { service: string; declaration: string; methods: string[] };
      bridge.serviceName = loaded.service;
      bridge.declarationText = loaded.declaration;
      bridge.methods = loaded.methods;
      return bridge;
    }
    const python = spec.kind === 'alfworld' ? process.env.ALFWORLD_PYTHON ?? vendor('alfworld-venv/bin/python') :
      process.env.SCIENCEWORLD_PYTHON ?? vendor('scienceworld-venv/bin/python');
    const env = spec.kind === 'alfworld' ? { ...process.env, ALFWORLD_DATA: process.env.ALFWORLD_DATA ?? dataset('alfworld/data') } : process.env;
    const bridge = new WorldBridge(spawn(python, [script(spec.kind), 'serve'], { stdio: ['pipe', 'pipe', 'pipe'], env }));
    await bridge.request('load', { task: spec.task, variation: spec.variation ?? 0, simplifications: spec.simplifications ?? 'easy' });
    return bridge;
  }

  request(op: string, args: Record<string, unknown> = {}): Promise<unknown> {
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.process.stdin.write(JSON.stringify({ id, op, ...args }) + '\n');
    });
  }

  /** The TypeScript declaration of `world` the program is shown: every method is asynchronous. */
  static declaration(kind: 'scienceworld' | 'alfworld'): string {
    const actions = kind === 'alfworld' ? '{ commands: string[] }' :
      '{ templates: string[]; objects: string[] }  // fill a template\'s OBJ slots with objects';
    return [
      'declare const world: {',
      '  /** The task to accomplish. */',
      '  task(): Promise<string>;',
      '  /** A description of where you are. */',
      '  look(): Promise<string>;',
      '  /** What you are carrying. */',
      '  inventory(): Promise<string>;',
      `  /** The commands possible right now. */`,
      `  actions(): Promise<${actions}>;`,
      '  /** Carry out one command, as written in actions(). */',
      '  act(command: string): Promise<{ observation: string; score: number; done: boolean }>;',
      '  /** How far the task is: done when score reaches 100. */',
      '  score(): Promise<{ score: number; done: boolean; moves: number }>;',
      '}',
    ].join('\n');
  }

  /** The declaration the program is shown for this world's service. */
  declarationFor(kind: WorldSpec['kind']): string {
    return this.declarationText ?? WorldBridge.declaration(kind as 'scienceworld' | 'alfworld');
  }

  /** The service the program sees (`world`, or the name a self-declaring world gives). The host-only `score` is never
   * offered to a self-declaring world's program. */
  service(): Record<string, (...args: unknown[]) => unknown> {
    if (this.methods) return Object.fromEntries(this.methods.filter(method => method !== 'score')
      .map(method => [method, (...args: unknown[]) => this.request(method, { args })]));
    return {
      task: () => this.request('task'),
      look: () => this.request('look'),
      inventory: () => this.request('inventory'),
      actions: () => this.request('actions'),
      act: (command: unknown) => this.request('act', { command: String(command) }),
      score: () => this.request('score'),
    };
  }

  close(): void { this.process.kill(); }
}
