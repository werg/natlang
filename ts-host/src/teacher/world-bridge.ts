/**
 * Interactive worlds that run in their own process (ScienceWorld, ALFWorld), exposed to a program as a host service.
 * The bridge speaks JSON lines with `scripts/inline-curriculum/<kind>_bridge.py serve`; the service's
 * methods are asynchronous, so eval code awaits them.
 */
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';

/** A ScienceWorld task variation, or an ALFWorld game (task: its path under ALFWORLD_DATA). */
export type WorldSpec = { kind: 'scienceworld' | 'alfworld'; task: string; variation?: number; simplifications?: string };

const script = (kind: WorldSpec['kind']) => fileURLToPath(new URL(`../../scripts/inline-curriculum/${kind}_bridge.py`, import.meta.url));
const vendor = (path: string) => fileURLToPath(new URL(`../../../vendor/${path}`, import.meta.url));

export class WorldBridge {
  private next = 1;
  private readonly pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
  private constructor(private readonly process: ChildProcessWithoutNullStreams) {
    createInterface({ input: process.stdout }).on('line', line => {
      const reply = JSON.parse(line) as { id: number; result?: unknown; error?: string };
      const waiter = this.pending.get(reply.id);
      if (!waiter) return;
      this.pending.delete(reply.id);
      if (reply.error !== undefined) waiter.reject(new Error(reply.error)); else waiter.resolve(reply.result);
    });
    process.on('exit', () => { for (const waiter of this.pending.values()) waiter.reject(new Error('the world process exited')); });
  }

  /**
   * Start a world process and load the task. SCIENCEWORLD_PYTHON / ALFWORLD_PYTHON name a Python with the package
   * installed; ALFWORLD_DATA points at ALFWorld's downloaded data.
   */
  static async open(spec: WorldSpec): Promise<WorldBridge> {
    const python = spec.kind === 'alfworld' ? process.env.ALFWORLD_PYTHON ?? vendor('alfworld-venv/bin/python') :
      process.env.SCIENCEWORLD_PYTHON ?? vendor('scienceworld-venv/bin/python');
    const env = spec.kind === 'alfworld' ? { ...process.env, ALFWORLD_DATA: process.env.ALFWORLD_DATA ?? vendor('datasets/alfworld/data') } : process.env;
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

  /** The service the program sees as `world`. */
  service(): Record<string, (...args: unknown[]) => unknown> {
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
