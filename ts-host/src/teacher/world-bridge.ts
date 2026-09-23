/**
 * Interactive worlds that run in their own process (ScienceWorld), exposed to a program as a host service.
 * The bridge speaks JSON lines with `scripts/inline-curriculum/scienceworld_bridge.py serve`; the service's
 * methods are asynchronous, so eval code awaits them.
 */
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';

export type WorldSpec = { kind: 'scienceworld'; task: string; variation: number; simplifications?: string };

const BRIDGE = fileURLToPath(new URL('../../scripts/inline-curriculum/scienceworld_bridge.py', import.meta.url));

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

  /** Start a world process and load the task. SCIENCEWORLD_PYTHON names a Python with scienceworld installed. */
  static async open(spec: WorldSpec): Promise<WorldBridge> {
    const python = process.env.SCIENCEWORLD_PYTHON ?? fileURLToPath(new URL('../../../vendor/scienceworld-venv/bin/python', import.meta.url));
    const bridge = new WorldBridge(spawn(python, [BRIDGE, 'serve'], { stdio: ['pipe', 'pipe', 'pipe'] }));
    await bridge.request('load', { task: spec.task, variation: spec.variation, simplifications: spec.simplifications ?? 'easy' });
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
