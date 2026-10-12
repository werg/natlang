/**
 * The Node host's parts of pi's coding agent: pi-durable's NodeExecutionEnv for each directory (the local file system
 * and shell) and pi's system prompt with resources loaded from the file system. main.ts composes them with the
 * platform-neutral core; the browser host composes its own (browser.ts).
 */
import type { NatlangRuntime } from '@natlang/node';
import type { Registry } from '../vendor/durable/src/harness/types.ts';
import { isAbsolute, relative, resolve } from 'node:path';
import type { Context } from '@earendil-works/chord';
import { NodeExecutionEnv } from '../vendor/durable/src/env/node.ts';
import { err, FileError, type Result } from '../vendor/durable/src/env/index.ts';
import { codingRegistry, ExecutionEnvs } from '../extensions/index.ts';
import { nodePiPrompt, type NodePiPromptOptions } from '../extensions/pi-prompt/node.ts';

/**
 * A NodeExecutionEnv whose file paths must stay inside `root` (eval runs: the task's repository copy), so a file tool
 * cannot reach the checkout pi runs from or anything else on the machine. Shell commands are not confined.
 */
export class ConfinedNodeExecutionEnv extends NodeExecutionEnv {
  readonly #root: string;
  constructor(options: { cwd: string; root: string }) { super({ cwd: options.cwd }); this.#root = resolve(options.root); }
  override async absolutePath(path: string, context: Context): Promise<Result<string, FileError>> {
    const resolved = await super.absolutePath(path, context);
    if (!resolved.ok) return resolved;
    const inside = relative(this.#root, resolved.value);
    if (inside === '' || (!inside.startsWith('..') && !isAbsolute(inside))) return resolved;
    return err(new FileError('permission_denied', `${path} is outside the task repository ${this.#root}; use a path inside it`, path));
  }
}

/** One NodeExecutionEnv per directory; with `confine`, every one is confined to that directory (ConfinedNodeExecutionEnv). */
export function createEnvs(defaultCwd: string, confine?: string): ExecutionEnvs<NodeExecutionEnv> {
  return new ExecutionEnvs(defaultCwd, cwd => confine ? new ConfinedNodeExecutionEnv({ cwd, root: confine }) : new NodeExecutionEnv({ cwd }));
}

/** pi's coding registry with the Node host's prompt resources. */
export function nodeCodingRegistry(natlang: NatlangRuntime, options: NodePiPromptOptions & { subagent?: boolean; tools?: 'nl' | 'crisp' }): Registry {
  return codingRegistry(natlang, { prompt: nodePiPrompt(options), ...(options.subagent === undefined ? {} : { subagent: options.subagent }),
    ...(options.tools ? { tools: options.tools } : {}) });
}
