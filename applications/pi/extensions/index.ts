/**
 * pi's coding agent on the port, composed as pi's durable frontend composes it (harness-setup.ts
 * `createCodingRegistry`, runtime.ts): a registry with the coding tools, pi's system prompt and the subagent, and one
 * execution environment per directory.
 */
import type { Context } from '@earendil-works/chord';
import type { NatlangRuntime } from '@natlang/node';
import { createRegistry } from '../vendor/durable/src/harness/registry.ts';
import type { EnvTarget, Registry } from '../vendor/durable/src/harness/types.ts';
import { NodeExecutionEnv } from '../vendor/durable/src/env/node.ts';
import { codingTools } from './coding-tools/index.ts';
import { piPrompt, type PiPromptOptions } from './pi-prompt/sections.ts';
import { subagent } from './subagent/index.ts';

export { codingTools } from './coding-tools/index.ts';
export { piPrompt } from './pi-prompt/sections.ts';
export { subagent } from './subagent/index.ts';

/** A registry with pi's coding tools, system prompt and subagent; their functions run on `natlang`. */
export function codingRegistry(natlang: NatlangRuntime, options: PiPromptOptions & { subagent?: boolean }): Registry {
  const registry = createRegistry();
  registry.install(codingTools(natlang));
  registry.install(piPrompt(options));
  if (options.subagent !== false) registry.install(subagent(natlang));
  return registry;
}

/** One execution environment per directory, shared by every conversation in it (pi's ExecutionEnvs). */
export class ExecutionEnvs {
  readonly #defaultCwd: string;
  readonly #envs = new Map<string, NodeExecutionEnv>();
  constructor(defaultCwd: string) { this.#defaultCwd = defaultCwd; }
  readonly env = ({ cwd = this.#defaultCwd }: EnvTarget): NodeExecutionEnv => {
    let env = this.#envs.get(cwd);
    if (!env) { env = new NodeExecutionEnv({ cwd }); this.#envs.set(cwd, env); }
    return env;
  };
  async cleanup(context: Context): Promise<void> {
    const envs = [...this.#envs.values()];
    this.#envs.clear();
    for (const env of envs) await env.cleanup(context);
  }
}

export function createEnvs(defaultCwd: string): ExecutionEnvs { return new ExecutionEnvs(defaultCwd); }
