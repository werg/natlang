/**
 * pi's coding agent on the port, composed as pi's durable frontend composes it (harness-setup.ts
 * `createCodingRegistry`, runtime.ts): a registry with the coding tools, pi's system prompt and the subagent, and one
 * execution environment per directory. Platform-neutral: each host supplies its environments (the Node host's
 * NodeExecutionEnv, host/node.ts; the browser host's virtual folder, host/folder-env.ts) and its prompt resources.
 */
import type { Context } from '@earendil-works/chord';
import type { NatlangRuntime } from 'natlang:runtime';
import { createRegistry } from '../vendor/durable/src/harness/registry.ts';
import type { EnvTarget, Extension, Registry } from '../vendor/durable/src/harness/types.ts';
import type { ExecutionEnv } from '../vendor/durable/src/env/index.ts';
import { codingTools } from './coding-tools/index.ts';
import { piPrompt, type PiPromptOptions } from './pi-prompt/sections.ts';
import { subagent } from './subagent/index.ts';

export { codingTools } from './coding-tools/index.ts';
export { piPrompt } from './pi-prompt/sections.ts';
export { subagent } from './subagent/index.ts';

export type CodingRegistryOptions = {
  /** pi's system prompt: the host's own (the Node host's loads resources from the file system), else `piPrompt(prompt)`. */
  prompt: Extension | PiPromptOptions;
  /** Install the subagent tool (default true). */
  subagent?: boolean;
};

/** A registry with pi's coding tools, system prompt and subagent; their functions run on `natlang`. */
export function codingRegistry(natlang: NatlangRuntime, options: CodingRegistryOptions): Registry {
  const registry = createRegistry();
  registry.install(codingTools(natlang));
  registry.install('name' in options.prompt ? options.prompt : piPrompt(options.prompt));
  if (options.subagent !== false) registry.install(subagent(natlang));
  return registry;
}

/** One execution environment per directory, shared by every conversation in it (pi's ExecutionEnvs). */
export class ExecutionEnvs<T extends ExecutionEnv = ExecutionEnv> {
  readonly #defaultCwd: string;
  readonly #create: (cwd: string) => T;
  readonly #envs = new Map<string, T>();
  /** `create` builds the environment of a directory, once per directory. */
  constructor(defaultCwd: string, create: (cwd: string) => T) { this.#defaultCwd = defaultCwd; this.#create = create; }
  readonly env = ({ cwd = this.#defaultCwd }: EnvTarget): T => {
    let env = this.#envs.get(cwd);
    if (!env) { env = this.#create(cwd); this.#envs.set(cwd, env); }
    return env;
  };
  async cleanup(context: Context): Promise<void> {
    const envs = [...this.#envs.values()];
    this.#envs.clear();
    for (const env of envs) await env.cleanup(context);
  }
}
