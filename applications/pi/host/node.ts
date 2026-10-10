/**
 * The Node host's parts of pi's coding agent: pi-durable's NodeExecutionEnv for each directory (the local file system
 * and shell) and pi's system prompt with resources loaded from the file system. main.ts composes them with the
 * platform-neutral core; the browser host composes its own (browser.ts).
 */
import type { NatlangRuntime } from '@natlang/node';
import type { Registry } from '../vendor/durable/src/harness/types.ts';
import { NodeExecutionEnv } from '../vendor/durable/src/env/node.ts';
import { codingRegistry, ExecutionEnvs } from '../extensions/index.ts';
import { nodePiPrompt, type NodePiPromptOptions } from '../extensions/pi-prompt/node.ts';

/** One NodeExecutionEnv per directory. */
export function createEnvs(defaultCwd: string): ExecutionEnvs<NodeExecutionEnv> {
  return new ExecutionEnvs(defaultCwd, cwd => new NodeExecutionEnv({ cwd }));
}

/** pi's coding registry with the Node host's prompt resources. */
export function nodeCodingRegistry(natlang: NatlangRuntime, options: NodePiPromptOptions & { subagent?: boolean }): Registry {
  return codingRegistry(natlang, { prompt: nodePiPrompt(options), ...(options.subagent === undefined ? {} : { subagent: options.subagent }) });
}
