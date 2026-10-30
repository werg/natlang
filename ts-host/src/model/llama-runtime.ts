/**
 * The Node binding of llama/runtime.ts for the CLI and `@natlang/node`: the same functions over the environment, with the
 * platform's directories, `child_process`, global `fetch` and stderr (`nodeLlamaHost`). Hosts with their own supervisor use
 * `@natlang/node/llama`.
 */
import * as llama from '../llama/runtime.js';
import { nodeLlamaHost } from '../llama/node.js';
import type { LlamaRuntimeRelease } from '../llama/runtime.js';

export type { LlamaRuntimeArtifact, LlamaRuntimeRelease, LlamaServerInspection, LlamaRuntimeDiscovery } from '../llama/runtime.js';
export { LLAMA_RUNTIME_RELEASE, LLAMA_FORK_SOURCE, isCompatibleLlamaVersion, describeLlamaRuntime } from '../llama/runtime.js';
export { defaultNatlangRuntimeDirectory } from '../llama/node.js';

export const inspectLlamaServer = (path: string, source: llama.LlamaServerInspection['source'], release?: LlamaRuntimeRelease,
  environment: NodeJS.ProcessEnv = process.env) => llama.inspectLlamaServer(nodeLlamaHost(environment), path, source, release);

export const discoverLlamaRuntime = (environment: NodeJS.ProcessEnv = process.env, release?: LlamaRuntimeRelease) =>
  llama.discoverLlamaRuntime(nodeLlamaHost(environment), release);

export const installManagedLlamaRuntime = (options: { environment?: NodeJS.ProcessEnv; release?: LlamaRuntimeRelease;
  error?: NodeJS.WritableStream } = {}) =>
  llama.installManagedLlamaRuntime(nodeLlamaHost(options.environment, options.error), options.release ? { release: options.release } : {});

export const buildForkLlamaRuntime = (options: { environment?: NodeJS.ProcessEnv; error?: NodeJS.WritableStream;
  cuda?: boolean; jobs?: number } = {}) =>
  llama.buildForkLlamaRuntime(nodeLlamaHost(options.environment, options.error),
    { ...(options.cuda !== undefined ? { cuda: options.cuda } : {}), ...(options.jobs !== undefined ? { jobs: options.jobs } : {}) });
