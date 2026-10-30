/** The Node binding of llama/shared.ts (the server record under the platform's state directory). */
import * as shared from '../llama/shared.js';
import { nodeLlamaHost } from '../llama/node.js';

export type { SharedServer } from '../llama/shared.js';

export const sharedServerPath = (environment: NodeJS.ProcessEnv = process.env) => shared.sharedServerPath(nodeLlamaHost(environment));
export const publishSharedServer = (server: Omit<shared.SharedServer, 'schema'>, environment: NodeJS.ProcessEnv = process.env) =>
  shared.publishSharedServer(nodeLlamaHost(environment), server);
export const withdrawSharedServer = (pid: number, environment: NodeJS.ProcessEnv = process.env) =>
  shared.withdrawSharedServer(nodeLlamaHost(environment), pid);
export const readSharedServer = (environment: NodeJS.ProcessEnv = process.env) => shared.readSharedServer(nodeLlamaHost(environment));
export const findSharedServer = (modelSha256: string, runtime: string, environment: NodeJS.ProcessEnv = process.env) =>
  shared.findSharedServer(nodeLlamaHost(environment), modelSha256, runtime);
