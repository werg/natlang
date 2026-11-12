/** The Node binding of llama/shared.ts (shared model servers under the platform's state directory). */
import * as shared from '../llama/shared.js';
import { nodeLlamaHost } from '../llama/node.js';

export type { SharedServer } from '../llama/shared.js';

/** The shared model servers running on this machine. */
export const listSharedServers = (environment: NodeJS.ProcessEnv = process.env) => shared.listSharedServers(nodeLlamaHost(environment));
/** Live leases on a shared server: the natlang processes using it. */
export const sharedServerLeases = (server: shared.SharedServer, environment: NodeJS.ProcessEnv = process.env) =>
  shared.liveLeases(shared.sharedServersDirectory(nodeLlamaHost(environment)), server.key);
export const stopSharedServer = shared.stopSharedServer;
