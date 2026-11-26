import { modelConnectionAttemptTimeout } from './node-connection-policy.js';

let installation: Promise<void> | undefined;
/** Pi's Node providers own HTTP and WebSocket clients, including adapters that
 * reject custom fetch. Configure only this Node process, once before requests;
 * respect an explicit nondefault Node setting unless our override is supplied. */
export function configureNodeModelConnections(): Promise<void> {
  if (installation) return installation;
  const operation = (async () => {
    if (typeof process === 'undefined' || !process.versions?.node) return;
    const milliseconds = modelConnectionAttemptTimeout(process.env);
    if (milliseconds === undefined) return;
    const net = await import('node:net');
    if (process.env.NATLANG_NETWORK_FAMILY_ATTEMPT_TIMEOUT_MS?.trim() || net.getDefaultAutoSelectFamilyAttemptTimeout() === 250)
      net.setDefaultAutoSelectFamilyAttemptTimeout(milliseconds);
  })();
  const pending = operation.catch(error => {
    if (installation === pending) installation = undefined;
    throw error;
  });
  installation = pending;
  return pending;
}
