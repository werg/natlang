/** Address racing is separate from the request deadline. Node's 250 ms
 * default can discard usable addresses before a delayed handshake settles. */
export function modelConnectionAttemptTimeout(environment: Record<string, string | undefined>): number | undefined {
  const configured = environment.NATLANG_NETWORK_FAMILY_ATTEMPT_TIMEOUT_MS?.trim();
  if (configured === '0') return undefined;
  const milliseconds = configured ? Number(configured) : 1500;
  if (!Number.isInteger(milliseconds) || milliseconds < 10 || milliseconds > 60000)
    throw new Error('NATLANG_NETWORK_FAMILY_ATTEMPT_TIMEOUT_MS must be 0 or an integer from 10 to 60000');
  return milliseconds;
}

