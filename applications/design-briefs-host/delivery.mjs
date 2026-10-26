/** Deliver atomically committed domain outbox packets only to explicitly configured targets. */
import { ApplicationError } from './runtime.mjs';

export async function dispatchOnce({ store, principal, appId, workerId, routes, fetchImpl = fetch,
  timeoutMs = 10_000, leaseMs = 30_000, limit = 10 }) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs >= leaseMs ||
      !Number.isSafeInteger(limit) || limit < 1 || limit > 200)
    throw new ApplicationError('invalid_delivery_configuration');
  const configured = new Map();
  for (const [topic, route] of Object.entries(routes?.[appId] ?? {})) {
    let endpoint;
    try { endpoint = new URL(route.endpoint); } catch { throw new ApplicationError('invalid_delivery_route'); }
    if (!['https:','http:'].includes(endpoint.protocol) || endpoint.username || endpoint.password)
      throw new ApplicationError('invalid_delivery_route');
    configured.set(topic, { endpoint, headers: new Headers(route.headers ?? {}) });
  }
  const results = [];
  // Lease only the item about to be sent. A slow earlier delivery must not exhaust later items' leases.
  for (let sent = 0; sent < limit; sent++) {
    const [item] = store.claimOutbox(principal, { tenantId: principal.tenantId, appId, workerId, leaseMs, limit: 1 });
    if (!item) break;
    const route = configured.get(item.topic);
    if (!route) {
      results.push(store.failOutbox(principal, item.id, workerId, { code: 'route_unconfigured', ambiguous: false }));
      continue;
    }
    const headers = new Headers(route.headers);
    headers.set('content-type', 'application/json');
    headers.set('idempotency-key', `${item.tenantId}:${item.appId}:${item.caseId}:${item.key}`);
    try {
      const response = await fetchImpl(route.endpoint, { method: 'POST', redirect: 'error',
        signal: AbortSignal.timeout(timeoutMs), headers: Object.fromEntries(headers),
        body: JSON.stringify({ idempotencyKey: item.key, caseId: item.caseId, payload: item.payload }) });
      const receipt = await response.json();
      if (response.ok && receipt?.status === 'delivered' && receipt?.idempotencyKey === item.key &&
          typeof receipt.receiptId === 'string' && receipt.receiptId) {
        results.push(store.ackOutbox(principal, item.id, workerId, receipt));
      } else {
        results.push(store.failOutbox(principal, item.id, workerId, { code: 'delivery_unconfirmed', ambiguous: true }));
      }
    } catch {
      // A failed transport may have delivered. Persist the unknown outcome for receipt reconciliation.
      results.push(store.failOutbox(principal, item.id, workerId, { code: 'delivery_unconfirmed', ambiguous: true }));
    }
  }
  return results;
}
