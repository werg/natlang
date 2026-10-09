/** Retry policy for operational failures only; incorrect model answers are ordinary results. */
const details = (error: unknown): any => error && typeof error === 'object' ? error : {};
const text = (error: unknown) => String(error instanceof Error ? error.message : error).toLowerCase();
export function rateLimited(error: unknown): boolean {
  const value = details(error);
  return Number(value.status ?? value.statusCode) === 429 ||
    /rate.?limit|too many requests|usage_limit_reached|\b429\b/.test(text(error));
}
/** Pi preserves the provider's finish_reason in these exact errors. Keep the allowlist narrow:
 * `error` and `network_error` mean the provider did not deliver a complete assistant turn;
 * content filters, malformed arguments, auth, and request validation are not retryable here.
 */
export function providerFinishReason(error: unknown): 'error' | 'network_error' | undefined {
  const match = text(error).trim().match(/^provider finish_reason: (error|network_error)$/);
  return match?.[1] as 'error' | 'network_error' | undefined;
}
export function transportFailure(error: unknown, scope: 'case' | 'request' = 'case'): boolean {
  const value = details(error);
  const code = String(value.providerCode ?? value.provider_code ?? value.code ?? '').toUpperCase();
  const neverRetryCodes = new Set(['NON_BRIDGE_TOOL_USE', 'NATLANG_PROVIDER_REQUEST_RETRIES_EXHAUSTED',
    'NATLANG_PROVIDER_REQUEST_TIMEOUT', 'NATLANG_PROVIDER_ACTION_CYCLE_TIMEOUT', 'INVALID_REQUEST',
    'INVALID_REQUEST_ERROR', 'INVALID_API_KEY', 'UNAUTHORIZED', 'AUTHENTICATION_ERROR', 'CONTENT_FILTER',
    'CONTENT_POLICY_VIOLATION', 'CONTEXT_LENGTH_EXCEEDED', 'SCHEMA_VALIDATION_ERROR', 'INVALID_SCHEMA']);
  if (neverRetryCodes.has(code) || value.name === 'AbortError' ||
      value.providerRetryable === false || value.provider_retryable === false ||
      (scope === 'request' && /context (?:size|window)|context_length|context length|maximum context|schema.{0,20}(?:invalid|error|violat)|(?:invalid|error|violat).{0,20}schema|content filter|content policy|invalid api key|unauthori[sz]ed|authentication error/i.test(text(error))))
    return false;
  return rateLimited(error) || [408, 500, 502, 503, 504].includes(Number(value.status ?? value.statusCode)) ||
    providerFinishReason(error) !== undefined ||
    text(error).trim() === 'provider returned an empty response' ||
    /^connection error\.?$/.test(text(error).trim()) ||
    /connection refused|connection reset|fetch failed|socket|timed out|timeout|econnreset|econnrefused|remote end closed|\b(?:http|status|server error)\D*(?:408|500|502|503|504)\b|context size has been exceeded/.test(text(error));
}
/** Same-request retries are stricter than whole-case retries: explicit provider non-retryability wins. */
export function providerRequestRetryable(error: unknown): boolean {
  return transportFailure(error, 'request');
}
/** Headers may be unavailable after a provider SDK flattens its error. Use only supplied delays. */
export function retryAfterMs(error: unknown, now = Date.now()): number {
  const value = details(error), headers = value.headers ?? value.response?.headers;
  const header = headers?.get?.('retry-after') ?? headers?.['retry-after'] ?? headers?.['Retry-After'];
  if (header !== undefined && header !== null) {
    const seconds = Number(header), milliseconds = Number.isFinite(seconds) ? seconds * 1000 : Date.parse(String(header)) - now;
    if (Number.isFinite(milliseconds) && milliseconds >= 0) return milliseconds;
  }
  // Google SDK errors can flatten RetryInfo into one or more JSON-encoded message strings.
  // Preserve the explicitly supplied protobuf duration rather than retrying a daily quota every minute.
  const suppliedDelays: number[] = [];
  const inspect = (node: unknown, depth = 0): void => {
    if (depth > 8 || node == null) return;
    if (typeof node === 'string') {
      if (node.length <= 65536 && node.trim().startsWith('{')) {
        try { inspect(JSON.parse(node), depth + 1); } catch { /* not encoded provider metadata */ }
      }
      return;
    }
    if (Array.isArray(node)) { for (const item of node) inspect(item, depth + 1); return; }
    if (typeof node !== 'object') return;
    const record = node as Record<string, unknown>;
    const duration = typeof record.retryDelay === 'string' ? /^(\d+(?:\.\d+)?)s$/.exec(record.retryDelay) : null;
    if (duration && Number.isFinite(Number(duration[1]))) suppliedDelays.push(Number(duration[1]) * 1000);
    for (const key of ['error', 'message', 'details']) inspect(record[key], depth + 1);
  };
  inspect(value); inspect(error instanceof Error ? error.message : error);
  if (suppliedDelays.length) return Math.max(...suppliedDelays);
  const message = text(error);
  const milliseconds = value.retry_after_ms ?? message.match(/"retry_after_ms"\s*:\s*(\d+(?:\.\d+)?)/)?.[1];
  const seconds = value.retry_after ?? message.match(/"(?:retry_after|resets_in_seconds)"\s*:\s*(\d+(?:\.\d+)?)/)?.[1] ??
    message.match(/(?:try|retry) again in\s*(\d+(?:\.\d+)?)\s*(?:s\b|seconds?)/)?.[1];
  const result = milliseconds !== undefined ? Number(milliseconds) : Number(seconds) * 1000;
  return Number.isFinite(result) && result >= 0 ? result : 0;
}
export function retryWaitMs(error: unknown, attempt: number, base = 5000, random = Math.random, now = Date.now()): number {
  const cap = rateLimited(error) ? 120000 : 30000;
  const exponential = Math.max(0, base) * (rateLimited(error) ? 3 : 1) * 2 ** Math.min(attempt, 20);
  // Clamp after jitter. A provider's minimum delay takes precedence over our policy cap.
  return Math.ceil(Math.max(retryAfterMs(error, now), Math.min(cap, exponential * (0.5 + random()))));
}
