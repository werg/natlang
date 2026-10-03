/** Retry policy for operational failures only; incorrect model answers are ordinary results. */
const details = (error: unknown): any => error && typeof error === 'object' ? error : {};
const text = (error: unknown) => String(error instanceof Error ? error.message : error).toLowerCase();
export function rateLimited(error: unknown): boolean {
  const value = details(error);
  return Number(value.status ?? value.statusCode) === 429 ||
    /rate.?limit|too many requests|usage_limit_reached|\b429\b/.test(text(error));
}
export function transportFailure(error: unknown): boolean {
  const value = details(error);
  return rateLimited(error) || [408, 500, 502, 503, 504].includes(Number(value.status ?? value.statusCode)) ||
    text(error).trim() === 'provider returned an empty response' ||
    /^connection error\.?$/.test(text(error).trim()) ||
    /connection refused|connection reset|fetch failed|socket|timed out|timeout|econnreset|econnrefused|remote end closed|\b(?:http|status|server error)\D*(?:408|500|502|503|504)\b|context size has been exceeded/.test(text(error));
}
/** Headers may be unavailable after a provider SDK flattens its error. Use only supplied delays. */
export function retryAfterMs(error: unknown, now = Date.now()): number {
  const value = details(error), headers = value.headers ?? value.response?.headers;
  const header = headers?.get?.('retry-after') ?? headers?.['retry-after'] ?? headers?.['Retry-After'];
  if (header !== undefined && header !== null) {
    const seconds = Number(header), milliseconds = Number.isFinite(seconds) ? seconds * 1000 : Date.parse(String(header)) - now;
    if (Number.isFinite(milliseconds) && milliseconds >= 0) return milliseconds;
  }
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
