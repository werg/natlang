/** Stop reasons whose assistant messages never enter model context (pi-durable context.ts EXCLUDED_STOP_REASONS). */
export const EXCLUDED_STOP_REASONS: string[] = ['aborted', 'error', 'deferred'];

/** The text of a synthesized result for a call whose result is missing (pi-durable context.ts MISSING_RESULT_TEXT). */
export const MISSING_RESULT_TEXT: string = 'Tool result unavailable: history ends before this call completed.';
