/**
 * A tool result's fixed formats (pi-durable tool.ts `finalResult` steps 1-2 and 4-5, `boundContent`, `truncated`, and
 * output.ts `boundOutput`): assembling the result from what the execution reported, and bounding its text for the
 * model. The `afterTool` chain between the two is `run`'s.
 */

type Text = { type: 'text'; text: string; textSignature?: string };
type Image = { type: 'image'; data: string; mimeType: string };
type Content = (Text | Image)[];
type Diagnostic = { severity: 'info' | 'warn' | 'error'; message: string; code?: string };
type Result = { content?: Content; isError?: boolean; details?: unknown; diagnostics?: Diagnostic[]; usage?: unknown; control?: unknown };
type Limits = { maxBytes: number; maxLines: number; retain: 'head' | 'tail' };
type Dropped = { droppedBytes: number; droppedLines: number };
/** The retained output used as content, and what its limits dropped; null when the result had its own content. */
type Retained = { content: Content; droppedBytes: number; droppedLines: number } | null;
type Execution = { retained: { text: string; droppedBytes: number; droppedLines: number }; diagnostics: Diagnostic[]; details?: unknown };

const NEWLINE = 0x0a;
const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8', { ignoreBOM: true });

function lineCount(bytes: Uint8Array): number {
  if (bytes.length === 0) return 0;
  let newlines = 0;
  for (let i = 0; i < bytes.length; i++) if (bytes[i] === NEWLINE) newlines++;
  return newlines + (bytes[bytes.length - 1] === NEWLINE ? 0 : 1);
}

/** The last character boundary at or before `index`. */
function characterEnd(bytes: Uint8Array, index: number): number {
  for (let step = 0; step < index; step++) if (((bytes[index - step] ?? 0) & 0xc0) !== 0x80) return index - step;
  return 0;
}

/** The first character boundary at or after `index`. */
function characterStart(bytes: Uint8Array, index: number): number {
  const length = bytes.length;
  for (let at = index; at < length; at++) if (((bytes[at] ?? 0) & 0xc0) !== 0x80) return at;
  return length;
}

/** Index of the last newline at or before `from`, or -1. */
function lastNewline(bytes: Uint8Array, from: number): number {
  for (let step = 0; step <= from; step++) if (bytes[from - step] === NEWLINE) return from - step;
  return -1;
}

/** Index of the first newline at or after `from`, or -1. */
function firstNewline(bytes: Uint8Array, from: number): number {
  const length = bytes.length;
  for (let at = Math.max(0, from); at < length; at++) if (bytes[at] === NEWLINE) return at;
  return -1;
}

function headRange(bytes: Uint8Array, limits: Limits): [number, number] {
  if (limits.maxLines === 0 || limits.maxBytes === 0) return [0, 0];
  let end = bytes.length;
  let lines = 0;
  const length = bytes.length;
  for (let i = 0; i < length; i++) {
    if (bytes[i] !== NEWLINE) continue;
    if (++lines === limits.maxLines) { end = i + 1; break; }
  }
  if (end > limits.maxBytes) {
    const newline = lastNewline(bytes, limits.maxBytes - 1);
    end = newline === -1 ? characterEnd(bytes, limits.maxBytes) : newline + 1;
  }
  return [0, end];
}

function tailRange(bytes: Uint8Array, limits: Limits): [number, number] {
  if (limits.maxLines === 0 || limits.maxBytes === 0) return [bytes.length, bytes.length];
  // A trailing newline ends the last line rather than starting another.
  const last = bytes[bytes.length - 1] === NEWLINE ? bytes.length - 2 : bytes.length - 1;
  let start = 0;
  let lines = 1;
  for (let step = 0; step <= last; step++) {
    const index = last - step;
    if (bytes[index] !== NEWLINE) continue;
    if (lines === limits.maxLines) { start = index + 1; break; }
    lines++;
  }
  if (bytes.length - start > limits.maxBytes) {
    const from = bytes.length - limits.maxBytes;
    const newline = firstNewline(bytes, from - 1);
    // The first line starting inside the byte window, or a cut of the last line when it alone is too long.
    start = newline !== -1 && newline + 1 < bytes.length ? newline + 1 : characterStart(bytes, from);
  }
  return [start, bytes.length];
}

/**
 * `text` bounded to whole lines within the limits: the first lines for head, the last for tail, trailing newline
 * included; a single line longer than maxBytes is cut at a character boundary.
 */
export function boundOutput(text: string, limits: Limits): { text: string; bytes: number; droppedBytes: number; droppedLines: number } {
  const bytes = encoder.encode(text);
  const [from, to] = limits.retain === 'head' ? headRange(bytes, limits) : tailRange(bytes, limits);
  const kept = bytes.subarray(from, to);
  return { text: kept.length === bytes.length ? text : decoder.decode(kept), bytes: kept.length,
    droppedBytes: bytes.length - kept.length, droppedLines: lineCount(bytes) - lineCount(kept) };
}

/** pi's truncation diagnostic; `retain` unknown (null) when rebuilt from a slot. */
export function truncated(dropped: Dropped, retain: 'head' | 'tail' | null): Diagnostic {
  const kept = retain === null ? '' : ` to its ${retain === 'head' ? 'beginning' : 'end'}`;
  return { severity: 'warn', code: 'truncated', message: `Output truncated${kept}: ${dropped.droppedLines} lines, ${dropped.droppedBytes} bytes dropped` };
}

/**
 * The joined text of `content` bounded by `limits`. When something is dropped, the first (head) or last (tail) text item
 * holds the bounded text, the other text items are dropped, and non-text items stay in place.
 */
export function boundContent(content: Content, limits: Limits): { content: Content; droppedBytes: number; droppedLines: number } {
  const texts = content.filter((item): item is Text => item.type === 'text');
  const bounded = boundOutput(texts.map(item => item.text).join(''), limits);
  if (bounded.droppedBytes === 0) return { content, droppedBytes: 0, droppedLines: 0 };
  const keep = limits.retain === 'head' ? texts[0] : texts[texts.length - 1];
  const result: Content = [];
  for (const item of content) {
    if (item.type !== 'text') result.push(item);
    else if (item === keep) result.push({ ...item, text: bounded.text });
  }
  return { content: result, droppedBytes: bounded.droppedBytes, droppedLines: bounded.droppedLines };
}

/**
 * Steps 1-2 of pi's final result. A result without `content` gets the retained output as its content ([] when empty,
 * else one text item) and `retained` reports it; `details` falls back to the last reported details; `diagnostics` are
 * the ones reported while running, then the result's own.
 */
export function assemble(result: Result, execution: Execution): { result: Result; retained: Retained } {
  const useRetained = result.content === undefined;
  const content: Content = useRetained ? (execution.retained.text === '' ? [] : [{ type: 'text', text: execution.retained.text }]) : result.content!;
  const details = result.details === undefined ? execution.details : result.details;
  const final: Result = { ...result, content, ...(details === undefined ? {} : { details }),
    diagnostics: [...execution.diagnostics, ...(result.diagnostics ?? [])] };
  if (details === undefined) delete final.details;
  return { result: final, retained: useRetained ? { content, droppedBytes: execution.retained.droppedBytes, droppedLines: execution.retained.droppedLines } : null };
}

/**
 * Steps 4-5 of pi's final result, after `afterTool`: when the retained output is still the content and dropped bytes,
 * a truncation diagnostic; then the content bounded, with a second truncation diagnostic when that dropped bytes. The
 * harness diagnostics come last.
 */
export function bound(result: Result, limits: Limits, retained: Retained): Result {
  const harness: Diagnostic[] = [];
  const content = result.content ?? [];
  if (retained !== null && retained.droppedBytes > 0 && JSON.stringify(content) === JSON.stringify(retained.content))
    harness.push(truncated(retained, limits.retain));
  const bounded = boundContent(content, limits);
  if (bounded.droppedBytes > 0) harness.push(truncated(bounded, limits.retain));
  return { ...result, content: bounded.content, diagnostics: [...(result.diagnostics ?? []), ...harness] };
}
