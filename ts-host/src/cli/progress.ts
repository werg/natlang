/**
 * What `natlang run|call|ask` shows while natural-language calls work (live events, runtime.ts `LiveEvent`): on a
 * terminal one status line that is cleared when no call is running; otherwise a line every 15 s while calls run.
 * A call that fails is reported in one line, whether or not the program catches its error; the request error it failed
 * on is not repeated, nor is the error when the command then reports it.
 * NATLANG_PROGRESS=off|line|auto (default auto) or `--progress` chooses.
 */
import { appendFileSync } from 'node:fs';
import type { LiveEvent, LiveSink } from '../runtime/runtime.js';

export type ProgressMode = 'auto' | 'line' | 'off';
type Active = { name: string; turn: number; started: number; requesting: boolean; thought: number; written: number;
  /** Consecutive actions the runtime rejected, and the first diagnostic code of the latest one. */
  rejected: number; rejection: string };

const FRAMES = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏';
const tokens = (count: number) => count >= 1000 ? `${(count / 1000).toFixed(1)}k` : String(count);
const seconds = (ms: number) => ms >= 60_000 ? `${Math.floor(ms / 60_000)}m${String(Math.round(ms / 1000) % 60).padStart(2, '0')}s` : `${Math.round(ms / 1000)}s`;

export function parseProgressMode(value: string | undefined): ProgressMode {
  if (value === undefined || value === '') return 'auto';
  if (value === 'auto' || value === 'line' || value === 'off') return value;
  throw new Error('progress must be auto, line or off');
}

/** Failures the display has printed (first lines), so the command's own error report does not repeat one. */
const reported = new Set<string>();
/** Whether `message` (an error the command is about to print) repeats a failure the progress display already printed. */
export const reportedFailure = (message: string) => [...reported].some(detail => message.includes(detail));

/**
 * `roots: false` leaves root-call failures to the command (natlang test prints each example's outcome). Generated
 * tokens are estimated from the streamed characters (about 4 per token) while a request runs, and counted exactly
 * when it ends; a terminal gets a one-line summary when the display closes.
 */
export function progressReporter(stream: NodeJS.WritableStream & { isTTY?: boolean }, mode: ProgressMode = 'auto',
  options: { roots?: boolean } = {}): { live: LiveSink | undefined; close(): void } {
  if (mode === 'off') return { live: undefined, close() {} };
  const terminal = mode === 'auto' && stream.isTTY === true;
  const active = new Map<string, Active>();
  let promptTokens = 0, completionTokens = 0, cachedTokens = 0, timedTokens = 0, generationMs = 0, finished = 0, frame = 0, shown = false, since = 0,
    lastLine = 0, first = 0;
  const describe = () => {
    const calls = [...active.values()];
    const now = Date.now(), oldest = calls.reduce((first, call) => Math.min(first, call.started), now);
    const focus = calls.at(-1)!;
    const others = calls.length > 1 ? ` +${calls.length - 1} more` : '';
    const doing = !focus.requesting ? '' : focus.written ? ` (writing, ${tokens(focus.written)} chars)` :
      focus.thought ? ` (thinking, ${tokens(focus.thought)} chars)` : ' (reading the prompt)';
    const stuck = focus.rejected >= 2 ? ` · last ${focus.rejected} actions rejected${focus.rejection ? ` (${focus.rejection})` : ''}` : '';
    const streaming = Math.round(calls.reduce((sum, call) => sum + (call.requesting ? call.thought + call.written : 0), 0) / 4);
    const generated = completionTokens + streaming;
    // Counters appear once known: prompt tokens when a request ends, generated ones (estimated) as soon as text streams.
    const counts = [promptTokens ? `${tokens(promptTokens)} prompt${cachedTokens ? ` (${tokens(cachedTokens)} cached)` : ''}` : '',
      generated ? `${streaming ? '~' : ''}${tokens(generated)} generated` : ''].filter(Boolean).join(' + ');
    return `${focus.name} · turn ${Math.max(1, focus.turn)}${doing}${stuck}${others} · ${finished} done` +
      `${counts ? ` · ${counts} tokens${generationMs > 0 ? ` (${(timedTokens * 1000 / generationMs).toFixed(1)}/s)` : ''}` : ''} · ${seconds(now - (since || oldest))}`;
  };
  const clear = () => { if (shown) { stream.write('\r\x1b[2K'); shown = false; } };
  const render = () => {
    if (!active.size) { clear(); return; }
    if (terminal) {
      const width = (stream as { columns?: number }).columns ?? 100;
      stream.write(`\r\x1b[2K${FRAMES[frame++ % FRAMES.length]} ${describe()}`.slice(0, width + 5));
      shown = true;
    } else if (Date.now() - lastLine >= 15_000) { lastLine = Date.now(); stream.write(`natlang: ${describe()}\n`); }
  };
  const timer = setInterval(render, terminal ? 120 : 1000); timer.unref();
  const persistent = (line: string) => { clear(); stream.write(line + '\n'); };
  const live: LiveSink = (event: LiveEvent) => {
    if (event.type === 'call_start') {
      if (!active.size) { since = Date.now(); lastLine = Date.now(); }
      first ||= Date.now();
      active.set(event.callId, { name: event.name, turn: 0, started: Date.now(), requesting: false, thought: 0, written: 0, rejected: 0, rejection: '' });
    } else if (event.type === 'call_end') {
      active.delete(event.callId); finished++;
      // A failed request that the call then fails on is reported once, as the call's failure.
      if (event.outcome !== 'done' && (event.parentCallId !== null || options.roots !== false)) {
        const detail = event.detail.split('\n')[0]!.slice(0, 300);
        reported.add(detail);
        persistent(`natlang: ${event.name} ${event.outcome === 'quiesced' ? 'stopped' : event.outcome}: ${detail}`);
      }
      if (!active.size) clear();
    } else if (event.type === 'model_delta') {
      const call = active.get(event.callId);
      if (!call) return;
      if (event.delta.type === 'reasoning') call.thought += event.delta.text.length;
      else if (event.delta.type === 'text') call.written += event.delta.text.length;
      else if (event.delta.type === 'tool_call') call.written += (event.delta.arguments ?? '').length;
      else if (event.delta.type === 'reset') { call.thought = 0; call.written = 0; }
    } else if (event.type === 'trace') {
      const call = active.get(event.callId), data = event.event;
      if (!call) return;
      if (data.kind === 'model_request' && data.phase === 'start') { call.turn = Number(data.turn ?? call.turn); call.requesting = true; call.thought = 0; call.written = 0; }
      if (data.kind === 'action') {
        if (data.outcome === 'rejected') { call.rejected++; call.rejection = /\b([a-z]+(?:-[a-z]+)+):/.exec(String(data.result_text ?? ''))?.[1] ?? ''; }
        else call.rejected = 0;
      }
      if (data.kind === 'model_request' && data.phase !== 'start') {
        call.requesting = false;
        promptTokens += Number(data.prompt_tokens ?? 0); completionTokens += Number(data.completion_tokens ?? 0); cachedTokens += Number(data.cached_tokens ?? 0);
        if (typeof data.generation_ms === 'number' && data.generation_ms > 0) { generationMs += data.generation_ms; timedTokens += Number(data.completion_tokens ?? 0); }
      }
    }
  };
  return { live, close() {
    clearInterval(timer); clear();
    if (terminal && finished) stream.write(`natlang: ${finished} call${finished === 1 ? '' : 's'}` +
      `${promptTokens || completionTokens ? ` · ${tokens(promptTokens)} prompt + ${tokens(completionTokens)} generated tokens` : ''} · ` +
      `${((Date.now() - first) / 1000).toFixed(1)} s\n`);
  } };
}

/**
 * Live events as JSON lines appended to `path` (NATLANG_LIVE_FILE; the dashboard supervises runs this way). Consecutive
 * text and reasoning deltas of one turn are joined, and lines are written at most every 150 ms.
 */
export function liveFileSink(path: string): { live: LiveSink; close(): void } {
  let pending: LiveEvent[] = [];
  const flush = () => {
    if (!pending.length) return;
    const lines = pending.map(event => JSON.stringify({ at: Date.now(), ...event })).join('\n') + '\n';
    pending = [];
    try { appendFileSync(path, lines); } catch { /* the supervisor went away; the run goes on */ }
  };
  const timer = setInterval(flush, 150); timer.unref();
  const live: LiveSink = event => {
    const last = pending.at(-1);
    if (event.type === 'model_delta' && (event.delta.type === 'text' || event.delta.type === 'reasoning') && last?.type === 'model_delta' &&
        last.callId === event.callId && last.turn === event.turn && last.delta.type === event.delta.type) {
      pending[pending.length - 1] = { ...last, delta: { type: event.delta.type, text: (last.delta as { text: string }).text + event.delta.text } };
      return;
    }
    pending.push(event);
    if (event.type === 'call_end' || event.type === 'call_start') flush();
  };
  return { live, close() { clearInterval(timer); flush(); } };
}

/** The progress display and, under a supervisor (NATLANG_LIVE_FILE), the live event file, as one sink. */
export function commandProgress(stream: NodeJS.WritableStream & { isTTY?: boolean }, mode: ProgressMode,
  environment: NodeJS.ProcessEnv = process.env, options: { roots?: boolean } = {}): { live: LiveSink | undefined; close(): void } {
  const display = progressReporter(stream, mode, options);
  const file = environment.NATLANG_LIVE_FILE ? liveFileSink(environment.NATLANG_LIVE_FILE) : undefined;
  if (!file) return display;
  return { live: event => { display.live?.(event); file.live(event); }, close() { display.close(); file.close(); } };
}
