/**
 * `pluggable({ crisp, nl }, mode)`: one entry point over a crisp and a natural-language implementation of the same
 * part. An application selects the mode by a setting; `shadow` runs both, serves one and records whether they agree.
 * A host and application helper, not part of what a model is shown.
 */
import { resolveFrame } from './runtime.js';
import { traceFor } from '../native/graph.js';
import { canonicalValue } from '../native/refinement.js';

export type PluggableMode = 'crisp' | 'nl' | 'shadow';
/**
 * Older spellings of `nl` that settings and configs already use. Deprecated: write `nl`. They are accepted everywhere a
 * mode is, so existing configs keep working; new settings use the three modes only.
 */
export type LegacyPluggableMode = 'natlang' | 'natural-language';
/** What a setting may hold: a mode, a deprecated spelling, or nothing (the part's default applies). */
export type PluggableSetting = PluggableMode | LegacyPluggableMode | undefined;
export type PluggableImplementations<A extends unknown[], R> = {
  crisp: (...args: A) => R | Promise<R>;
  nl: (...args: A) => R | Promise<R>;
};
export type PluggableOptions<R> = {
  /** The mode when the setting is absent. Default `nl`. */
  default?: PluggableMode;
  /** Names the part in the trace. */
  name?: string;
  /** In shadow mode, the side whose result is returned; the other is only compared. Default `nl`. */
  serve?: 'crisp' | 'nl';
  /** Whether two results agree. Default: equal canonical values. */
  same?: (crisp: R, nl: R) => boolean;
};

/** The mode a setting names: `crisp`, `nl` or `shadow`; `natlang` and `natural-language` (deprecated) mean `nl`; absent means `fallback`. */
export function pluggableMode(setting: unknown, fallback: PluggableMode = 'nl'): PluggableMode {
  if (setting === undefined || setting === null) return fallback;
  if (setting === 'crisp' || setting === 'nl' || setting === 'shadow') return setting;
  if (setting === 'natlang' || setting === 'natural-language') return 'nl';
  throw new TypeError(`pluggable mode is "crisp", "nl" or "shadow", not ${JSON.stringify(setting)}; pass one of those three ` +
    '("natlang" and "natural-language" are accepted for "nl" but deprecated).');
}

const clip = (text: string): string => text.length > 400 ? `${text.slice(0, 400)} … (${text.length} chars)` : text;

/** Record a trace event on the current call, or on the task when no call trace is open; nothing outside a task. */
function emit(kind: string, data: Record<string, unknown>): void {
  let frame: ReturnType<typeof resolveFrame>;
  try { frame = resolveFrame(); } catch { return; }
  const trace = traceFor(frame.parentCallId);
  if (trace) { trace.emit(kind, { call_id: frame.parentCallId ?? null, ...data }); return; }
  // Host code outside any call: a small trace of its own carries the event, as an iteration's does.
  const task = frame.task;
  task.record({ callId: `${task.id}/pluggable-${++recorded}`, parentCallId: null, taskId: task.id,
    definitionId: `pluggable:${data.name ?? 'part'}`, name: `pluggable ${data.name ?? 'part'}`, outcome: 'done', detail: '',
    events: [{ kind, call_id: null, ...data }] } as never);
}
let recorded = 0;

/**
 * The part as one function. `crisp` and `nl` run that implementation alone. `shadow` runs both on the same arguments
 * (concurrently), returns the served side's result, and records a `pluggable_shadow` trace event with both results and
 * `agree`, as `refinement_shadow` does for refinements. A failure of the side that is not served is recorded as a
 * disagreement and does not fail the call.
 */
export function pluggable<A extends unknown[], R>(implementations: PluggableImplementations<A, R>, setting: PluggableSetting,
    options: PluggableOptions<R> = {}): (...args: A) => Promise<R> {
  const mode = pluggableMode(setting, options.default);
  const serve = options.serve ?? 'nl';
  const same = options.same ?? ((crisp: R, nl: R) => canonicalValue(crisp) === canonicalValue(nl));
  return async (...args: A): Promise<R> => {
    if (mode !== 'shadow') return implementations[mode](...args);
    type Side = { ok: true; value: R } | { ok: false; error: unknown };
    const settle = async (run: () => R | Promise<R>): Promise<Side> => {
      try { return { ok: true, value: await run() }; } catch (error) { return { ok: false, error }; }
    };
    const [crisp, nl] = await Promise.all([settle(() => implementations.crisp(...args)), settle(() => implementations.nl(...args))]);
    const served = serve === 'crisp' ? crisp : nl;
    const shown = (side: Side) => side.ok ? clip(canonicalValue(side.value)) : `error: ${(side.error as Error)?.message ?? String(side.error)}`;
    const agree = crisp.ok && nl.ok && same(crisp.value, nl.value);
    emit('pluggable_shadow', { name: options.name ?? null, served: serve, crisp: shown(crisp), nl: shown(nl), agree });
    if (!served.ok) throw served.error;
    return served.value;
  };
}
