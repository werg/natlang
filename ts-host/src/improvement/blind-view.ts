/**
 * Reward-blind views (plans/neuralese/LEARNING_CONTINUUM.md §10.1): what a reward-blind improver may see of a
 * function's attempts, built by schema from runtime traces, and the gate check that a view carries no feedback.
 *
 * - `reward-blind-observations`: the function's source, its inputs, and its attempt trajectories, including the
 *   environment's natural responses (tool results, effect results, runtime errors).
 * - `reward-blind-strict`: the same without environment responses: only what the function was asked and what it did.
 *
 * Both exclude expected outputs, scores, gates, pass/fail flags, checker and judge outputs, and anything derived from
 * them. Trace events are kept by an allow-list per kind (unknown kinds are dropped), so a new event kind cannot leak
 * feedback by default. `checkBlindView` is the gate: no forbidden key anywhere and no sealed-answer fragment.
 */
import { createHash } from 'node:crypto';
import type { Visibility } from './step-record.js';

export type BlindVisibility = Exclude<Visibility, 'full'>;

/** Fields kept per event kind; `environment` fields only in `reward-blind-observations`. */
const EVENTS: Record<string, { own: readonly string[]; environment?: readonly string[] }> = {
  invocation: { own: ['call_id', 'definition', 'signature', 'inputs'] },
  model_request: { own: ['call_id', 'turn', 'messages'] },
  proposal: { own: ['call_id', 'turn', 'text', 'calls'] },
  model_turn: { own: ['call_id', 'turn', 'calls'] },
  action: { own: ['call_id', 'name', 'arguments'], environment: ['result_text', 'outcome', 'diagnostics'] },
  effect: { own: ['call_id', 'service', 'method', 'args'], environment: ['result'] },
  state: { own: ['value'] },
  folder: { own: ['call_id', 'mode', 'changes'] },
};

/** Keys that name feedback; none may appear anywhere in a blind view. */
export const FORBIDDEN_KEYS = ['expected', 'expectedFiles', 'expected_files', 'score', 'scores', 'quality', 'passed',
  'gates', 'gatesPassed', 'reward', 'rewards', 'gold', 'judge', 'verdict', 'grade', 'feedback', 'correct', 'label',
  'labels', 'probabilities_gold', 'effect_size', 'wins', 'losses', 'disposition'] as const;
const FORBIDDEN = new Set<string>(FORBIDDEN_KEYS);

export type BlindView = {
  readonly schema: 'natlang.blind-view/1';
  readonly visibility: BlindVisibility;
  /** The function under improvement: its files (source and context documents). */
  readonly function: Readonly<Record<string, string>>;
  readonly attempts: readonly { readonly inputs: unknown; readonly events: readonly Record<string, unknown>[] }[];
  readonly background?: Readonly<Record<string, string>>;
};

/** Messages without environment responses (tool-role content withheld) for the strict class. */
function strictMessages(messages: unknown): unknown {
  if (!Array.isArray(messages)) return messages;
  return messages.map(message => message && typeof message === 'object' && (message as { role?: string }).role === 'tool'
    ? { ...(message as object), content: '[environment response withheld]' } : message);
}

/** One trace event as the visibility class allows, or null when its kind is not shown. */
export function blindEvent(event: Record<string, unknown>, visibility: BlindVisibility): Record<string, unknown> | null {
  const spec = EVENTS[String(event.kind)];
  if (!spec) return null;
  const keep = [...spec.own, ...(visibility === 'reward-blind-observations' ? spec.environment ?? [] : [])];
  const out: Record<string, unknown> = { kind: event.kind, seq: event.seq };
  for (const key of keep) if (key in event) out[key] = key === 'messages' && visibility === 'reward-blind-strict' ? strictMessages(event[key]) : event[key];
  return stripForbidden(out) as Record<string, unknown>;
}

/** Remove forbidden keys at any depth (a tool's own JSON may carry e.g. `score`; the improver must not see it). */
function stripForbidden(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stripForbidden);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value)
    .filter(([key]) => !FORBIDDEN.has(key)).map(([key, item]) => [key, stripForbidden(item)]));
  return value;
}

/** Build the view of a function's attempts: `traces` are runtime invocation traces (their `events`), one per attempt. */
export function blindView(options: { visibility: BlindVisibility; function: Record<string, string>;
    attempts: readonly { inputs: unknown; traces: readonly { events: readonly Record<string, unknown>[] }[] }[];
    background?: Record<string, string> }): BlindView {
  return { schema: 'natlang.blind-view/1', visibility: options.visibility, function: { ...options.function },
    attempts: options.attempts.map(attempt => ({ inputs: stripForbidden(attempt.inputs),
      events: attempt.traces.flatMap(trace => trace.events).map(event => blindEvent(event, options.visibility))
        .filter((event): event is Record<string, unknown> => event !== null) })),
    ...(options.background ? { background: { ...options.background } } : {}) };
}

export const blindViewDigest = (view: BlindView) => createHash('sha256').update(JSON.stringify(view)).digest('hex');

/**
 * The gate (§10.1): structural (no forbidden key at any depth, no event kind outside the allow-list, strict views
 * carry no environment fields) and fragments (no sealed-answer fragment appears; `fragments` come from
 * `answerFragments` of the sealed cases and any privileged text). Returns the problems found; empty means admissible.
 */
export function checkBlindView(view: BlindView, fragments: readonly string[] = []): string[] {
  const problems: string[] = [];
  const visit = (value: unknown, path: string) => {
    if (Array.isArray(value)) value.forEach((item, i) => visit(item, `${path}[${i}]`));
    else if (value && typeof value === 'object') for (const [key, item] of Object.entries(value)) {
      if (FORBIDDEN.has(key)) problems.push(`forbidden key ${key} at ${path}`);
      visit(item, `${path}.${key}`);
    }
  };
  visit(view, 'view');
  view.attempts.forEach((attempt, a) => attempt.events.forEach((event, e) => {
    const spec = EVENTS[String(event.kind)];
    if (!spec) { problems.push(`event kind ${String(event.kind)} is not shown in blind views (attempt ${a}, event ${e})`); return; }
    if (view.visibility === 'reward-blind-strict') for (const key of spec.environment ?? [])
      if (key in event) problems.push(`strict view carries environment field ${key} (attempt ${a}, event ${e})`);
  }));
  const text = JSON.stringify(view);
  for (const fragment of fragments) if (fragment && text.includes(JSON.stringify(fragment).slice(1, -1))) {
    problems.push(`sealed fragment ${JSON.stringify(fragment.slice(0, 60))} appears in the view`);
  }
  return problems;
}
