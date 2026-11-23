/**
 * Decision readout for calls with a finite result type (`"a" | "b"`, `boolean`, a union of literals, a record whose
 * fields are all finite, a union of such records): `decisionDomain`. Up to DECISION_JOINT_OPTIONS values are scored
 * together, each as the whole reply; a larger record type is scored field by field on the reply prefix chosen so far.
 *
 * Instead of the tool loop, the runtime asks the model's driver to score every allowed result as the reply to the
 * call's opening, and returns the most probable one. The full distribution is kept in the trace, so a decision is a
 * scored probability vector: it can be scored with proper scoring rules (Brier, ranked
 * probability score) and, on a Neuralese server, differentiated. A call declares `readout: decision`.
 *
 * `readout: decision` reasons first (DECISIONS 2026-11-23 "reasoned decisions"): the model replies to the readout's
 * request `decisionSamples` times (its own reasoning, sampled at the call's temperature), and the options are scored
 * after each reasoning. A value's probability is the mean over the samples of its softmax after that reasoning: the
 * model's probability of answering with it when it reasons first, as it does when it generates (temperature shapes
 * the reasoning drawn; the scores are the model's own log-probabilities, not tempered). `readout: instant` scores the
 * options right after the request, without reasoning: one forward pass, for latency-critical calls; its
 * probabilities describe the model answering without reasoning.
 * Without scoring, the same prompt generates one actual JSON choice, with no inferred probabilities.
 * A call opts in through frontmatter, or every finite-typed call does when the model config sets `decisionReadout: 'finite-returns'`.
 */
import type { DecisionRequest, DecisionScorer, DecisionScores } from '../contracts.js';
import type { Type, TypeEnv } from './types.js';

export type DecisionValue = string | number | boolean | null;
export const DECISION_READOUT_MODES = ['decision', 'instant'] as const;
export type DecisionReadout = typeof DECISION_READOUT_MODES[number];

/** Every value of a finite type, in declaration order; null when the type is not finite. */
export function finiteValues(type: Type, env: TypeEnv): DecisionValue[] | null {
  const out: DecisionValue[] = [];
  const visit = (item: Type, depth: number): boolean => {
    if (depth > 32) return false;
    const resolved = item.kind === 'name' ? env.resolve(item) : item;
    switch (resolved.kind) {
      case 'refined': case 'untrusted': return visit(resolved.base, depth + 1);
      case 'lit': out.push(resolved.value); return true;
      case 'union': return resolved.members.every(member => visit(member, depth + 1));
      case 'prim':
        if (resolved.name === 'boolean') { out.push(true, false); return true; }
        if (resolved.name === 'null') { out.push(null); return true; }
        return false;
      default: return false;
    }
  };
  if (!visit(type, 0)) return null;
  return [...new Map(out.map(value => [JSON.stringify(value), value])).values()];
}

/** One field of a record arm: its key and the JSON texts of its values (with the values), in declaration order. */
export type DecisionSlot = { key: string; alternatives: { text: string; value: unknown }[] };
/** One arm of a finite type: a single value (a literal, `true`, `null`) or a record whose every field is finite. */
export type DecisionArm = { kind: 'value'; text: string; value: unknown } | { kind: 'record'; slots: DecisionSlot[] };
/** A finite result type as the readout scores it: its arms in declaration order and how many values they hold. */
export type DecisionDomain = { arms: DecisionArm[]; size: number };

/** Records hold at most this many values to be scored jointly (every record one option); larger ones go field by field. */
export const DECISION_JOINT_OPTIONS = 32;

const recordsEnabled = () => (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env?.NATLANG_DECISION_RECORDS !== '0';

/**
 * The finite domain of `type`: literal unions and booleans (as `finiteValues`), finite records (every field finite,
 * none optional) and unions of them, tagged or not. Null when the type is not finite. A record field's values are its
 * own domain enumerated (a nested record is one alternative per value). `NATLANG_DECISION_RECORDS=0` restores the
 * flat domain only.
 */
export function decisionDomain(type: Type, env: TypeEnv): DecisionDomain | null {
  const flat = finiteValues(type, env);
  if (flat) return { arms: flat.map(value => ({ kind: 'value' as const, text: JSON.stringify(value), value })), size: flat.length };
  if (!recordsEnabled()) return null;
  const arms: DecisionArm[] = [];
  const visit = (item: Type, depth: number): boolean => {
    if (depth > 32) return false;
    const resolved = item.kind === 'name' ? env.resolve(item) : item;
    switch (resolved.kind) {
      case 'refined': case 'untrusted': return visit(resolved.base, depth + 1);
      case 'union': return resolved.members.every(member => visit(member, depth + 1));
      case 'record': {
        if (resolved.fields.some(field => field.optional)) return false;
        const slots: DecisionSlot[] = [];
        for (const field of resolved.fields) {
          const inner = decisionDomain(field.type, env);
          if (!inner) return false;
          slots.push({ key: field.name, alternatives: [...domainValues(inner)] });
        }
        arms.push({ kind: 'record', slots });
        return true;
      }
      default: {
        const values = finiteValues(resolved, env);
        if (!values) return false;
        arms.push(...values.map(value => ({ kind: 'value' as const, text: JSON.stringify(value), value })));
        return true;
      }
    }
  };
  if (!visit(type, 0)) return null;
  const unique = [...new Map(arms.map(arm => [armForm(arm), arm])).values()];
  return { arms: unique, size: unique.reduce((sum, arm) => sum + armSize(arm), 0) };
}

/** Whether a readout scores the domain's values together, each as the whole reply: a flat domain always (as before
 * records), a domain with records up to DECISION_JOINT_OPTIONS values; otherwise field by field. */
export const scoredJointly = (domain: DecisionDomain) =>
  domain.size <= DECISION_JOINT_OPTIONS || domain.arms.every(arm => arm.kind === 'value');

const armSize = (arm: DecisionArm) => arm.kind === 'value' ? 1 : arm.slots.reduce((product, slot) => product * slot.alternatives.length, 1);
const pairText = (slot: DecisionSlot, text: string) => JSON.stringify(slot.key) + ':' + text;

/** Every value of a domain with its JSON text (compact, keys in declaration order), arms in order, first field slowest. */
export function* domainValues(domain: DecisionDomain): Generator<{ text: string; value: unknown }> {
  const seen = new Set<string>();
  for (const arm of domain.arms) {
    if (arm.kind === 'value') { if (!seen.has(arm.text)) { seen.add(arm.text); yield { text: arm.text, value: arm.value }; } continue; }
    const walk = function* (index: number, pairs: string[], value: Record<string, unknown>): Generator<{ text: string; value: unknown }> {
      if (index === arm.slots.length) { yield { text: '{' + pairs.join(',') + '}', value: { ...value } }; return; }
      const slot = arm.slots[index]!;
      for (const alternative of slot.alternatives)
        yield* walk(index + 1, [...pairs, pairText(slot, alternative.text)], { ...value, [slot.key]: alternative.value });
    };
    for (const item of walk(0, [], {})) if (!seen.has(item.text)) { seen.add(item.text); yield item; }
  }
}

/** One arm as the field-by-field prompt writes it: compact JSON with each field's values as `<A | B>`. */
export function armForm(arm: DecisionArm): string {
  if (arm.kind === 'value') return arm.text;
  return '{' + arm.slots.map(slot => pairText(slot, slot.alternatives.length === 1 ? slot.alternatives[0]!.text :
    '<' + slot.alternatives.map(alternative => alternative.text).join(' | ') + '>')).join(',') + '}';
}

/** A partial reply of a field-by-field readout: an arm and the alternatives chosen for its first fields. */
export type DecisionPath = { arm: number; chosen: number[] };
/** One candidate of a step: the reply text through its field (open), and whether that completes the value. */
export type DecisionCandidate = { text: string; complete: boolean; paths: DecisionPath[]; key?: string; value?: unknown };

/**
 * The candidates of the next step of a field-by-field readout from the paths still open. A candidate is the reply
 * from its start through the next field's value, then `,"` when another field follows (the cut lands where the next
 * key's name starts, so a tokenizer that merges punctuation runs, `","` or `":"`, tokenizes it as a prefix of the
 * whole reply), or `}` after the last; a value arm is its JSON text. Paths that write the same text share a candidate.
 */
export function stepCandidates(domain: DecisionDomain, paths: readonly DecisionPath[]): DecisionCandidate[] {
  const out = new Map<string, DecisionCandidate>();
  const add = (candidate: Omit<DecisionCandidate, 'paths'>, path: DecisionPath) => {
    const found = out.get(candidate.text);
    if (found) found.paths.push(path); else out.set(candidate.text, { ...candidate, paths: [path] });
  };
  for (const path of paths) {
    const arm = domain.arms[path.arm]!;
    if (arm.kind === 'value') { add({ text: arm.text, complete: true, value: arm.value }, path); continue; }
    if (!arm.slots.length) { add({ text: '{}', complete: true }, path); continue; }
    const index = path.chosen.length, slot = arm.slots[index]!;
    const before = path.chosen.map((choice, at) => pairText(arm.slots[at]!, arm.slots[at]!.alternatives[choice]!.text));
    slot.alternatives.forEach((alternative, choice) => {
      const last = index === arm.slots.length - 1;
      add({ text: '{' + [...before, pairText(slot, alternative.text)].join(',') + (last ? '}' : ',"'), complete: last,
        key: slot.key, value: alternative.value }, { arm: path.arm, chosen: [...path.chosen, choice] });
    });
  }
  return [...out.values()];
}

/** The value a completed path spells. */
export function pathValue(domain: DecisionDomain, path: DecisionPath): unknown {
  const arm = domain.arms[path.arm]!;
  return arm.kind === 'value' ? arm.value : Object.fromEntries(arm.slots.map((slot, at) => [slot.key, slot.alternatives[path.chosen[at]!]!.value]));
}

/** The domain's value equal to `value` (by JSON structure, keys in any order), with its text; undefined when none is. */
export function domainMatch(domain: DecisionDomain, value: unknown): { text: string; value: unknown } | undefined {
  const same = (a: unknown, b: unknown): boolean => {
    if (a === null || b === null || typeof a !== 'object' || typeof b !== 'object') return a === b;
    if (Array.isArray(a) || Array.isArray(b)) return false;
    const keys = Object.keys(a as object);
    return keys.length === Object.keys(b as object).length &&
      keys.every(key => Object.hasOwn(b as object, key) && same((a as Record<string, unknown>)[key], (b as Record<string, unknown>)[key]));
  };
  for (const arm of domain.arms) {
    if (arm.kind === 'value') { if (same(arm.value, value)) return { text: arm.text, value: arm.value }; continue; }
    if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).length !== arm.slots.length) continue;
    const pairs: string[] = [], out: Record<string, unknown> = {};
    const fits = arm.slots.every(slot => {
      if (!Object.hasOwn(value, slot.key)) return false;
      const alternative = slot.alternatives.find(item => same(item.value, (value as Record<string, unknown>)[slot.key]));
      if (!alternative) return false;
      pairs.push(pairText(slot, alternative.text)); out[slot.key] = alternative.value;
      return true;
    });
    if (fits) return { text: '{' + pairs.join(',') + '}', value: out };
  }
  return undefined;
}

/** The readout's system prompt: the call's opening stays, the tool manual does not (no tools are offered). */
export const DECISION_SYSTEM_PROMPT = 'You are running one call of a natural-language function. Carry out its ' +
  "instructions yourself for this call's argument values, judging the visible inputs. Your reply is the function's " +
  'return value: one JSON value of the declared type, with nothing else.';

export { decisionPrompt } from '../decision-frame.js';

/** The driver's decision scorer (contracts.ts `DecisionRequest`), if it carries the optional capability. */
/** Reasoning samples a `readout: decision` call draws by default: enough to estimate the reasoned answer
 * distribution (a floor or an escalation reads it), drawn in parallel so a batching server decodes them together. */
export const DECISION_SAMPLES = 4;

/** The reasoning samples of a `readout: decision` call: the configured number (`decisionSamples`), else
 * `NATLANG_DECISION_SAMPLES`, else DECISION_SAMPLES; one at temperature 0 (every draw is the same). 0 reads every
 * decision out without reasoning, as `readout: instant` does. */
export function decisionSamples(configured?: number, temperature?: number): number {
  const env = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env?.NATLANG_DECISION_SAMPLES;
  const value = configured ?? (env !== undefined && env !== '' ? Number(env) : DECISION_SAMPLES);
  if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`decisionSamples must be a whole number of reasoning samples (0 or more), not ${value}`);
  return temperature === 0 ? Math.min(value, 1) : value;
}

/** The temperature a `readout: decision` call draws its reasoning samples at when the call sets none: the sampling law
 * (T=1, as collection and the calibration that chose DECISION_SAMPLES). Left to the backend, the samples took its
 * default, which is greedy on the native engine (and on a stock model's greedy request settings): every sample was the
 * same reasoning, so a readout cost DECISION_SAMPLES decodes for one draw and its probabilities were one conditional
 * (B14 triage, DECISIONS 2026-11-23 "instant readout collapse"). `NATLANG_DECISION_REASONING_TEMPERATURE` sets
 * another value; `backend` restores the backend default. */
export const DECISION_REASONING_TEMPERATURE = 1;

/** The temperature of a readout's reasoning samples: the call's, else the `NATLANG_DECISION_REASONING_TEMPERATURE`
 * value, else DECISION_REASONING_TEMPERATURE; undefined (the backend's default) for `backend`. */
export function decisionReasoningTemperature(callTemperature?: number): number | undefined {
  if (callTemperature !== undefined) return callTemperature;
  const env = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env?.NATLANG_DECISION_REASONING_TEMPERATURE;
  if (env === undefined || env === '') return DECISION_REASONING_TEMPERATURE;
  if (env === 'backend') return undefined;
  const value = Number(env);
  if (!Number.isFinite(value) || value < 0) throw new RangeError(`NATLANG_DECISION_REASONING_TEMPERATURE must be a temperature (0 or more) or backend, not ${env}`);
  return value;
}

/** Scorers that declined (`decision-unsupported`: a backend that cannot score replies). */
const declinedScorers = new WeakSet<DecisionScorer>();

/** Record that a scorer declined: from then on it is no scorer, so readouts generate their value without first sampling
 * reasoning for a score that never comes, and refinement judges run as ordinary calls. */
export function declineScorer(scorer: DecisionScorer): void { declinedScorers.add(scorer); }

/** The driver's decision scorer (contracts.ts `DecisionRequest`), if it carries the capability and has not declined it. */
export function decisionScorer(driver: { decide?: DecisionScorer } | null | undefined): DecisionScorer | undefined {
  return typeof driver?.decide === 'function' && !declinedScorers.has(driver.decide) ? driver.decide : undefined;
}

/**
 * A scripted (non-model) driver's successful answer to one call. A decision readout run without a scoring driver
 * generates its choice on a tool-free request and reads it as JSON text (agent.ts `generatedDecision`), so a scripted
 * answer there is the value's JSON text; every other call returns through `return_result`.
 */
export function scriptedSuccess(request: { tools?: readonly unknown[] }, value: unknown):
    { text: string; calls: [] } | { calls: [string, Record<string, unknown>][] } {
  return Array.isArray(request.tools) && request.tools.length === 0 ? { text: JSON.stringify(value), calls: [] } :
    { calls: [['return_result', { status: 'success', value }]] };
}

/**
 * Score several decision requests at once. A scorer with its own `scoreMany` (batched on the server, or coalescing
 * the readouts of concurrent calls) is used; otherwise the items are issued concurrently, so the model scheduler and
 * the server see them together. One entry per item; an item that fails does so alone.
 */
export function scoreMany(scorer: DecisionScorer, items: DecisionRequest[], signal?: AbortSignal): Promise<PromiseSettledResult<DecisionScores>[]> {
  return scorer.scoreMany ? scorer.scoreMany(items, signal) : Promise.allSettled(items.map(item => scorer(item, signal)));
}

/** Normalised probabilities from log-probabilities (stable softmax). */
export function softmax(logProbs: number[]): number[] {
  const top = Math.max(...logProbs);
  const weights = logProbs.map(value => Math.exp(value - top));
  const total = weights.reduce((sum, value) => sum + value, 0);
  return weights.map(value => value / total);
}
