/**
 * The specializer's policies, separate from its orchestration so each can be tested without a store or a model.
 *
 * Pluggable (the `promotionPolicy` pattern, runtime/promotion.ts): which definitions are worth a look (`targetPolicy`)
 * and the stored text of a decline (`declinePolicy`), each `crisp` (default), `nl` or `shadow` (both run, crisp
 * serves, agreement is traced). Crisp bounds stay in force whatever the natural-language side answers.
 *
 * Loop control: the rounds loop is bounded by a named resource limit (`rounds`) and stops early at a fixed point
 * (`madeProgress`), see there.
 */
import { pluggable, type PluggableMode } from '@natlang/node';
import type { DeclineGroup, WorthAnswer, WorthCandidate } from './types.js';

/** Defaults of the CLI options and limits, each with its reason. */
export const DEFAULTS = {
  /** Resource limit of the repair loop: a writer's attempts per group. Three tries fix most reports; more is spend with little return. */
  rounds: 3,
  /** Shadow replays and audits per cycle, so one cycle stays short enough to be interrupted and to look at new targets. */
  jobs: 50,
  /** Seconds between cycles with --loop: new calls accumulate slowly. */
  interval: 600,
  /** Executor load (vLLM running plus waiting requests) above which the specializer waits, so it does not compete with the programs it serves. */
  maxBusy: 2,
  /** Seconds to wait for an idle executor before going ahead, so a permanently busy executor does not starve specializing. */
  idleWait: 1800,
  /** Groups per definition that get a writer: bounds spend per definition. */
  maxGroups: 8,
};

/** The crisp rule for a worthwhile definition, spelled the way worthLooking.nl spells it. */
export function crispWorth(candidate: WorthCandidate, minCalls: number): WorthAnswer {
  if (candidate.agent_calls < minCalls) return { look: false, reason: 'too few calls' };
  // Calls that used no model tokens (scripted test drivers) have nothing to save.
  if (!candidate.tokens) return { look: false, reason: 'nothing to save' };
  if (candidate.decline && candidate.agent_calls < 2 * candidate.decline.calls_at_decline)
    return { look: false, reason: 'the earlier decline stands until the volume doubles' };
  if (candidate.has_compilation && candidate.new_calls_since_compilation < minCalls)
    return { look: false, reason: 'too few new calls since the last compilation' };
  return { look: true, reason: 'enough unspecialized volume' };
}

/** A natural-language answer as the policy uses it: the crisp bound (at least `minCalls` agent calls) holds whatever it says. */
export function sanitizeWorth(value: unknown, candidate: WorthCandidate, minCalls: number): WorthAnswer {
  const answer = value as Partial<WorthAnswer> | null;
  const reason = typeof answer?.reason === 'string' && answer.reason ? answer.reason : '(no reason given)';
  if (typeof answer?.look !== 'boolean') return { look: false, reason: 'the policy gave no valid answer' };
  if (candidate.agent_calls < minCalls) return { look: false, reason: 'too few calls' };
  return { look: answer.look, reason };
}

export type WorthFunction = (candidate: WorthCandidate, minCalls: number) => WorthAnswer | Promise<WorthAnswer>;

/** The target policy as one function: `crisp`, `nl` or `shadow` (crisp serves, agreement is traced). */
export function worthPolicy(mode: PluggableMode, judge: WorthFunction): (candidate: WorthCandidate, minCalls: number) => Promise<WorthAnswer> {
  return pluggable<[WorthCandidate, number], WorthAnswer>({ crisp: (candidate, minCalls) => crispWorth(candidate, minCalls),
    nl: async (candidate, minCalls) => sanitizeWorth(await judge(candidate, minCalls), candidate, minCalls) }, mode,
  { name: 'specializer.worthLooking', serve: 'crisp', same: (crisp, other) => crisp.look === other.look });
}

/** One group's line in the crisp decline text. */
export function groupLine(group: DeclineGroup): string {
  return `${group.id} (${group.label}, ${group.calls} calls): ${group.reason ?? 'not accepted'}${group.why ? `: ${group.why}` : ''}`;
}

/** The crisp decline text: the group lines, joined. */
export function crispDeclineText(groups: readonly DeclineGroup[]): string {
  return `no case was accepted: ${groups.map(groupLine).join('; ') || 'no cases written'}`.slice(0, 2000);
}

export type DeclineFunction = (groups: DeclineGroup[]) => string | Promise<string>;

/**
 * The decline text as one function. Only stored text; no decision depends on it. Under `shadow` the crisp text is stored
 * and the natural-language text is compared only for being a usable sentence.
 */
export function declinePolicy(mode: PluggableMode, judge: DeclineFunction): (groups: DeclineGroup[]) => Promise<string> {
  return pluggable<[DeclineGroup[]], string>({ crisp: groups => crispDeclineText(groups),
    nl: async groups => {
      const text = String(await judge(groups)).trim();
      return text ? `no case was accepted: ${text}`.slice(0, 2000) : crispDeclineText(groups);
    } }, mode,
  { name: 'specializer.summarizeDecline', serve: 'crisp', same: (_crisp, other) => other.length > 0 });
}

/**
 * Whether the last round of the repair loop changed anything a writer could use: a group became done, a group's report
 * differs, or a group's written case differs. States are summarized as strings by `stateKey`. When none changed, the
 * next round would run the same writers on the same evidence.
 *
 * The loop is bounded by `rounds` (a resource limit, always honoured) and also ends at this fixed point. That is the
 * most general sound choice: it terminates (the limit), it never stops while any group still gets new information
 * (progress), and it spends nothing on rounds whose input is identical to the last round's. A rule that stopped on
 * "no group improved" alone would end rounds in which a writer produced a new case or a new report that the next round
 * could repair.
 */
export function madeProgress(before: readonly string[], after: readonly string[]): boolean {
  return before.length !== after.length || after.some((key, index) => key !== before[index]);
}

/** A group state as one comparable string. */
export const stateKey = (state: { done?: string; report?: string; part?: string }): string =>
  JSON.stringify([state.done ?? null, state.report ?? null, state.part ?? null]);

/** Longest guidance kept: the student reads it on every call, so it stays short. */
export const GUIDANCE_LIMIT = 1500;
/** Words that state a point as a prohibition. Guidance states every point as an action. */
const NEGATION = /\b(?:do not|don't|dont|never|must not|mustn't|should not|shouldn't|cannot|can't|avoid|refrain|no longer|not)\b/i;

/** The problems of a piece of guidance: its length, and the sentences written as prohibitions (empty when it is fine). */
export function guidanceProblems(text: string): { tooLong: boolean; empty: boolean; negative: string[] } {
  const sentences = text.split(/(?<=[.!?])\s+|\n+/).map(sentence => sentence.trim()).filter(Boolean);
  return { tooLong: text.length > GUIDANCE_LIMIT, empty: text.trim().length < 20, negative: sentences.filter(sentence => NEGATION.test(sentence)) };
}

/** The feedback that goes back to writeGuidance, or '' when the guidance is fine. */
export function guidanceFeedback(text: string): string {
  const { tooLong, empty, negative } = guidanceProblems(text);
  const parts: string[] = [];
  if (empty) parts.push('The guidance is empty or too short to guide.');
  if (tooLong) parts.push(`The guidance has ${text.length} characters; keep it within ${GUIDANCE_LIMIT}.`);
  if (negative.length) parts.push(`Rewrite these sentences as actions to take:\n${negative.map(sentence => `- ${sentence}`).join('\n')}`);
  return parts.join('\n');
}

/**
 * Guidance for the compilation's `instructions.md`: asks `write` once, returns its text when it passes the crisp
 * checks, asks once more with the feedback when it does not, and returns undefined when the second text fails too.
 */
export async function checkedGuidance(write: (feedback: string) => Promise<string>): Promise<string | undefined> {
  let feedback = '';
  for (let attempt = 0; attempt < 2; attempt++) {
    const text = String(await write(feedback)).trim();
    feedback = guidanceFeedback(text);
    if (!feedback) return text + '\n';
  }
  return undefined;
}
