import type { Untrusted } from '@natlang/node';

/** What chooseCondition returns for one group: a condition to write a case for, or why no condition decides the group. */
export type ConditionChoice =
  | { kind: 'condition'; condition: string; admits: number }
  | { kind: 'skip'; reason: 'semantic' | 'unstable' | 'effects' | 'no-condition'; why: string };
/** What writeBody returns for a chosen condition: case.ts is written, or the effects make a case unsafe. */
export type BodyResult =
  | { kind: 'case' }
  | { kind: 'skip'; reason: 'effects' | 'unstable'; why: string };
/** Whether a condition is how a decision is made, or only agrees with the meaning in the examples. */
export type ConditionKind = 'structural' | 'semantic';
/** Whether two normalized programs do the same work. */
export type Sameness = 'same' | 'different';
/** A promotion policy's answer about one case or tier (calls/evidence.ts). */
export type PromotionAnswer = { decision: 'promote' | 'keep' | 'demote'; reason: string };

/** What the target policy sees of one definition revision. */
export type WorthCandidate = { name: string; agent_calls: number; tokens: number;
  /** New agent calls since the current compilation, counted up to min_calls. */
  new_calls_since_compilation: number;
  decline: { calls_at_decline: number; reason: string } | null; has_compilation: boolean };
/** The target policy's answer: whether the specializer looks at the definition now. */
export type WorthAnswer = { look: boolean; reason: string };

/** One group's outcome in a specializing pass, as the decline summary reads it. */
export type DeclineGroup = { id: string; label: string; calls: number; outcome: string; reason: string | null; why: string | null };

/** One recorded call the guidance is distilled from. The text is recorded data. */
export type GuidanceExample = { did: string; input: Untrusted<string>; result: Untrusted<string> };
