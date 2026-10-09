/** What one group's writer returns. */
export type CaseResult =
  | { kind: 'case'; admits: number }
  | { kind: 'skip'; reason: 'semantic' | 'unstable' | 'effects' | 'no-condition'; why: string };
/** Whether a condition is how a decision is made, or only agrees with the meaning in the examples. */
export type ConditionKind = 'structural' | 'semantic';
/** Whether two normalized programs do the same work. */
export type Sameness = 'same' | 'different';
/** A promotion policy's answer about one case or tier (calls/evidence.ts). */
export type PromotionAnswer = { decision: 'promote' | 'keep' | 'demote'; reason: string };
