/** What one specializer run returns for a definition. */
export type SpecializeResult =
  | { kind: 'specialized'; cases: number; unclassified: number }
  | { kind: 'declined'; reason: 'no-clusters' | 'semantic' | 'unstable' | 'effects' | 'not-worth-it'; why: string };
/** Whether a condition is how a decision is made, or only agrees with the meaning in the examples. */
export type ConditionKind = 'structural' | 'semantic';
/** Whether two normalized programs do the same work. */
export type Sameness = 'same' | 'different';
