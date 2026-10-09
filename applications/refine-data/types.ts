/** The value of a nearMiss call: the edited copy and the one change made. */
export type NearMiss = { edited: unknown, edit: string };

/** The second teacher pass over a (value, edited) pair. */
export type Verification = {
  /** Whether the original value is the property. */
  original_holds: boolean,
  /** Whether the edited value is the property. */
  edited_holds: boolean,
  /** Whether the edit is the smallest change that matters: the two values differ in one detail. */
  minimal: boolean,
  /** One sentence naming what decides the two verdicts. */
  reason: string,
};

/** Satisfying values for one predicate, written to seed near-miss pairs. */
export type Examples = { values: unknown[] };
