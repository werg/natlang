/** What one step of a `.nl` body does. judgment: weighs evidence or meaning and decides; mechanism: exact work a
 * program does the same way every time (set operations, counting, copying fields, calling a fixed sequence of
 * functions, assembling a record); guard-for-type: restates what the declared types already guarantee; redundant:
 * repeats another step or asks again for what is already known. */
export type StepKind = "judgment" | "mechanism" | "guard-for-type" | "redundant"; // @natlang/std StepKind, named here for the .nl signatures

/** Whether a legacy program is migrated into the new style or deleted. */
export type Keep = "migrate" | "delete";

/** After a rejected equivalence comparison: write another candidate, or stop and record the rejection. */
export type Next = "retry" | "give-up";

/** One step of a body, with what it does. */
export type KindedStep = { text: string, kind: StepKind };

/** An idiom-floor or check finding in a program: its file, line, code and message as `natlang check` reports them. */
export type Finding = { file: string, line: number, code: string, message: string };

/** How a program did on its own cases before migration: cases measured, cases passed at least once, and the passes
 * over the scored samples. */
export type Baseline = { cases: number, passingCases: number, passes: number, scored: number };

/** A driver rewrite: the TypeScript helper module (its default export runs the fixed steps) and the new instructions,
 * which call the helper by its name. */
export type DriverRewrite = { helper: string, instructions: string };

/** The comparison of the migrated version with the original on the same cases and seeds (07 §4.2). */
export type Comparison = { accepted: boolean, problems: string[], expectedPassingFirst: number, expectedPassingSecond: number };
