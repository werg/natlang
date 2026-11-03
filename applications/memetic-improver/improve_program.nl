---
description: Improve a natlang program for its future invocations by measured, typed changes.
args:
  program: ProgramBrief
returns: Improvement
types:
  ProgramBrief: "{ family: string; root: string; functions: string[]; supportAttempts: number; supportSuccess: number; budget: { evaluations: number; fits: number } }"
  Improvement: "{ committed: string[]; reason: string }"
  RefineState: "{ kept: string[]; best: number; tried: { id: string; success: number; kept: boolean }[]; misses: number; done: boolean; notes: string }"
---
1. program runs on many invocations. workspace holds its source, how its recent support invocations went, and the operations that change it and measure a change.
2. Read workspace.failures() and the source of the functions they involve. Open an attempt with workspace.attempt(id) when its outcome alone does not show what went wrong.
3. Improve the program step by step with refine: await refine.iterateOn({ kept: [], best: program.supportSuccess, tried: [], misses: 0, done: false, notes: "" }, program).withLimit({ maxSteps: program.budget.evaluations }).until(state => state.done).
4. Commit the final state's kept ids with workspace.commit(kept, reason), where reason says in one sentence what they fix, and return { committed, reason }. Commit an empty list when no change helped.
