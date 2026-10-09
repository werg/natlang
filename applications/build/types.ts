import type { Is } from '@natlang/node';

/**
 * A declared build task: a trusted argv command with the files it reads and writes. The workspace runs it without a
 * shell, inside the root. needs lists the tasks that must finish first. inputs and outputs are paths relative to the
 * root, and every output is a path no earlier build left behind unless the ledger recorded it.
 */
export type Task = { id: string, needs: string[], description: string, argv: string[], inputs: string[], outputs: string[] };

/** What the executor reports for one task. status unknown means the process was interrupted: its effects may have happened. */
export type TaskResult = { id: string, status: 'ok' | 'failed' | 'unknown', exit_code: number, input_sha256: string,
  output_sha256: string, detail: string };

/**
 * One task as the dependency graph sees it. deps = needs plus producers: a task depends on every task it names in
 * needs and on every task whose output it reads. sources are inputs that no task produces: files that must exist before
 * the build. dependents is the reverse of deps. All lists are sorted.
 */
export type Node = { id: string, needs: string[], producers: string[], deps: string[], dependents: string[], sources: string[] };

/**
 * A fault in the declarations. code names the rule: empty-id, duplicate-id, unknown-goal, unknown-need,
 * self-dependency, duplicate-need, no-command, no-output, duplicate-output, self-read, undeclared-dependency.
 * tasks are the ids involved. detail says what to change, in one sentence.
 */
export type Problem = { code: string, tasks: string[], detail: string };

/** The declarations understood: one node per task, sorted by id, and every fault found. An empty problems list is a valid graph. */
export type Graph = { nodes: Node[], problems: Problem[] };

/**
 * The build's plan for one goal. needed is the goal's closure: the goal and everything it depends on, sorted.
 * order is one valid order of needed (each task after its deps, ties by smaller id first). cycle lists the tasks that
 * can never become ready because they wait on each other; empty when the plan is acyclic.
 */
export type Plan = { goal: string, needed: string[], order: string[], cycle: string[] };

/** What the workspace records and sees for a task: the evidence that cache validity is judged from. */
export type Evidence = {
  task: string,
  /** Digest of the declaration (argv, inputs, outputs). It changes when the task is redefined. */
  fingerprint: string,
  /** The task's inputs now; sha256 is null when the file is missing. */
  inputs: { path: string, sha256: string | null }[],
  /** The task's outputs now; sha256 is null when the file is missing. */
  outputs: { path: string, sha256: string | null }[],
  /** The ledger entry of the last successful run in this root, or null when there is none. */
  recorded: { fingerprint: string, inputs: { path: string, sha256: string }[], outputs: { path: string, sha256: string }[] } | null,
};

/** Whether a task's recorded outputs may be reused, and the reason in one short phrase. */
export type Validity = { valid: boolean, reason: string };

/**
 * One round's decision, as data. run: execute task. reuse: its recorded outputs are valid, so settle it without
 * running. stop: nothing more can run (task is empty). reject: the chosen task was not ready.
 */
export type Decision = { kind: 'run' | 'reuse' | 'stop' | 'reject', task: string, why: string };

/** The cause of a failed or interrupted task, as a person fixing the build needs it. */
export type Diagnosis = {
  task: string,
  /** command-failed, missing-input, missing-tool, output-conflict, input-mutated, path-escape, declaration-error, interrupted or other. */
  cause: string,
  /** The file, tool or task the evidence points at; empty when none. */
  culprit: string,
  summary: string,
  fix: string,
  /** no: running again changes nothing. after-fix: run again once the fix is made. inspect-first: the outcome is unknown; check the outputs before running again. */
  retry: 'no' | 'after-fix' | 'inspect-first',
};

/** Why a task ran or was reused in this build. */
export type Judgment = { task: string, action: 'ran' | 'reused', reason: string };

/**
 * The build as one value. A round decides from this snapshot, performs its effect, and commits a new state.
 * order lists the tasks that finished, built or reused, in order. attempted lists every task acted on, failed ones
 * too. status is running until the goal is done or a round stops the build.
 */
export type BuildState = {
  goal: string, tasks: Task[], graph: Graph, needed: string[],
  order: string[], attempted: string[], results: TaskResult[], judgments: Judgment[], blocked: string[],
  status: 'running' | 'done' | 'failed' | 'unknown' | 'blocked' | 'invalid', detail: string,
  diagnosis?: Diagnosis,
};

/** The closing words of a build report: what happened, and what to do next. */
export type Summary = { summary: string, next: string[] };

/** The build's report. Everything but summary and next is the final state's own data. */
export type BuildReport = {
  goal: string, status: 'running' | 'done' | 'failed' | 'unknown' | 'blocked' | 'invalid', detail: string,
  order: string[], results: TaskResult[], blocked: string[],
  built: string[], reused: string[], not_run: string[], judgments: Judgment[],
  diagnosis?: Diagnosis, problems: Problem[], plan?: Plan,
  summary: string, next: string[],
};

// ---------------------------------------------------------------- refined results
// What a stage returns carries the property its value shows by itself. Declarations (Task) and the state the commit
// builds keep plain types: `declare` and `commit` enforce them exactly. Each predicate below has a crisp checker in
// refinements.ts. (Summary.summary would take a judged predicate; it is proposed in DECOMPOSITION.md and not wired.)

/** The result of `order`: needed is sorted without duplicates and holds the goal; order and cycle split it. */
export type CheckedPlan = Is<Plan, "a plan whose needed list is sorted without duplicates and contains the goal, and whose order and cycle together hold each needed task exactly once">;

/** The result of the validity judgment: valid exactly when the reason says the record is up to date. */
export type CheckedValidity = Is<Validity, "valid exactly when the reason is up to date; an invalid verdict's reason is no record of an earlier run, declaration changed, or input changed, output missing or output modified followed by a colon, a space and a path">;

/** The result of `diagnose`: a known cause, and the retry advice that follows from it. */
export type CheckedDiagnosis = Is<Diagnosis, "a diagnosis whose cause is command-failed, missing-input, missing-tool, output-conflict, input-mutated, path-escape, declaration-error, interrupted or other, and whose retry is inspect-first for interrupted, no for other and after-fix for every other cause">;

/** The build's report: done only when the goal was built or reused and every result is ok. */
export type CheckedReport = Is<BuildReport, "a report whose status is done only when order contains the goal and every result has status ok">;
