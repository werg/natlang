import type { Untrusted } from '@natlang/node';

/**
 * One verified fact of a training world (never a development, confirmation or held world). `invocation` names the one
 * call it belongs to (earlier-attempt feedback, fit hindsight); `match` holds the exact prompt strings that identify the
 * child call a `child_value` answers.
 */
export type GuidanceFact = {
  id: string,
  kind: "expected_result" | "expected_stop" | "expected_files" | "expected_effects" | "reference_failure"
    | "reference_solution" | "child_value" | "minimum_sequence" | "decisive" | "teacher_trajectory"
    | "attempt_feedback" | "fit_hindsight",
  text: Untrusted<string>,
  invocation?: string | null,
  match?: string[],
};

/**
 * The call the guidance is for. `root` is the episode's root call, `child` a call it made, `binding` a learned-optimizer
 * instruction or skill state (it sees what a root sees). `text` is the call's own prompt text.
 */
export type GuidanceConsumer = { name: string, role: "root" | "child" | "binding", text: Untrusted<string> };

/**
 * One scored student group of a JOLT+ collection (planTeacherAttempts): how its students did on one training world,
 * the runtime feedback their attempts received, and how many verified facts the world has.
 */
export type TeacherGroup = {
  group: number, taskId: string, family: string | null, students: number, successes: number, meanQuality: number,
  feedback: Untrusted<string>[], facts: number,
};
/** Privileged teacher attempts planned for one group. */
export type TeacherAttempts = { group: number, attempts: number };

/**
 * One tool call of a recorded execution, in the shape the runtime's own transcript gives a call
 * (`TranscriptEntry`, ts-host/src/native/runtime.ts): the model turn it was made in, the reasoning that led to it, the
 * tool and its arguments (`code` for eval), its outcome `status` (ok, rejected, error, completed, ...), an eval's
 * returned `value` as data, what it printed (`console`) and the full text the model was shown (`output`).
 */
export type TrajectoryStep = {
  turn: number, tool: string, code?: Untrusted<string>, arguments: Record<string, unknown>, status: string,
  value?: unknown, console?: Untrusted<string>, output: Untrusted<string>, reasoning?: Untrusted<string>,
};

/**
 * One call of a verified successful execution of a task: its function, its prompt text, its own steps in order, how it
 * finished, and the executions of the child calls it made (in the order it made them). A trajectory is a call tree.
 */
export type TrajectoryCall = {
  call: string, function: string, role: "root" | "child", prompt: Untrusted<string>, steps: TrajectoryStep[],
  result: { status: string, value?: unknown, reason?: Untrusted<string> }, children: TrajectoryCall[],
};

/** A crisp structural overview of one call of a trajectory (trajectoryOverview.ts): its own steps and its children's sizes. */
export type TrajectoryOverview = {
  call: string, function: string, role: "root" | "child", promptChars: number, turns: number,
  steps: { index: number, turn: number, tool: string, status: string, chars: number }[],
  errors: number, result: string,
  children: { call: string, function: string, steps: number, descendants: number, result: string }[],
};
