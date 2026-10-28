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
