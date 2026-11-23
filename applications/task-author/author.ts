/**
 * Semantic-FP task author and qualifier (WS3 T3, WS6 §2 Tier 0 flagship; plans/semantic-fp/03 §4).
 * `draftTask.nl` writes a task with a reference answer, negative answers and a keyword-baseline answer. Exact checks
 * (structure, held custody, and replaying every answer through the task's oracle) feed `semanticRepair`. Admission
 * also needs `isSemantic.nl` and `reviewOracle.nl`, escalated when unsure.
 */
import { escalate, semanticRepair } from '@natlang/node/std';
import draftTask from './draftTask.nl';
import isSemantic from './isSemantic.nl';
import reviewOracle from './reviewOracle.nl';
import type { AuthoredTask, QualificationResult, TaskSeed } from './types.js';

export type AuthorOptions = {
  /** Whether a split group is reserved for held evaluation (held-v3/held-v4 custody; `held_eval.reserved`). */
  isReserved: (splitGroup: string) => boolean;
  /** Run the task's oracle on one answer: true when it accepts it. The outside world (verifier or exact match). */
  replay: (task: AuthoredTask, answer: string) => boolean | Promise<boolean>;
  /** Maximum repair attempts (default `3`). */
  attempts?: number;
  /** Stronger model to escalate uncertain qualification decisions to (default `'teacher'`). */
  escalateTo?: string;
};

/** Exact structure and custody checks of a draft. */
export function checkTaskStructure(task: AuthoredTask, seed: TaskSeed, isReserved: (group: string) => boolean): string[] {
  const issues: string[] = [];
  if (task.id !== seed.id) issues.push(`task.id must equal seed.id (${seed.id}).`);
  if (task.split_group !== seed.splitGroup) issues.push(`task.split_group must equal seed.splitGroup (${seed.splitGroup}).`);
  if (isReserved(task.split_group))
    issues.push(`split_group "${task.split_group}" is reserved for held evaluation and cannot enter training.`);
  if (!Array.isArray(task.constructs) || !task.constructs.includes(seed.construct))
    issues.push(`task.constructs must include "${seed.construct}".`);
  if (!task.prompt || task.prompt.trim().length < 20) issues.push('task.prompt must describe a concrete task (at least 20 characters).');
  const files = task.files ?? {};
  if (!('natlang.json' in files)) issues.push('task.files must include natlang.json.');
  if (!Object.keys(files).some(p => p.endsWith('.ts') || p.endsWith('.nl'))) issues.push('task.files must include at least one .ts or .nl source file.');
  if (task.oracle === 'verifier' && !task.verifier)
    issues.push('a verifier oracle task names its verifier (code_test.mjs, proper_score.mjs, service_trace.mjs or policy_eval.mjs).');
  if (!task.reference?.trim()) issues.push('task.reference must be a correct answer.');
  if (!Array.isArray(task.negatives) || task.negatives.filter(n => n?.trim()).length < 2) issues.push('task.negatives must hold at least two wrong answers.');
  if (!task.keywordBaseline?.trim()) issues.push('task.keywordBaseline must be the answer a cue-word rule gives.');
  return issues;
}

/** Replay the draft's answers through its oracle: the reference passes, every negative and the keyword baseline fail. */
export async function replayProblems(task: AuthoredTask, replay: AuthorOptions['replay']): Promise<string[]> {
  const problems: string[] = [];
  if (!await replay(task, task.reference)) problems.push('The oracle rejects task.reference; fix the oracle or the reference.');
  for (const [index, negative] of task.negatives.entries())
    if (await replay(task, negative)) problems.push(`The oracle accepts task.negatives[${index}]; make the oracle reject it or write a wrong answer.`);
  if (await replay(task, task.keywordBaseline))
    problems.push('The oracle accepts task.keywordBaseline: a cue-word rule solves the task; rewrite it so meaning, not wording, decides.');
  return problems;
}

export async function authorAndQualifyTask(seed: TaskSeed, options: AuthorOptions): Promise<QualificationResult> {
  const check = async (candidate: AuthoredTask): Promise<string[]> => {
    const structural = checkTaskStructure(candidate, seed, options.isReserved);
    return structural.length ? structural : replayProblems(candidate, options.replay);
  };
  const repaired = await semanticRepair(await draftTask(seed, []), {
    check,
    fix: (_previous, problems) => draftTask(seed, problems),
    attempts: options.attempts ?? 3,
  });
  const task = repaired.value;
  const problems = [...repaired.problems];
  if (problems.length) return { admitted: false, task, problems, semanticVerdict: false, oracleVerdict: false, escalated: false };

  const to = options.escalateTo ?? 'teacher';
  const [semantic, oracle] = await Promise.all([
    escalate(isSemantic, { to, floor: 0.8 })(task),
    escalate(reviewOracle, { to, floor: 0.8 })(task),
  ]);
  return {
    admitted: semantic.value && oracle.value,
    task,
    problems,
    semanticVerdict: semantic.value,
    oracleVerdict: oracle.value,
    escalated: semantic.escalated || oracle.escalated,
  };
}
