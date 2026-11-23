/**
 * Semantic-FP task author & qualifier (WS3 T3, WS6 §2 Tier 0 flagship).
 * Combines `semanticRepair` drafting against deterministic structural/custody/baseline checks
 * with calibrated `readout: decision` semantic & oracle reviews (`escalate`).
 */
import { escalate, semanticRepair } from '@natlang/node/std';
import draftTask from './draftTask.nl';
import isSemantic from './isSemantic.nl';
import reviewOracle from './reviewOracle.nl';
import type { AuthoredTask, QualificationResult, TaskSeed } from './types.js';

export type AuthorOptions = {
  /** Reserved split groups (held-v3/held-v4 custody) that must never be authored into train. */
  reservedGroups?: ReadonlySet<string>;
  /** Optional custom keyword-baseline runner: returns true if a naive baseline passes (which rejects the task). */
  keywordBaselinePasses?: (task: AuthoredTask) => boolean;
  /** Maximum repair attempts (default `3`). */
  attempts?: number;
  /** Stronger model to escalate uncertain qualification decisions to (default `'teacher'`). */
  escalateTo?: string;
};

/** Exact structural, custody, and construct checks matching authored_checks.py. */
export function checkTaskStructure(
  task: AuthoredTask,
  seed: TaskSeed,
  reservedGroups?: ReadonlySet<string>,
): string[] {
  const issues: string[] = [];
  if (!task.id || task.id !== seed.id) issues.push(`task.id must equal seed.id (${seed.id}).`);
  if (!task.split_group || task.split_group !== seed.splitGroup)
    issues.push(`task.split_group must equal seed.splitGroup (${seed.splitGroup}).`);
  if (reservedGroups?.has(task.split_group))
    issues.push(`split_group "${task.split_group}" is reserved for held evaluation and cannot enter training.`);
  if (!Array.isArray(task.constructs) || !task.constructs.includes(seed.construct))
    issues.push(`task.constructs must include "${seed.construct}".`);
  if (!task.prompt || task.prompt.trim().length < 20)
    issues.push('task.prompt must describe a concrete task (at least 20 characters).');
  const files = task.files ?? {};
  if (!('natlang.json' in files)) issues.push('task.files must include natlang.json.');
  if (!Object.keys(files).some(p => p.endsWith('.ts') || p.endsWith('.nl')))
    issues.push('task.files must include at least one .ts or .nl source file.');
  if (task.oracle === 'verifier' && !task.verifier)
    issues.push('verifier oracle tasks must specify task.verifier (e.g. code_test.mjs, proper_score.mjs, service_trace.mjs, policy_eval.mjs).');
  return issues;
}

export async function authorAndQualifyTask(seed: TaskSeed, options: AuthorOptions = {}): Promise<QualificationResult> {
  const initial = await draftTask(seed, []);
  const check = (candidate: AuthoredTask): string[] => {
    const issues = checkTaskStructure(candidate, seed, options.reservedGroups);
    if (options.keywordBaselinePasses?.(candidate))
      issues.push('Task is solved by the trivial keyword baseline; rewrite with genuine paraphrase/semantic nuance.');
    return issues;
  };
  const repaired = await semanticRepair(initial, {
    check,
    fix: async (_prev, problems) => draftTask(seed, problems),
    attempts: options.attempts ?? 3,
  });

  const structuralIssues = check(repaired.value);
  const baselineDefeated = !(options.keywordBaselinePasses?.(repaired.value) ?? false);
  const to = options.escalateTo ?? 'teacher';
  const judgeSemantic = escalate(isSemantic, { to, floor: 0.8 });
  const judgeOracle = escalate(reviewOracle, { to, floor: 0.8 });

  const [sem, orc] = await Promise.all([
    judgeSemantic(repaired.value),
    judgeOracle(repaired.value),
  ]);

  const admitted = structuralIssues.length === 0 && baselineDefeated && sem.value && orc.value;
  return {
    admitted,
    task: repaired.value,
    structuralIssues,
    semanticVerdict: sem.value,
    oracleVerdict: orc.value,
    baselineDefeated,
    escalated: sem.escalated || orc.escalated,
  };
}
