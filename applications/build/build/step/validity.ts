/**
 * Cache validity: whether a task's recorded outputs may be reused, from what the workspace saw and recorded. A
 * pluggable hot path of every round. The setting `build.implementation('validity')` selects the digest comparison
 * of the workspace (crisp) or `validity/judge.nl` (natural-language).
 */
import { pluggable } from '@natlang/node';
import { build } from 'natlang:services';
import judge from './validity/judge.nl';
import type { Evidence, Task, Validity } from '../../types.js';

/**
 * The crisp implementation: the first mismatch between the ledger entry and the files, in the order declaration, inputs,
 * outputs. The comparison itself lives once, in the workspace (`build.mismatch`, which `reuse` also refuses on).
 */
export async function crisp(task: Task, _evidence: Evidence): Promise<Validity> {
  const problem = await build.mismatch(task);
  return problem === null ? { valid: true, reason: 'up to date' } : { valid: false, reason: problem };
}

export default async function validity(task: Task, evidence: Evidence): Promise<Validity> {
  return pluggable({ crisp: () => crisp(task, evidence), nl: () => judge(task, evidence) }, await build.implementation('validity'),
    { name: 'build.validity', same: (exact, judged) => exact.valid === judged.valid })();
}
