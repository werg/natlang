/** The crisp implementation of the `settled` point (`settled.nl` is the natural-language one): the same four rules. */
import type { RepairState } from '../types.js';

export default function settledCrisp(state: RepairState): boolean {
  if (state.validation?.status === 'passed') return true;
  if (state.remaining <= 0) return true;
  if (state.findings.length > 0 && state.findings.every(finding => !finding.repairable)) return true;
  return new Set(state.revisions).size < state.revisions.length;
}
