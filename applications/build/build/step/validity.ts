/**
 * Cache validity: whether a task's recorded outputs may be reused, from what the workspace saw and recorded. A
 * pluggable hot path of every round. The setting `build.implementation('validity')` selects the digest comparison
 * below (crisp) or `validity/judge.nl` (natural-language).
 */
import { build } from 'natlang:services';
import judge from './validity/judge.nl';
import type { Evidence, Task, Validity } from '../../types.js';

/** The first mismatch between the evidence and the record, in the order declaration, inputs, outputs. */
export function crisp(_task: Task, evidence: Evidence): Validity {
  const record = evidence.recorded;
  if (!record) return { valid: false, reason: 'no record of an earlier run' };
  if (record.fingerprint !== evidence.fingerprint) return { valid: false, reason: 'declaration changed' };
  for (const input of evidence.inputs) {
    const before = record.inputs.find(row => row.path === input.path);
    if (!before || before.sha256 !== input.sha256) return { valid: false, reason: `input changed: ${input.path}` };
  }
  for (const output of evidence.outputs) {
    const before = record.outputs.find(row => row.path === output.path);
    if (output.sha256 === null) return { valid: false, reason: `output missing: ${output.path}` };
    if (!before || before.sha256 !== output.sha256) return { valid: false, reason: `output modified: ${output.path}` };
  }
  return { valid: true, reason: 'up to date' };
}

export default async function validity(task: Task, evidence: Evidence): Promise<Validity> {
  if (await build.implementation('validity') === 'natural-language') return judge(task, evidence);
  return crisp(task, evidence);
}
