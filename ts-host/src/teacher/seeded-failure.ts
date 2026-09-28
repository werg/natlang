import type { ProgramRecord } from './program.js';
import type { Handoff } from './handoff.js';

/** Replacing the first planted action may prevent its failure rather than observe it. */
export function replacesPlantedFailure(record: ProgramRecord): boolean {
  const seed = record.semantics.failure_seed, handoff = record.handoff as Handoff | undefined;
  return !!seed && handoff?.kind === 'failed_action' && handoff.call === 0 &&
    !(handoff.prefix[0]?.length) && !!handoff.rejected.calls?.some(([tool, args]) => tool === 'eval' &&
      (args as { code?: string })?.code === seed.code);
}
