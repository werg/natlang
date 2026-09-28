/** Shared collection/admission policy for exercises whose premise no longer exists in the runtime. */
import type { ProgramRecord } from './program.js';

export const RETIRED_FAMILIES: ReadonlySet<string> = new Set(['inline_type_repair']);
export function retiredFamily(record: ProgramRecord): string | undefined {
  const curriculum = record.curriculum as { family?: string } | undefined;
  return curriculum?.family && RETIRED_FAMILIES.has(curriculum.family) ? curriculum.family : undefined;
}
