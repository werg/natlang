/** Shared collection/admission policy for exercises whose premise no longer exists in the runtime. */
import type { ProgramRecord } from './program.js';

// Untyped nl results now run open, so inline_type_repair no longer triggers its required compiler refusal.
export const RETIRED_FAMILIES: ReadonlySet<string> = new Set(['inline_type_repair']);
export function retiredFamily(record: ProgramRecord): string | undefined {
  const curriculum = record.curriculum as { family?: string } | undefined;
  return curriculum?.family && RETIRED_FAMILIES.has(curriculum.family) ? curriculum.family : undefined;
}
