/**
 * pi-durable tool.ts `fromSlot`: the error result of a call that cannot finish normally (aborted, or interrupted after
 * intent), built from the durable partial its slot holds.
 */

type Diagnostic = { severity: 'info' | 'warn' | 'error'; message: string; code?: string };
type Slot = { output?: string; droppedBytes?: number; droppedLines?: number; details?: unknown; diagnostics?: Diagnostic[] };
type Result = { content: { type: 'text'; text: string }[]; isError: true; details?: unknown; diagnostics: Diagnostic[] };

/**
 * content: one text item with the slot's output when it is not empty, else []. isError true. details: the slot's, if
 * any. diagnostics: the slot's, then "Output truncated: <lines> lines, <bytes> bytes dropped" (warn, code truncated)
 * when the slot dropped bytes, then the error `code` with `message`. `slot` null: no slot is listed.
 */
export default function fromSlot(slot: Slot | null, code: string, message: string): Result {
  const diagnostics: Diagnostic[] = [...(slot?.diagnostics ?? [])];
  const droppedBytes = slot?.droppedBytes ?? 0;
  if (droppedBytes > 0) diagnostics.push({ severity: 'warn', code: 'truncated',
    message: `Output truncated: ${slot?.droppedLines ?? 0} lines, ${droppedBytes} bytes dropped` });
  diagnostics.push({ severity: 'error', code, message });
  return {
    content: slot?.output === undefined || slot.output === '' ? [] : [{ type: 'text', text: slot.output }],
    isError: true,
    ...(slot?.details === undefined ? {} : { details: slot.details }),
    diagnostics,
  };
}
