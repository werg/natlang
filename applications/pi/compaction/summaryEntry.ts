/** pi-durable compaction.ts `placeSummary`: the summary entry a compaction places. */

export const SUMMARY_PREFIX = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";
export const SUMMARY_SUFFIX = "\n</summary>";

type Entry = { kind: 'pi.compaction'; head: number; model: { role: 'user'; content: { type: 'text'; text: string }[]; timestamp: number }[]; data: { reason: string } };

/**
 * The "pi.compaction" entry draft for `summary`: head = firstKept, one user message whose text is SUMMARY_PREFIX +
 * summary + SUMMARY_SUFFIX stamped `now`, data {reason}.
 */
export default function summaryEntry(summary: string, firstKept: number, reason: string, now: number): Entry {
  return { kind: 'pi.compaction', head: firstKept,
    model: [{ role: 'user', content: [{ type: 'text', text: `${SUMMARY_PREFIX}${summary}${SUMMARY_SUFFIX}` }], timestamp: now }],
    data: { reason } };
}
