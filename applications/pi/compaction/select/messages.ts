/** The messages a compaction summarizes, for the beforeCompact hook. */

const MISSING_RESULT_TEXT = 'Tool result unavailable: history ends before this call completed.';

type Text = { type: 'text'; text: string };
type Call = { type: 'toolCall'; id: string; name: string; arguments: Record<string, unknown> };
type Message = { role: string; content: unknown; timestamp: number; toolCallId?: string; toolName?: string };
type View = { contributions: Message[][] };

/**
 * Each assistant's tool results directly after it, in call order (pi-durable context.ts `orderToolResults`): results are
 * taken from the messages before the next assistant; a missing one is synthesized; unmatched results are dropped.
 */
function orderToolResults(messages: Message[]): Message[] {
  const ordered: Message[] = [];
  const count = messages.length;
  for (let index = 0; index < count; index++) {
    const message = messages[index]!;
    if (message.role === 'toolResult') continue;
    ordered.push(message);
    if (message.role !== 'assistant') continue;
    const calls = (message.content as { type: string }[]).filter((content): content is Call => content.type === 'toolCall');
    if (calls.length === 0) continue;
    const results = new Map<string, number>();
    for (let next = index + 1; next < count; next++) {
      const candidate = messages[next]!;
      if (candidate.role === 'assistant') break;
      if (candidate.role === 'toolResult' && !results.has(candidate.toolCallId!)) results.set(candidate.toolCallId!, next);
    }
    for (const call of calls) {
      const at = results.get(call.id);
      ordered.push(at === undefined ? { role: 'toolResult', toolCallId: call.id, toolName: call.name,
        content: [{ type: 'text', text: MISSING_RESULT_TEXT }], isError: true, details: { reason: 'missing_result' },
        timestamp: message.timestamp } as Message : messages[at]!);
    }
  }
  return ordered;
}

/**
 * The model messages of `view.entries[0..cut)` (pi-durable compaction.ts `summarizedMessages`): the head marker's
 * first, ordered as in model context, so an earlier summary is summarized again.
 */
export function summarizedMessages(view: View, cut: number): Message[] {
  return orderToolResults(view.contributions.slice(0, cut).flat());
}
