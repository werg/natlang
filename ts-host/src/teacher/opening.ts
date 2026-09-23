/** The opening of a model conversation: what a call shows the model before its first own turn. */
export type Message = { role: string; content?: unknown; tool_calls?: { id?: string; function?: { name?: string; arguments?: string } }[];
  tool_call_id?: string };

/** Length of the opening: system, user, and the runtime's pre-filled `scope_` exchanges. */
export function openingLength(context: Message[]): number {
  let length = 2;
  while (context[length]?.role === 'assistant' && (context[length]!.tool_calls ?? []).length &&
      context[length]!.tool_calls!.every(call => String(call.id).startsWith('scope_')))
    length += 1 + context[length]!.tool_calls!.length;
  return length;
}
/** Message content as text. */
export const text = (value: unknown) => typeof value === 'string' ? value : JSON.stringify(value ?? '');
/** The opening's text: instructions, the pre-filled evals, and their results (argument values). */
export const openingText = (context: Message[]) => context.slice(1, openingLength(context))
  .map(message => text(message.content) + (message.tool_calls ?? []).map(call => call.function?.arguments ?? '').join('\n')).join('\n');
