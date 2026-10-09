/**
 * The summarizer's request (pi-durable compaction.ts): pi's verbatim prompts, the transcript serialization, and the
 * messages before the cut.
 */

export const TOOL_RESULT_MAX_CHARS = 2000;

export const SUMMARIZATION_SYSTEM_PROMPT = `You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.`;

export const SUMMARIZATION_PROMPT = `The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work. If the conversation starts with an earlier summary, preserve its information and fold the newer messages into it.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages.`;

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

/** The text of message content; a Neuralese block (types.ts NeuraleseContent), which a text reader cannot read, is named. */
function contentText(content: unknown): string {
  if (typeof content === 'string') return content;
  return (content as { type: string; text?: string; id?: string }[])
    .flatMap(block => (block.type === 'text' && block.text !== undefined ? [block.text] :
      block.type === 'neuralese' ? [`[Neuralese block ${block.id}]`] : []))
    .join('\n');
}

function truncate(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  return `${text.slice(0, maxChars)}\n\n[... ${text.length - maxChars} more characters truncated]`;
}

/**
 * Messages as a plain transcript, so the summarizer reads it instead of continuing it: "[User]: …", "[Assistant
 * thinking]: …", "[Assistant]: …", "[Assistant tool calls]: name(key=<JSON>, …); …", "[Tool result]: …" (cut to 2000
 * characters), parts joined by blank lines. System messages are omitted.
 */
export function serializeConversation(messages: Message[]): string {
  const parts: string[] = [];
  for (const message of messages) {
    if (message.role === 'user') {
      const text = contentText(message.content);
      if (text.length > 0) parts.push(`[User]: ${text}`);
    } else if (message.role === 'assistant') {
      const content = message.content as { type: string; thinking?: string; text?: string; name?: string; arguments?: Record<string, unknown> }[];
      const thinking = content.flatMap(item => (item.type === 'thinking' ? [item.thinking!] : []));
      const text = content.flatMap(item => (item.type === 'text' ? [item.text!] : []));
      const calls = content.flatMap(item => item.type === 'toolCall' ?
        [`${item.name}(${Object.entries(item.arguments ?? {}).map(([key, value]) => `${key}=${JSON.stringify(value)}`).join(', ')})`] : []);
      if (thinking.length > 0) parts.push(`[Assistant thinking]: ${thinking.join('\n')}`);
      if (text.length > 0) parts.push(`[Assistant]: ${text.join('\n')}`);
      if (calls.length > 0) parts.push(`[Assistant tool calls]: ${calls.join('; ')}`);
    } else if (message.role === 'toolResult') {
      const text = contentText(message.content);
      if (text.length > 0) parts.push(`[Tool result]: ${truncate(text, TOOL_RESULT_MAX_CHARS)}`);
    }
  }
  return parts.join('\n\n');
}

/** The summarizer's user text: the serialized conversation, the prompt, and "Additional focus: <instructions>" when given. */
export function summaryPrompt(messages: Message[], instructions: string | null): string {
  const focus = instructions === null ? '' : `\n\nAdditional focus: ${instructions}`;
  return `<conversation>\n${serializeConversation(messages)}\n</conversation>\n\n${SUMMARIZATION_PROMPT}${focus}`;
}

/**
 * The whole summary request for `view.entries[0..cut)`: the system prompt, then one user message whose text is
 * summaryPrompt over the summarized messages, both stamped `now`. No tools.
 */
export function summaryMessages(view: View, cut: number, instructions: string | null, now: number): Message[] {
  return [
    { role: 'system', content: SUMMARIZATION_SYSTEM_PROMPT, timestamp: now },
    { role: 'user', content: [{ type: 'text', text: summaryPrompt(summarizedMessages(view, cut), instructions) }], timestamp: now },
  ];
}
