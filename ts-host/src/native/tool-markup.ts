/** Text that is actually native tool syntax must never be accepted as a plain answer. */
export const TOOL_MARKUP = /<tool_call>|<\/tool_call>|<function=|<\/function>|<parameter=|<\/parameter>|<\|tool_call_start\|>|<\|tool_call_end\|>/;

/** The close of a reply's reasoning block (thinking templates open the block in the generation prompt). */
const THINK_CLOSE = '</think>';
/** The one sentence a reply that closed its reasoning a second time is answered with (`ModelTurn.malformed_call`). */
export const SECOND_THINK_CLOSE = 'Your last reply closed its reasoning a second time, so nothing ran; close the reasoning once, ' +
  'then make your tool call and end the reply.';

/**
 * Whether a reply closed its reasoning more than once. Every natlang engine (managed llama-server and the page through
 * the fork's chat parser, the Python native engine and the SGLang actor through `serve/chat.py`) splits the reply at
 * its first `</think>`, drops that close and returns the rest as content, so a close left in the content or the
 * reasoning is a second one. A server that splits nothing leaves a whole leading `<think>…</think>` block in the
 * content: that block's close is the first. grpo-v4-run2 learned to write a second close after short answers instead of
 * ending its turn (DECISIONS 2026-11-19); read as content, the reply's calls ran and the malformed form was rewarded.
 */
export function secondThinkClose(content: string, reasoning: unknown): boolean {
  if (typeof reasoning === 'string' && reasoning.includes(THINK_CLOSE)) return true;
  const opened = /^\s*<think>/.test(content) ? content.indexOf(THINK_CLOSE) : -1;
  return content.includes(THINK_CLOSE, opened < 0 ? 0 : opened + THINK_CLOSE.length);
}
