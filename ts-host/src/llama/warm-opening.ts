/**
 * Warm prompt openings, the engine-independent part: what a request's opening is and where its rendered text ends. The
 * managed server (llama/warm.ts) and the in-page engine (browser/warm-openings.ts) both prefill an opening once, save
 * the slot state and restore it later; both key a state by the same opening text (`openingText`), hashed to SHA-256.
 * Pure functions, safe in a page.
 */

/** A request's opening: its system prompt and tools; `text` is what identifies it (hash it for a key). */
export type OpeningParts = { system: { role: 'system'; content: string }; tools: unknown[]; text: string };

/** The opening of a request whose first message is a system prompt of at least 256 characters. */
export function openingParts(request: { messages: readonly unknown[]; tools?: readonly unknown[] }): OpeningParts | undefined {
  const system = request.messages[0] as { role?: unknown; content?: unknown } | undefined;
  if (system?.role !== 'system' || typeof system.content !== 'string' || system.content.length < 256) return undefined;
  const tools = [...(request.tools ?? [])];
  return { system: { role: 'system', content: system.content }, tools, text: JSON.stringify([system.content, tools]) };
}

/**
 * The longest text every conversation with this opening starts with, cut just after a special token so that the rest
 * of any prompt tokenizes on from it.
 */
export function sharedPrefix(first: string, second: string): string {
  let length = 0;
  while (length < first.length && length < second.length && first[length] === second[length]) length++;
  const common = first.slice(0, length);
  let cut = 0;
  for (const match of common.matchAll(/<\|[^|<>\s]{1,40}\|>|<\/?[a-z_]{2,24}>/g)) cut = match.index! + match[0].length;
  return common.slice(0, cut);
}

/** The two user messages whose renderings, after the opening, differ from their first character (`sharedPrefix`). */
export const OPENING_PROBES = ['⁣A', '⁣B'] as const;
