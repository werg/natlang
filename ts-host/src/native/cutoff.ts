/**
 * How the runtime shortens what the model is shown. Every shortening leaves the same kind of note, `<<…>>`, in text
 * and in code alike: what was left out, where all of it is in the eval scope, and how to see more. The marker is
 * deliberately not TypeScript: it stands out in a listing, and a cut-off literal pasted into an eval fails to
 * compile at that spot instead of running on part of the data.
 *
 * Values are cut by structure (items, fields) by the value renderer (agent.ts renderValue); text output is cut by
 * position here, keeping its start and its end, where logs usually say how things went.
 */

/** The fewest characters of one output or value a message shows (the page size of a session with no context budget). */
export const SHOWN_CHARS = 2000;

/** Scope and input values may be longer than tool output; larger values remain available in eval. */
export const SCOPE_VALUE_CHARS = 4096;

/**
 * A value whose whole literal fits in this many characters is never shortened, whatever budget is left for it (a
 * `Stored local …` line, a late field of a long record): a small record shown as its field names, or a short string cut
 * after 17 characters, read as missing data.
 */
export const SMALL_VALUE_CHARS = 300;

/** Whether plain data (strings, numbers, booleans, null, arrays, plain records) has a literal of at most `limit` characters; bounded work. */
export function literalFits(value: unknown, limit: number): boolean {
  let left = limit;
  const visit = (item: unknown, depth: number): boolean => {
    if (item === null || typeof item === 'boolean' || typeof item === 'number') { left -= String(item).length; return left >= 0; }
    if (typeof item === 'string') {
      if (item.length > left) return false;
      left -= JSON.stringify(item).length; return left >= 0;
    }
    if (typeof item !== 'object' || depth > 8) return false;
    if (Array.isArray(item)) {
      left -= 2;
      for (const element of item) { left -= 2; if (left < 0 || !visit(element, depth + 1)) return false; }
      return left >= 0;
    }
    const proto = Object.getPrototypeOf(item);
    if (Object.prototype.toString.call(item) !== '[object Object]' || (proto !== null && Object.getPrototypeOf(proto) !== null)) return false;
    left -= 4;
    for (const key of Object.keys(item)) {
      left -= key.length + 4;
      if (left < 0 || !visit((item as Record<string, unknown>)[key], depth + 1)) return false;
    }
    return left >= 0;
  };
  return visit(value, 0);
}

/**
 * The share of the context budget one tool output may fill. Compaction starts past three quarters of the budget and
 * brings the prompt back under half, so an eighth leaves room for the opening, several reads and the reply.
 */
export const OUTPUT_CONTEXT_SHARE = 1 / 8;
/** Characters per prompt token for sizing pages (the agent's initial estimate, before server calibration). */
export const PAGE_CHARS_PER_TOKEN = 3.5;

/**
 * How many characters of one output a message shows, from the call's context budget in prompt tokens (no fixed page
 * size, owner 2026-10-11): a long text fits in one read where the context allows, and only text that truly exceeds an
 * eighth of the context is cut off and paged (read_page). Never below SHOWN_CHARS; `null` (no budget) gives that floor.
 */
export function shownCharsFor(contextTokens: number | null | undefined): number {
  if (contextTokens === null || contextTokens === undefined || !Number.isFinite(contextTokens)) return SHOWN_CHARS;
  return Math.max(SHOWN_CHARS, Math.floor(contextTokens * OUTPUT_CONTEXT_SHARE * PAGE_CHARS_PER_TOKEN));
}

/** Parts of a note: `holder` is the scope expression that holds the complete value; `next` a call that shows the next part. */
export type Where = { holder?: string; next?: string };

/**
 * The words of a note: `cut off: …; only this display is shortened, holder holds the complete value; next shows the
 * next part`. It says that the value is whole: models read the earlier `holder holds all of it` as data that was lost.
 */
export function noteText(what: string, where: Where = {}): string {
  return [what, ...(where.holder ? [`only this display is shortened, ${where.holder} holds the complete value`] : []),
    ...(where.next ? [`${where.next} shows the next part`] : [])].join('; ');
}
/** The note, the same in text and in code: `<<cut off: …; only this display is shortened, customers holds the complete value>>`. */
export const note = (what: string, where: Where = {}) => `<<${noteText(what, where)}>>`;

/**
 * Where a long text is cut: it shows [0, head) and [tail, end). Cuts move to a nearby line break when there is one,
 * so lines are not split. read_page's first page is exactly the head.
 */
export function cutPoints(text: string, budget = SHOWN_CHARS): { head: number; tail: number } {
  let head = Math.floor(budget * 0.75), tail = text.length - (budget - head);
  const headBreak = text.lastIndexOf('\n', head), tailBreak = text.indexOf('\n', tail);
  if (headBreak > head - 200) head = headBreak;
  if (tailBreak >= 0 && tailBreak < tail + 200) tail = tailBreak + 1;
  return { head, tail };
}

/** A text as it fits in a message: whole if short, else its head and its tail with the note between them. */
export function cutText(text: string, where: Where = {}, budget = SHOWN_CHARS): string {
  if (text.length <= budget) return text;
  const { head, tail } = cutPoints(text, budget);
  return `${text.slice(0, head)}\n${note(`cut off: ${tail - head} of ${text.length} characters not shown`, where)}\n${text.slice(tail)}`;
}
