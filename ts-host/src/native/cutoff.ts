/**
 * How the runtime shortens what the model is shown. Every shortening leaves the same kind of note, `<<…>>`, in text
 * and in code alike: what was left out, where all of it is in the eval scope, and how to see more. The marker is
 * deliberately not TypeScript: it stands out in a listing, and a cut-off literal pasted into an eval fails to
 * compile at that spot instead of running on part of the data.
 *
 * Values are cut by structure (items, fields) by the value renderer (agent.ts renderValue); text output is cut by
 * position here, keeping its start and its end, where logs usually say how things went.
 */

/** How many characters of one output or value a message shows. */
export const SHOWN_CHARS = 2000;

/** Parts of a note: `holder` is the scope expression that holds all of it; `next` a call that shows the next part. */
export type Where = { holder?: string; next?: string };

/** The words of a note: `cut off: …; holder holds all of it; next shows the next part`. */
export function noteText(what: string, where: Where = {}): string {
  return [what, ...(where.holder ? [`${where.holder} holds all of it`] : []),
    ...(where.next ? [`${where.next} shows the next part`] : [])].join('; ');
}
/** The note, the same in text and in code: `<<cut off: …; customers holds all of it>>`. */
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
