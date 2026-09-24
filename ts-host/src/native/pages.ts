import { SHOWN_CHARS, cutPoints, cutText, note } from './cutoff.js';

/** Short, common words: page IDs a tokenizer handles as one token each, handed out in this order. */
const WORDS = ['amber', 'birch', 'cedar', 'delta', 'ember', 'fern', 'grove', 'harbor', 'iris', 'juniper', 'kelp',
  'lark', 'maple', 'nectar', 'oak', 'pebble', 'quartz', 'river', 'sage', 'thistle', 'umber', 'violet', 'willow',
  'yarrow', 'zinc', 'anchor', 'basil', 'canyon', 'dune', 'elm', 'flint', 'garnet', 'heron', 'ivy', 'jade', 'lotus',
  'meadow', 'north', 'olive', 'pine', 'quill', 'reed', 'slate', 'tide', 'vale', 'wren'];

/**
 * Output that a tool result cut off, kept whole so the model can read the rest with read_page.
 * IDs are assigned in a fixed order per call and never reused, so a page always shows the same text.
 */
export class PageStore {
  private readonly pages = new Map<string, string[]>();

  /**
   * The text as it fits in a tool result (see cutoff.ts): whole if short, else its first page and a note naming
   * `holder`, which holds all of it, and the read_page call that shows the next page.
   */
  show(text: string, holder?: string, budget = SHOWN_CHARS): string {
    if (text.length <= budget) return text;
    const { id } = this.add(text, budget);
    return cutText(text, { holder, next: `read_page("${id}", 2)` }, budget);
  }

  /** Keep a text in pages and return its ID. The first page is the head a cut-off shows (cutPoints). */
  add(text: string, budget = SHOWN_CHARS): { id: string; count: number } {
    const index = this.pages.size;
    const id = WORDS[index % WORDS.length]! + (index < WORDS.length ? '' : String(Math.floor(index / WORDS.length) + 1));
    const { head } = cutPoints(text, budget);
    const pages: string[] = [text.slice(0, head)];
    for (let start = head; start < text.length; start += budget) pages.push(text.slice(start, start + budget));
    this.pages.set(id, pages);
    return { id, count: pages.length };
  }

  read(id: string, page: number): string {
    const pages = this.pages.get(id);
    if (!pages) throw new Error(`no cut-off output is named ${JSON.stringify(id)}; use an ID from a cut-off message`);
    if (!Number.isInteger(page) || page < 1 || page > pages.length)
      throw new Error(`${id} has pages 1 to ${pages.length}`);
    return pages[page - 1]! + '\n' + (page < pages.length ?
      note(`page ${page} of ${pages.length} shown`, { next: `read_page("${id}", ${page + 1})` }) : note(`page ${page} of ${pages.length}, the last`));
  }
}
