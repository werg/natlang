/**
 * Verifiable writing constraints: the oracle for writing and chat-style natlang functions whose answer is open
 * text. Each constraint is checked by code (no judge), in the spirit of IFEval's verifiable instructions, so a
 * writing case can be admitted with the same exactness as a computed answer.
 */
export type WritingConstraint =
  | { kind: 'word_count'; min?: number; max?: number }
  | { kind: 'sentence_count'; min?: number; max?: number }
  | { kind: 'paragraph_count'; count: number }
  | { kind: 'bullet_count'; count: number }
  | { kind: 'include_words'; words: string[] }
  | { kind: 'exclude_words'; words: string[] }
  | { kind: 'word_frequency'; word: string; min: number }
  | { kind: 'starts_with'; text: string }
  | { kind: 'ends_with'; text: string }
  | { kind: 'no_commas' }
  | { kind: 'all_lowercase' }
  | { kind: 'title'; }
  | { kind: 'placeholders'; min: number }
  | { kind: 'json_keys'; keys: string[] };

export type ConstraintResult = { passed: boolean; score: number; failed: string[] };

const words = (text: string) => text.match(/[\p{L}\p{N}'’-]+/gu) ?? [];
const lowerWords = (text: string) => words(text).map(w => w.toLowerCase());
/** Sentences: runs ending in . ! ? (or the end of the text), bullets and lines count as sentence ends too. */
const sentences = (text: string) => text.split(/(?<=[.!?])\s+|\n+/).map(s => s.trim()).filter(s => words(s).length > 0);
const paragraphs = (text: string) => text.split(/\n\s*\n/).map(p => p.trim()).filter(Boolean);
const bullets = (text: string) => text.split('\n').filter(line => /^\s*[-*•]\s+\S/.test(line));

function holds(text: string, c: WritingConstraint): boolean {
  switch (c.kind) {
    case 'word_count': {
      const n = words(text).length;
      return (c.min === undefined || n >= c.min) && (c.max === undefined || n <= c.max);
    }
    case 'sentence_count': {
      const n = sentences(text).length;
      return (c.min === undefined || n >= c.min) && (c.max === undefined || n <= c.max);
    }
    case 'paragraph_count': return paragraphs(text).length === c.count;
    case 'bullet_count': return bullets(text).length === c.count;
    case 'include_words': {
      const present = new Set(lowerWords(text));
      return c.words.every(w => lowerWords(w).every(part => present.has(part)));
    }
    case 'exclude_words': {
      const present = new Set(lowerWords(text));
      return c.words.every(w => !lowerWords(w).every(part => present.has(part)));
    }
    case 'word_frequency': return lowerWords(text).filter(w => w === c.word.toLowerCase()).length >= c.min;
    case 'starts_with': return text.trimStart().startsWith(c.text);
    case 'ends_with': return text.trimEnd().endsWith(c.text);
    case 'no_commas': return !text.includes(',');
    case 'all_lowercase': return text === text.toLowerCase() && /\p{L}/u.test(text);
    case 'title': return /<<[^<>\n]+>>/.test(text);
    case 'placeholders': return (text.match(/\[[^\[\]\n]+\]/g) ?? []).length >= c.min;
    case 'json_keys': {
      try {
        const body = text.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
        const value = JSON.parse(body);
        return value !== null && typeof value === 'object' && !Array.isArray(value) && c.keys.every(k => k in value);
      } catch { return false; }
    }
  }
}

/** Check every constraint; `score` is the fraction that hold, `failed` describes the ones that do not. */
export function checkConstraints(text: unknown, constraints: WritingConstraint[]): ConstraintResult {
  if (typeof text !== 'string') return { passed: false, score: 0, failed: ['the answer is not text'] };
  const failed = constraints.filter(c => !holds(text, c)).map(describeConstraint);
  return { passed: failed.length === 0, score: constraints.length ? 1 - failed.length / constraints.length : 1, failed };
}

/** The constraint as an instruction a writer can follow (used in the function's instructions and in feedback). */
export function describeConstraint(c: WritingConstraint): string {
  switch (c.kind) {
    case 'word_count':
      return c.min !== undefined && c.max !== undefined ? `Use between ${c.min} and ${c.max} words.`
        : c.min !== undefined ? `Use at least ${c.min} words.` : `Use at most ${c.max} words.`;
    case 'sentence_count':
      return c.min !== undefined && c.max !== undefined ? (c.min === c.max ? `Write exactly ${c.min} sentences.` : `Write between ${c.min} and ${c.max} sentences.`)
        : c.min !== undefined ? `Write at least ${c.min} sentences.` : `Write at most ${c.max} sentences.`;
    case 'paragraph_count': return `Write exactly ${c.count} paragraphs, separated by a blank line.`;
    case 'bullet_count': return `Include exactly ${c.count} bullet points, each on its own line starting with "- ".`;
    case 'include_words': return `Use these words: ${c.words.map(w => `"${w}"`).join(', ')}.`;
    case 'exclude_words': return `Do not use these words: ${c.words.map(w => `"${w}"`).join(', ')}.`;
    case 'word_frequency': return `Use the word "${c.word}" at least ${c.min} times.`;
    case 'starts_with': return `Start with exactly: ${JSON.stringify(c.text)}.`;
    case 'ends_with': return `End with exactly: ${JSON.stringify(c.text)}.`;
    case 'no_commas': return 'Do not use any commas.';
    case 'all_lowercase': return 'Write everything in lowercase letters.';
    case 'title': return 'Give it a title wrapped in double angle brackets, like <<title>>.';
    case 'placeholders': return `Include at least ${c.min} placeholders in square brackets, like [name].`;
    case 'json_keys': return `Answer with a JSON object that has the keys ${c.keys.map(k => `"${k}"`).join(', ')}.`;
  }
}
