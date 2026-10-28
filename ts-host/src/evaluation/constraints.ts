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
  | { kind: 'json_keys'; keys: string[] }
  | EditFidelityConstraint;

/**
 * Edit fidelity (edit_stream, 2026-10-28): the answer is the document after exactly the requested edits. The generator
 * applies every edit deterministically, so it knows the edited document. The candidate must have the reference's
 * paragraphs in order, each equal to the reference paragraph up to formatting the task leaves open (whitespace and line
 * wrapping inside a paragraph, a single or blank line between paragraphs, straight or typographic quotes) and at most
 * `tolerance` of its tokens edited (rounded down, so short paragraphs such as an added title or closing line are exact);
 * every rename source and target occurs exactly as often as in the reference (applied everywhere, nowhere else); and no
 * token 4-gram of a deleted paragraph that the reference lacks reappears. Keyword lists and length bands alone accepted
 * 88 unrelated repeated words on a held case (2026-10-26 audit).
 */
export type EditFidelityConstraint = { kind: 'edit_fidelity'; version: 'edit-fidelity/1';
  /** The reference document's paragraphs, in order, after every edit. */
  paragraphs: string[];
  /** Original text of every paragraph an instruction deleted. */
  deleted: string[];
  /** Every rename instruction, in order. */
  renames: { from: string; to: string }[];
  /** Largest fraction of a paragraph's reference tokens that may differ (token edit distance, floored). */
  tolerance: number };

export type ConstraintResult = { passed: boolean; score: number; failed: string[] };

const words = (text: string) => text.match(/[\p{L}\p{N}'’-]+/gu) ?? [];
const lowerWords = (text: string) => words(text).map(w => w.toLowerCase());
/** Sentences: runs ending in . ! ? (or the end of the text), bullets and lines count as sentence ends too. */
const sentences = (text: string) => text.split(/(?<=[.!?])\s+|\n+/).map(s => s.trim()).filter(s => words(s).length > 0);
const paragraphs = (text: string) => text.split(/\n\s*\n/).map(p => p.trim()).filter(Boolean);
const bullets = (text: string) => text.split('\n').filter(line => /^\s*[-*•]\s+\S/.test(line));

const normalizeEditText = (text: string) => text.normalize('NFC').replace(/\r\n?/g, '\n').replace(/[\u2018\u2019\u201b]/g, "'")
  .replace(/[\u201c\u201d\u201f]/g, '"').replace(/[\u00a0\u2007\u202f]/g, ' ').trim();
const editTokens = (text: string) => normalizeEditText(text).match(/[\p{L}\p{N}]+|[^\s\p{L}\p{N}]/gu) ?? [];
const nameCount = (text: string, name: string, flags: string) =>
  (text.match(new RegExp(`\\b${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, flags)) ?? []).length;
/** Token edit distance, or Infinity once it exceeds `limit` (every row minimum above the limit). */
function tokenDistance(a: string[], b: string[], limit: number): number {
  if (Math.abs(a.length - b.length) > limit) return Infinity;
  let previous = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const row = [i];
    for (let j = 1; j <= b.length; j++)
      row[j] = Math.min(previous[j]! + 1, row[j - 1]! + 1, previous[j - 1]! + (a[i - 1] === b[j - 1] ? 0 : 1));
    if (Math.min(...row) > limit) return Infinity;
    previous = row;
  }
  return previous[b.length]!;
}
const fourGrams = (tokens: string[]) => new Set(tokens.map(t => t.toLowerCase()).flatMap((_, i, all) =>
  i + 4 <= all.length ? [all.slice(i, i + 4).join(' ')] : []));

/** The ways the candidate departs from the reference edit (empty when it is the edited document). */
export function editFidelityProblems(text: string, c: EditFidelityConstraint): string[] {
  const problems: string[] = [];
  const reference = c.paragraphs.map(p => editTokens(p));
  const candidate = normalizeEditText(text);
  // A blank line separates paragraphs; a single line break also does when that is how the candidate lays them out.
  const split = (pattern: RegExp) => candidate.split(pattern).map(p => p.trim()).filter(Boolean);
  let paragraphs = split(/\n\s*\n/);
  if (paragraphs.length !== reference.length) {
    const lines = split(/\n+/);
    if (lines.length === reference.length) paragraphs = lines;
  }
  if (paragraphs.length !== reference.length)
    problems.push(`has ${paragraphs.length} paragraphs where the edited document has ${reference.length}`);
  else paragraphs.forEach((paragraph, i) => {
    const want = reference[i]!, limit = Math.floor(c.tolerance * want.length);
    const got = editTokens(paragraph);
    if (got.length > 2 * want.length + 8 || tokenDistance(got, want, limit) > limit)
      problems.push(`paragraph ${i + 1} is not the edited document's paragraph ${i + 1}`);
  });
  const referenceText = c.paragraphs.join('\n\n');
  for (const { from, to } of c.renames) for (const name of [from, to]) for (const flags of ['g', 'gi']) {
    const got = nameCount(candidate, name, flags), want = nameCount(referenceText, name, flags);
    if (got !== want) { problems.push(`"${name}" occurs ${got} times where the edited document has it ${want} times${flags === 'gi' ? ' (ignoring case)' : ''}`); break; }
  }
  const kept = fourGrams(reference.flat()), present = fourGrams(editTokens(candidate));
  c.deleted.forEach((paragraph, i) => {
    const distinctive = [...fourGrams(editTokens(paragraph))].filter(gram => !kept.has(gram));
    if (distinctive.some(gram => present.has(gram))) problems.push(`text of deleted paragraph ${i + 1} remains`);
  });
  return problems;
}

function holds(text: string, c: WritingConstraint): boolean {
  switch (c.kind) {
    case 'edit_fidelity': return editFidelityProblems(text, c).length === 0;
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
    case 'edit_fidelity': return 'Return the whole document with exactly the requested edits applied and nothing else changed.';
  }
}
