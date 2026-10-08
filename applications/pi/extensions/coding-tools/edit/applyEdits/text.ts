/**
 * The text mechanics of the edit tool's matching (pi edit-diff.ts): LF normalization, the tolerant normalization,
 * finding (exact, then tolerant), counting, and applying replacements, including the replacement that keeps
 * unchanged lines' original bytes. Index arithmetic and character classes, so crisp.
 */

/** A match of one edit: its index in the edits list, where it starts and how long it is in the content matched against, and its new text. */
export type Match = { editIndex: number; matchIndex: number; matchLength: number; newText: string };

/** Every CRLF and lone CR becomes LF. */
export function normalizeToLF(text: string): string {
  return text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
}

/**
 * The tolerant form of a text: NFKC, trailing whitespace stripped from each line, smart quotes as ASCII quotes, Unicode
 * dashes as "-", special spaces as " ".
 */
export function normalizeForFuzzyMatch(text: string): string {
  return text.normalize('NFKC').split('\n').map(line => line.trimEnd()).join('\n')
    .replace(/[‘’‚‛]/g, "'")
    .replace(/[“”„‟]/g, '"')
    .replace(/[‐‑‒–—―−]/g, '-')
    .replace(/[  -   　]/g, ' ');
}

/**
 * Find oldText in content: exactly first, else in the tolerant forms of both. `index` and `matchLength` are in content
 * when exact, in the tolerant content when `usedFuzzyMatch`.
 */
export function fuzzyFindText(content: string, oldText: string): { found: boolean; index: number; matchLength: number; usedFuzzyMatch: boolean } {
  const exact = content.indexOf(oldText);
  if (exact !== -1) return { found: true, index: exact, matchLength: oldText.length, usedFuzzyMatch: false };
  const fuzzyContent = normalizeForFuzzyMatch(content), fuzzyOld = normalizeForFuzzyMatch(oldText);
  const index = fuzzyContent.indexOf(fuzzyOld);
  if (index === -1) return { found: false, index: -1, matchLength: 0, usedFuzzyMatch: false };
  return { found: true, index, matchLength: fuzzyOld.length, usedFuzzyMatch: true };
}

/** How often the tolerant form of oldText occurs in the tolerant form of content. */
export function countOccurrences(content: string, oldText: string): number {
  return normalizeForFuzzyMatch(content).split(normalizeForFuzzyMatch(oldText)).length - 1;
}

/** Apply matches to content, from the last to the first so indexes stay valid; indexes are shifted by -offset. */
export function applyReplacements(content: string, replacements: Match[], offset?: number): string {
  const shift = offset ?? 0;
  let result = content;
  for (let i = replacements.length - 1; i >= 0; i--) {
    const replacement = replacements[i]!;
    const at = replacement.matchIndex - shift;
    result = result.substring(0, at) + replacement.newText + result.substring(at + replacement.matchLength);
  }
  return result;
}

function splitLinesWithEndings(content: string): string[] {
  return content.match(/[^\n]*\n|[^\n]+/g) ?? [];
}

function lineSpans(content: string): { start: number; end: number }[] {
  let offset = 0;
  return splitLinesWithEndings(content).map(line => {
    const span = { start: offset, end: offset + line.length };
    offset = span.end;
    return span;
  });
}

function replacementLineRange(lines: { start: number; end: number }[], replacement: Match): { startLine: number; endLine: number } {
  const start = replacement.matchIndex, end = replacement.matchIndex + replacement.matchLength;
  const startLine = lines.findIndex(line => start >= line.start && start < line.end);
  if (startLine === -1) throw new Error('Replacement range is outside the base content.');
  let endLine = startLine;
  for (let i = startLine; i < lines.length; i++) {
    if (lines[i]!.end >= end) break;
    endLine = i + 1;
  }
  if (endLine >= lines.length) throw new Error('Replacement range is outside the base content.');
  return { startLine, endLine: endLine + 1 };
}

/**
 * Apply matches found in `baseContent` (the tolerant form of `originalContent`) while keeping every line no match
 * touches as it is in `originalContent`: each touched group of lines is rewritten from the base, the rest copied.
 */
export function applyReplacementsPreservingUnchangedLines(originalContent: string, baseContent: string, replacements: Match[]): string {
  const originalLines = splitLinesWithEndings(originalContent);
  const baseLines = lineSpans(baseContent);
  if (originalLines.length !== baseLines.length) throw new Error('Cannot preserve unchanged lines because the base content has a different line count.');
  const groups: { startLine: number; endLine: number; replacements: Match[] }[] = [];
  for (const replacement of [...replacements].sort((a, b) => a.matchIndex - b.matchIndex)) {
    const range = replacementLineRange(baseLines, replacement);
    const current = groups[groups.length - 1];
    if (current && range.startLine < current.endLine) {
      current.endLine = Math.max(current.endLine, range.endLine);
      current.replacements.push(replacement);
      continue;
    }
    groups.push({ ...range, replacements: [replacement] });
  }
  let originalLine = 0;
  let result = '';
  for (const group of groups) {
    result += originalLines.slice(originalLine, group.startLine).join('');
    const from = baseLines[group.startLine]!.start, to = baseLines[group.endLine - 1]!.end;
    result += applyReplacements(baseContent.slice(from, to), group.replacements, from);
    originalLine = group.endLine;
  }
  return result + originalLines.slice(originalLine).join('');
}
