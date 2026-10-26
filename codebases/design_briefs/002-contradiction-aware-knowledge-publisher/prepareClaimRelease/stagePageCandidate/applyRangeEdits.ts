export default function applyRangeEdits(text: string, edits: { start: number; end: number; expected: string; replacement: string }[]): { text: string | null; error: string | null } {
  const sorted = [...edits].sort((a, b) => a.start - b.start || a.end - b.end);
  let cursor = 0, output = "";
  for (const edit of sorted) {
    if (!Number.isSafeInteger(edit.start) || !Number.isSafeInteger(edit.end) || edit.start < cursor || edit.end < edit.start || edit.end > text.length)
      return { text: null, error: "invalid_or_overlapping_range" };
    if (text.slice(edit.start, edit.end) !== edit.expected) return { text: null, error: "anchor_changed" };
    output += text.slice(cursor, edit.start) + edit.replacement; cursor = edit.end;
  }
  return { text: output + text.slice(cursor), error: null };
}
