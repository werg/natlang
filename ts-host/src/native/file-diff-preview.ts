import { hexDigest } from './hash.js';
import type { ChangeSet } from './scoped-fs.js';

function describe(bytes: Uint8Array | undefined): Record<string, unknown> | undefined {
  if (bytes === undefined) return undefined;
  const result: Record<string, unknown> = { bytes: bytes.byteLength,
    sha256: hexDigest(bytes) };
  try {
    const text = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes);
    if (!text.includes('\0')) result.text = text;
  } catch { /* Binary changes remain inspectable by size and digest, without lossy decoding. */ }
  return result;
}

/** Text for the model; the filesystem's exact byte-valued ChangeSet stays unchanged. */
export function fileDiffPreview(diff: ChangeSet): string {
  const changes = diff.changes.map(change => {
    const before = describe(change.before), after = describe(change.after);
    const left = before?.text, right = after?.text;
    if (typeof left === 'string' && typeof right === 'string') {
      let prefix = 0, suffix = 0;
      while (prefix < left.length && prefix < right.length && left[prefix] === right[prefix]) prefix++;
      while (suffix < left.length - prefix && suffix < right.length - prefix &&
        left[left.length - 1 - suffix] === right[right.length - 1 - suffix]) suffix++;
      // Keep a little literal context, including whitespace. JSON escaping makes newlines visible.
      const start = Math.max(0, prefix - 80);
      const leftEnd = Math.min(left.length, left.length - suffix + 80);
      const rightEnd = Math.min(right.length, right.length - suffix + 80);
      delete before!.text; delete after!.text;
      return { path: change.path, kind: change.kind, before, after,
        unchanged_prefix_characters: prefix, unchanged_suffix_characters: suffix,
        context_start_character: start,
        before_text: left.slice(start, leftEnd), after_text: right.slice(start, rightEnd),
        omitted_before_tail_characters: left.length - leftEnd,
        omitted_after_tail_characters: right.length - rightEnd };
    }
    return { path: change.path, kind: change.kind, before, after };
  });
  return JSON.stringify({ version: 'natlang.file_diff_preview/1',
    description: 'Exact changed text with surrounding context. Escaped whitespace is significant; omitted regions are unchanged. Binary files show size and digest.',
    moves: diff.moves, changes }, null, 1);
}
