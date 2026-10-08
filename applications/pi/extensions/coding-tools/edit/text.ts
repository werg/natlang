/** Byte-order mark and line endings of a file the edit tool rewrites (pi edit-diff.ts). */

/** The UTF-8 byte-order mark ("﻿" or "") and the text without it. */
export function stripBom(content: string): { bom: string; text: string } {
  return content.startsWith('﻿') ? { bom: '﻿', text: content.slice(1) } : { bom: '', text: content };
}

/** "\r\n" when the first line ending is CRLF, else "\n". */
export function detectLineEnding(content: string): string {
  const crlf = content.indexOf('\r\n'), lf = content.indexOf('\n');
  if (lf === -1 || crlf === -1) return '\n';
  return crlf < lf ? '\r\n' : '\n';
}

/** Every CRLF and lone CR becomes LF. */
export function normalizeToLF(text: string): string {
  return text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
}

/** LF text with `ending` restored. */
export function restoreLineEndings(text: string, ending: string): string {
  return ending === '\r\n' ? text.replace(/\n/g, '\r\n') : text;
}
