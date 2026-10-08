/**
 * The read tool's selection (pi tools/read.ts `readText`, `readHead`, and truncate.ts): the result for an opened file,
 * exactly as pi computes it. It equals decoding the whole file, splitting it on "\n", taking the requested lines and
 * bounding them to 2000 lines or 50KB, while reading only one line scan's worth plus the head. Byte counts, slicing,
 * truncation and the continuation messages are exact rules, so they stay crisp.
 *
 * Reads go through the env service's reader (`env.readBytes`, `env.scanLines`). A failed read throws its message.
 */
import { env } from 'natlang:services';
import { detectSupportedImageMimeTypeOf } from './image.js';

const DEFAULT_MAX_LINES = 2000;
const DEFAULT_MAX_BYTES = 50 * 1024;
const READ_CHUNK = 64 * 1024;

type Diagnostic = { severity: 'info' | 'warn' | 'error'; message: string; code?: string };
type Truncation = {
  truncated: boolean; truncatedBy: 'lines' | 'bytes' | null; totalLines: number; totalBytes: number; outputLines: number;
  outputBytes: number; lastLinePartial: boolean; firstLineExceedsLimit: boolean; maxLines: number; maxBytes: number;
};
type ReadResult = {
  content: { type: 'text'; text: string }[]; isError?: boolean; details?: { truncation: Truncation }; diagnostics: Diagnostic[];
};
type LineScan = { newlines: number; start: number; end: number; firstLineEnd: number; lastLineStart: number; selectedBytes: number; firstLineBytes: number };

function value<T>(result: { value?: T; error?: { message: string } }): T {
  if (result.error) throw new Error(result.error.message);
  return result.value as T;
}

/** UTF-8 byte length of a string. */
export function utf8ByteLength(content: string): number {
  return new TextEncoder().encode(content).length;
}

/** pi's human-readable size: 512B, 1.5KB, 2.0MB. */
export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

/** The last character boundary at or before index. */
function characterEnd(bytes: Uint8Array, index: number): number {
  let end = index;
  for (let step = 0; step < 4; step++) {
    if (end <= 0 || ((bytes[end] ?? 0) & 0xc0) !== 0x80) break;
    end--;
  }
  return end;
}

function splitLinesForCounting(content: string): string[] {
  if (content.length === 0) return [];
  const lines = content.split('\n');
  if (content.endsWith('\n')) lines.pop();
  return lines;
}

/** pi's truncateHeadOf: the head of a text known by a prefix and its totals, in whole lines within the limits. */
export function truncateHeadOf(prefix: string, totals: { lines: number; bytes: number }): Truncation & { content: string } {
  const maxLines = DEFAULT_MAX_LINES, maxBytes = DEFAULT_MAX_BYTES;
  const totalBytes = totals.bytes, totalLines = totals.lines;
  const lines = splitLinesForCounting(prefix);
  const base = { totalLines, totalBytes, lastLinePartial: false, maxLines, maxBytes };
  if (totalLines <= maxLines && totalBytes <= maxBytes)
    return { content: prefix, truncated: false, truncatedBy: null, outputLines: totalLines, outputBytes: totalBytes, firstLineExceedsLimit: false, ...base };
  if (utf8ByteLength(lines[0] ?? '') > maxBytes)
    return { content: '', truncated: true, truncatedBy: 'bytes', outputLines: 0, outputBytes: 0, firstLineExceedsLimit: true, ...base };
  const output: string[] = [];
  let bytes = 0;
  let truncatedBy: 'lines' | 'bytes' = 'lines';
  const count = Math.min(lines.length, maxLines);
  for (let i = 0; i < count; i++) {
    const lineBytes = utf8ByteLength(lines[i]!) + (i > 0 ? 1 : 0);
    if (bytes + lineBytes > maxBytes) { truncatedBy = 'bytes'; break; }
    output.push(lines[i]!);
    bytes += lineBytes;
  }
  if (truncatedBy !== 'bytes') truncatedBy = output.length < totalLines ? 'lines' : 'bytes';
  const content = output.join('\n');
  return { content, truncated: true, truncatedBy, outputLines: output.length, outputBytes: utf8ByteLength(content), firstLineExceedsLimit: false, ...base };
}

/** `Array.prototype.slice`'s conversion of an index: NaN is 0, other values truncate toward zero. */
function sliceIndex(index: number): number {
  return Number.isNaN(index) ? 0 : Math.trunc(index);
}

/**
 * The decoded start of bytes [start, end) of the file, decoded as part of the whole file: all of it, or enough for
 * truncateHeadOf (more than 50KB + 1 bytes, or 2000 newlines).
 */
async function readHead(reader: number, start: number, end: number, skipBom: boolean): Promise<string> {
  const decoder = new TextDecoder('utf-8', { ignoreBOM: true });
  let text = '';
  let newlines = 0;
  let position = skipBom && start === 0 ? 3 : start;
  const chunks = Math.ceil(Math.max(0, end - position) / READ_CHUNK) + 1;
  for (let chunk = 0; chunk < chunks; chunk++) {
    if (position >= end) break;
    const bytes = value(await env.readBytes(reader, position, Math.min(READ_CHUNK, end - position)));
    if (bytes.length === 0) break;
    position += bytes.length;
    const decoded = decoder.decode(bytes, { stream: true });
    text += decoded;
    newlines += decoded.split('\n').length - 1;
    if (newlines >= DEFAULT_MAX_LINES || utf8ByteLength(text) > DEFAULT_MAX_BYTES + 1) return text;
  }
  return text + decoder.decode();
}

/**
 * The read result for the opened file `reader` (opened with env.openReader; `info` from env.readerInfo), for the
 * call's path as the model gave it and its offset and limit. Throws `Offset <offset> is beyond end of file (<n> lines
 * total)` when offset is past the last line.
 */
export default async function selection(reader: number, info: { size: number; mtimeMs: number }, path: string,
  offset?: number, limit?: number): Promise<ReadResult> {
  const mimeType = await detectSupportedImageMimeTypeOf({ size: info.size,
    read: async (position, length) => value(await env.readBytes(reader, position, length)) });
  if (mimeType) return { content: [], isError: true,
    diagnostics: [{ severity: 'error', code: 'unsupported_image', message: `${path} is an image (${mimeType}); reading images is not supported` }] };

  const startLine = offset ? Math.max(0, offset - 1) : 0;
  const startLineDisplay = startLine + 1;
  const sliceStart = sliceIndex(startLine);
  const scanStart = Number.isSafeInteger(sliceStart) ? sliceStart : 0;
  const requestedEnd = limit === undefined || limit === null ? undefined : Math.max(scanStart + 1, sliceIndex(startLine + limit));
  const scanEnd = requestedEnd !== undefined && Number.isSafeInteger(requestedEnd) ? requestedEnd : undefined;
  const scanOf = async (endLine: number | undefined): Promise<LineScan> => value(await env.scanLines(reader, scanStart, endLine));
  let scan = await scanOf(scanEnd);
  const totalFileLines = scan.newlines + 1;
  if (startLine >= totalFileLines) throw new Error(`Offset ${offset} is beyond end of file (${totalFileLines} lines total)`);

  let userLimitedLines: number | undefined;
  let selectedLineCount = totalFileLines - sliceStart;
  if (limit !== undefined && limit !== null) {
    const endLine = Math.min(startLine + limit, totalFileLines);
    userLimitedLines = endLine - startLine;
    const relativeEnd = sliceIndex(endLine);
    const sliceEnd = relativeEnd < 0 ? Math.max(totalFileLines + relativeEnd, 0) : relativeEnd;
    selectedLineCount = Math.max(0, sliceEnd - sliceStart);
    if (selectedLineCount > 0 && relativeEnd < 0) scan = await scanOf(sliceEnd);
  }
  const empty = selectedLineCount === 0;
  const endsWithNewline = !empty && scan.lastLineStart === scan.end && scan.lastLineStart > scan.start;
  const totals = { lines: empty || scan.selectedBytes === 0 ? 0 : selectedLineCount - (endsWithNewline ? 1 : 0), bytes: empty ? 0 : scan.selectedBytes };
  const firstBytes = value(await env.readBytes(reader, 0, 3));
  const bom = firstBytes[0] === 0xef && firstBytes[1] === 0xbb && firstBytes[2] === 0xbf;
  const head = empty ? '' : await readHead(reader, scan.start, scan.end, bom);

  const { content: headText, ...truncation } = truncateHeadOf(head, totals);
  const diagnostics: Diagnostic[] = [];
  let outputText = headText;
  let details: { truncation: Truncation } | undefined;
  if (truncation.firstLineExceedsLimit) {
    const integral = Number.isInteger(startLine);
    const lineBytes = new TextEncoder().encode(integral ? (head.split('\n')[0] ?? '') : '');
    const lineSize = integral ? scan.firstLineBytes : 0;
    const end = characterEnd(lineBytes, DEFAULT_MAX_BYTES);
    outputText = new TextDecoder().decode(lineBytes.subarray(0, end));
    diagnostics.push({ severity: 'warn', code: 'truncated',
      message: `Line ${startLineDisplay} is ${formatSize(lineSize)}, exceeds the ${formatSize(DEFAULT_MAX_BYTES)} limit; showing its first ${formatSize(end)}. Use bash: sed -n '${startLineDisplay}p' ${path} | tail -c +${end + 1}` });
    details = { truncation: { ...truncation, outputBytes: end, outputLines: 1 } };
  } else if (truncation.truncated) {
    const endLineDisplay = startLineDisplay + truncation.outputLines - 1;
    const limitText = truncation.truncatedBy === 'lines' ? '' : ` (${formatSize(DEFAULT_MAX_BYTES)} limit)`;
    diagnostics.push({ severity: 'info', code: 'truncated',
      message: `Showing lines ${startLineDisplay}-${endLineDisplay} of ${totalFileLines}${limitText}. Use offset=${endLineDisplay + 1} to continue.` });
    details = { truncation };
  } else if (userLimitedLines !== undefined && startLine + userLimitedLines < totalFileLines) {
    const remaining = totalFileLines - (startLine + userLimitedLines);
    diagnostics.push({ severity: 'info', message: `${remaining} more lines in file. Use offset=${startLine + userLimitedLines + 1} to continue.` });
  }
  return { content: outputText === '' ? [] : [{ type: 'text', text: outputText }], ...(details === undefined ? {} : { details }), diagnostics };
}
