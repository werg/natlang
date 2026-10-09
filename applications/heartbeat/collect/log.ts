// Source "log": the tail of a watched log, the age of its last progress line and its error-looking lines. The
// arithmetic and the pattern extraction are crisp so the model reads numbers and a short list.
import { untrusted } from '@natlang/node';
import type { Host } from '../host.js';
import type { ErrorCandidate, WatchEntry } from '../types.js';
import { failure, makeReading, type Reading } from './reading.js';

const TAIL_BYTES = 64 << 10;
const TAIL_LINES = 80;
const TAIL_CHARS = 7500;
const ERROR_PATTERN = /Error|Traceback|OOM|refused|Killed|CUDA/;
const LINE_CHARS = 400;
const STAMP = /(\d{4}-\d\d-\d\d)[T ](\d\d:\d\d:\d\d)(\.\d+)?(Z|[+-]\d\d:?\d\d)?/;

/** The moment a log line states, when it states one. A stamp without a zone is the machine's local time. */
export function lineTime(line: string): number | null {
  const found = STAMP.exec(line);
  if (!found) return null;
  const zone = found[4] ?? '';
  const parsed = Date.parse(`${found[1]}T${found[2]}${found[3] ?? ''}${zone.length === 5 && !zone.includes(':') && zone !== 'Z' ? `${zone.slice(0, 3)}:${zone.slice(3)}` : zone}`);
  return Number.isNaN(parsed) ? null : parsed;
}

export type LogFindings = { reading: Reading | null, minutes_since_progress: number | null, error_candidates: ErrorCandidate[] };

export async function collectLog(host: Host, entry: WatchEntry): Promise<LogFindings> {
  if (!entry.log) return { reading: null, minutes_since_progress: null, error_candidates: [] };
  const file = await host.readTail(entry.log, TAIL_BYTES);
  if (!file) return { reading: failure(host, 'log', entry.run_id, `the log ${entry.log} does not exist`), minutes_since_progress: null, error_candidates: [] };
  let lines = file.text.split('\n').filter(line => line.trim() !== '').slice(-TAIL_LINES).map(line => line.length > LINE_CHARS ? `${line.slice(0, LINE_CHARS)} …` : line);
  while (lines.join('\n').length > TAIL_CHARS && lines.length > 1) lines = lines.slice(1);
  const tail = lines.join('\n');
  const reading = makeReading(host, 'log', entry.run_id, true, tail || '(the log is empty)', TAIL_CHARS + 200);
  let marker: RegExp | null = null;
  try { marker = new RegExp(entry.progress_marker); } catch { marker = null; }
  const now = host.now().getTime();
  let minutes: number | null = null;
  if (marker) {
    const matched = [...lines].reverse().find(line => marker!.test(line));
    if (matched !== undefined) {
      // The line's own time when it has one; otherwise the log's modification time, which is when it last grew.
      const at = lineTime(matched) ?? file.mtimeMs;
      minutes = Math.max(0, Math.round((now - at) / 60_000));
    }
  }
  const candidates = lines.filter(line => ERROR_PATTERN.test(line)).slice(-5)
    .map(line => ({ reading_id: reading.id, line: untrusted(line, `log ${entry.run_id}`) }));
  return { reading, minutes_since_progress: minutes, error_candidates: candidates };
}
