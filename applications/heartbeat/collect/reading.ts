// A Reading is one timestamped, hashed observation of the outside world. A failing collector still returns one
// (ok: false, with the error text): a missing reading is a finding.
import { createHash } from 'node:crypto';
import type { Host } from '../host.js';
import type { Source } from '../types.js';

export type Reading = {
  /** "<source>:<subject>:<hhmm>" (UTC). */
  id: string,
  source: Source,
  subject: string,
  /** ISO time of collection. */
  at: string,
  ok: boolean,
  /** Outside text: the model sees it as data. */
  text: string,
  truncated: boolean,
  /** SHA-256 of the full text before any cut. */
  sha256: string,
};

export const READING_CHARS = 8000;

export function makeReading(host: Host, source: Source, subject: string, ok: boolean, text: string, limit = READING_CHARS): Reading {
  const at = host.now();
  const hhmm = at.toISOString().slice(11, 16).replace(':', '');
  const truncated = text.length > limit;
  return { id: `${source}:${subject}:${hhmm}`, source, subject, at: at.toISOString(), ok,
    text: truncated ? text.slice(0, limit) : text, truncated, sha256: createHash('sha256').update(text).digest('hex') };
}

/** A reading of a command's output; a failed command keeps its error text as the reading. */
export function failure(host: Host, source: Source, subject: string, error: string): Reading {
  return makeReading(host, source, subject, false, error.trim() || 'no output', 2000);
}

export const sha256 = (text: string) => createHash('sha256').update(text).digest('hex');
