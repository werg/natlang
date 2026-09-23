import { randomUUID } from 'node:crypto';
import { appendFileSync, existsSync, mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import type { AppEvent, Commit } from '../app/event-loop.js';

export type TerminalCheckpoint<S> = { schema: 'natlang-terminal-session/v1';
  revision: number; state: S; seen_event_ids: string[]; updated_at: string };

/** Single-writer durable state with an append-only reduction journal. */
export class TerminalSessionStore<S, E extends AppEvent = AppEvent> {
  constructor(readonly path: string) {
    if (!path) throw new Error('session path is required');
  }
  load(initialState: S): TerminalCheckpoint<S> {
    if (!existsSync(this.path)) return { schema: 'natlang-terminal-session/v1', revision: 0,
      state: structuredClone(initialState), seen_event_ids: [], updated_at: new Date(0).toISOString() };
    const value = JSON.parse(readFileSync(this.path, 'utf8')) as TerminalCheckpoint<S>;
    if (value.schema !== 'natlang-terminal-session/v1' || !Number.isSafeInteger(value.revision) ||
        value.revision < 0 || !Array.isArray(value.seen_event_ids)) throw new Error('invalid terminal session checkpoint');
    return structuredClone(value);
  }
  commit(commit: Commit<S, E>, priorSeen: Iterable<string>): void {
    mkdirSync(dirname(this.path), { recursive: true });
    const current = this.load(commit.state);
    if (current.revision !== commit.revision - 1)
      throw new Error(`session revision conflict: found ${current.revision}, expected ${commit.revision - 1}`);
    const seen = [...new Set([...priorSeen, commit.event.id])];
    const checkpoint: TerminalCheckpoint<S> = { schema: 'natlang-terminal-session/v1',
      revision: commit.revision, state: structuredClone(commit.state), seen_event_ids: seen,
      updated_at: new Date().toISOString() };
    const temporary = `${this.path}.${randomUUID()}.building`;
    writeFileSync(temporary, JSON.stringify(checkpoint, null, 2) + '\n', { flag: 'wx' });
    try {
      appendFileSync(`${this.path}.events.jsonl`, JSON.stringify({ schema: 'natlang-terminal-event/v1',
        revision: commit.revision, event: commit.event, phase: 'prepared', at: checkpoint.updated_at }) + '\n');
      renameSync(temporary, this.path);
    } catch (error) { try { unlinkSync(temporary); } catch {} throw error; }
  }
}
