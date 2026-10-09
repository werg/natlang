// Source "inbox": the coordination messages for this machine, read through scripts/coord.py without moving any cursor
// (`inbox --json`), and the peer machine's status page.
import { untrusted } from '@natlang/node';
import type { Host } from '../host.js';
import type { CoordMessage } from '../types.js';
import { failure, makeReading, type Reading } from './reading.js';

type RawMessage = Partial<CoordMessage> & { id: string };
const normalize = (raw: RawMessage): CoordMessage => ({ id: raw.id, from: raw.from ?? '', to: raw.to ?? [], kind: raw.kind ?? 'note', subject: raw.subject ?? '',
  body: untrusted(raw.body ?? '', `coordination message ${raw.id}`), reply_to: raw.reply_to ?? null, urgent: raw.urgent === true, sent_at: raw.sent_at ?? '' });

export type InboxReading = { reading: Reading, messages: CoordMessage[], resolved: string[] };

export async function collectInbox(host: Host, repo: string, machine: string, hours: number): Promise<InboxReading> {
  const result = await host.exec(['python3', `${repo}/scripts/coord.py`, '--repo', repo, '--as', `${machine}-heartbeat`, 'inbox', '--json', '--hours', String(hours)],
    { timeoutMs: 60_000, cwd: repo, env: { COORD_MACHINE: machine } });
  if (!result.ok) return { reading: failure(host, 'inbox', machine, result.stderr || result.stdout), messages: [], resolved: [] };
  try {
    const shown = JSON.parse(result.stdout) as { unread: RawMessage[], open_requests: RawMessage[], recent: RawMessage[], resolved: string[] };
    const seen = new Map<string, CoordMessage>();
    for (const raw of [...shown.open_requests, ...shown.unread, ...shown.recent]) if (!seen.has(raw.id)) seen.set(raw.id, normalize(raw));
    // Replies and closes that settle a request are history; the request itself, resolved or not, stays visible with its state.
    const messages = [...seen.values()].sort((a, b) => a.sent_at.localeCompare(b.sent_at));
    const resolved = new Set(shown.resolved);
    const text = messages.map(message => `${message.id} ${message.kind}${message.urgent ? ' URGENT' : ''} from ${message.from}${resolved.has(message.id) ? ' (resolved)' : ''}: ${message.subject}`).join('\n');
    return { reading: makeReading(host, 'inbox', machine, true, text || '(no messages)'), messages, resolved: shown.resolved };
  } catch (error) {
    return { reading: failure(host, 'inbox', machine, `unreadable coord output: ${(error as Error).message}`), messages: [], resolved: [] };
  }
}

export async function collectPeerStatus(host: Host, repo: string, machine: string): Promise<Reading> {
  const result = await host.exec(['python3', `${repo}/scripts/coord.py`, '--repo', repo, '--as', `${machine}-heartbeat`, 'status'],
    { timeoutMs: 60_000, cwd: repo, env: { COORD_MACHINE: machine } });
  return result.ok ? makeReading(host, 'peer-status', machine, true, result.stdout.trim() || '(no status pages)') : failure(host, 'peer-status', machine, result.stderr || result.stdout);
}
