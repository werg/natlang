/**
 * Log investigator. The investigation is natural language (investigate.nl and its folder: see DECOMPOSITION.md).
 * This file is what stays outside it: the exact log index (a service), the idempotent alert sink (a service), the
 * pure bounded `commit` of a decision into the incident state, and `step`, which folds one event:
 * snapshot, decide (model calls), carry out the effects as data, commit.
 */
import { untrusted, type FolderHandle } from '@natlang/node';
import investigate from './investigate.nl';
import type { Alert, ClosedIncident, Decision, Effect, Evidence, Incident, IncidentState, LogEvent, LogSettings, Observation,
  SearchQuery, SearchResult } from './types.js';

export type * from './types.js';
export type AlertSink = (alert: { key: string, service: string, code: string, claim: string, evidence_ids: string[] }) =>
  Promise<{ status?: string, detail?: string } | undefined>;

export const defaultSettings: LogSettings = { significance: 'natlang', window_ms: 60_000, threshold: 3, quiet_ms: 300_000, max_members: 50 };
export const emptyIncidentState = (): IncidentState =>
  ({ cursor: -1, observed: 0, alerts: [], unknowns: [], status: 'idle', incidents: [], closed: [] });

/** What the model learns of the index service. */
export const indexDeclaration = `/** Search the exact log index. Every field given must match; contains is a case-insensitive substring of the message; from and to are inclusive event times. total counts all matches, evidence holds the first limit by event time. */
export function search(query: { service?: string, code?: string, level?: string, contains?: string, from: number, to: number, limit: number }): Promise<{ total: number, evidence: { id: string, service: string, code: string, occurred_at: number, level: string, message: Untrusted<string> }[] }>;`;

/** The exact index of log lines: event time and arrival time kept apart, ingest by unique ID, windowed search. */
export class LogIndex {
  private readonly lines = new Map<string, LogEvent>();
  readonly events: Record<string, unknown>[] = [];
  constructor(private readonly windowMs: number) {}

  observe(item: LogEvent): Observation {
    if (item.kind !== 'log') throw new Error('only log events can be observed');
    if (!item.id || !item.service || !item.code || !Number.isFinite(item.occurred_at) || !Number.isFinite(item.arrived_at))
      throw new Error('invalid log event');
    const existing = this.lines.get(item.id);
    if (existing) {
      if (JSON.stringify(existing) !== JSON.stringify(item)) throw new Error('log ID reused with different content');
      return this.summary(item, 'duplicate');
    }
    this.lines.set(item.id, structuredClone(item));
    this.events.push({ operation: 'logs.ingest', id: item.id, service: item.service, code: item.code,
      occurred_at: item.occurred_at, arrived_at: item.arrived_at });
    return this.summary(item, 'new');
  }

  summary(item: LogEvent, status: Observation['status']): Observation {
    const count = this.search({ service: item.service, code: item.code, from: item.occurred_at - this.windowMs, to: item.occurred_at, limit: 1 }, false).total;
    return { id: item.id, status, service: item.service, code: item.code, occurred_at: item.occurred_at, count,
      late: item.arrived_at - item.occurred_at > this.windowMs };
  }

  has(ids: string[]): boolean { return ids.every(id => this.lines.has(id)); }

  search(query: SearchQuery, record = true): SearchResult {
    const needle = query.contains?.toLowerCase();
    const matches = [...this.lines.values()].filter(row =>
      (query.service === undefined || row.service === query.service) && (query.code === undefined || row.code === query.code) &&
      (query.level === undefined || row.level.toLowerCase() === query.level.toLowerCase()) &&
      (needle === undefined || row.message.toLowerCase().includes(needle)) &&
      row.occurred_at >= query.from && row.occurred_at <= query.to)
      .sort((a, b) => a.occurred_at - b.occurred_at || a.id.localeCompare(b.id));
    const evidence: Evidence[] = matches.slice(0, Math.max(0, query.limit)).map(row =>
      ({ id: row.id, service: row.service, code: row.code, occurred_at: row.occurred_at, level: row.level, message: untrusted(row.message, 'index.search') }));
    if (record) this.events.push({ operation: 'logs.query', query, ids: evidence.map(row => row.id) });
    return { total: matches.length, evidence };
  }
}

/** The alert sink: idempotent by key, and it accepts only alerts that cite distinct records the index holds. */
export class AlertSinkService {
  private readonly receipts = new Map<string, Alert>();
  readonly events: Record<string, unknown>[] = [];
  constructor(private readonly index: LogIndex, private readonly send: AlertSink | null) {}

  async deliver(effect: Effect): Promise<Alert> {
    const ids = [...new Set(effect.evidence_ids)].sort();
    if (!effect.key || !ids.length || !this.index.has(ids)) return { status: 'insufficient', key: effect.key, detail: 'the alert cites no records, or records the index does not hold' };
    const prior = this.receipts.get(effect.key);
    if (prior) return { ...prior, status: 'duplicate' };
    const receipt: Alert = { status: 'local', key: effect.key, detail: effect.claim };
    this.receipts.set(effect.key, receipt);
    if (this.send) {
      try {
        const delivered = await this.send({ key: effect.key, service: effect.service, code: effect.code, claim: effect.claim, evidence_ids: ids });
        receipt.status = delivered?.status === 'sent' ? 'sent' : 'unknown';
        receipt.detail = String(delivered?.detail ?? effect.claim);
      } catch (error) {
        receipt.status = 'unknown';
        receipt.detail = error instanceof Error ? error.message : String(error);
      }
    }
    this.events.push({ operation: 'logs.alert', key: effect.key, status: receipt.status, evidence_ids: ids });
    return { ...receipt };
  }
}

/** The index and the alert sink with the settings that go with them. */
export class LogWorkspace {
  readonly settings: LogSettings;
  readonly index: LogIndex;
  readonly sink: AlertSinkService;

  constructor({ windowMs, threshold, quietMs, significance, sendAlert = null }:
    { windowMs?: number, threshold?: number, quietMs?: number, significance?: LogSettings['significance'], sendAlert?: AlertSink | null } = {}) {
    this.settings = { ...defaultSettings, ...windowMs !== undefined ? { window_ms: windowMs } : {}, ...threshold !== undefined ? { threshold } : {},
      ...quietMs !== undefined ? { quiet_ms: quietMs } : {}, ...significance ? { significance } : {} };
    this.index = new LogIndex(this.settings.window_ms);
    this.sink = new AlertSinkService(this.index, sendAlert);
  }

  /** The runtime options that give natural-language stages the index service. */
  runOptions() {
    return { services: { index: { search: (query: SearchQuery) => this.index.search(query) } }, serviceDeclarations: { index: indexDeclaration } };
  }

  drainEvents(): Record<string, unknown>[] { return [...this.index.events.splice(0), ...this.sink.events.splice(0)]; }
}

const keepNewest = <T>(rows: T[], max: number) => rows.slice(Math.max(0, rows.length - max));

/**
 * The commit: apply a decision and the receipts of its effects to the state. Pure and bounded: it retires incidents whose
 * timer (closes_at) has passed at the event's time, and caps the open and closed lists.
 */
export function commit(state: IncidentState, item: LogEvent, decision: Decision, receipts: Alert[], settings: LogSettings = defaultSettings): IncidentState {
  const cursor = item.cursor;
  if (item.kind === 'gap') {
    const note = decision.gap;
    const incidents = state.incidents.map(row => note?.affected.includes(row.id) ? { ...row, closes_at: Math.max(row.closes_at, item.occurred_at + settings.quiet_ms) } : row);
    return { ...state, cursor, status: 'gap', incidents, unknowns: [...state.unknowns, note?.unknown ?? `Source gap at cursor ${cursor}: ${item.message}`] };
  }
  const observed = state.observed + 1;
  const unknowns = decision.escalation.uncertainty ? [...state.unknowns, `${item.id}: ${decision.escalation.uncertainty}`] : state.unknowns;
  const refused = receipts.some(row => row.status === 'insufficient');
  let incidents = state.incidents.filter(row => !decision.folded.includes(row.id));
  if (decision.incident) {
    const next: Incident = refused ? { ...decision.incident, alert_key: null } : decision.incident;
    incidents = incidents.some(row => row.id === next.id) ? incidents.map(row => row.id === next.id ? next : row) : [...incidents, next];
  }
  const live = incidents.filter(row => row.closes_at >= item.occurred_at);
  const newest = new Set(keepNewest([...live].sort((x, y) => x.last_seen - y.last_seen), 20));
  const open = live.filter(row => newest.has(row));
  const retired = incidents.filter(row => !newest.has(row));
  const closed: ClosedIncident[] = keepNewest([...state.closed, ...retired.map(row => ({ id: row.id, closed_at: item.occurred_at, count: row.count,
    severity: row.severity, alert_key: row.alert_key, summary: row.summary }))], 20);
  const kept = receipts.filter(row => row.status !== 'insufficient' && row.status !== 'duplicate');
  const status = kept.some(row => row.status === 'unknown') ? 'delivery-unknown' : kept.length ? 'alerted'
    : decision.escalation.action === 'ignore' ? 'observing' : 'investigating';
  return { cursor, observed, unknowns, incidents: open, closed, alerts: [...state.alerts, ...kept], status };
}

/** Fold one log or source-gap event into the incident state. */
export async function step(logs: LogWorkspace, state: IncidentState, item: LogEvent, files?: FolderHandle): Promise<IncidentState> {
  if (item.cursor <= state.cursor) return item.kind === 'gap' ? state : { ...state, status: 'duplicate' };
  const observation = item.kind === 'log' ? logs.index.observe(item) : undefined;
  if (observation?.status === 'duplicate') return { ...state, status: 'duplicate' };
  const decision = await investigate(item, observation, state.incidents, logs.settings, files);
  const receipts: Alert[] = [];
  for (const effect of decision.effects ?? []) receipts.push(await logs.sink.deliver(effect));
  return commit(state, item, decision, receipts, logs.settings);
}
