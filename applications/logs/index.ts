/**
 * Log investigator. `LogWorkspace` retains an exact index of log lines (event time and arrival time
 * kept apart), answers windowed queries by service and code, and guards an optional idempotent alert
 * sink. `step` folds one log or source-gap event into incident state: natlang judges significance from
 * exact evidence, and an escalation still needs three distinct matching source records.
 */
import type { FolderHandle } from '@natlang/node';
import assess from './assess.nl';
import type { Evidence, Judgement, LogEvent, Observation } from './types.js';

export type * from './types.js';
export type Alert = { status: 'local' | 'sent' | 'unknown' | 'duplicate' | 'insufficient', key: string, detail: string };
export type IncidentState = { cursor: number, observed: number, alerts: Alert[], unknowns: string[],
  status: 'idle' | 'observing' | 'investigating' | 'alerted' | 'delivery-unknown' | 'duplicate' | 'gap' };
export type AlertSink = (alert: { key: string, service: string, code: string, claim: string, evidence_ids: string[] }) =>
  Promise<{ status?: string, detail?: string } | undefined>;

export const emptyIncidentState = (): IncidentState => ({ cursor: -1, observed: 0, alerts: [], unknowns: [], status: 'idle' });

export class LogWorkspace {
  private readonly windowMs: number;
  private readonly threshold: number;
  private readonly sendAlert: AlertSink | null;
  private readonly lines = new Map<string, LogEvent>();
  private readonly receipts = new Map<string, Alert>();
  private readonly activeIncidents = new Map<string, { key: string, last: number }>();
  private readonly events: Record<string, unknown>[] = [];

  constructor({ windowMs = 60_000, threshold = 3, sendAlert = null }: { windowMs?: number, threshold?: number, sendAlert?: AlertSink | null } = {}) {
    this.windowMs = windowMs; this.threshold = threshold; this.sendAlert = sendAlert;
  }

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
    return { id: item.id, status, service: item.service, code: item.code, occurred_at: item.occurred_at,
      count: this.window(item).length, late: item.arrived_at - item.occurred_at > this.windowMs };
  }

  query(observation: Observation): Evidence[] {
    const matches = this.window(observation).sort((a, b) => a.occurred_at - b.occurred_at || a.id.localeCompare(b.id));
    this.events.push({ operation: 'logs.query', source_id: observation.id, ids: matches.map(row => row.id) });
    return matches.map(row => ({ id: row.id, occurred_at: row.occurred_at, level: row.level, message: row.message }));
  }

  async alert(item: LogEvent, observation: Observation, evidence: Evidence[], judgement: Judgement): Promise<Alert> {
    const ids = new Set(evidence.map(row => row.id));
    const actual = this.query(observation);
    if (observation.status !== 'new' || ids.size < this.threshold || actual.length < this.threshold ||
        evidence.some(row => !actual.some(source => JSON.stringify(source) === JSON.stringify(row))))
      return { status: 'insufficient', key: '', detail: 'not enough distinct current source evidence' };
    const family = `${item.service}:${item.code}`;
    const active = this.activeIncidents.get(family);
    if (active && item.occurred_at < active.last - this.windowMs)
      return { status: 'insufficient', key: '', detail: 'late evidence predates the active incident window' };
    const key = active && Math.abs(item.occurred_at - active.last) <= this.windowMs ? active.key : `${family}:${actual[0]!.id}`;
    this.activeIncidents.set(family, { key, last: Math.max(item.occurred_at, active?.last ?? item.occurred_at) });
    const prior = this.receipts.get(key);
    if (prior) return { ...prior, status: 'duplicate' };
    const receipt: Alert = { status: 'local', key, detail: judgement.claim };
    this.receipts.set(key, receipt);
    if (this.sendAlert) {
      try {
        const delivered = await this.sendAlert({ key, service: item.service, code: item.code, claim: judgement.claim,
          evidence_ids: [...ids].sort() });
        receipt.status = delivered?.status === 'sent' ? 'sent' : 'unknown';
        receipt.detail = String(delivered?.detail ?? judgement.claim);
      } catch (error) {
        receipt.status = 'unknown';
        receipt.detail = error instanceof Error ? error.message : String(error);
      }
    }
    this.events.push({ operation: 'logs.alert', key, status: receipt.status, evidence_ids: [...ids].sort() });
    return { ...receipt };
  }

  drainEvents(): Record<string, unknown>[] { return this.events.splice(0); }

  private window(item: { service: string, code: string, occurred_at: number }): LogEvent[] {
    return [...this.lines.values()].filter(row => row.service === item.service && row.code === item.code &&
      row.occurred_at >= item.occurred_at - this.windowMs && row.occurred_at <= item.occurred_at);
  }
}

/** Fold one log or source-gap event into the incident state. */
export async function step(logs: LogWorkspace, state: IncidentState, item: LogEvent, files?: FolderHandle): Promise<IncidentState> {
  if (item.kind === 'gap') return item.cursor <= state.cursor ? state :
    { ...state, cursor: item.cursor, status: 'gap', unknowns: [...state.unknowns, `Source gap at cursor ${item.cursor}: ${item.message}`] };
  const observation = logs.observe(item);
  if (item.cursor <= state.cursor || observation.status === 'duplicate') return { ...state, status: 'duplicate' };
  const evidence = logs.query(observation);
  const judgement = await assess(item, observation, evidence, files);
  const cursor = item.cursor, observed = state.observed + 1;
  const unknowns = judgement.uncertainty ? [...state.unknowns, `${item.id}: ${judgement.uncertainty}`] : state.unknowns;
  if (judgement.action !== 'escalate')
    return { ...state, cursor, observed, unknowns, status: judgement.action === 'investigate' ? 'investigating' : 'observing' };
  const receipt = await logs.alert(item, observation, evidence, judgement);
  return { ...state, cursor, observed, unknowns,
    alerts: receipt.status === 'insufficient' || receipt.status === 'duplicate' ? state.alerts : [...state.alerts, receipt],
    status: receipt.status === 'insufficient' ? 'investigating' : receipt.status === 'unknown' ? 'delivery-unknown' : 'alerted' };
}
