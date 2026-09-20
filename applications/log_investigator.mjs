/** Retained exact log index and optional idempotent alert sink. */
export class LogWorkspace {
  constructor({ windowMs = 60_000, threshold = 3, sendAlert = null } = {}) {
    this.windowMs = windowMs;
    this.threshold = threshold;
    this.sendAlert = sendAlert;
    this.lines = new Map();
    this.receipts = new Map();
    this.activeIncidents = new Map();
    this.events = [];
  }

  observe(item) {
    if (item.kind !== 'log') throw new Error('only log events can be observed');
    if (!item.id || !item.service || !item.code || !Number.isFinite(item.occurred_at) ||
        !Number.isFinite(item.arrived_at)) throw new Error('invalid log event');
    const existing = this.lines.get(item.id);
    if (existing) {
      if (JSON.stringify(existing) !== JSON.stringify(item)) throw new Error('log ID reused with different content');
      return this.summary(item, 'duplicate');
    }
    this.lines.set(item.id, structuredClone(item));
    this.events.push({ operation: 'logs.ingest', id: item.id, service: item.service,
      code: item.code, occurred_at: item.occurred_at, arrived_at: item.arrived_at });
    return this.summary(item, 'new');
  }

  summary(item, status) {
    const matches = [...this.lines.values()].filter(row => row.service === item.service &&
      row.code === item.code && row.occurred_at >= item.occurred_at - this.windowMs &&
      row.occurred_at <= item.occurred_at);
    return { id: item.id, status, service: item.service, code: item.code,
      occurred_at: item.occurred_at, count: matches.length,
      late: item.arrived_at - item.occurred_at > this.windowMs };
  }

  query(observation) {
    const matches = [...this.lines.values()].filter(row => row.service === observation.service &&
      row.code === observation.code &&
      row.occurred_at >= observation.occurred_at - this.windowMs &&
      row.occurred_at <= observation.occurred_at)
      .sort((a, b) => a.occurred_at - b.occurred_at || a.id.localeCompare(b.id));
    this.events.push({ operation: 'logs.query', source_id: observation.id,
      ids: matches.map(row => row.id) });
    return matches.map(row => ({ id: row.id, occurred_at: row.occurred_at,
      level: row.level, message: row.message }));
  }

  async alert(item, observation, evidence, judgement) {
    const ids = new Set(evidence.map(row => row.id));
    const actual = this.query(observation);
    if (observation.status !== 'new' || ids.size < this.threshold ||
        actual.length < this.threshold || evidence.some(row =>
          !actual.some(source => JSON.stringify(source) === JSON.stringify(row))))
      return { status: 'insufficient', key: '', detail: 'not enough distinct current source evidence' };
    const family = `${item.service}:${item.code}`;
    const active = this.activeIncidents.get(family);
    if (active && item.occurred_at < active.last - this.windowMs)
      return { status: 'insufficient', key: '', detail: 'late evidence predates the active incident window' };
    const key = active && Math.abs(item.occurred_at - active.last) <= this.windowMs ?
      active.key : `${family}:${actual[0].id}`;
    this.activeIncidents.set(family, { key, last: Math.max(item.occurred_at, active?.last ?? item.occurred_at) });
    const prior = this.receipts.get(key);
    if (prior) return { ...prior, status: 'duplicate' };
    const receipt = { status: 'local', key, detail: judgement.claim };
    this.receipts.set(key, receipt);
    if (this.sendAlert) {
      try {
        const delivered = await this.sendAlert({ key, service: item.service,
          code: item.code, claim: judgement.claim, evidence_ids: [...ids].sort() });
        receipt.status = delivered?.status === 'sent' ? 'sent' : 'unknown';
        receipt.detail = String(delivered?.detail ?? judgement.claim);
      } catch (error) {
        receipt.status = 'unknown';
        receipt.detail = error instanceof Error ? error.message : String(error);
      }
    }
    this.events.push({ operation: 'logs.alert', key, status: receipt.status,
      evidence_ids: [...ids].sort() });
    return { ...receipt };
  }

  drainEvents() { return this.events.splice(0); }
}
