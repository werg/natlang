// All the collectors, read-only, into one snapshot: the Evidence every decision of a cycle is a function of.
import { untrusted } from '@natlang/node';
import type { Settings } from '../actions.js';
import type { Host } from '../host.js';
import type { CoordMessage, ReadingView, Resources, RunEvidence, UnitState, WatchEntry } from '../types.js';
import { collectGates } from './gate.js';
import { collectInbox, collectPeerStatus } from './inbox.js';
import { collectLedger, type LedgerStatus } from './ledger.js';
import { collectLog } from './log.js';
import { makeReading, type Reading } from './reading.js';
import { collectDisk, collectGit, collectGpu, collectTeacher } from './system.js';
import { collectUnit, collectUnitList } from './unit.js';

export * from './reading.js';

export type Snapshot = {
  machine: string,
  at: string,
  readings: Reading[],
  runs: RunEvidence[],
  messages: CoordMessage[],
  resolved: string[],
  resources: Resources,
  ledger: LedgerStatus | null,
};

const view = (reading: Reading): ReadingView => ({ id: reading.id, source: reading.source, ok: reading.ok, text: untrusted(reading.text, `${reading.source} reading ${reading.id}`) });

/** Run the collectors for the machine's watch entries. Each collector failing is a finding in the readings, never an exception. */
export async function collect(host: Host, settings: Settings, repo: string, watch: WatchEntry[]): Promise<Snapshot> {
  const machine = host.machine();
  const readings: Reading[] = [];
  const keep = <T extends Reading | Reading[] | null>(reading: T): T => { if (reading) readings.push(...(Array.isArray(reading) ? reading : [reading])); return reading; };
  const [ledger, list, gpu, teacher, inbox] = await Promise.all([
    collectLedger(host, repo), collectUnitList(host), collectGpu(host),
    collectTeacher(host, settings.teacher_metrics_url, settings.teacher_window_file ? resolveIn(repo, settings.teacher_window_file) : null),
    collectInbox(host, repo, machine, settings.inbox_hours)]);
  keep(ledger.reading); keep(list.reading); keep(gpu.reading); keep(teacher.reading); keep(inbox.reading);
  keep(await collectGit(host, repo)); keep(await collectDisk(host, settings.data_roots)); keep(await collectPeerStatus(host, repo, machine));

  const runs: RunEvidence[] = [];
  for (const entry of watch) {
    const unit = await collectUnit(host, entry.unit); keep(unit.reading);
    const absolute = { ...entry, log: entry.log ? resolveIn(repo, entry.log) : null, gate_paths: entry.gate_paths.map(path => resolveIn(repo, path)) };
    const log = await collectLog(host, absolute); keep(log.reading);
    const gates = await collectGates(host, absolute); keep(gates.readings);
    const claim = ledger.status?.claims[entry.unit] ?? null;
    const ledgerView = ledger.status ? makeReading(host, 'ledger', entry.unit, true, JSON.stringify({ unit: entry.unit, claim, headroom_gb: ledger.status.headroom_gb, available_gb: ledger.status.available_gb }, null, 2)) : ledger.reading;
    if (ledger.status) keep(ledgerView);
    const state: UnitState = unit.state ?? { active_state: 'unknown', sub_state: 'unknown', result: 'unknown', exit_status: null, started_at: '' };
    runs.push({ entry, unit: state, minutes_since_progress: log.minutes_since_progress, error_candidates: log.error_candidates, gates: gates.gates,
      ledger_claim: claim ? { budget_gb: claim.budget_gb, used_gb: claim.used_gb, peak_gb: claim.peak_gb, hold_budget: claim.hold_budget === true } : null,
      headroom_gb: ledger.status?.headroom_gb ?? null,
      readings: [unit.reading, ...(log.reading ? [log.reading] : []), ...gates.readings, ledgerView].map(view) });
  }

  const watchedUnits = new Set(watch.map(entry => entry.unit));
  const ignore = settings.ignore_units.map(pattern => new RegExp(pattern));
  const candidates = new Set([...list.units.filter(unit => unit.active_state === 'active' || unit.active_state === 'failed' || unit.active_state === 'activating').map(unit => unit.unit),
    ...Object.keys(ledger.status?.claims ?? {})]);
  const unwatched = [...candidates].filter(name => !watchedUnits.has(name) && !ignore.some(pattern => pattern.test(name))).sort();
  return { machine, at: host.now().toISOString(), readings, runs, messages: inbox.messages, resolved: inbox.resolved, ledger: ledger.status,
    resources: { headroom_gb: ledger.status?.headroom_gb ?? null, gpu_utilization: gpu.utilization, teacher_load: teacher.load, unwatched } };
}

export const resolveIn = (repo: string, path: string) => path.startsWith('/') ? path : `${repo}/${path}`;
