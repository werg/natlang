// The comparison with what the agents did (plans/HEARTBEAT_PROGRAM.md section 7). Crisp and exact: agent actions are
// derived from observable effects and mapped to allowlist ids; a proposal matches an agent action with the same action id
// and target in the same cycle window or the next one.
import type { Host } from './host.js';
import type { Cause, Health, Proposal, WatchEntry } from './types.js';

export type AgentAction = { at: string, action: string, target: string, source: 'ledger' | 'git' | 'coord' | 'status' | 'explicit' };
export type CycleSummary = {
  cycle: string,
  at: string,
  runs: { run_id: string, unit: string, health: Health, cause: Cause }[],
  proposals: Proposal[],
};
export type Counted = { cycle: string, action: string, target: string, cause: Cause | null };
export type Comparison = {
  cycles: number,
  /** Proposals the agent also took. */
  agreement: Counted[],
  /** Proposals the agent did not take. */
  extra: Counted[],
  /** Agent actions of proposable kinds the program did not propose. */
  missed: Counted[],
  /** Agent actions that are not commands the program can propose (code edits). */
  other: Counted[],
  /** Runs the program called progressing or finished-ok while the agent acted on them. */
  disagreements: { cycle: string, run_id: string, health: Health, action: string }[],
  /** Per cause: agreement, extra and missed counts. */
  by_cause: Record<string, { agreement: number, extra: number, missed: number }>,
  /** After a proposed or taken action on a run, did its health improve within three cycles. */
  outcomes: { cycle: string, run_id: string, action: string, taken: boolean, outcome: 'improved' | 'not-improved' | 'pending' }[],
};

const UNHEALTHY: Health[] = ['stalled', 'finished-failed', 'blocked', 'gate-failed', 'unknown'];
const HEALTHY: Health[] = ['progressing', 'finished-ok'];
const NOT_PROPOSED = new Set(['edit-code', 'record-heartbeat']);

/** The cycle window an instant falls in: the last cycle that started at or before it; -1 before the first. */
function windowOf(cycles: CycleSummary[], at: string): number {
  const time = Date.parse(at);
  let found = -1;
  cycles.forEach((cycle, index) => { if (Date.parse(cycle.at) <= time) found = index; });
  return found;
}

/** Target names an agent action by unit or run id; both resolve to the run id when a watch entry knows the unit. */
function runOf(target: string, watch: WatchEntry[]): string {
  return watch.find(entry => entry.unit === target || entry.run_id === target)?.run_id ?? target;
}

export function compare(cycles: CycleSummary[], actions: AgentAction[], watch: WatchEntry[]): Comparison {
  const sorted = [...cycles].sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
  const taken = actions.map(action => ({ ...action, target: runOf(action.target, watch), window: windowOf(sorted, action.at) })).filter(action => action.window >= 0);
  const used = new Set<number>();
  const result: Comparison = { cycles: sorted.length, agreement: [], extra: [], missed: [], other: [], disagreements: [], by_cause: {}, outcomes: [] };
  const causeOf = (cycle: CycleSummary, target: string): Cause | null => cycle.runs.find(run => run.run_id === runOf(target, watch))?.cause ?? null;
  const bump = (cause: Cause | null, key: 'agreement' | 'extra' | 'missed') => {
    const name = cause ?? 'none';
    (result.by_cause[name] ??= { agreement: 0, extra: 0, missed: 0 })[key]++;
  };
  sorted.forEach((cycle, index) => {
    for (const proposal of cycle.proposals) {
      const target = runOf(proposal.target, watch);
      const matchAt = taken.findIndex((action, at) => !used.has(at) && action.action === proposal.action && action.target === target && (action.window === index || action.window === index + 1));
      const counted: Counted = { cycle: cycle.cycle, action: proposal.action, target, cause: causeOf(cycle, target) };
      if (matchAt >= 0) { used.add(matchAt); result.agreement.push(counted); bump(counted.cause, 'agreement'); }
      else { result.extra.push(counted); bump(counted.cause, 'extra'); }
    }
  });
  taken.forEach((action, at) => {
    if (used.has(at)) return;
    const cycle = sorted[action.window]!;
    const counted: Counted = { cycle: cycle.cycle, action: action.action, target: action.target, cause: causeOf(cycle, action.target) };
    if (NOT_PROPOSED.has(action.action)) { result.other.push(counted); return; }
    result.missed.push(counted); bump(counted.cause, 'missed');
  });
  for (const action of taken) {
    const cycle = sorted[action.window]!;
    const run = cycle.runs.find(item => item.run_id === action.target);
    if (run && HEALTHY.includes(run.health)) result.disagreements.push({ cycle: cycle.cycle, run_id: run.run_id, health: run.health, action: action.action });
  }
  // Outcome labels: the run's health in the next three cycles, after an action that was proposed or taken in this one.
  const seen = new Set<string>();
  sorted.forEach((cycle, index) => {
    const acted = [...cycle.proposals.map(proposal => ({ action: proposal.action, target: runOf(proposal.target, watch), taken: false })),
      ...taken.filter(action => action.window === index).map(action => ({ action: action.action, target: action.target, taken: true }))];
    for (const item of acted) {
      const run = cycle.runs.find(candidate => candidate.run_id === item.target);
      if (!run || !UNHEALTHY.includes(run.health)) continue;
      const key = `${cycle.cycle}|${item.action}|${item.target}`;
      if (seen.has(key)) { const existing = result.outcomes.find(entry => entry.cycle === cycle.cycle && entry.action === item.action && entry.run_id === item.target); if (existing && item.taken) existing.taken = true; continue; }
      seen.add(key);
      const later = sorted.slice(index + 1, index + 4).map(next => next.runs.find(candidate => candidate.run_id === run.run_id)?.health).filter((health): health is Health => !!health);
      const outcome = later.some(health => HEALTHY.includes(health)) ? 'improved' : later.length >= 3 ? 'not-improved' : 'pending';
      result.outcomes.push({ cycle: cycle.cycle, run_id: run.run_id, action: item.action, taken: item.taken, outcome });
    }
  });
  return result;
}

export type DeriveOptions = { repo: string, machine: string, since: string, until: string, watch: WatchEntry[], ignoreUnits: RegExp[], ledgerPath: string, recordDir: string };

/** Agent actions in a time window, from the ledger's events, commits, coordination messages, the status page and explicit records. */
export async function deriveAgentActions(host: Host, options: DeriveOptions): Promise<AgentAction[]> {
  const from = Date.parse(options.since), to = Date.parse(options.until);
  const within = (iso: string) => { const time = Date.parse(iso); return time >= from && time < to; };
  const found: AgentAction[] = [];
  const ledger = await host.readText(options.ledgerPath);
  if (ledger) {
    try {
      const events = (JSON.parse(ledger.text) as { events?: { time: number, event: string, unit: string }[] }).events ?? [];
      const mapped: Record<string, string> = { admitted: 'relaunch-run', stopped: 'stop-unit', adopted: 'adopt-unit' };
      for (const event of events) {
        const at = new Date(event.time * 1000).toISOString();
        if (mapped[event.event] && within(at) && !options.ignoreUnits.some(pattern => pattern.test(event.unit)))
          found.push({ at, action: mapped[event.event]!, target: event.unit, source: 'ledger' });
      }
    } catch { /* an unreadable ledger is simply no evidence */ }
  }
  const commits = await host.exec(['git', '-C', options.repo, 'log', `--since=${options.since}`, `--until=${options.until}`, '--format=%cI%x09%H'], { timeoutMs: 20_000 });
  if (commits.ok) for (const line of commits.stdout.split('\n').filter(Boolean)) {
    const [at, hash] = line.split('\t');
    if (at && hash) found.push({ at, action: 'edit-code', target: hash.slice(0, 12), source: 'git' });
  }
  const messages = await host.exec(['python3', `${options.repo}/scripts/coord.py`, '--repo', options.repo, '--as', `${options.machine}-heartbeat`, 'log', '-n', '200', '--json'],
    { timeoutMs: 30_000, env: { COORD_MACHINE: options.machine } });
  if (messages.ok) {
    try {
      for (const message of JSON.parse(messages.stdout) as { from: string, kind: string, sent_at: string, reply_to: string | null }[]) {
        if (!message.from.startsWith(`${options.machine}-`) || message.from === `${options.machine}-heartbeat` || !within(message.sent_at)) continue;
        if (message.kind === 'reply' || message.kind === 'close') found.push({ at: message.sent_at, action: 'reply-note', target: message.reply_to ?? '', source: 'coord' });
        else found.push({ at: message.sent_at, action: 'send-note', target: options.machine, source: 'coord' });
      }
    } catch { /* ditto */ }
  }
  const page = await host.readText(`${options.repo}/.coordination/status/${options.machine}.json`);
  if (page) try {
    const shown = JSON.parse(page.text) as { updated_at: string, by: string };
    if (within(shown.updated_at) && shown.by !== `${options.machine}-heartbeat`) found.push({ at: shown.updated_at, action: 'set-status-page', target: options.machine, source: 'status' });
  } catch { /* ditto */ }
  const explicit = await host.readText(`${options.repo}/${options.recordDir}/actions.jsonl`);
  for (const line of explicit?.text.split('\n').filter(Boolean) ?? []) try {
    const item = JSON.parse(line) as { at: string, action: string, target: string };
    if (within(item.at)) found.push({ ...item, source: 'explicit' });
  } catch { /* a damaged line is skipped */ }
  return found.sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
}

/** A session records an action that left no visible effect: `heartbeat record --action ID --target T`. */
export async function recordAction(host: Host, repo: string, recordDir: string, action: string, target: string): Promise<AgentAction> {
  const item: AgentAction = { at: host.now().toISOString(), action, target, source: 'explicit' };
  await host.appendText(`${repo}/${recordDir}/actions.jsonl`, `${JSON.stringify({ at: item.at, action, target })}\n`);
  return item;
}
