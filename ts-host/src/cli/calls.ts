/**
 * `natlang traces ...` and `natlang compilations ...`: the standard view of the machine's call record store and of the
 * compilations stored beside it (plans/TRACE_SPECIALIZATION.md §3.5, §7.4). `natlang specialize ...` runs the
 * specializer application on the store.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { CallStore } from '../calls/store.js';
import { expandedRecord } from '../calls/expanded.js';
import { machineStoreRoot, openMachineStore } from '../machine-store.js';
import { exportSpecializationCorpus } from '../calls/corpus.js';
import { renderEvidence, renderHistory, study } from '../calls/specializer.js';
import { enclosingProgram } from '../calls/feedback.js';
import { renderTrace } from '../calls/trace-view.js';
import { TRACES_HELP } from './traces-help.js';
import { emptyEvidence, foldEvent, newTierEvidence, setTierState, type Evidence, type EvidenceState, type TierEvidence } from '../calls/evidence.js';
import type { CallRecord, CallStoreSettings, CaseRole, CaseTier } from '../calls/types.js';

/** A mistake in how a command was called: printed as `natlang: MESSAGE` and exit code 2. */
export class UsageError extends Error {}

type Args = { words: string[]; options: Map<string, string | true> };
/** The options an action takes: `values` take an argument, `flags` do not. Every action also takes --store DIR and --json. */
type Spec = { values?: readonly string[]; flags?: readonly string[] };

/** Levenshtein distance, for "did you mean". */
function distance(a: string, b: string): number {
  let previous = Array.from({ length: b.length + 1 }, (_, index) => index);
  for (let i = 1; i <= a.length; i++) {
    const row = [i];
    for (let j = 1; j <= b.length; j++) row[j] = Math.min(previous[j]! + 1, row[j - 1]! + 1, previous[j - 1]! + (a[i - 1] === b[j - 1] ? 0 : 1));
    previous = row;
  }
  return previous[b.length]!;
}
/** The candidate closest to `word`, when it is close enough to be a likely typo. */
export function closest(word: string, candidates: readonly string[]): string | undefined {
  const ranked = candidates.map(candidate => ({ candidate, d: distance(word, candidate) })).sort((x, y) => x.d - y.d)[0];
  return ranked && ranked.d <= Math.max(2, Math.floor(word.length / 3)) ? ranked.candidate : undefined;
}

/** The first word of `argv` that is not an option or an option's value: the action. */
function actionOf(argv: string[], valued: ReadonlySet<string>): string | undefined {
  for (let index = 0; index < argv.length; index++) {
    const item = argv[index]!;
    if (!item.startsWith('-')) return item;
    if (valued.has(item)) index++;
  }
  return undefined;
}

function parse(argv: string[], command: string, spec: Spec): Args {
  const values = new Set(['--store', ...spec.values ?? []]), flags = new Set(['--json', ...spec.flags ?? []]);
  const words: string[] = [], options = new Map<string, string | true>();
  for (let index = 0; index < argv.length; index++) {
    const item = argv[index]!;
    if (item === '--') { words.push(...argv.slice(index + 1)); break; }
    if (!item.startsWith('-') || item === '-' || /^-\d/.test(item)) { words.push(item); continue; }
    const [name, inline] = item.includes('=') ? [item.slice(0, item.indexOf('=')), item.slice(item.indexOf('=') + 1)] : [item, undefined];
    if (flags.has(name) && inline === undefined) { options.set(name, true); continue; }
    if (values.has(name)) {
      const value = inline ?? argv[++index];
      if (value === undefined || (inline === undefined && value.startsWith('--'))) throw new UsageError(`${name} needs a value (natlang ${command} ${name} VALUE)`);
      options.set(name, value); continue;
    }
    const known = [...values, ...flags];
    const guess = closest(name, known);
    throw new UsageError(`unknown option ${name} for natlang ${command}${guess ? ` (did you mean ${guess}?)` : ''}; it takes ${known.sort().join(', ')}`);
  }
  return { words, options };
}
const text = (args: Args, name: string) => { const value = args.options.get(name); return typeof value === 'string' ? value : undefined; };
function number(args: Args, name: string, fallback: number): number {
  const value = text(args, name);
  if (value === undefined) return fallback;
  if (!/^\d+$/.test(value) || Number(value) < 1) throw new UsageError(`${name} takes a positive whole number, not ${JSON.stringify(value)}`);
  return Number(value);
}
/** `--since` as an ISO time: an ISO date or time, or an age (`30m`, `2h`, `7d`) back from now. */
function since(args: Args): string | undefined {
  const value = text(args, '--since');
  if (value === undefined) return undefined;
  const age = /^(\d+)([mhd])$/.exec(value);
  if (age) return new Date(Date.now() - Number(age[1]) * { m: 60_000, h: 3_600_000, d: 86_400_000 }[age[2] as 'm' | 'h' | 'd']).toISOString();
  if (/^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$/.test(value) && !Number.isNaN(Date.parse(value)))
    return new Date(value.length === 10 ? `${value}T00:00:00Z` : value).toISOString();
  throw new UsageError(`--since takes an ISO date or time (2026-11-13, 2026-11-13T09:00Z) or an age (30m, 2h, 7d), not ${JSON.stringify(value)}`);
}
function choice<T extends string>(args: Args, name: string, allowed: readonly T[]): T | undefined {
  const value = text(args, name);
  if (value !== undefined && !allowed.includes(value as T)) throw new UsageError(`${name} is one of ${allowed.join(', ')}, not ${JSON.stringify(value)}`);
  return value as T | undefined;
}

function openStore(args: Args): CallStore {
  const root = text(args, '--store') ?? machineStoreRoot();
  if (!root) throw new Error('the call store is off on this machine (NATLANG_CALL_STORE=off); pass --store DIR to read one');
  return openMachineStore(root);
}

/** The call a typed ID means: the exact ID, a unique prefix, or a unique short form (without `task-N-`). Among the
 * calls of one task, the task's root call wins. */
export function resolveCall(store: CallStore, word: string | undefined, usage: string): string {
  if (!word) throw new UsageError(`usage: ${usage}`);
  const matches = store.matchCalls(word, 1000);
  if (matches.length === 1) return matches[0]!.call_id;
  if (!matches.length) throw new UsageError(`no call matches ${word}; natlang traces list shows recent calls (--all for every program's)`);
  const tasks = new Set(matches.map(row => row.call_id.split('/')[0]));
  const roots = matches.filter(row => !row.parent_call_id);
  if (tasks.size === 1 && roots.length === 1) return roots[0]!.call_id;
  throw new UsageError(`${word} matches ${matches.length === 1000 ? '1000 or more' : matches.length} calls; give more of the ID:\n` +
    matches.slice(0, 5).map(row => `  ${row.call_id}  ${row.definition_name} ${row.outcome} ${row.started_at}`).join('\n'));
}
const print = (value: unknown, json: boolean) => process.stdout.write(json ? `${JSON.stringify(value, null, 2)}\n` :
  `${String(value)}${String(value).endsWith('\n') ? '' : '\n'}`);
const table = (rows: Record<string, unknown>[], columns: string[]): string => {
  if (!rows.length) return '(none)';
  const cells = [columns, ...rows.map(row => columns.map(column => String(row[column] ?? '')))];
  const widths = columns.map((_, index) => Math.min(60, Math.max(...cells.map(row => row[index]!.length))));
  return cells.map(row => row.map((cell, index) => cell.slice(0, 60).padEnd(widths[index]!)).join('  ').trimEnd()).join('\n');
};

/** The stored compilation's cases of one function, as the tier-3 row of the ladder: the same evidence, summed over its cases. */
export type CaseTierRow = { function: string; served: number; ms_per_call: number; state: string; evidence: Evidence;
  saved_tokens: number; spent_tokens: number };

/** Sum the evidence of a function's current compilation cases and join the store's savings (one accounting, `CallStore.savings`). */
export function caseTiersOf(store: CallStore, definition?: string): CaseTierRow[] {
  return store.savings({ definition }).flatMap(saving => {
    const compilation = store.currentCompilation(saving.definition_key);
    const evidence = emptyEvidence();
    let active = 0, shadow = 0;
    for (const item of compilation?.cases ?? []) {
      const summary = store.caseSummary(item.hash);
      if (!summary) continue;
      for (const key of Object.keys(evidence) as (keyof Evidence)[]) evidence[key] += summary.evidence[key];
      if (item.tier === 'active') active++; else if (item.tier === 'shadow') shadow++;
    }
    if (!compilation && !saving.served) return [];
    return [{ function: saving.definition_name, served: saving.served, ms_per_call: saving.crisp_ms_per_call,
      state: active ? 'active' : shadow ? 'shadow' : compilation ? 'demoted' : '-', evidence, saved_tokens: saving.saved_tokens, spent_tokens: saving.spent_tokens }];
  });
}

/**
 * One view of the ladder per function and tier. Event rows come from the tier events kept with calls (calls/tiers.ts):
 * calls served, deopts and their rate, the cost of a served call, tokens wasted by attempts that were then deoptimized,
 * the state the events imply and the evidence folded by the same module the cases use (calls/evidence.ts). The tier-3
 * row for stored cases (`cases`) takes its calls, evidence, state and savings from the compilations and `CallStore.savings`
 * instead, so served and saved are counted once, there.
 */
export function tiersOf(events: { value: unknown; definition_name: string | null; tokens_in: number; tokens_out: number; wall_ms: number }[],
  definition?: string, cases: CaseTierRow[] = []) {
  type Row = { function: string; tier: string; calls: number; deopts: number; deopt_rate: number; tokens_per_call: number; ms_per_call: number;
    wasted_tokens: number; state: string; compared: number; worse: number; live: number; audited: number; handed_off: number; guard_misses: number;
    net_tokens: number | null; tokens: number; ms: number; fold: TierEvidence; seeded: boolean };
  const rows = new Map<string, Row>();
  const open = (fn: string, tier: string): Row => {
    const key = `${fn}\0${tier}`;
    const row = rows.get(key) ?? { function: fn, tier, calls: 0, deopts: 0, deopt_rate: 0, tokens_per_call: 0, ms_per_call: 0, wasted_tokens: 0,
      state: 'active', compared: 0, worse: 0, live: 0, audited: 0, handed_off: 0, guard_misses: 0, net_tokens: null, tokens: 0, ms: 0, fold: newTierEvidence('active'), seeded: false };
    rows.set(key, row);
    return row;
  };
  for (const { value, tokens_in, tokens_out, wall_ms } of events) {
    const ev = value as { fn?: string; tier?: string; event?: string; kind?: string; own?: boolean; state?: EvidenceState };
    if (!ev?.fn || !ev.tier || (definition && ev.fn !== definition)) continue;
    const row = open(ev.fn, ev.tier);
    const tokens = Number(tokens_in) + Number(tokens_out);
    if (ev.event === 'served') { row.calls++; row.tokens += tokens; row.ms += Number(wall_ms); }
    else if (ev.event === 'deopt') { row.deopts++; if (ev.own) row.wasted_tokens += tokens; }
    else if (ev.event === 'promoted') setTierState(row.fold, 'active');
    else if (ev.event === 'demoted') setTierState(row.fold, 'demoted');
    if (ev.event !== 'promoted' && ev.event !== 'demoted') {
      if (!row.seeded && ev.state) { row.fold.state = ev.state; row.seeded = true; }
      foldEvent(row.fold, ev as { event: string; kind?: string });
    }
  }
  for (const item of cases) {
    const row = open(item.function, 'tier3');
    row.calls = item.served; row.ms_per_call = item.ms_per_call; row.state = item.state; row.net_tokens = item.saved_tokens - item.spent_tokens;
    row.fold.evidence = { ...item.evidence, guard_misses: row.fold.evidence.guard_misses, handed_off: item.evidence.handed_off };
    row.fold.state = 'active';
  }
  return [...rows.values()].map(({ tokens, ms, fold, seeded: _seeded, ...row }) => {
    const fromCases = cases.some(item => item.function === row.function) && row.tier === 'tier3';
    const e = fold.evidence;
    return { ...row, state: fromCases ? row.state : fold.state, compared: e.compared, worse: e.worse, live: e.live_compared, audited: e.audited,
      handed_off: e.handed_off, guard_misses: e.guard_misses,
      deopt_rate: row.calls + row.deopts ? Math.round(1000 * row.deopts / (row.calls + row.deopts)) / 1000 : 0,
      tokens_per_call: fromCases ? 0 : row.calls ? Math.round(tokens / row.calls) : 0, ms_per_call: fromCases ? row.ms_per_call : row.calls ? Math.round(ms / row.calls) : 0 };
  }).sort((x, y) => x.function.localeCompare(y.function) || y.tier.localeCompare(x.tier));
}

/**
 * Batch occupancy of the model requests in a set of events (plans/BATCHED_EXECUTION.md §3.6): how many requests the
 * scheduler had in flight when each was sent, the size of the batch it left in, and how long it queued.
 */
export function occupancyOf(events: readonly Record<string, unknown>[]) {
  const sent = events.filter(event => event.kind === 'model_request' && event.phase === 'end' && typeof event.batch_id === 'string');
  const num = (event: Record<string, unknown>, key: string) => Number(event[key]) || 0;
  const waits = sent.map(event => num(event, 'queue_wait_ms')).sort((x, y) => x - y);
  const mean = (values: number[]) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
  const at = (p: number) => waits.length ? waits[Math.min(waits.length - 1, Math.ceil(p * waits.length) - 1)]! : 0;
  const round = (value: number) => Math.round(value * 100) / 100;
  return { requests: events.filter(event => event.kind === 'model_request' && event.phase === 'end').length, scheduled: sent.length,
    batches: new Set(sent.map(event => event.batch_id)).size, mean_in_flight: round(mean(sent.map(event => num(event, 'in_flight')))),
    max_in_flight: Math.max(0, ...sent.map(event => num(event, 'in_flight'))), mean_batch_size: round(mean(sent.map(event => num(event, 'batch_size')))),
    queue_wait_ms_p50: round(at(0.5)), queue_wait_ms_p95: round(at(0.95)), running_turns: sent.filter(event => event.schedule_priority === 'running').length };
}

/** The options of each `natlang traces` action. */
const TRACES: Record<string, Spec> = {
  status: {}, help: {},
  hot: { values: ['--program', '--since', '--by', '--limit'] },
  list: { values: ['--program', '--definition', '--executor', '--outcome', '--case', '--since', '--limit', '--after'], flags: ['--all', '--audits'] },
  show: { flags: ['--events', '--full'] },
  occupancy: { values: ['--definition', '--executor', '--since', '--limit'], flags: ['--audits'] },
  tiers: { values: ['--definition'] },
  export: { values: ['--definition', '--executor', '--limit'], flags: ['--audits'] },
  pin: {}, unpin: {}, annotate: {},
  good: { values: ['--group', '--program'], flags: ['--dry-run'] },
  bad: { values: ['--expected', '--group', '--program'], flags: ['--dry-run'] },
  fix: { values: ['--expected', '--group', '--program'], flags: ['--dry-run'] },
  config: {}, evict: { values: ['--bytes', '--work-limit'] },
};
const valuedOptions = (specs: Record<string, Spec>) => new Set(['--store', ...Object.values(specs).flatMap(spec => spec.values ?? [])]);

/** Usage errors print one line and exit 2; other failures are the caller's (exit 1). */
async function usageExit(run: () => Promise<number>): Promise<number> {
  try { return await run(); }
  catch (error) {
    if (!(error instanceof UsageError)) throw error;
    process.stderr.write(`natlang: ${error.message}\n`);
    return 2;
  }
}

/** `natlang traces ...` */
export function tracesCommand(argv: string[]): Promise<number> {
  return usageExit(async () => {
    if (argv.includes('--help') || argv.includes('-h') || argv[0] === 'help') { print(TRACES_HELP, false); return 0; }
    const action = actionOf(argv, valuedOptions(TRACES)) ?? 'status';
    if (!TRACES[action]) {
      const guess = closest(action, Object.keys(TRACES));
      throw new UsageError(`unknown action natlang traces ${action}${guess ? ` (did you mean ${guess}?)` : ''}; natlang help traces lists them`);
    }
    const args = parse(argv, `traces ${action}`, TRACES[action]!);
    const words = args.words.slice(1);
    return traces(action, words, args);
  });
}

async function traces(action: string, words: string[], args: Args): Promise<number> {
  const json = args.options.has('--json');
  const extra = (count: number, usage: string) => { if (words.length > count) throw new UsageError(`unexpected argument ${words[count]}; usage: ${usage}`); };
  if (['status', 'hot', 'list', 'occupancy', 'tiers', 'export', 'evict'].includes(action)) extra(0, `natlang traces ${action} [OPTIONS]`);
  const store = openStore(args);
  if (action === 'status') {
    const settings = store.settings();
    const count = (store.db.prepare('SELECT COUNT(*) AS n FROM calls').get() as { n: number }).n;
    const status = { root: store.root, calls: count, bytes: store.diskBytes(), settings, pendingJobs: store.pendingJobs(1000).length,
      compilations: store.compilations({ status: 'current' }).length, findings: store.findings({ limit: 1000 }).length };
    print(json ? status : [`store ${status.root}`, `${status.calls} calls, ${(status.bytes / 2 ** 20).toFixed(1)} MiB`,
      `${status.compilations} current compilations, ${status.pendingJobs} pending shadow/audit jobs, ${status.findings} open findings (natlang compilations findings)`,
      `settings: ${JSON.stringify(settings)}`].join('\n'), json);
    return 0;
  }
  if (action === 'hot') {
    const rows = store.hot({ program: text(args, '--program') ? resolve(text(args, '--program')!) : undefined, since: since(args),
      by: choice(args, '--by', ['calls', 'tokens', 'wall_ms'] as const) ?? 'calls', limit: number(args, '--limit', 20) });
    print(json ? rows : table(rows, ['definition_name', 'calls', 'agent_calls', 'crisp_calls', 'tokens', 'wall_ms', 'definition_source', 'definition_key']), json);
    return 0;
  }
  if (action === 'list') {
    if (args.options.has('--all') && text(args, '--program')) throw new UsageError('--all lists every program\'s calls and --program DIR one program\'s; pass one of them');
    // Inside a program, its calls by default; --all for the machine.
    const program = text(args, '--program') ? resolve(text(args, '--program')!) : args.options.has('--all') ? undefined : enclosingProgram();
    if (text(args, '--program') && !existsSync(program!)) throw new UsageError(`--program ${text(args, '--program')}: no such directory`);
    const rows = store.calls({ definition: text(args, '--definition'), executor: text(args, '--executor'), outcome: text(args, '--outcome'),
      caseHash: text(args, '--case'), since: since(args), limit: number(args, '--limit', 30), after: text(args, '--after'),
      audits: args.options.has('--audits'), program });
    if (json) { print(rows, true); return 0; }
    const scope = program ? `calls of ${program}${text(args, '--program') ? '' : ' (--all for every program\'s)'}` : 'calls on this machine';
    print(rows.length ? `${table(rows, ['call_id', 'definition_name', 'executor', 'outcome', 'started_at', 'wall_ms'])}\n${scope}, newest first; ` +
      `natlang traces show CALL (an ID prefix will do) shows one` : `no ${scope}`, false);
    return 0;
  }
  if (action === 'show') {
    extra(1, 'natlang traces show CALL [--full] [--events] [--json]');
    const id = resolveCall(store, words[0], 'natlang traces show CALL [--full] [--events] [--json]');
    if (args.options.has('--events')) { print((store.events(id) ?? []).map(event => JSON.stringify(event)).join('\n'), false); return 0; }
    const record = store.call(id);
    if (record) { print(json ? expandedRecord(store, record) : renderTrace(store, { record }, { full: args.options.has('--full') }), json); return 0; }
    const running = store.db.prepare('SELECT * FROM calls WHERE call_id = ?').get(id) as Record<string, unknown>;
    print(json ? running : renderTrace(store, { running }, { full: args.options.has('--full') }), json);
    return 0;
  }
  if (action === 'occupancy') {
    const rows = store.calls({ definition: text(args, '--definition'), executor: text(args, '--executor'), since: since(args),
      limit: number(args, '--limit', 200), audits: args.options.has('--audits') });
    const per = rows.map(row => ({ call_id: row.call_id, definition_name: row.definition_name, ...occupancyOf(store.events(row.call_id) ?? []) }))
      .filter(row => row.requests > 0);
    const total = occupancyOf(rows.flatMap(row => (store.events(row.call_id) ?? []).map(event => ({ ...event }))));
    print(json ? { total, calls: per } : [`${rows.length} calls, ${total.requests} model requests, ${total.scheduled} with scheduler records in ${total.batches} batches`,
      `in flight when sent: mean ${total.mean_in_flight}, max ${total.max_in_flight}; batch size mean ${total.mean_batch_size}; queue wait p50 ${total.queue_wait_ms_p50} ms, p95 ${total.queue_wait_ms_p95} ms; ${total.running_turns} turns of running calls`,
      table(per, ['call_id', 'definition_name', 'requests', 'scheduled', 'batches', 'mean_in_flight', 'max_in_flight', 'mean_batch_size', 'queue_wait_ms_p95'])].join('\n'), json);
    return 0;
  }
  if (action === 'tiers') {
    const rows = tiersOf(store.tierRows(), text(args, '--definition'), caseTiersOf(store, text(args, '--definition')));
    print(json ? rows : rows.length ? [table(rows, ['function', 'tier', 'state', 'calls', 'deopts', 'deopt_rate', 'tokens_per_call', 'ms_per_call', 'wasted_tokens',
      'compared', 'worse', 'live', 'audited', 'net_tokens']), 'tier 3 rows are the stored compilation: calls, evidence and net_tokens are the ones `natlang compilations list|savings` show; promotion policy: ' +
      store.settings().promotionPolicy].join('\n') :
      'no compilations or tier events recorded (plans/TIERED_ENGINE.md)', json);
    return 0;
  }
  if (action === 'export') {
    for (const row of store.calls({ definition: text(args, '--definition'), executor: text(args, '--executor'), limit: number(args, '--limit', 1000),
      audits: args.options.has('--audits') })) {
      const record = store.call(row.call_id);
      if (record) process.stdout.write(`${JSON.stringify(expandedRecord(store, record))}\n`);
    }
    return 0;
  }
  if (action === 'pin' || action === 'unpin') {
    extra(1, `natlang traces ${action} CALL`);
    const id = resolveCall(store, words[0], `natlang traces ${action} CALL`);
    store.pin(id, action === 'pin'); print(`${action}ned ${id}`, false); return 0;
  }
  if (action === 'annotate') {
    const [call, kind, ...value] = words;
    if (!call || !kind || !value.length) throw new UsageError('usage: natlang traces annotate CALL KIND VALUE');
    const id = resolveCall(store, call, 'natlang traces annotate CALL KIND VALUE');
    let parsed: unknown = value.join(' ');
    try { parsed = JSON.parse(String(parsed)); } catch { /* text */ }
    store.annotate(id, kind, parsed, 'cli'); print(`annotated ${id}`, false); return 0;
  }
  if (action === 'good' || action === 'bad' || action === 'fix') {
    const usage = `natlang traces ${action} CALL${action === 'fix' ? ' --expected JSON' : action === 'bad' ? ' [--expected JSON]' : ''} [--group G] [--program DIR] [--dry-run]`;
    extra(1, usage);
    const id = resolveCall(store, words[0], usage);
    if (action === 'fix' && text(args, '--expected') === undefined) throw new UsageError(`natlang traces fix needs the right answer: ${usage} (quote strings: '"billing"')`);
    const { recordFeedback } = await import('../calls/feedback.js');
    print(recordFeedback(store, { verdict: action, call: id, expected: text(args, '--expected'), group: text(args, '--group') || process.env.NATLANG_FEEDBACK_GROUP || undefined,
      program: text(args, '--program'), dryRun: args.options.has('--dry-run'), authorizedProducers: await authorizedProducers() }), false); return 0;
  }
  if (action === 'config') {
    const changes: Partial<CallStoreSettings> = {};
    for (const pair of words) {
      const [key, value] = pair.split('=');
      if (!key || value === undefined) throw new UsageError('usage: natlang traces config [KEY=VALUE...]');
      (changes as Record<string, unknown>)[key] = ['specialization', 'promotionPolicy', 'targetPolicy', 'declinePolicy'].includes(key) ? value : Number(value);
    }
    print(Object.keys(changes).length ? store.writeSettings(changes) : store.settings(), true);
    return 0;
  }
  // evict
  print(`freed ${store.evict(text(args, '--bytes') ? number(args, '--bytes', 0) : undefined,
    text(args, '--work-limit') ? number(args, '--work-limit', 0) : undefined)} bytes`, false); return 0;
}

/** The compilation a word names: an ID, a definition key, name or source. */
function findCompilation(store: CallStore, word: string) {
  return store.compilation(word) ?? store.currentCompilation(word) ??
    (() => { const row = store.compilations({ definition: word, status: 'current', limit: 1 })[0]; return row ? store.compilation(row.id) : undefined; })();
}

/** `natlang compilations ...` */
export function compilationsCommand(argv: string[]): Promise<number> {
  return usageExit(async () => {
    const args = parse(argv, 'compilations', { values: ['--program', '--limit', '--role', '--definition', '--out'], flags: ['--all'] });
    const [action = 'list', ...words] = args.words;
    return compilations(action, words, args);
  });
}

async function compilations(action: string, words: string[], args: Args): Promise<number> {
  const json = args.options.has('--json');
  const store = openStore(args);
  if (action === 'list') {
    const rows = store.compilations({ program: text(args, '--program') ? resolve(text(args, '--program')!) : undefined,
      status: args.options.has('--all') ? undefined : 'current', limit: number(args, '--limit', 100) }).map(row => {
      const cases = store.cases(row.id);
      const tiers = cases.reduce<Record<string, number>>((count, item) => ({ ...count, [item.tier]: (count[item.tier] ?? 0) + 1 }), {});
      return { ...row, cases: cases.length, tiers: Object.entries(tiers).map(([tier, n]) => `${n} ${tier}`).join(', '),
        served: cases.reduce((sum, item) => sum + item.served, 0), handed_off: cases.reduce((sum, item) => sum + item.handed_off, 0) };
    });
    print(json ? rows : table(rows, ['definition_name', 'id', 'status', 'cases', 'tiers', 'served', 'handed_off', 'created_at', 'definition_source']), json);
    return 0;
  }
  if (action === 'declines') {
    const rows = store.declines(number(args, '--limit', 100));
    print(json ? rows : table(rows as unknown as Record<string, unknown>[], ['definition_id', 'reason', 'why', 'calls_at_decline', 'created_at']), json);
    return 0;
  }
  if (action === 'show') {
    const compilation = words[0] && findCompilation(store, words[0]);
    if (!compilation) throw new Error(`usage: natlang compilations show DEFINITION|ID (nothing found for ${words[0] ?? ''})`);
    if (json) { print(compilation, true); return 0; }
    print([`compilation ${compilation.id} of ${compilation.definition_name} (${compilation.definition_source ?? 'inline'}), ${compilation.status}, ${compilation.created_at}`,
      `definition key ${compilation.definition_key}, interface ${compilation.interface_hash}${compilation.parent_id ? `, follows ${compilation.parent_id}` : ''}`, '',
      ...compilation.cases.map(item => `case ${item.position + 1} ${item.hash}: ${item.tier}; served ${item.served}, handed back ${item.handed_off}, ` +
        `compared ${item.compared} (${item.worse} worse, ${item.better} better; ${item.live_compared ?? 0} live, ${item.live_worse ?? 0} worse), audited ${item.audited} (${item.audit_worse} worse)` +
        `${item.note ? `; ${item.note}` : ''}\n  examples: ${store.caseCalls(item.hash, undefined, 5).map(link => `${link.call_id} (${link.role}${link.verdict ? ` ${link.verdict}` : ''})`).join(', ') || 'none'}`),
      '', '--- cases.ts', compilation.files['cases.ts'] ?? '', '--- report.md', compilation.files['report.md'] ?? ''].join('\n'), false);
    return 0;
  }
  if (action === 'why') {
    const record = store.call(resolveCall(store, words[0], 'natlang compilations why CALL'));
    if (!record) throw new Error(`call ${words[0]} has not finished; natlang traces show ${words[0]} shows it`);
    const compilation = store.currentCompilation(record.definition.key);
    const lines = [`${record.definition.name}: ${record.executor.kind}`];
    if (record.executor.case_hash) lines.push(`served by case ${record.executor.case_hash}${record.executor.case_error ? `, which stopped (${record.executor.case_error}); the agent finished the call` : ''}`);
    else if (!compilation) lines.push(store.declineFor(record.definition.key) ? `no compilation: declined (${store.declineFor(record.definition.key)!.reason}: ${store.declineFor(record.definition.key)!.why})` :
      'no compilation for this definition revision');
    else if (compilation.interface_hash !== record.definition.interface) lines.push(`compilation ${compilation.id} was not used: the function's context changed since it was compiled`);
    else {
      const active = compilation.cases.filter(item => item.tier === 'active').length;
      lines.push(`compilation ${compilation.id} has ${compilation.cases.length} case(s), ${active} active; no active case's guard admitted these inputs` +
        `${record.audit_of ? ' (this is an audit run, which never uses compilations)' : ''}`);
    }
    for (const link of store.callCases(record.call_id)) lines.push(`linked to case ${link.case_hash} as ${link.role}${link.verdict ? ` (${link.verdict})` : ''}`);
    print(lines.join('\n'), false);
    return 0;
  }
  if (action === 'calls') {
    if (!words[0]) throw new Error('usage: natlang compilations calls CASE [--role ROLE]');
    const rows = store.caseCalls(words[0], text(args, '--role') as CaseRole | undefined, number(args, '--limit', 100));
    print(json ? rows : table(rows, ['call_id', 'role', 'verdict', 'created_at']), json);
    return 0;
  }
  if (action === 'history') {
    if (!words[0]) throw new Error('usage: natlang compilations history DEFINITION');
    const rows = store.compilations({ definition: words[0], limit: 50 });
    const decline = rows[0] ? store.declineFor(rows[0].definition_key) : store.declines(500).find(row => row.definition_id === words[0] || row.definition_key === words[0]);
    print(json ? { compilations: rows, decline } : [table(rows, ['id', 'status', 'created_at', 'parent_id']),
      ...(decline ? [`declined ${decline.created_at}: ${decline.reason}: ${decline.why}`] : []),
      ...rows.slice(0, 1).flatMap(row => store.jobs(undefined, 20).filter(job => store.cases(row.id).some(item => item.hash === job.case_hash))
        .map(job => `${job.kind} ${job.case_hash} on ${job.call_id}: ${job.status}${job.verdict ? ` ${job.verdict}` : ''}`))].join('\n'), json);
    return 0;
  }
  if (action === 'export') {
    const [word, target] = words;
    if (!word || !target) throw new Error('usage: natlang compilations export DEFINITION DIR');
    const compilation = findCompilation(store, word);
    const key = compilation?.definition_key ?? store.hot({ limit: 500 }).find(row => [row.definition_key, row.definition_id, row.definition_name, row.definition_source].includes(word))?.definition_key;
    if (!key) throw new Error(`nothing recorded for ${word}`);
    const write = (path: string, body: string) => { const file = join(target, path); mkdirSync(dirname(file), { recursive: true }); writeFileSync(file, body); };
    for (const [path, body] of Object.entries(compilation?.files ?? {})) write(path, body);
    const subject = study(store, key);
    if (subject) {
      const { history, previousCases } = renderHistory(store, key);
      for (const [path, body] of Object.entries(renderEvidence(store, subject, { history, previousCases, report: compilation?.files['report.md'] })))
        write(join('evidence', path), body);
    }
    print(`wrote ${compilation ? `compilation ${compilation.id} and ` : ''}its evidence to ${target}`, false);
    return 0;
  }
  if (action === 'disable' || action === 'enable') {
    const word = words[0];
    if (!word) throw new Error(`usage: natlang compilations ${action} CASE|COMPILATION`);
    const compilation = store.compilation(word);
    if (compilation) store.setCompilationStatus(word, action === 'disable' ? 'disabled' : 'current');
    else store.setTier(word, (action === 'disable' ? 'disabled' : 'shadow') as CaseTier, `${action}d by hand`);
    print(`${action}d ${word}`, false);
    return 0;
  }
  if (action === 'savings') {
    const rows = store.savings({ definition: words[0] });
    const shown = rows.map(row => ({ ...row, net_tokens: row.saved_tokens - row.spent_tokens,
      saved_time: `${(row.saved_ms / 60_000).toFixed(1)} min`, spent_time: `${(row.spent_ms / 60_000).toFixed(1)} min` }));
    print(json ? shown : table(shown, ['definition_name', 'served', 'agent_tokens_per_call', 'agent_ms_per_call', 'crisp_ms_per_call', 'saved_tokens',
      'spent_tokens', 'net_tokens', 'saved_time', 'spent_time', 'definition_source']) + (json ? '' : '\nper-tier evidence and cost: natlang traces tiers'), json);
    return 0;
  }
  if (action === 'findings') {
    const rows = store.findings({ definition: words[0], all: args.options.has('--all'), limit: number(args, '--limit', 100) });
    print(json ? rows : rows.map(row => `#${row.id} ${row.definition_name} (${row.definition_source ?? row.definition_key}) ${row.kind}${row.status === 'open' ? '' : ` [${row.status}]`}: ` +
      `${row.summary} (seen ${row.seen}×, ${row.updated_at})\n  ${JSON.stringify(row.detail).slice(0, 600)}`).join('\n') || 'no open findings', json);
    return 0;
  }
  if (action === 'acknowledge') {
    const id = Number(words[0]);
    if (!Number.isInteger(id)) throw new Error('usage: natlang compilations acknowledge FINDING-ID');
    print(store.acknowledgeFinding(id) ? `acknowledged #${id}` : `no finding #${id}`, false);
    return 0;
  }
  if (action === 'export-corpus') {
    const out = text(args, '--out');
    if (!out) throw new Error('usage: natlang compilations export-corpus --out DIR [--definition NAME]');
    const summary = exportSpecializationCorpus(store, resolve(out), { definition: text(args, '--definition') });
    print(json ? summary : `wrote ${summary.cases} cases, ${summary.declines} declines and ${summary.findings} findings (${summary.calls} calls) to ${resolve(out)}`, json);
    return 0;
  }
  throw new UsageError('usage: natlang compilations list|show|why|calls|history|export|disable|enable|declines|savings|findings|acknowledge|export-corpus');
}

/** The specializer application in this checkout. */
export function specializerDirectory(): string | undefined {
  const candidates = [process.env.NATLANG_SPECIALIZER, fileURLToPath(new URL('../../../applications/specializer', import.meta.url))];
  return candidates.find(item => item && existsSync(join(item, 'natlang.json')));
}

/** `natlang specialize [MODEL OPTIONS] [SPECIALIZER OPTIONS]`: `natlang run` of the specializer application. */
export function specializeArguments(argv: string[]): string[] {
  const directory = specializerDirectory();
  if (!directory) throw new Error('the specializer application was not found; set NATLANG_SPECIALIZER to its directory');
  const runOptions = new Set(['--profile', '--provider', '--model', '--yes', '--state', '--workspace']);
  const run: string[] = [], rest: string[] = [];
  for (let index = 0; index < argv.length; index++) {
    const item = argv[index]!;
    if (runOptions.has(item)) { run.push(item); if (item !== '--yes') run.push(argv[++index] ?? ''); }
    else rest.push(item);
  }
  return ['run', directory, ...run, '--', ...rest];
}

/** The models whose outputs may become tune labels: the tuning teachers (configured, or a native student's `escalate`)
 * and every native profile's model (our own checkpoints). Without any tuning configuration nothing is restricted. */
async function authorizedProducers(): Promise<ReadonlySet<string> | undefined> {
  const { loadModelConfiguration } = await import('../model/config.js');
  const { loadTuneSettings } = await import('../tune/settings.js');
  let teachers: string[];
  try { const { settings } = loadTuneSettings(); teachers = [settings.teacher, ...(settings.teachers ?? [])]; }
  catch { return undefined; }
  const models = new Set<string>();
  const add = (profile: string) => { try { models.add(loadModelConfiguration(profile).choice.model); } catch { /* unusable profile */ } };
  teachers.forEach(add);
  const { defaultNatlangConfigDirectory } = await import('../package/store.js');
  const path = join(defaultNatlangConfigDirectory(), 'config.json');
  const profiles = existsSync(path) ? (JSON.parse(readFileSync(path, 'utf8')) as { profiles?: Record<string, { native?: unknown }> }).profiles ?? {} : {};
  for (const [name, profile] of Object.entries(profiles)) if (profile.native) add(name);
  return models;
}

