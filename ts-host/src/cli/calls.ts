/**
 * `natlang traces ...` and `natlang compilations ...`: the standard view of the machine's call record store and of the
 * compilations stored beside it (plans/TRACE_SPECIALIZATION.md §3.5, §7.4). `natlang specialize ...` runs the
 * specializer application on the store.
 */
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CallStore, machineStoreRoot } from '../calls/store.js';
import { exportSpecializationCorpus } from '../calls/corpus.js';
import { renderEvidence, renderHistory, study } from '../calls/specializer.js';
import type { CallRecord, CallStoreSettings, CaseRole, CaseTier } from '../calls/types.js';

type Args = { words: string[]; options: Map<string, string | true> };
function parse(argv: string[]): Args {
  const words: string[] = [], options = new Map<string, string | true>();
  const flags = new Set(['--json', '--events', '--values', '--all']);
  for (let index = 0; index < argv.length; index++) {
    const item = argv[index]!;
    if (item.startsWith('--')) {
      if (flags.has(item) || index + 1 >= argv.length) options.set(item, true);
      else options.set(item, argv[++index]!);
    } else words.push(item);
  }
  return { words, options };
}
const text = (args: Args, name: string) => { const value = args.options.get(name); return typeof value === 'string' ? value : undefined; };
const number = (args: Args, name: string, fallback: number) => Number(text(args, name) ?? fallback);

function openStore(args: Args): CallStore {
  const root = text(args, '--store') ?? machineStoreRoot();
  if (!root) throw new Error('the call store is off on this machine (NATLANG_CALL_STORE=off); pass --store DIR to read one');
  return CallStore.open(root);
}
const print = (value: unknown, json: boolean) => process.stdout.write(json ? `${JSON.stringify(value, null, 2)}\n` :
  `${String(value)}${String(value).endsWith('\n') ? '' : '\n'}`);
const table = (rows: Record<string, unknown>[], columns: string[]): string => {
  if (!rows.length) return '(none)';
  const cells = [columns, ...rows.map(row => columns.map(column => String(row[column] ?? '')))];
  const widths = columns.map((_, index) => Math.min(60, Math.max(...cells.map(row => row[index]!.length))));
  return cells.map(row => row.map((cell, index) => cell.slice(0, 60).padEnd(widths[index]!)).join('  ').trimEnd()).join('\n');
};
const short = (value: unknown, limit = 200) => { const shown = JSON.stringify(value); return shown === undefined ? 'undefined' : shown.length > limit ? `${shown.slice(0, limit)}…` : shown; };

/** A record with its values in place of their references. */
function expanded(store: CallStore, record: CallRecord): Record<string, unknown> {
  const value = (ref: Parameters<CallStore['value']>[0]) => ref?.complete ? store.value(ref) : ref;
  return { ...record, inputs: Object.fromEntries(Object.entries(record.inputs).map(([name, ref]) => [name, value(ref)])),
    captures: Object.fromEntries(Object.entries(record.captures).map(([name, ref]) => [name, value(ref)])),
    output: value(record.output), effects: record.effects.map(effect => ({ ...effect, args: value(effect.args), result: value(effect.result) })),
    capture_writes: record.capture_writes.map(write => ({ ...write, after: value(write.after) })),
    definition: { ...record.definition, instructions: value(record.definition.instructions) } };
}

function describeCall(store: CallStore, record: CallRecord): string {
  const view = expanded(store, record);
  const cases = store.callCases(record.call_id);
  return [`call ${record.call_id}${record.parent_call_id ? ` (in ${record.parent_call_id}, eval ${record.parent_action_index ?? '?'})` : ''}`,
    `definition: ${record.definition.name} (${record.definition.source ?? record.definition.site}) key ${record.definition.key}`,
    `executor: ${record.executor.kind}${record.executor.case_hash ? ` case ${record.executor.case_hash}` : ''}${record.executor.model_id ? ` model ${record.executor.model_id}` : ''}`,
    ...(record.executor.case_error ? [`case stopped: ${record.executor.case_error}`] : []),
    `outcome: ${record.outcome}${record.detail ? ` (${record.detail.slice(0, 200)})` : ''}`,
    `started ${record.started_at}, ${record.cost.wall_ms} ms, ${record.cost.model_requests} model requests, ${record.cost.tokens_in}+${record.cost.tokens_out} tokens, ${record.cost.evals} evals`,
    `inputs: ${short(view.inputs, 600)}`, ...(Object.keys(record.captures).length ? [`captures: ${short(view.captures, 400)}`] : []),
    `output: ${short(view.output, 600)}`,
    ...(record.effects.length ? ['effects:', ...(view.effects as { service: string; method: string; args: unknown; result?: unknown; error?: string; by: string }[])
      .map(effect => `  ${effect.by} ${effect.service}.${effect.method}(${short(effect.args, 160).slice(1, -1)}) ${effect.error ? `failed: ${effect.error}` : `→ ${short(effect.result, 160)}`}`)] : []),
    ...(record.approach.evals.length ? [`approach ${record.approach.hash ?? '-'}:`, ...record.approach.evals.map((code, index) => `  eval ${index + 1}: ${code.replace(/\s+/g, ' ').slice(0, 300)}`)] : []),
    ...(cases.length ? ['cases:', ...cases.map(item => `  ${item.case_hash} ${item.role}${item.verdict ? ` ${item.verdict}` : ''}`)] : []),
    ...(store.annotations(record.call_id).map(item => `annotation ${item.kind}: ${short(item.value)}`)),
    ...(store.children(record.call_id).map(child => `child ${child.call_id} ${child.definition_name} ${child.executor} ${child.outcome}`))].join('\n');
}

/**
 * Per function and tier from the tier events kept with calls (calls/tiers.ts): calls served, deopts and their rate, the
 * cost of a served call, tokens wasted by attempts that were then deoptimized, and the last state the events imply.
 */
export function tiersOf(events: { value: unknown; definition_name: string | null; tokens_in: number; tokens_out: number; wall_ms: number }[],
  definition?: string) {
  type Row = { function: string; tier: string; calls: number; deopts: number; deopt_rate: number; tokens_per_call: number; ms_per_call: number;
    wasted_tokens: number; state: string; tokens: number; ms: number };
  const rows = new Map<string, Row>();
  for (const { value, tokens_in, tokens_out, wall_ms } of events) {
    const ev = value as { fn?: string; tier?: string; event?: string; own?: boolean };
    if (!ev?.fn || !ev.tier || (definition && ev.fn !== definition)) continue;
    const key = `${ev.fn}\0${ev.tier}`;
    const row = rows.get(key) ?? { function: ev.fn, tier: ev.tier, calls: 0, deopts: 0, deopt_rate: 0, tokens_per_call: 0, ms_per_call: 0,
      wasted_tokens: 0, state: 'active', tokens: 0, ms: 0 };
    rows.set(key, row);
    const tokens = Number(tokens_in) + Number(tokens_out);
    if (ev.event === 'served') { row.calls++; row.tokens += tokens; row.ms += Number(wall_ms); }
    else if (ev.event === 'deopt') { row.deopts++; if (ev.own) row.wasted_tokens += tokens; }
    else if (ev.event === 'promoted') row.state = 'active';
    else if (ev.event === 'demoted') row.state = 'demoted';
  }
  return [...rows.values()].map(({ tokens, ms, ...row }) => ({ ...row, deopt_rate: row.calls + row.deopts ? Math.round(1000 * row.deopts / (row.calls + row.deopts)) / 1000 : 0,
    tokens_per_call: row.calls ? Math.round(tokens / row.calls) : 0, ms_per_call: row.calls ? Math.round(ms / row.calls) : 0 }))
    .sort((a, b) => a.function.localeCompare(b.function) || b.tier.localeCompare(a.tier));
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

/** `natlang traces ...` */
export async function tracesCommand(argv: string[]): Promise<number> {
  const args = parse(argv);
  const json = args.options.has('--json');
  const [action = 'status', ...words] = args.words;
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
    const rows = store.hot({ program: text(args, '--program') ? resolve(text(args, '--program')!) : undefined, since: text(args, '--since'),
      by: (text(args, '--by') as 'calls' | 'tokens' | 'wall_ms' | undefined) ?? 'calls', limit: number(args, '--limit', 20) });
    print(json ? rows : table(rows, ['definition_name', 'calls', 'agent_calls', 'crisp_calls', 'tokens', 'wall_ms', 'definition_source', 'definition_key']), json);
    return 0;
  }
  if (action === 'list') {
    const rows = store.calls({ definition: text(args, '--definition'), executor: text(args, '--executor'), outcome: text(args, '--outcome'),
      caseHash: text(args, '--case'), since: text(args, '--since'), limit: number(args, '--limit', 30), after: text(args, '--after'),
      audits: args.options.has('--all') });
    print(json ? rows : table(rows, ['call_id', 'definition_name', 'executor', 'outcome', 'started_at', 'wall_ms', 'approach_hash']), json);
    return 0;
  }
  if (action === 'show') {
    const record = words[0] && store.call(words[0]);
    if (!record && words[0] && args.options.has('--events')) {
      print((store.events(words[0]) ?? []).map(event => JSON.stringify(event)).join('\n'), false);
      return 0;
    }
    if (!record) {
      const running = words[0] ? store.db.prepare("SELECT * FROM calls WHERE call_id = ? AND outcome IN ('running', 'interrupted')").get(words[0]) as Record<string, unknown> | undefined : undefined;
      if (!running) throw new Error(`usage: natlang traces show CALL (no call ${words[0] ?? ''})`);
      print(json ? running : [`call ${running.call_id}: ${running.definition_name} ${running.outcome === 'running' ? 'is running' : 'was interrupted (its process ended)'} since ${running.started_at}`,
        ...store.children(String(running.call_id)).map(child => `child ${child.call_id} ${child.definition_name} ${child.executor} ${child.outcome}`)].join('\n'), json);
      return 0;
    }
    if (args.options.has('--events')) { print((store.events(record.call_id) ?? []).map(event => JSON.stringify(event)).join('\n'), false); return 0; }
    print(json ? expanded(store, record) : describeCall(store, record), json);
    return 0;
  }
  if (action === 'occupancy') {
    const rows = store.calls({ definition: text(args, '--definition'), executor: text(args, '--executor'), since: text(args, '--since'),
      limit: number(args, '--limit', 200), audits: args.options.has('--all') });
    const per = rows.map(row => ({ call_id: row.call_id, definition_name: row.definition_name, ...occupancyOf(store.events(row.call_id) ?? []) }))
      .filter(row => row.requests > 0);
    const total = occupancyOf(rows.flatMap(row => (store.events(row.call_id) ?? []).map(event => ({ ...event }))));
    print(json ? { total, calls: per } : [`${rows.length} calls, ${total.requests} model requests, ${total.scheduled} with scheduler records in ${total.batches} batches`,
      `in flight when sent: mean ${total.mean_in_flight}, max ${total.max_in_flight}; batch size mean ${total.mean_batch_size}; queue wait p50 ${total.queue_wait_ms_p50} ms, p95 ${total.queue_wait_ms_p95} ms; ${total.running_turns} turns of running calls`,
      table(per, ['call_id', 'definition_name', 'requests', 'scheduled', 'batches', 'mean_in_flight', 'max_in_flight', 'mean_batch_size', 'queue_wait_ms_p95'])].join('\n'), json);
    return 0;
  }
  if (action === 'tiers') {
    const rows = tiersOf(store.tierRows(), text(args, '--definition'));
    print(json ? rows : rows.length ? table(rows, ['function', 'tier', 'calls', 'deopts', 'deopt_rate', 'tokens_per_call', 'ms_per_call', 'wasted_tokens', 'state']) :
      'no tier events recorded (configure the tiered engine: plans/TIERED_ENGINE.md)', json);
    return 0;
  }
  if (action === 'export') {
    for (const row of store.calls({ definition: text(args, '--definition'), executor: text(args, '--executor'), limit: number(args, '--limit', 1000),
      audits: args.options.has('--all') })) {
      const record = store.call(row.call_id);
      if (record) process.stdout.write(`${JSON.stringify(expanded(store, record))}\n`);
    }
    return 0;
  }
  if (action === 'pin' || action === 'unpin') {
    if (!words[0]) throw new Error(`usage: natlang traces ${action} CALL`);
    store.pin(words[0], action === 'pin'); print(`${action}ned ${words[0]}`, false); return 0;
  }
  if (action === 'annotate') {
    const [call, kind, ...value] = words;
    if (!call || !kind || !value.length) throw new Error('usage: natlang traces annotate CALL KIND VALUE');
    let parsed: unknown = value.join(' ');
    try { parsed = JSON.parse(String(parsed)); } catch { /* text */ }
    store.annotate(call, kind, parsed, 'cli'); print(`annotated ${call}`, false); return 0;
  }
  if (action === 'config') {
    const changes: Partial<CallStoreSettings> = {};
    for (const pair of words) {
      const [key, value] = pair.split('=');
      if (!key || value === undefined) throw new Error('usage: natlang traces config [KEY=VALUE...]');
      (changes as Record<string, unknown>)[key] = key === 'specialization' ? value : Number(value);
    }
    print(Object.keys(changes).length ? store.writeSettings(changes) : store.settings(), true);
    return 0;
  }
  if (action === 'evict') { print(`freed ${store.evict(text(args, '--bytes') ? Number(text(args, '--bytes')) : undefined)} bytes`, false); return 0; }
  throw new Error('usage: natlang traces status|hot|list|show|occupancy|tiers|export|pin|unpin|annotate|config|evict');
}

/** The compilation a word names: an ID, a definition key, name or source. */
function findCompilation(store: CallStore, word: string) {
  return store.compilation(word) ?? store.currentCompilation(word) ??
    (() => { const row = store.compilations({ definition: word, status: 'current', limit: 1 })[0]; return row ? store.compilation(row.id) : undefined; })();
}

/** `natlang compilations ...` */
export async function compilationsCommand(argv: string[]): Promise<number> {
  const args = parse(argv);
  const json = args.options.has('--json');
  const [action = 'list', ...words] = args.words;
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
    const record = words[0] && store.call(words[0]);
    if (!record) throw new Error('usage: natlang compilations why CALL');
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
      'spent_tokens', 'net_tokens', 'saved_time', 'spent_time', 'definition_source']), json);
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
  throw new Error('usage: natlang compilations list|show|why|calls|history|export|disable|enable|declines|savings|findings|acknowledge|export-corpus');
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
