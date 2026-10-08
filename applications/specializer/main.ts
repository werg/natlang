/**
 * `natlang run applications/specializer -- [--definition NAME|KEY] [--program DIR] [--loop] [--interval SECONDS]`
 *
 * Compiles recorded executions of natural-language functions into guarded crisp cases (plans/TRACE_SPECIALIZATION.md).
 * For each hot definition revision in the machine's call store: study its recorded calls (crisp), let specialize.nl
 * write cases.ts over the rendered evidence or decline, verify the cases on the recorded calls (replay plus judge), and
 * store the accepted ones as the definition's compilation, where runtimes pick them up in shadow. Then run pending
 * shadow replays and audits. With --loop it repeats; every step can be interrupted and resumes from the store.
 */
import { resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { CallStore, Folder, crispDecline, machineStoreRoot, renderEvidence, renderHistory, renderReport, runJobs, saveAccepted,
  study, verifyCases, type CaseCheck, type HotDefinition } from '@natlang/node';
import specialize from './specialize.nl';
import type { SpecializeResult } from './types.js';

type Options = { definition?: string; program?: string; loop: boolean; interval: number; rounds: number; jobs: number; minCalls?: number;
  includeSelf: boolean; dryRun: boolean };

function parse(args: string[]): Options {
  const value = (name: string) => { const at = args.indexOf(name); return at >= 0 ? args[at + 1] : undefined; };
  return { definition: value('--definition'), program: value('--program') ? resolve(value('--program')!) : undefined,
    loop: args.includes('--loop'), interval: Number(value('--interval') ?? 600), rounds: Number(value('--rounds') ?? 3),
    jobs: Number(value('--jobs') ?? 50), minCalls: value('--min-calls') ? Number(value('--min-calls')) : undefined,
    includeSelf: args.includes('--include-self'), dryRun: args.includes('--dry-run') };
}

/** Definitions worth a look now: enough agent calls, no standing decline, and new calls since the last compilation. */
export function targets(store: CallStore, options: Options, self: string | null): HotDefinition[] {
  const minCalls = options.minCalls ?? store.settings().minCalls;
  return store.hot({ program: options.program, limit: 200 }).filter((item: HotDefinition) => {
    if (options.definition && ![item.definition_key, item.definition_id, item.definition_name, item.definition_source].includes(options.definition)) return false;
    if (!options.includeSelf && self && item.program_root === self) return false;
    if (item.agent_calls < minCalls) return false;
    // Calls that used no model tokens (scripted test drivers) have nothing to save.
    if (!options.definition && !item.tokens) return false;
    const decline = store.declineFor(item.definition_key);
    if (decline && !options.definition && item.agent_calls < 2 * decline.calls_at_decline) return false;
    const current = store.currentCompilation(item.definition_key);
    if (current && !options.definition && store.calls({ key: item.definition_key, executor: 'agent', since: current.created_at, limit: minCalls }).length < minCalls)
      return false;
    return true;
  });
}

/** Specialize one definition revision: at most `rounds` rounds of write, verify, report. */
export async function specializeOne(context: TargetContext, store: CallStore, key: string, options: Options): Promise<string> {
  const minCalls = options.minCalls ?? store.settings().minCalls;
  const subject = study(store, key);
  if (!subject) return 'no usable calls';
  const name = subject.definition.name;
  const crisp = crispDecline(subject, minCalls);
  if (crisp) {
    if (!options.dryRun) store.decline({ definitionKey: key, definitionId: subject.definition.id, reason: crisp.reason, why: crisp.why, calls: subject.examples.length });
    return `${name}: declined without a model (${crisp.reason}: ${crisp.why})`;
  }
  const { history, previousCases } = renderHistory(store, key);
  let report: string | undefined, text = previousCases ?? '', checks: CaseCheck[] = [];
  for (let round = 1; round <= options.rounds; round++) {
    const evidence = renderEvidence(store, subject, { report, history, previousCases });
    const files: Record<string, string> = Object.fromEntries(Object.entries(evidence).map(([path, body]) => [`evidence/${path}`, String(body)]));
    if (text) files['cases.ts'] = text;
    const workspace = Folder.fromFiles(files);
    const result = await context.runtime.run(() => workspace.root().apply(specialize, name)) as SpecializeResult;
    if (result.kind === 'declined') {
      if (!options.dryRun) store.decline({ definitionKey: key, definitionId: subject.definition.id, reason: result.reason, why: result.why, calls: subject.examples.length });
      return `${name}: declined (${result.reason}: ${result.why})`;
    }
    text = workspace.filePaths().includes('cases.ts') ? new TextDecoder().decode(workspace.readBytesSync('cases.ts')) : '';
    if (!text.trim()) { report = '# Report\n\nThere is no cases.ts. Write it, or return kind "declined" with a reason.\n'; continue; }
    const verification = await verifyCases(context.runtime, store, subject, text, { bound: store.settings().acceptanceBound });
    report = renderReport(store, subject, verification);
    checks = verification.checks;
    if (checks.length && checks.every(check => check.accepted)) break;
  }
  if (options.dryRun) return `${name}: ${checks.filter(check => check.accepted).length} of ${checks.length} cases would be accepted\n${report ?? ''}`;
  const saved = checks.length ? saveAccepted(store, subject, text, checks, report ?? '', { rounds: options.rounds }) : undefined;
  if (saved) return `${name}: compilation ${saved.id} with ${saved.cases} case(s), in shadow`;
  store.decline({ definitionKey: key, definitionId: subject.definition.id, reason: 'no-clusters',
    why: `no case passed verification in ${options.rounds} rounds: ${checks.map(check => check.reason).join('; ') || 'no cases written'}`,
    calls: subject.examples.length });
  return `${name}: no case passed verification; declined`;
}

export async function main(context: TargetContext): Promise<number> {
  const options = parse(context.args);
  const root = machineStoreRoot();
  if (!root) { context.io.error.write('the call store is off on this machine (NATLANG_CALL_STORE=off)\n'); return 2; }
  const store = CallStore.open(root);
  const self = context.package?.root ?? null;
  const log = (line: string) => context.io.output.write(`${new Date().toISOString()} ${line}\n`);
  let stopping = false;
  const stop = () => { stopping = true; log('stopping after the current step'); };
  process.once('SIGTERM', stop); process.once('SIGINT', stop);
  do {
    for (const target of targets(store, options, self)) {
      if (stopping) break;
      try { log(await specializeOne(context, store, target.definition_key, options)); }
      catch (error) { log(`${target.definition_name}: failed: ${error instanceof Error ? error.stack ?? error.message : String(error)}`); }
    }
    if (!stopping && !options.dryRun) {
      const ran = await runJobs(context.runtime, store, { limit: options.jobs, log });
      if (ran) log(`${ran} shadow/audit job(s) done`);
    }
    if (!options.loop || stopping) break;
    for (let waited = 0; waited < options.interval && !stopping; waited++) await new Promise(done => setTimeout(done, 1000));
  } while (!stopping);
  return 0;
}
