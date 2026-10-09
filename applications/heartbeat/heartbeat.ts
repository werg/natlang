// One heartbeat cycle: collect (crisp, read-only), decide (diagnoseRun per run, triageInbox, planNext - each pluggable
// crisp|nl|shadow, each answer checked by an exact verifier), apply (only what the allowlist lets run by itself at this
// stage), record. When no executor answers, the crisp ladder produces the report and says so.
import { pluggable, untrusted, type Untrusted } from '@natlang/node';
import diagnoseRun from './diagnoseRun.nl';
import planNext from './planNext.nl';
import triageInbox from './triageInbox.nl';
import { loadActions, loadWatch, lowerStage, summarize, type ActionDef, type Known, type Mode, type Settings, type Stage } from './actions.js';
import { applyPlan, type Disposition } from './apply.js';
import { deriveAgentActions } from './compare.js';
import { collect, resolveIn, type Snapshot } from './collect/index.js';
import { vllmLoad } from './collect/system.js';
import type { Host } from './host.js';
import { crispDiagnose, crispPlan, crispTriage, idleResource, type PlanArgs } from './ladder.js';
import { appendRecord, dayOf, memoryOf, readRecords, writeLatest, type CycleRecord, type StageNote } from './record.js';
import type { CycleMemory, InboxTriage, Plan, RunDiagnosis, RunEvidence } from './types.js';
import { diagnosisProblems, planProblems, triageProblems } from './verify.js';

/** Runs natural-language work in a task (`(fn) => runtime.run(fn)`). */
export type Run = <T>(fn: () => Promise<T>) => Promise<T>;
export type CycleOptions = {
  host: Host,
  run: Run,
  /** Where actions.json and watch.json live. */
  configDir: string,
  /** The repository the collectors and scripts run in. */
  repo: string,
  /** A stage that holds the cycle back below the file's stage; it never promotes. */
  stage?: Stage,
  /** The endpoint to probe before calling the functions; absent when the model is not reached over HTTP. */
  executorEndpoint?: string,
  /** False: collect and decide, but write no record and apply nothing. */
  record?: boolean,
  /** Overrides the file's modes (for tests and one-off runs). */
  modes?: Partial<Settings['modes']>,
  log?: (line: string) => void,
};

const parts = ['diagnoseRun', 'triageInbox', 'planNext'] as const;

/** Whether the executor answers: any HTTP answer from its health or model list counts. */
export async function executorUp(host: Host, endpoint: string): Promise<boolean> {
  for (const path of ['/health', '/v1/models']) if ((await host.fetchText(new URL(path, endpoint).href, 5000)) !== null) return true;
  return false;
}

/**
 * Wait until the executor is not busy (vLLM's running plus waiting requests at most `max_busy`), so the heartbeat does not
 * compete with the programs it serves; goes ahead after `idle_wait_seconds`. An endpoint without metrics is never waited for.
 */
export async function waitForIdle(host: Host, endpoint: string, settings: Settings, log: (line: string) => void, sleep: (ms: number) => Promise<void>): Promise<void> {
  const busy = async () => { const text = await host.fetchText(new URL('/metrics', endpoint).href, 5000); const load = text === null ? null : vllmLoad(text); return load ? load.running + load.waiting : undefined; };
  const started = host.now().getTime();
  for (let load = await busy(); load !== undefined && load > settings.max_busy; load = await busy()) {
    if (host.now().getTime() - started > settings.idle_wait_seconds * 1000) { log(`the executor is still busy (${load} requests); going ahead`); return; }
    log(`waiting for the executor to be idle (${load} requests, at most ${settings.max_busy})`);
    await sleep(30_000);
  }
}

const compactTime = (iso: string) => iso.replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z');

/**
 * One part of the cycle in the mode the settings choose. `nl` calls the function and, when it cannot produce a verified
 * answer (a model error, or problems left after the repairs), serves the crisp side and says so. `shadow` serves crisp and
 * records whether the two agree.
 */
async function decide<A extends unknown[], R>(name: string, mode: Mode, up: boolean, crisp: (...args: A) => R,
    viaNl: (...args: A) => Promise<{ value: R, attempts: number, problems: string[] }>, same: (crisp: R, other: R) => boolean, summary: (value: R) => string, args: A): Promise<{ value: R, note: StageNote }> {
  if (mode === 'crisp') return { value: crisp(...args), note: { via: 'crisp', attempts: 0, problems: [] } };
  if (!up) return { value: crisp(...args), note: { via: 'crisp-degraded', attempts: 0, problems: [], error: 'the executor did not answer' } };
  let captured: { value: R, attempts: number, problems: string[] } | undefined;
  const part = pluggable<A, R>({ crisp, nl: async (...inner: A) => { captured = await viaNl(...inner); return captured.value; } }, mode, { name, serve: mode === 'shadow' ? 'crisp' : 'nl', same });
  try {
    const value = await part(...args);
    if (mode === 'shadow') return { value, note: { via: 'shadow', attempts: captured?.attempts ?? 0, problems: captured?.problems ?? [], agree: captured !== undefined && same(value, captured.value),
      nl_summary: captured ? summary(captured.value) : undefined } };
    if (captured && captured.problems.length) return { value: crisp(...args), note: { via: 'crisp-fallback', attempts: captured.attempts, problems: captured.problems } };
    return { value, note: { via: 'nl', attempts: captured?.attempts ?? 1, problems: [] } };
  } catch (error) {
    return { value: crisp(...args), note: { via: 'crisp-fallback', attempts: captured?.attempts ?? 1, problems: captured?.problems ?? [], error: (error instanceof Error ? error.message : String(error)).slice(0, 400) } };
  }
}

/** Ask, check, and ask again with the checker's sentences until the answer has no problems or the repairs are used. */
async function repairing<R>(repairs: number, ask: (feedback: string) => Promise<R>, check: (value: R) => string[], tidy?: (value: R, problems: string[]) => R): Promise<{ value: R, attempts: number, problems: string[] }> {
  let feedback = '', last: R | undefined, problems: string[] = [];
  for (let attempt = 1; attempt <= repairs + 1; attempt++) {
    last = await ask(feedback);
    problems = check(last);
    if (!problems.length) return { value: last, attempts: attempt, problems: [] };
    feedback = `The previous answer had these problems. ${problems.join(' ')}`;
  }
  // With a tidy step the leftover is salvaged (what is valid stays) and accepted; without one the problems stand.
  return tidy ? { value: tidy(last!, problems), attempts: repairs + 1, problems: [] } : { value: last!, attempts: repairs + 1, problems };
}

const sameDiagnosis = (a: RunDiagnosis, b: RunDiagnosis) => a.health === b.health && a.cause === b.cause;
const sameTriage = (a: InboxTriage, b: InboxTriage) => JSON.stringify(a.obligations.map(item => [item.message_id, item.kind, item.reply_needed])) === JSON.stringify(b.obligations.map(item => [item.message_id, item.kind, item.reply_needed]));
const samePlan = (a: Plan, b: Plan) => JSON.stringify(a.proposals.map(item => [item.action, item.target]).sort()) === JSON.stringify(b.proposals.map(item => [item.action, item.target]).sort());

export async function runCycle(options: CycleOptions): Promise<CycleRecord> {
  const { host, repo } = options;
  const log = options.log ?? (() => undefined);
  const loaded = await loadActions(host, `${options.configDir}/actions.json`);
  const settings: Settings = loaded.value.settings;
  const defs: ActionDef[] = loaded.value.actions;
  const machine = host.machine();
  const stage = lowerStage(settings.stage, options.stage);
  const modes = { ...settings.modes, ...options.modes };
  const watch = await loadWatch(host, `${options.configDir}/watch.json`, machine);
  const endpoint = options.executorEndpoint ?? undefined;
  const up = endpoint ? await executorUp(host, endpoint) : true;
  log(`${machine}: ${watch.length} watched run(s), stage ${stage}${up ? '' : ', executor unreachable'}`);

  const snapshot: Snapshot = await collect(host, settings, repo, watch);
  const at = snapshot.at, cycle = `${machine}-${compactTime(at)}`;
  const day = dayOf(at), yesterday = dayOf(new Date(Date.parse(at) - 86_400_000).toISOString());
  const earlier = await readRecords(host, repo, settings.record_dir, machine, [yesterday, day]);
  const history: CycleMemory[] = memoryOf(earlier).slice(0, settings.history_cycles);
  const lastAt = earlier.at(-1)?.at ?? new Date(Date.parse(at) - settings.timer_interval_minutes * 60_000).toISOString();
  const agentActions = await deriveAgentActions(host, { repo, machine, since: lastAt, until: at, watch, ignoreUnits: settings.ignore_units.map(pattern => new RegExp(pattern)),
    ledgerPath: process.env.NATLANG_MEMORY_LEDGER ?? `${process.env.HOME ?? ''}/.local/state/natlang/memory-ledger.json`, recordDir: settings.record_dir });

  // Diagnoses: one call per watched run.
  const runs = await Promise.all(snapshot.runs.map(async (evidence: RunEvidence) => {
    const result = await decide('diagnoseRun', modes.diagnoseRun, up, crispDiagnose,
      (run: RunEvidence) => repairing(settings.repairs, feedback => options.run(() => diagnoseRun(run, feedback)), value => diagnosisProblems(value, run)),
      sameDiagnosis, value => `${value.health}/${value.cause}`, [evidence]);
    return { evidence, ...result };
  }));
  const diagnoses = runs.map(item => item.value);
  const watched = watch.map(entry => ({ run_id: entry.run_id, unit: entry.unit }));

  const triaged = await decide('triageInbox', modes.triageInbox, up, crispTriage,
    (messages: typeof snapshot.messages, machineName: string, list: typeof watched) => repairing(settings.repairs,
      feedback => options.run(() => triageInbox(messages, machineName as 'dgx' | 'pop', list, feedback)), value => triageProblems(value, messages, list)),
    sameTriage, value => `${value.obligations.length} obligation(s)`, [snapshot.messages, machine, watched] as [typeof snapshot.messages, string, typeof watched]);

  const nextSteps: Untrusted<string>[] = [];
  for (const entry of watch) if (entry.next_steps_doc) {
    const doc = await host.readText(resolveIn(repo, entry.next_steps_doc));
    if (doc) nextSteps.push(untrusted(`Next steps of run ${entry.run_id}:\n${doc.text.slice(0, 6000)}`, `next steps of ${entry.run_id}`));
  }
  const known: Known = { runs: watch.map(entry => entry.run_id), units: watch.map(entry => entry.unit), messages: snapshot.messages.map(message => message.id), machine,
    timerIntervalMinutes: settings.timer_interval_minutes, repo };
  const planArgs: PlanArgs = { diagnoses, triage: triaged.value, resources: snapshot.resources, next_steps: nextSteps, allowlist: defs.filter(def => def.proposable).map(summarize), history };
  const check = { defs, known, diagnoses, triage: triaged.value, resources: snapshot.resources, settings };
  let rejected: { proposal: unknown, problems: string[] }[] = [];
  const planned = await decide('planNext', modes.planNext, up, (args: PlanArgs) => crispPlan(args, { defs, settings, machine, repo, runs: snapshot.runs }),
    (args: PlanArgs) => repairing(settings.repairs, feedback => options.run(() => planNext(args.diagnoses, args.triage, args.resources, args.next_steps, args.allowlist, args.history, feedback)),
      plan => { const found = planProblems(plan, check); return [...found.proposals.flatMap(problems => problems ?? []), ...found.plan]; },
      plan => {
        // Past the repairs, proposals that pass the allowlist checks stay and the rest are dropped with their problems; the crisp summary replaces an unfit one.
        const found = planProblems(plan, check);
        rejected = plan.proposals.flatMap((proposal, index) => found.proposals[index] ? [{ proposal, problems: found.proposals[index]! }] : []);
        const idle = idleResource(snapshot.resources, settings.idle);
        return { proposals: plan.proposals.filter((_, index) => !found.proposals[index]), idle_resources: found.plan.length ? idle : plan.idle_resources,
          summary: found.plan.some(problem => problem.startsWith('summary')) ? crispPlan(args, { defs, settings, machine, repo, runs: snapshot.runs }).summary : plan.summary };
      }),
    samePlan, value => `${value.proposals.length} proposal(s)`, [planArgs]);
  const plan = planned.value;
  if (planned.note.via !== 'nl') rejected = [];

  // The program records every cycle. Everything else the allowlist lets run by itself at this stage.
  const recordAction = defs.find(def => def.id === 'record-heartbeat');
  const dispositions: Disposition[] = [];
  if (options.record !== false)
    dispositions.push({ proposal: { action: 'record-heartbeat', target: machine, params: {}, why: 'Every cycle is recorded.', cites: [], expected_effect: 'A line is appended to the heartbeat record.' },
      status: 'applied', owner: recordAction?.owner ?? 'program', reversible: true, reason: 'recorded', command: null, receipt: null });
  dispositions.push(...(options.record === false ? plan.proposals.map((proposal): Disposition => ({ proposal, status: 'proposed', owner: defs.find(def => def.id === proposal.action)?.owner ?? 'unknown',
    reversible: defs.find(def => def.id === proposal.action)?.reversible ?? null, reason: 'a dry run applies nothing', command: null, receipt: null })) : await applyPlan(host, plan, defs, known, settings, stage)));

  const record: CycleRecord = { schema: 'natlang.heartbeat-cycle/1', cycle, at, machine, stage, modes,
    degraded: { executor_unreachable: !up, natural_language_unavailable: parts.some(part => modes[part] !== 'crisp') &&
      [...runs.map(item => item.note), triaged.note, planned.note].every(note => note.via === 'crisp-degraded' || note.via === 'crisp-fallback') },
    readings: snapshot.readings.map(reading => ({ ...reading, untrusted: true as const })),
    runs: runs.map(item => ({ run_id: item.evidence.entry.run_id, unit: item.evidence.entry.unit, diagnosis: item.value, note: item.note })),
    triage: { ...triaged.value, note: triaged.note }, plan: { ...plan, note: planned.note, rejected },
    dispositions, unwatched: snapshot.resources.unwatched, warnings: loaded.warnings, agent_actions: agentActions };
  if (options.record !== false) {
    const path = await appendRecord(host, repo, settings.record_dir, record);
    const latest = await writeLatest(host, repo, settings.record_dir, record);
    log(`recorded ${path}; report ${latest}`);
  }
  return record;
}
