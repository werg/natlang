/**
 * `natlang run applications/specializer -- [--definition NAME|KEY] [--program DIR] [--loop] [--interval SECONDS]`
 *
 * Compiles recorded executions of natural-language functions into guarded crisp cases (plans/TRACE_SPECIALIZATION.md).
 * For each hot definition revision in the machine's call store: study its recorded calls and split them into groups by
 * what they did (crisp), let chooseCondition.nl and writeBody.nl write each group's case or say why not, verify the cases on the recorded calls
 * (replay plus judge), and store the accepted ones as the definition's compilation, where runtimes pick them up in shadow. Then run pending
 * shadow replays and audits. With --loop it repeats; every step can be interrupted and resumes from the store.
 */
import { resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { CallStore, Folder, pluggableMode, untrusted, TRACES_DECLARATIONS, reviewPromotions, assembleCases, createNatlangRuntime, crispDecline, groupsOf, machineStoreRoot, measure,
  betterFinding, detectFindings, renderFunction, renderGroup, renderReport, runJob, saveAccepted, study, tracesService, verifyCases, type CaseCheck, type DeclineReason,
  type Group, type HotDefinition, type NatlangRuntime, type Study } from '@natlang/node';
import chooseCondition from './chooseCondition.nl';
import writeBody from './writeBody.nl';
import decidePromotion from './promote.nl';
import worthLooking from './worthLooking.nl';
import summarizeDecline from './summarizeDecline.nl';
import writeGuidance from './writeGuidance.nl';
import { DEFAULTS, checkedGuidance, declinePolicy, groupLine, madeProgress, stateKey, worthPolicy } from './policy.js';
import type { BodyResult, ConditionChoice, DeclineGroup, GuidanceExample, WorthCandidate } from './types.js';

type Options = { definition?: string; program?: string; loop: boolean; interval: number; rounds: number; jobs: number; minCalls?: number;
  includeSelf: boolean; dryRun: boolean; jobsOnly: boolean; maxBusy: number; idleWait: number; maxGroups: number; guidance: boolean };

function parse(args: string[]): Options {
  const value = (name: string) => { const at = args.indexOf(name); return at >= 0 ? args[at + 1] : undefined; };
  return { definition: value('--definition'), program: value('--program') ? resolve(value('--program')!) : undefined,
    loop: args.includes('--loop'), interval: Number(value('--interval') ?? DEFAULTS.interval), rounds: Number(value('--rounds') ?? DEFAULTS.rounds),
    jobs: Number(value('--jobs') ?? DEFAULTS.jobs), minCalls: value('--min-calls') ? Number(value('--min-calls')) : undefined,
    includeSelf: args.includes('--include-self'), dryRun: args.includes('--dry-run'), jobsOnly: args.includes('--jobs-only'),
    maxBusy: Number(value('--max-busy') ?? DEFAULTS.maxBusy), idleWait: Number(value('--idle-wait') ?? DEFAULTS.idleWait),
    maxGroups: Number(value('--max-groups') ?? DEFAULTS.maxGroups), guidance: !args.includes('--no-guidance') };
}

/**
 * Definitions worth a look now, by the store's `targetPolicy`: `crisp` the count rule (policy.ts `crispWorth`), `nl`
 * worthLooking.nl per candidate, `shadow` both with the crisp answer serving. An explicitly named definition needs only
 * the minimum of agent calls. The natural-language side never lowers the crisp bound of `minCalls` agent calls.
 * `runtime` runs worthLooking.nl; it is needed only when the policy is not `crisp`.
 */
export async function targets(store: CallStore, options: Options, self: string | null, runtime?: NatlangRuntime): Promise<HotDefinition[]> {
  const minCalls = options.minCalls ?? store.settings().minCalls;
  const mode = pluggableMode(store.settings().targetPolicy, 'crisp');
  const policy = worthPolicy(mode, (candidate, min) => runtime!.run(() => worthLooking(candidate, min)) as Promise<never>);
  // By tokens: what serving a function crisply can save is what its agent calls cost.
  const hot = store.hot({ program: options.program, by: 'tokens', limit: 200 }).filter((item: HotDefinition) => {
    if (options.definition && ![item.definition_key, item.definition_id, item.definition_name, item.definition_source].includes(options.definition)) return false;
    return options.includeSelf || !self || item.program_root !== self;
  });
  const chosen: HotDefinition[] = [];
  for (const item of hot) {
    if (options.definition) { if (item.agent_calls >= minCalls) chosen.push(item); continue; }
    const decline = store.declineFor(item.definition_key), current = store.currentCompilation(item.definition_key);
    const candidate: WorthCandidate = { name: item.definition_name, agent_calls: item.agent_calls, tokens: item.tokens,
      new_calls_since_compilation: current ? store.calls({ key: item.definition_key, executor: 'agent', since: current.created_at, limit: minCalls }).length : 0,
      decline: decline ? { calls_at_decline: decline.calls_at_decline, reason: decline.reason } : null, has_compilation: !!current };
    if ((await policy(candidate, minCalls)).look) chosen.push(item);
  }
  return chosen;
}

/**
 * One writer call's budget. Actions bound the work; the time bound only catches a stalled call, since a shared executor
 * can be slow for hours. A group whose writer runs out is left to the function.
 */
const WRITER_LIMITS = { maxActions: 150, timeoutMs: 2 * 60 * 60_000 };
/** Writer calls running at once for one function. */
const WRITERS = 2;

const GROUP_DECLARATIONS = `/** Exactly how a condition (a JavaScript expression over args) does on the recorded training calls: how many of this
 * group's calls it admits (ofGroup of groupSize), how many calls of other groups (others), and a few of those. */
export function measure(condition: string): { valid: boolean; error?: string; ofGroup: number; groupSize: number; others: number;
  counterexamples: { call: string; did: string; inputs: unknown }[] };
`;

/** Whether a writer call stopped on its budget (actions, episodes or time), not on a failure around it. */
function exhausted(error: unknown): boolean {
  const outcome = (error as { outcome?: string } | null)?.outcome;
  const message = error instanceof Error ? error.message : String(error);
  return outcome === 'budget' || /budget exhausted|timed out/.test(message) && !/ECONN|fetch failed|model request/.test(message);
}

/** Tokens a model driver used, counted as it goes (the specializer's spend per function). */
type Meter = { tokens: number };
type ModelOption = NonNullable<Parameters<typeof createNatlangRuntime>[0]>['model'];
function metered(model: ModelOption, meter: Meter): ModelOption {
  if (!model) return model;
  const config = typeof model === 'function' ? { driver: model } : { ...model };
  const inner = config.driver;
  const driver = async (...args: Parameters<typeof inner>) => {
    const turn = await inner(...args);
    meter.tokens += (turn.prompt_tokens ?? 0) + (turn.completion_tokens ?? 0);
    return turn;
  };
  // Everything the driver tells the runtime stays: its model, its context window (compaction depends on it), its name.
  for (const name of new Set([...Object.getOwnPropertyNames(inner), ...Object.getOwnPropertyNames(Object.getPrototypeOf(inner) ?? {})])) {
    if (['length', 'prototype', 'arguments', 'caller', 'constructor', 'apply', 'call', 'bind', 'toString'].includes(name)) continue;
    const value = (inner as unknown as Record<string, unknown>)[name];
    Object.defineProperty(driver, name, { value: typeof value === 'function' ? value.bind(inner) : value, configurable: true });
  }
  return { ...config, driver: driver as typeof inner };
}

/** The runtime verification and offline jobs judge in, metered. */
function judgeRuntime(context: TargetContext, meter: Meter): NatlangRuntime {
  return createNatlangRuntime({ model: metered(context.model, meter), executorIdentity: context.executorIdentity, programRoot: context.package?.root });
}

/**
 * Wait until the executor is not busy (vLLM's running plus waiting requests at most `maxBusy`), so specializing does not
 * compete with the programs it serves. Gives up waiting after `idleWait` seconds and goes ahead; never waits when the
 * endpoint has no metrics.
 */
async function waitForIdle(context: TargetContext, options: Options, log: (line: string) => void, stopping: () => boolean): Promise<void> {
  const endpoint = (context.executorIdentity?.configuration as { endpoint?: unknown } | undefined)?.endpoint;
  if (typeof endpoint !== 'string' || options.idleWait <= 0) return;
  const url = new URL('/metrics', endpoint).href;
  const busy = async (): Promise<number | undefined> => {
    try {
      const text = await (await fetch(url, { signal: AbortSignal.timeout(5000) })).text();
      let total = 0, found = false;
      for (const line of text.split('\n')) {
        const match = /^vllm:num_requests_(running|waiting)\{[^}]*\} ([\d.]+)$/.exec(line);
        if (match) { total += Number(match[2]); found = true; }
      }
      return found ? total : undefined;
    } catch { return undefined; }
  };
  const started = Date.now();
  let said = false;
  for (let load = await busy(); load !== undefined && load > options.maxBusy && !stopping(); load = await busy()) {
    if (Date.now() - started > options.idleWait * 1000) { log(`the executor is still busy (${load} requests); going ahead`); return; }
    if (!said) { log(`waiting for the executor to be idle (${load} requests, at most ${options.maxBusy})`); said = true; }
    await new Promise(done => setTimeout(done, 30_000));
  }
}

/** The runtime one group's writer runs in: the launcher's model, the store as `traces`, the group's exact `measure`. */
function writerRuntime(context: TargetContext, store: CallStore, subject: Study, group: Group, meter: Meter): NatlangRuntime {
  return createNatlangRuntime({ model: metered(context.model, meter), executorIdentity: context.executorIdentity, programRoot: context.package?.root,
    services: { traces: tracesService(store), group: { measure: (condition: string) => measure(subject, group, String(condition)) } },
    serviceDeclarations: { traces: TRACES_DECLARATIONS, group: GROUP_DECLARATIONS }, limits: WRITER_LIMITS });
}

type Written = { kind: 'case'; text: string } | { kind: 'skip'; reason: string; why: string } | { kind: 'budget'; why: string } | { kind: 'report'; report: string } | { kind: 'empty' };

/** One group's writer call over its folder. */
async function writeGroup(context: TargetContext, store: CallStore, subject: Study, group: Group, functionMd: string, meter: Meter, report?: string): Promise<Written> {
  const workspace = Folder.fromFiles(renderGroup(store, subject, group, functionMd, { report }));
  const runtime = writerRuntime(context, store, subject, group, meter);
  const budget = (error: unknown): Written => {
    if (!exhausted(error)) throw error;
    return { kind: 'budget', why: error instanceof Error ? error.message.slice(0, 300) : String(error) };
  };
  // 1. chooseCondition decides the condition; the host measures it exactly, so a claim of the model never stands in for the measure.
  let choice: ConditionChoice;
  try { choice = await runtime.run(() => workspace.root().apply(chooseCondition, subject.definition.name, group.id)) as ConditionChoice; }
  catch (error) { return budget(error); }
  if (choice.kind === 'skip') return { kind: 'skip', reason: choice.reason, why: choice.why };
  const measured = measure(subject, group, String(choice.condition));
  if (!measured.valid || measured.others > 0 || measured.ofGroup < 1)
    return { kind: 'report', report: '# Report\n\n' + (!measured.valid ? `The condition is not a valid expression over args: ${measured.error ?? 'it did not evaluate'}.\n` :
      `The condition admits ${measured.ofGroup} of this group's calls and ${measured.others} calls of other groups. A condition for a case admits at least one of this group's calls and none of the others.\n`) };
  // 2. writeBody writes case.ts for that condition.
  let body: BodyResult;
  try { body = await runtime.run(() => workspace.root().apply(writeBody, subject.definition.name, group.id, choice.condition)) as BodyResult; }
  catch (error) { return budget(error); }
  if (body.kind === 'skip') return { kind: 'skip', reason: body.reason, why: body.why };
  if (!workspace.filePaths().includes('case.ts')) return { kind: 'empty' };
  const text = new TextDecoder().decode(workspace.readBytesSync('case.ts'));
  return text.trim() ? { kind: 'case', text } : { kind: 'empty' };
}

/** Run `work` over `items`, at most `limit` at a time, in order of completion. */
async function pool<T>(items: readonly T[], limit: number, work: (item: T) => Promise<void>): Promise<void> {
  const queue = [...items];
  await Promise.all(Array.from({ length: Math.min(limit, queue.length) }, async () => { for (let item = queue.shift(); item !== undefined; item = queue.shift()) await work(item); }));
}

const DECLINE_FOR: Record<string, DeclineReason> = { semantic: 'semantic', unstable: 'unstable', effects: 'effects', 'no-condition': 'no-clusters', budget: 'budget' };

/**
 * Specialize one definition revision. The host splits its calls into groups by what they did; each group gets a writer
 * call that writes that group's case or says why not. Each round, the cases written so far are assembled and verified
 * together; a case that fails goes back to its group with its report, up to `rounds` attempts per group.
 */
export async function specializeOne(context: TargetContext, store: CallStore, key: string, options: Options): Promise<string> {
  const meter: Meter = { tokens: 0 }, started = Date.now();
  try { return await specializeMetered(context, store, key, options, meter); }
  finally { if (!options.dryRun) store.spent(key, 'specialize', meter.tokens, Date.now() - started); }
}

async function specializeMetered(context: TargetContext, store: CallStore, key: string, options: Options, meter: Meter): Promise<string> {
  const judge = judgeRuntime(context, meter);
  const minCalls = options.minCalls ?? store.settings().minCalls;
  const subject = study(store, key);
  if (!subject) return `${store.calls({ key, limit: 1, audits: true })[0]?.definition_name ?? key}: no usable calls (none from a model that spent tokens)`;
  const name = subject.definition.name;
  const decline = (reason: DeclineReason, why: string) => {
    if (!options.dryRun) store.decline({ definitionKey: key, definitionId: subject.definition.id, reason, why, calls: subject.examples.length });
  };
  const crisp = crispDecline(subject, minCalls);
  if (crisp) { decline(crisp.reason, crisp.why); return `${name}: declined without a model (${crisp.reason}: ${crisp.why})`; }
  if (!options.dryRun) for (const finding of detectFindings(store, subject)) store.finding(finding);
  const groups = groupsOf(subject, { max: options.maxGroups });
  if (!groups.length) { decline('no-clusters', 'no group of calls that did the same thing is large enough'); return `${name}: declined without a model (no-clusters)`; }
  const functionMd = renderFunction(subject);
  type State = { group: Group; attempts: number; part?: string; report?: string; done?: 'accepted' | 'skipped' | 'budget'; why?: string; reason?: string };
  const states = groups.map((group): State => ({ group, attempts: 0 }));
  let checks: CaseCheck[] = [], verified: { id: string; text: string }[] = [], report = '';
  // The repair loop: `rounds` is the resource limit; the loop also ends at a fixed point, when the last round left every
  // group's outcome, report and case as they were (policy.ts `madeProgress`).
  let roundStart: string[] | undefined;
  for (let round = 1; round <= options.rounds; round++) {
    const pending = states.filter(state => !state.done && state.attempts < options.rounds);
    if (!pending.length) break;
    const keys = states.map(stateKey);
    if (roundStart && !madeProgress(roundStart, keys)) break;
    roundStart = keys;
    await pool(pending, WRITERS, async state => {
      state.attempts++;
      const written = await writeGroup(context, store, subject, state.group, functionMd, meter, state.report);
      if (written.kind === 'case') { state.part = written.text; return; }
      state.part = undefined;
      if (written.kind === 'skip') Object.assign(state, { done: 'skipped', reason: written.reason, why: written.why });
      else if (written.kind === 'budget') Object.assign(state, { done: 'budget', reason: 'budget', why: written.why });
      else state.report = '# Report\n\nThere is no case.ts. Write it, or return kind "skip" with a reason.\n';
    });
    const parts = states.filter(state => state.part).map(state => ({ id: state.group.id, text: state.part! }));
    if (!parts.length) continue;
    const assembled = assembleCases(parts);
    for (const error of assembled.errors) {
      const state = states.find(item => item.group.id === error.id)!;
      state.part = undefined;
      state.report = `# Report\n\ncase.ts does not have the expected shape: ${error.error}.\n`;
    }
    const verification = await verifyCases(judge, store, subject, assembled.text, { bound: store.settings().acceptanceBound });
    if (verification.error) {
      // The parts load together or not at all; check each alone to find the one that does not load.
      for (const part of parts) {
        const alone = assembleCases([part]);
        const single = await verifyCases(judge, store, subject, alone.text, { bound: 0, maxJudged: 0 });
        if (single.error) { const state = states.find(item => item.group.id === part.id)!; state.part = undefined; state.report = `# Report\n\ncase.ts does not load: ${single.error}\n`; }
      }
      continue;
    }
    assembled.included.forEach((id, position) => {
      const state = states.find(item => item.group.id === id)!;
      const check = verification.checks[position];
      if (!check) return;
      const better = betterFinding(subject, state.group.label, check.results.filter(result => result.verdict === 'better').map(result => result.callId));
      if (better && !options.dryRun) store.finding(better);
      if (check.accepted) state.done = 'accepted';
      else { state.part = undefined; state.report = renderReport(store, subject, { checks: [check] }); }
    });
    checks = verification.checks; verified = parts.filter(part => assembled.included.includes(part.id)); report = renderReport(store, subject, verification);
  }
  const accepted = states.filter(state => state.done === 'accepted' && state.part).map(state => ({ id: state.group.id, text: state.part! }));
  const skipped = states.filter(state => state.done !== 'accepted');
  const outcomes: DeclineGroup[] = states.map(state => ({ id: state.group.id, label: state.group.label, calls: state.group.training.length,
    outcome: state.done ?? 'not accepted', reason: state.reason ?? null, why: state.why ?? null }));
  const summary = outcomes.filter(group => group.outcome !== 'accepted').map(groupLine);
  for (const state of skipped.filter(item => item.reason === 'unstable'))
    store.finding({ definitionKey: key, definitionId: subject.definition.id, definitionName: name, definitionSource: subject.definition.source ?? null,
      kind: 'unstable', summary: `the executor's results disagree for calls that did ${state.group.label}`, detail: { group: state.group.label, why: state.why,
        calls: state.group.training.slice(0, 8).map(example => example.callId) } });
  if (options.dryRun) return `${name}: ${accepted.length} of ${groups.length} groups would get a case\n${report}\n${summary.join('\n')}`;
  if (accepted.length) {
    // The accepted cases alone, verified once more: a rejected case earlier in the file can no longer take their calls.
    const final = assembleCases(accepted);
    const verification = await verifyCases(judge, store, subject, final.text, { bound: store.settings().acceptanceBound });
    // Tier 2: short guidance for the student, distilled from the traces the cases were built from (instructions.md).
    const guidance = verification.error || !options.guidance ? undefined : await guidanceFor(judge, store, subject, outcomes, states.filter(state => state.done === 'accepted').map(state => state.group));
    const saved = verification.error ? undefined : saveAccepted(store, subject, final.text, verification.checks, renderReport(store, subject, verification),
      { rounds: options.rounds, groups: outcomes }, guidance ? { 'instructions.md': guidance } : {});
    if (saved) return `${name}: compilation ${saved.id} with ${saved.cases} case(s), in shadow${summary.length ? `; left to the function: ${summary.join('; ')}` : ''}`;
  }
  const reasons = skipped.map(state => DECLINE_FOR[state.reason ?? ''] ?? 'no-clusters');
  const reason = reasons.sort((a, b) => reasons.filter(item => item === b).length - reasons.filter(item => item === a).length)[0] ?? 'no-clusters';
  const mode = pluggableMode(store.settings().declinePolicy, 'crisp');
  const text = declinePolicy(mode, groups => judge.run(() => summarizeDecline(groups)) as Promise<string>);
  decline(reason, await (mode === 'crisp' ? text(outcomes.filter(group => group.outcome !== 'accepted')) :
    judge.run(() => text(outcomes.filter(group => group.outcome !== 'accepted')))));
  void checks; void verified;
  return `${name}: declined (${reason}): ${summary.join('; ')}`;
}

/**
 * The compilation's `instructions.md`: writeGuidance.nl distils the accepted groups' recorded calls into short guidance
 * for the student; the crisp checks (length, positive wording) decide whether it is kept. One repair attempt; none
 * stored when the second text fails too, so tier 2 appears only with guidance that passed. A failed model call stores nothing.
 */
async function guidanceFor(judge: NatlangRuntime, store: CallStore, subject: Study, outcomes: DeclineGroup[], accepted: Group[]): Promise<string | undefined> {
  const examples: GuidanceExample[] = accepted.flatMap(group => group.training.slice(0, 2).map(example => {
    const record = subject.records.get(example.callId);
    const cell = (value: unknown) => untrusted((JSON.stringify(value) ?? 'null').slice(0, 600), 'recorded call');
    return { did: example.behavior ?? 'none', input: cell(example.args), result: cell(record ? store.value(record.output) : null) };
  })).slice(0, 8);
  try { return await checkedGuidance(feedback => judge.run(() => writeGuidance(subject.definition.body, outcomes, examples, feedback)) as Promise<string>); }
  catch { return undefined; }
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
    for (const target of options.jobsOnly ? [] : await targets(store, options, self, judgeRuntime(context, { tokens: 0 }))) {
      if (stopping) break;
      await waitForIdle(context, options, log, () => stopping);
      try { log(await specializeOne(context, store, target.definition_key, options)); }
      catch (error) { log(`${target.definition_name}: failed: ${error instanceof Error ? error.stack ?? error.message : String(error)}`); }
    }
    if (!stopping && !options.dryRun) {
      let ran = 0;
      for (const job of store.pendingJobs(options.jobs)) {
        if (stopping) break;
        await waitForIdle(context, options, log, () => stopping);
        const meter: Meter = { tokens: 0 }, started = Date.now();
        try {
          const result = await runJob(judgeRuntime(context, meter), store, job);
          store.finishJob(job.id, result.status, result.verdict, result.detail);
          log(`${job.kind} ${job.case_hash} on ${job.call_id}: ${result.status}${result.verdict ? ` ${result.verdict}` : ''}${result.detail ? ` (${result.detail})` : ''}`);
        } catch (error) {
          store.finishJob(job.id, 'failed', null, error instanceof Error ? error.message : String(error));
          log(`${job.kind} ${job.case_hash} on ${job.call_id}: failed: ${error instanceof Error ? error.message : String(error)}`);
        }
        const key = store.call(job.call_id)?.definition.key;
        if (key) store.spent(key, job.kind, meter.tokens, Date.now() - started);
        ran++;
      }
      if (ran) log(`${ran} shadow/audit job(s) done`);
    }
    // Promotion and demotion by the store's policy, off the hot path; under `crisp` the store already applied its rule.
    if (!stopping && !options.dryRun && store.settings().promotionPolicy !== 'crisp') {
      try {
        const meter: Meter = { tokens: 0 }, runtime = judgeRuntime(context, meter);
        const reviewed = await reviewPromotions(store, summary => runtime.run(() => decidePromotion(JSON.stringify(summary))) as Promise<never>);
        for (const item of reviewed.filter(entry => entry.applied || entry.decision.decision !== 'keep'))
          log(`promotion ${item.subject} ${item.id} (${item.state}): ${item.decision.decision}${item.applied ? ' applied' : ''}: ${item.decision.reason}`);
      } catch (error) { log(`promotion review failed: ${error instanceof Error ? error.message : String(error)}`); }
    }
    if (!options.loop || stopping) break;
    for (let waited = 0; waited < options.interval && !stopping; waited++) await new Promise(done => setTimeout(done, 1000));
  } while (!stopping);
  return 0;
}
