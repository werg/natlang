/** Native search loop derived from Ax GEPA at b780a14a3cb94d5ac572db04038399aef655c76c.
 * Modified for natlang contracts, mandatory native evaluation, budgets and durable runs.
 * Apache-2.0 attribution and extraction inventory: vendor/ax-gepa/UPSTREAM.json. */
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { fingerprint, canonical } from '../adaptation/identity.js';
import { validateCandidate } from '../adaptation/compatibility.js';
import type { Candidate } from '../adaptation/types.js';
import type { EvaluationBatch, PreparedSuite } from '../evaluation/types.js';
import { evaluate, candidateArtifact } from '../evaluation/runner.js';
import { validateBudget } from '../evaluation/suite.js';
import { trainingFeedback, EvaluationFeedbackError } from '../evaluation/feedback.js';
import { UsageGateway, BudgetExhausted } from '../evaluation/usage.js';
import { pairedChanges, evaluationSummary } from '../evaluation/report.js';
import { summarize } from '../evaluation/metrics.js';
import { propose } from './proposer.js';
import { RunStore } from './run-store.js';
import { ENGINE_VERSION, readCheckpoint, saveCheckpoint } from './checkpoint.js';
import { frontierParents, mergeCandidates, meanBetter, selectionEligible } from './strategies/gepa.js';
import { ComponentSelector } from './vendor/ax-gepa/gepaSelection.js';
import { getUpdateGroup } from './vendor/ax-gepa/gepaDependencies.js';
import { promote } from './promotion.js';
import { optimizationMarkdown } from './report.js';
import { exportAdaptationPatch } from './export-patch.js';
import type { OptimizationOptions, OptimizationResult, OptimizationTarget, SearchCandidate, SearchState } from './types.js';
function nativeTarget(prepared: PreparedSuite, options: OptimizationOptions): OptimizationTarget {
  const components = prepared.program.components.filter(component => prepared.components.includes(component.key));
  return { components: () => components,
    validate: async candidate => { try {
      validateCandidate(candidate, components);
      // Compile an escaped AST projection as well as checking portable contracts.
      exportAdaptationPatch(candidateArtifact(prepared, candidate), prepared.program, prepared.suite.executorIdentity, prepared.suite.program.root);
      return { valid: true }; }
      catch (error) { return { valid: false, feedback: String(error) }; } },
    evaluate: (candidate, caseIds, context) => evaluate(prepared, { driver: options.executor, judge: options.judge, candidate, caseIds,
      signal: context.signal, gateway: context.gateway, runId: context.runId, seed: context.seed }),
    feedback: async (batch, keys) => trainingFeedback(prepared, batch, keys) };
}
export async function optimize(prepared: PreparedSuite, options: OptimizationOptions): Promise<OptimizationResult> {
  return search(prepared, options, false);
}
export async function resumeOptimization(prepared: PreparedSuite, directory: string, options: OptimizationOptions): Promise<OptimizationResult> {
  return search(prepared, { ...options, out: directory }, true);
}
async function search(prepared: PreparedSuite, options: OptimizationOptions, resume: boolean): Promise<OptimizationResult> {
  const strategy = options.strategy ?? 'gepa', seed = options.seed ?? 0;
  const limits = options.budget ?? prepared.suite.budget;
  validateBudget(limits);
  const settings = { strategy, seed, limits, minibatchSize: options.minibatchSize ?? 4, maxPopulation: options.maxPopulation ?? 16,
    maxHistory: options.maxHistory ?? 1000,
    maxRepairs: options.maxRepairs ?? 1, dependencies: options.dependencies ?? {}, finalTest: options.finalTest ?? true,
    reflectionIdentity: options.reflectionIdentity ?? null, judgeIdentity: options.judgeIdentity ?? null,
    selection: prepared.suite.selection ?? null, holdoutReservation: options.holdoutReservation ?? {} };
  if (!Number.isSafeInteger(seed) || !Number.isSafeInteger(settings.minibatchSize) || settings.minibatchSize < 1 ||
    !Number.isSafeInteger(settings.maxPopulation) || settings.maxPopulation < 2 ||
    !Number.isSafeInteger(settings.maxHistory) || settings.maxHistory < 1 ||
    !Number.isSafeInteger(settings.maxRepairs) || settings.maxRepairs < 0) throw new Error('invalid search settings');
  const optionsHash = fingerprint(settings);
  const store = new RunStore(options.out ?? join(prepared.suite.program.root, '.natlang/adaptation/runs', options.runId ?? randomUUID()));
  let state: SearchState | undefined;
  const target = options.target ?? nativeTarget(prepared, options);
  const descriptors = target.components();
  const train = prepared.cases.filter(testCase => testCase.split === 'train').map(testCase => testCase.id).sort();
  const validation = prepared.cases.filter(testCase => testCase.split === 'validation').map(testCase => testCase.id).sort();
  const test = prepared.cases.filter(testCase => testCase.split === 'test').map(testCase => testCase.id).sort();
  if (!train.length || !validation.length) { store.close(); throw new Error('optimization requires independent train and validation cases'); }
  let gateway: UsageGateway;
  let attachedGateway: UsageGateway | undefined;
  try {
    if (canonical(descriptors.map(component => component.key).sort()) !== canonical([...prepared.components].sort())) throw new Error('optimization target must retain the prepared component inventory');
    for (const [key, dependencies] of Object.entries(settings.dependencies)) {
      if (!prepared.components.includes(key) || dependencies.some(dependency => !prepared.components.includes(dependency))) throw new Error('unknown update dependency: ' + key);
    }
    if (resume && store.has('checkpoint.json')) state = readCheckpoint(store, prepared.suiteHash, prepared.program.buildHash, optionsHash);
    if (state && !state.selected) delete state.stopReason;
    if (!resume && store.has('checkpoint.json')) throw new Error('run already exists; use resumeOptimization');
    store.manifest({ schema: 'natlang.adaptation-run/v1', engine: ENGINE_VERSION, suiteHash: prepared.suiteHash,
      programHash: prepared.program.buildHash, settings, components: descriptors.map(component => component.key),
      splits: { train, validation, test }, executor: prepared.suite.executorIdentity });
    gateway = new UsageGateway(limits, resume && store.has('ledger.json') ? store.read('ledger.json') : state?.ledger);
    if (settings.finalTest && !state?.selected) gateway.protect({ modelCalls: test.length, ...settings.holdoutReservation });
    attachedGateway = gateway;
    gateway.onUpdate = ledger => store.write('ledger.json', ledger);
    const batch = async (candidate: Candidate, ids: readonly string[]): Promise<EvaluationBatch> => {
      options.signal?.throwIfAborted();
      const key = fingerprint({ candidate, ids, seed, suite: prepared.suiteHash, program: prepared.program.buildHash,
        executor: prepared.suite.executorIdentity, policy: prepared.suite.policy ?? null, trace: 'complete' });
      const cacheFile = 'cache-' + key + '.json';
      if (store.has(cacheFile)) {
        const cached = store.read<EvaluationBatch>(cacheFile);
        if (cached.digest !== summarize(cached.results).digest) throw new Error('evaluation cache integrity mismatch: ' + key);
        store.event({ type: 'cache-hit', key }); return cached;
      }
      store.event({ type: 'evaluation-started', key, ids });
      const holdoutReserve = state?.selected || !settings.finalTest ? 0 : test.length;
      if (gateway.ledger.rollouts + ids.length + holdoutReserve > limits.maxRollouts) throw new BudgetExhausted('rollouts (holdout reserved)');
      const result = await target.evaluate(candidate, ids, { runId: state?.runId ?? options.runId ?? store.directory,
        evaluationId: key, seed, signal: options.signal, gateway });
      if (result.results.length !== ids.length || new Set(result.results.map(result => result.caseId)).size !== ids.length ||
          ids.some(id => !result.results.some(result => result.caseId === id)) || result.results.some(result => result.status !== 'scored' ||
            result.quality === null || !Number.isFinite(result.quality) || result.quality < 0 || result.quality > 1))
        throw new Error('evaluator returned invalid/incomplete results');
      const normalized = summarize(result.results); store.write(cacheFile, normalized); store.blob(normalized); store.event({ type: 'evaluation-completed', key }); return normalized;
    };
    if (!state) {
      const baselineValue = validateCandidate(Object.fromEntries(descriptors.map(component => [component.key, component.baseline])), descriptors);
      const baseline: SearchCandidate = { id: fingerprint(baselineValue), value: baselineValue, parents: [],
        train: await batch(baselineValue, train), validation: await batch(baselineValue, validation) };
      state = { schema: 'natlang.adaptation-run/v1', engine: ENGINE_VERSION, runId: options.runId ?? store.directory.split('/').at(-1)!,
        suiteHash: prepared.suiteHash, programHash: prepared.program.buildHash, optionsHash, strategy, seed,
        rng: seed >>> 0 || 0x9e3779b9, iteration: 0, population: [baseline], baseline, incumbent: baseline.id,
        selector: {}, history: [], ledger: gateway.snapshot(), selected: false };
      saveCheckpoint(store, state);
    }
    const random = () => { let x = state!.rng; x ^= x << 13; x ^= x >>> 17; x ^= x << 5; state!.rng = x >>> 0; return state!.rng / 4294967296; };
    const targets = descriptors.map(component => ({ key: component.key, dependsOn: settings.dependencies[component.key] ?? [] }));
    for (const component of targets) for (const dependency of component.dependsOn) if (!targets.some(target => target.key === dependency)) throw new Error('unknown update dependency: ' + dependency);
    let selector = new ComponentSelector(targets, state.selector);
    const checkpoint = () => { state!.selector = selector.snapshot(); state!.ledger = gateway.snapshot(); saveCheckpoint(store, state!); };
    const emit = (event: Record<string, unknown>) => {
      state!.history.push(event);
      if (state!.history.length > settings.maxHistory) {
        const removed = state!.history.length - settings.maxHistory;
        state!.history.splice(0, removed); state!.historyTruncated = (state!.historyTruncated ?? 0) + removed;
      }
      store.event(event); options.progress?.(event);
    };
    if (!state.selected) {
      try {
        while (gateway.ledger.proposals < limits.maxProposals) {
          gateway.check(options.signal);
          const covered = new Set(state.population.flatMap(candidate => candidate.train.results.flatMap(result => [...result.coverage])));
          const eligible = targets.filter(target => covered.has(target.key));
          if (!eligible.length) { state.stopReason = 'coverage-gap'; break; }
          const winners = strategy === 'gepa' ? frontierParents(state.population) : [state.incumbent];
          const parentId = winners[Math.floor(random() * winners.length)] ?? state.incumbent;
          const parent = state.population.find(candidate => candidate.id === parentId)!;
          let selected = selector.pick(state.iteration, random);
          if (!eligible.some(component => component.key === selected.key)) selected = eligible[Math.floor(random() * eligible.length)]!;
          const keys = getUpdateGroup(selected, targets).map(component => component.key);
          gateway.reserve('proposals', 1, options.signal); selector.recordProposal(selected.key);
          const iteration = state.iteration++;
          let proposal: Candidate | undefined;
          try {
            // Every fourth iteration attempts compatible composition before reflection.
            if (strategy === 'gepa' && iteration % 4 === 3 && state.population.length > 1) {
              const partner = state.population[Math.floor(random() * state.population.length)]!;
              proposal = mergeCandidates(state.baseline.value, parent.value, partner.value) ?? undefined;
            }
            proposal ??= await propose({ components: descriptors, candidate: parent.value, keys,
              feedback: await target.feedback(parent.train, keys), driver: options.reflection, gateway,
              seed: seed + iteration, signal: options.signal, maxRepairs: settings.maxRepairs });
            const valid = await target.validate(proposal); if (!valid.valid) throw new Error('invalid candidate: ' + valid.feedback);
          } catch (error) {
            if (error instanceof BudgetExhausted || error instanceof EvaluationFeedbackError || options.signal?.aborted) throw error;
            selector.recordResult(selected.key, false, iteration); emit({ iteration, type: 'invalid-proposal', error: String(error) }); checkpoint(); continue;
          }
          const id = fingerprint(proposal);
          if (state.population.some(candidate => candidate.id === id)) { selector.recordResult(selected.key, false, iteration); checkpoint(); continue; }
          const shuffled = [...train]; for (let index = shuffled.length - 1; index > 0; index--) {
            const chosen = Math.floor(random() * (index + 1)); [shuffled[index], shuffled[chosen]] = [shuffled[chosen]!, shuffled[index]!]; }
          const mini = shuffled.slice(0, settings.minibatchSize).sort();
          const parentBatch = await batch(parent.value, mini), proposedBatch = await batch(proposal, mini);
          const accepted = proposedBatch.gatesPassed && proposedBatch.quality! > parentBatch.quality!;
          selector.recordResult(selected.key, accepted, iteration);
          if (accepted) {
            const candidate: SearchCandidate = { id, value: proposal, parents: [parent.id],
              train: await batch(proposal, train), validation: await batch(proposal, validation) };
            state.population.push(candidate);
            const incumbent = state.population.find(item => item.id === state!.incumbent)!;
            const guidanceChanged = descriptors.some(component => component.kind === 'program.guidance' &&
              canonical(candidate.value[component.key]) !== canonical(component.baseline));
            const requiredCoverage = prepared.program.components.filter(component => component.kind === 'lambda.instructions');
            const missingGuidanceCoverage = guidanceChanged ? requiredCoverage.filter(component =>
              !candidate.validation.results.some(result => result.coverage.includes(component.key))).map(component => component.key) : [];
            if (missingGuidanceCoverage.length) emit({ iteration, type: 'insufficient-guidance-coverage', components: missingGuidanceCoverage });
            else if (meanBetter(candidate, incumbent, prepared.suite.selection)) state.incumbent = candidate.id;
            // Protect incumbent and baseline while preserving validation case winners.
            if (state.population.length > settings.maxPopulation) {
              const frontier = new Set(frontierParents(state.population));
              const removable = state.population.filter(item => item.id !== state!.incumbent && item.id !== state!.baseline.id)
                .sort((a, b) => Number(frontier.has(a.id)) - Number(frontier.has(b.id)) || (a.validation.quality ?? 0) - (b.validation.quality ?? 0) || a.id.localeCompare(b.id));
              const remove = removable[0]; if (remove) state.population = state.population.filter(item => item.id !== remove.id);
            }
          }
          emit({ iteration, type: accepted ? 'accepted' : 'rejected', candidate: id, parent: parent.id, keys,
            pairedQuality: proposedBatch.quality, parentQuality: parentBatch.quality }); checkpoint();
        }
        state.stopReason ??= 'completed';
      } catch (error) {
        if (options.signal?.aborted || error instanceof BudgetExhausted) {
          // Restore the last complete decision; charged in-flight usage lives in the separate ledger.
          const committed = readCheckpoint(store, prepared.suiteHash, prepared.program.buildHash, optionsHash);
          Object.assign(state, committed); selector = new ComponentSelector(targets, committed.selector);
        }
        if (options.signal?.aborted) state.stopReason = 'cancelled';
        else if (error instanceof BudgetExhausted) state.stopReason = 'budget-exhausted';
        else { state.stopReason = 'infrastructure-error'; checkpoint(); throw error; }
      }
      checkpoint();
      if (state.stopReason !== 'cancelled') { state.selected = true; checkpoint(); }
    }
    if (state.selected) gateway.protect({});
    // Freeze the incumbent before exposing any holdout result. Test never feeds proposals or selection.
    if (state.selected && settings.finalTest && test.length && !state.lockedTest) {
      try { state.lockedTest = await batch(state.population.find(candidate => candidate.id === state!.incumbent)!.value, test); checkpoint(); }
      catch (error) { if (!(error instanceof BudgetExhausted) && !options.signal?.aborted) throw error; emit({ type: 'holdout-incomplete', reason: String(error) }); checkpoint(); }
    }
    const selected = state.population.find(candidate => candidate.id === state!.incumbent)!;
    const uncovered = descriptors.filter(component => !state!.population.some(candidate => candidate.train.results.some(result => result.coverage.includes(component.key)))).map(component => component.key);
    if (uncovered.length && state.stopReason === 'completed') { state.stopReason = 'coverage-gap'; checkpoint(); }
    const holdoutComplete = !settings.finalTest || !test.length || !!state.lockedTest;
    const holdoutPassed = state.lockedTest?.gatesPassed ?? true;
    const finalized = state.selected && holdoutComplete && holdoutPassed;
    const artifact = promote(prepared, state, finalized);
    const report = { runId: state.runId, strategy, stopReason: state.stopReason, improvement: selected.validation.quality! - state.baseline.validation.quality!,
      baseline: state.baseline.validation, selectedValidation: selected.validation, lockedTest: state.lockedTest ?? null,
      pairedChanges: pairedChanges(state.baseline.validation, selected.validation),
      summaries: { baseline: evaluationSummary(state.baseline.validation, prepared.cases), validation: evaluationSummary(selected.validation, prepared.cases),
        lockedTest: state.lockedTest ? evaluationSummary(state.lockedTest, prepared.cases) : null },
      promptLengths: Object.fromEntries(Object.entries(selected.value).map(([key, value]) => [key, value.kind === 'program.guidance' ? value.text.length : value.template.segments.join('').length])),
      uncovered, ledger: state.ledger, history: state.history, historyTruncated: state.historyTruncated ?? 0, executor: prepared.suite.executorIdentity,
      reflection: settings.reflectionIdentity, judge: settings.judgeIdentity,
      randomness: { searchSeed: seed, caseSeeds: 'case ID and replicate', modelSeeding: 'Requested; provider support is caller declared.' },
      instructions: { baseline: state.baseline.value, selected: selected.value }, compatibility: 'validated',
      promotion: artifact.provenance.promotion, holdoutComplete,
      holdoutStatus: !settings.finalTest ? 'disabled' : !test.length ? 'not-configured' :
        state.lockedTest ? holdoutPassed ? 'passed' : 'failed' : 'incomplete' };
    if (selectionEligible(selected, prepared.suite.selection) && holdoutPassed) store.write(finalized ? 'selected.json' : 'incumbent.json', artifact); store.write('report.json', report);
    store.writeText('report.md', optimizationMarkdown(report));
    if (!selectionEligible(selected, prepared.suite.selection)) throw new Error('evaluation regression: no candidate passes required validation gates and selection constraints');
    if (!holdoutPassed) throw new Error('evaluation regression: frozen finalist failed locked test gates; no artifact was promoted');
    return { artifact, report, state, directory: store.directory };
  } catch (error) {
    store.write('failure-report.json', { status: error instanceof Error && error.message.startsWith('evaluation regression:') ?
      'regression' : 'infrastructure-or-input-error', error: String(error),
      committedIteration: state?.iteration ?? null, incumbent: state?.incumbent ?? null, ledger: attachedGateway?.snapshot() ?? null });
    throw error;
  } finally { if (attachedGateway) attachedGateway.onUpdate = undefined; store.close(); }
}
