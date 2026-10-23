/**
 * `natlang:learning` (spec/SPEC.md, Neuralese chapter "Learning"; S0 §9): `grad`, `valueAndGrad`, `stopGradient`,
 * objectives, immutable optimisers and `save`, backed by a Neuralese server's gradient replay sessions.
 *
 * `grad(f, a)` runs `f(a)` while recording every model turn the Neuralese driver makes (`recording.ts`). The loss
 * that `f` returns is a list of terms over those recorded turns; the server replays them with discrete choices held
 * fixed and returns gradients for the Neuralese values in `a` (spec/NEURALESE_GRAPH.md, "Replay"). Recorded turns
 * that wrote blocks go along as producers, so a block one call wrote and another read carries gradient back into
 * the writing call's context. Values wrapped in
 * `stopGradient`, and gradients computed by an inner `grad` (first-order nested differentiation), are constants. With
 * `{ order: 2 }` the inner gradient sessions and optimiser steps the loss runs are recorded and recomputed by the server
 * as functions of the arguments: the gradient passes through the inner updates (meta-learning through update steps).
 * Optimisers never change a value: a step returns new Neuralese values and new optimiser state.
 *
 * Only callers given the learning service can use it: in natlang code, `import … from 'natlang:learning'` resolves
 * against the task's `learning` service (`learningService(...)`); host code calls `createLearning(service)`.
 */
import { AsyncLocalStorage } from 'node:async_hooks';
import { readFileSync, writeFileSync } from 'node:fs';
import { currentFrame } from '../runtime/context.js';
import { isNativeStateId, isNativeAdapterRef, isNeuraleseRef, nativeAdapterRef, neuraleseRef, type NativeAdapterRef, type NeuraleseRef } from '../native/neuralese.js';
import { type NeuraleseBlockMeta, type NeuraleseStore } from '../native/neuralese-store.js';
import { distributionOf, loadNz, saveNz, type NzSaveExport } from '../native/nz-file.js';
import { HttpNeuraleseStore } from '../model/neuralese-server.js';
import * as deltaOps from './deltas.js';
import { setSystemPromptSource, systemPromptBank, type SystemPromptBank } from '../native/system-prompts.js';
import { setNativeAdapterSource, setNativeStateSource, setRecorderSource, type NativeAdapterBinding, type NativeStateBinding, type RecordedTurn, ModelTurnRecorder } from '../native/model-recording.js';
import { archiveNativeArtifactFrom, importNativeArtifactBundle } from '../native/neuralese-store.js';
import { createNeuraleseLibrary, type StandardLibrary } from './combinators.js';
import type { ReplayRecordSink } from './replay-records.js';
import { nativeGraphGradient, nativeRecordedGraphGradient, nativeStateOptimizerStep, nativeAdapterCreate, nativeAdapterOptimizerStep,
  nativeRecordedGraphGradients, nativeDerivativeProgramGradient, type NativeGraphRow, type NativeOptimizerSpec, type NativeGradientTarget,
  type NativeRecordedGradientTarget } from './native-learning-client.js';
import { nativeModelIdentityEqual, type NativeModelIdentity, type NativeTensorArtifact, type NativeTensorValue } from '../native/neuralese-store.js';

type Json = Record<string, unknown>;

const SERVICE = Symbol('natlang.learning-service');
export type LearningService = {
  readonly [SERVICE]: true;
  /** The Neuralese server that holds the blocks and runs gradient sessions. */
  readonly endpoint: string;
  readonly headers?: Record<string, string>;
  /** The runtime's store, for uploading arguments the server lacks and for `save`. */
  readonly store?: NeuraleseStore;
  readonly dialect?: string;
  /** Keeps every gradient session as a replay record (replay-records.ts) for whole-program training. */
  readonly replayRecords?: ReplayRecordSink;
};

export function learningService(options: Omit<LearningService, typeof SERVICE>): LearningService {
  return { ...options, [SERVICE]: true };
}
export const isLearningService = (value: unknown): value is LearningService =>
  !!value && typeof value === 'object' && SERVICE in value;

export class LearningError extends Error {
  constructor(readonly code: string, message: string) { super(`${code}: ${message}`); this.name = 'LearningError'; }
}

// Recording ------------------------------------------------------------------------------------------
class Recorder extends ModelTurnRecorder {}

const recording = new AsyncLocalStorage<Recorder>();
setRecorderSource(() => recording.getStore());

function recorder(where: string): Recorder {
  const active = recording.getStore();
  if (!active) throw new LearningError('learning-outside-grad', `${where} must run inside grad or valueAndGrad`);
  return active;
}

// Losses ---------------------------------------------------------------------------------------------
type Term = Json & { kind: string; weight?: number };

/** An opaque loss: terms over recorded turns. After `valueAndGrad` it has a value, and compares as a number. */
export class Loss {
  value?: number;
  constructor(readonly terms: readonly Term[]) {}
  valueOf(): number {
    if (this.value === undefined) throw new LearningError('learning-loss-unevaluated', 'a loss has a value only after valueAndGrad');
    return this.value;
  }
  toJSON(): unknown { return { $loss: { terms: this.terms.length, value: this.value ?? null } }; }
}

/** A recorded trajectory of one execution, for `logLikelihood`. */
export type Trajectory = { readonly id: string; readonly turns: readonly RecordedTurn[];
  readonly failure?: { readonly name?: string; readonly code?: string; readonly message: string } };

/** Run an output to its end. A failure is returned, not thrown: the turns it recorded still count. */
const settle = async (output: unknown): Promise<unknown> => {
  try { await (typeof output === 'function' ? (output as () => unknown)() : output); return undefined; } catch (error) { return error; }
};
const failureNote = (error: unknown) => error === undefined ? '' :
  `; the output failed: ${error instanceof Error ? error.message : String(error)}`.slice(0, 600);

/**
 * The turns one output made. A function output runs under its own recorder, so objectives evaluated concurrently
 * (`Promise.all`) cannot take each other's turns. A promise output already runs in the caller's recording, where
 * turns are told apart only by order, so a promise objective that overlaps another fails instead of guessing.
 */
async function capture(rec: Recorder, output: unknown, where: string): Promise<RecordedTurn[] & { failure?: unknown }> {
  if (typeof output === 'function') {
    const own = new Recorder();
    const failure = await recording.run(own, () => settle(output));
    rec.nested.push(...own.turns, ...own.nested);
    rec.absorb(own);
    return Object.assign(own.turns, { failure });
  }
  if (rec.active) rec.overlapped = true;
  rec.active++;
  try {
    const failure = await settle(output);
    if (rec.overlapped) throw new LearningError('learning-concurrent-objectives',
      `${where}: objectives over promises overlapped, so their turns cannot be told apart; pass the output as a function (() => runtime.run(...))`);
    return Object.assign(rec.claim(), { failure });
  } finally {
    if (--rec.active === 0) rec.overlapped = false;
  }
}

/** The first fragment of `privileged` (its JSON, or a string leaf of 8+ characters) found in the text of `view`. */
export function privilegeLeak(privileged: unknown, view: unknown): string | undefined {
  if (privileged === undefined || privileged === null) return undefined;
  const texts: string[] = [];
  const visit = (value: unknown) => {
    if (typeof value === 'string') texts.push(value);
    else if (Array.isArray(value)) value.forEach(visit);
    else if (value && typeof value === 'object') Object.values(value).forEach(visit);
  };
  visit(view);
  const haystack = texts.join('\n');
  const fragments: string[] = [];
  const json = typeof privileged === 'string' ? privileged : JSON.stringify(privileged);
  if (json.length >= 8) fragments.push(json);
  const leaves = (value: unknown) => {
    if (typeof value === 'string') { if (value.length >= 8) fragments.push(value); }
    else if (Array.isArray(value)) value.forEach(leaves);
    else if (value && typeof value === 'object') Object.values(value).forEach(leaves);
  };
  leaves(privileged);
  return fragments.find(fragment => haystack.includes(fragment));
}

/** The adapters a turn ran with, as a term field: replay scores the same adapted model. */
const adapted = (turn: RecordedTurn): Json => ({ ...(turn.adapters?.length ? { adapters: turn.adapters } : {}),
  ...(turn.native_state_modes ? { native_state_modes: turn.native_state_modes } : {}),
  ...(turn.native ? {native_turn_id:turn.native.id} : {}) });

function lastTurn(turns: readonly RecordedTurn[] & { failure?: unknown }, where: string): RecordedTurn {
  const turn = turns.filter(item => !item.decision).at(-1);
  if (!turn) throw new LearningError('learning-no-turns', `${where}: the output made no recorded model turn (is the model a Neuralese server?)${failureNote(turns.failure)}`);
  return turn;
}

function returnTarget(expected: unknown): Json {
  return { role: 'assistant', content: null, tool_calls: [{ id: 'expected', type: 'function', function: {
    name: 'return_result', arguments: JSON.stringify({ status: 'success', value: expected }) } }] };
}

let trajectories = 0;

export const objectives = {
  /** Cross-entropy of the expected result as the output call's final answer. */
  async crossEntropy(output: Promise<unknown> | (() => Promise<unknown>), expected: unknown): Promise<Loss> {
    const rec = recorder('objectives.crossEntropy');
    const turn = lastTurn(await capture(rec, output, 'crossEntropy'), 'crossEntropy');
    return new Loss([{ kind: 'crossEntropy', messages: turn.messages, tools: turn.tools, target: returnTarget(expected), ...adapted(turn) }]);
  },
  /**
   * A proper scoring rule on a decision readout (a call with `readout: decision`): the output's scored options,
   * normalised, against `expected` probabilities by option value (`{ "yes": 0.8, "no": 0.2 }`) or one value (all
   * mass on it). `rule`: `logLoss` (default), `brier`, or `rps` for ordered options (declaration order).
   */
  async decision(output: Promise<unknown> | (() => Promise<unknown>), expected: unknown, rule: 'logLoss' | 'brier' | 'rps' = 'logLoss'): Promise<Loss> {
    const rec = recorder('objectives.decision');
    const turns = await capture(rec, output, 'decision');
    const turn = turns.filter(item => item.decision).at(-1);
    if (!turn?.decision) throw new LearningError('learning-no-decision', 'objectives.decision: the output made no decision readout (does its function declare readout: decision?)' + failureNote(turns.failure));
    const options = turn.decision.options;
    const weights: Record<string, number> = expected && typeof expected === 'object' && !Array.isArray(expected)
      ? Object.fromEntries(Object.entries(expected as Record<string, unknown>).map(([key, value]) => [JSON.stringify(key), Number(value)]))
      : { [JSON.stringify(expected)]: 1 };
    // Keys of an object are strings; a non-string option ("true", "3") matches its JSON text either way.
    const probability = (option: string) => weights[option] ?? weights[JSON.stringify(option)] ?? 0;
    const probabilities = options.map(probability);
    const total = probabilities.reduce((sum, value) => sum + value, 0);
    if (!(total > 0) || probabilities.some(value => !(value >= 0)))
      throw new LearningError('learning-decision-target', `objectives.decision: expected must put mass on the options ${options.join(', ')}`);
    return new Loss([{ kind: 'decision', messages: turn.messages, tools: turn.tools, options: [...options], rule,
      target: { probabilities: probabilities.map(value => value / total) }, ...adapted(turn) }]);
  },
  /** KL from the same model given the full source to the output's final turn. */
  async selfDistill(output: Promise<unknown> | (() => Promise<unknown>), withFullSource: () => Promise<unknown>): Promise<Loss> {
    const rec = recorder('objectives.selfDistill');
    const student = lastTurn(await capture(rec, output, 'selfDistill'), 'selfDistill');
    const teacher = lastTurn(await capture(rec, withFullSource, 'selfDistill'), 'selfDistill (full source)');
    return new Loss([{ kind: 'selfDistill', messages: student.messages, tools: student.tools, target: student.reply,
      teacher_messages: teacher.messages, ...adapted(student),
      ...(teacher.native ? { teacher_native_turn_id: teacher.native.id } : {}),
      ...(teacher.native_state_modes ? { teacher_native_state_modes: teacher.native_state_modes } : {}),
      ...(teacher.adapters?.length ? { teacher_adapters: teacher.adapters } : {}) }]);
  },
  /**
   * Conditioned distillation (LEARNING_CONTINUUM §4.3): `teacher` is the same program run with knowledge the student
   * does not see (a skill text, a document, a worked solution, the outcome); the student is the run with the artifact
   * being trained and without it. A decision readout distils exactly: the teacher's readout over the same options is
   * the target (computed by the server, without gradient). A generated reply distils by KL(teacher ‖ student) over
   * the student's reply, as `selfDistill`. `privileged` is the knowledge itself: if any fragment of it (string leaves
   * of 8+ characters, or its JSON) appears in what the student sees, the objective fails (the privilege check).
   */
  async conditionedDistill(student: Promise<unknown> | (() => Promise<unknown>), teacher: Promise<unknown> | (() => Promise<unknown>),
      options: { privileged?: unknown; rule?: 'logLoss' | 'brier' | 'rps' } = {}): Promise<Loss> {
    const rec = recorder('objectives.conditionedDistill');
    const studentTurns = await capture(rec, student, 'conditionedDistill');
    const teacherTurns = await capture(rec, teacher, 'conditionedDistill (teacher)');
    const own = studentTurns.at(-1), privileged = teacherTurns.at(-1);
    if (!own) throw new LearningError('learning-no-turns', 'conditionedDistill: the student made no recorded model turn' + failureNote(studentTurns.failure));
    if (!privileged) throw new LearningError('learning-no-turns', 'conditionedDistill: the teacher made no recorded model turn' + failureNote(teacherTurns.failure));
    if (JSON.stringify([own.messages, own.adapters ?? []]) === JSON.stringify([privileged.messages, privileged.adapters ?? []]))
      throw new LearningError('learning-teacher-unconditioned', 'conditionedDistill: the teacher saw exactly what the student saw; give it the privileged context');
    const leak = privilegeLeak(options.privileged, own.messages);
    if (leak) throw new LearningError('learning-privilege-leak', `conditionedDistill: the student's view contains privileged text ${JSON.stringify(leak.slice(0, 80))}`);
    const teacherSide = { messages: privileged.messages, ...(privileged.tools ? { tools: privileged.tools } : {}),
      ...adapted(privileged) };
    if (own.decision) {
      if (!privileged.decision || JSON.stringify(privileged.decision.options) !== JSON.stringify(own.decision.options))
        throw new LearningError('learning-teacher-mismatch', 'conditionedDistill: the teacher must make the same decision readout (same options) as the student');
      return new Loss([{ kind: 'decision', messages: own.messages, tools: own.tools, options: [...own.decision.options], rule: options.rule ?? 'logLoss',
        teacher: teacherSide, ...adapted(own) }]);
    }
    if (privileged.decision) throw new LearningError('learning-teacher-mismatch', 'conditionedDistill: the student generated a reply but the teacher made a decision readout');
    return new Loss([{ kind: 'selfDistill', messages: own.messages, tools: own.tools, target: own.reply,
      teacher_messages: privileged.messages, ...adapted(own),
      ...(privileged.native ? { teacher_native_turn_id: privileged.native.id } : {}),
      ...(privileged.native_state_modes ? { teacher_native_state_modes: privileged.native_state_modes } : {}),
      ...(privileged.adapters?.length ? { teacher_adapters: privileged.adapters } : {}) }]);
  },
  /**
   * Reinforcement on a decision readout: `reward(value)` scores each option's value (host-side; the model never sees
   * it), and the loss is −Σ pᵢ rᵢ under the readout. A finite decision's expected reward is exact, so this is the
   * policy gradient without sampling. Rewards are centred on their mean so only differences between options matter.
   */
  async expectedReward(output: Promise<unknown> | (() => Promise<unknown>), reward: (value: unknown) => number | Promise<number>): Promise<Loss> {
    const rec = recorder('objectives.expectedReward');
    const turns = await capture(rec, output, 'expectedReward');
    const turn = turns.filter(item => item.decision).at(-1);
    if (!turn?.decision) throw new LearningError('learning-no-decision', 'objectives.expectedReward: the output made no decision readout (does its function declare readout: decision?)' + failureNote(turns.failure));
    const options = turn.decision.options;
    const values = options.map(option => { try { return JSON.parse(option); } catch { return option; } });
    const rewards = await Promise.all(values.map(async value => Number(await reward(value))));
    if (rewards.some(value => !Number.isFinite(value))) throw new LearningError('learning-reward', 'objectives.expectedReward: rewards must be finite numbers');
    const mean = rewards.reduce((sum, value) => sum + value, 0) / rewards.length;
    return new Loss([{ kind: 'decision', messages: turn.messages, tools: turn.tools, options: [...options], rule: 'expectedReward',
      target: { rewards: rewards.map(value => value - mean) }, ...adapted(turn) }]);
  },
  /**
   * REINFORCE over sampled trajectories (run at Neuralese and text temperature > 0 so they differ): each trajectory's
   * negative log-likelihood is weighted by its advantage, reward − baseline (the mean reward of the group by default,
   * as in group-relative policy optimisation), divided by the group size. Trajectories with equal rewards contribute
   * nothing.
   */
  async policyGradient(samples: readonly { trajectory: Trajectory; reward: number }[], options: { baseline?: 'mean' | number } = {}): Promise<Loss> {
    if (!samples.length) throw new LearningError('learning-reward', 'objectives.policyGradient needs at least one sample');
    if (samples.some(sample => !Number.isFinite(sample.reward))) throw new LearningError('learning-reward', 'objectives.policyGradient: rewards must be finite numbers');
    const baseline = typeof options.baseline === 'number' ? options.baseline :
      samples.reduce((sum, sample) => sum + sample.reward, 0) / samples.length;
    const terms: Term[] = [];
    for (const { trajectory, reward } of samples) {
      const weight = (reward - baseline) / samples.length;
      if (weight === 0) continue;
      for (const turn of trajectory.turns) if (!turn.decision)
        terms.push({ kind: 'logLikelihood', messages: turn.messages, tools: turn.tools, target: turn.reply, weight,
          likelihood_law: 'sampled-policy', ...adapted(turn) });
    }
    return new Loss(terms);
  },
  /** Negative log-probability of a recorded trajectory: text, stop decisions, sampled payloads. */
  async logLikelihood(trajectory: Trajectory, weight = 1): Promise<Loss> {
    return new Loss(trajectory.turns.map(turn => ({ kind: 'logLikelihood', messages: turn.messages, tools: turn.tools,
      target: turn.reply, weight, likelihood_law: 'model-conditional', ...adapted(turn) })));
  },
  /** KL of Gaussian blocks to N(0, I): written blocks use their recorded mean and scale. */
  async klPrior(blocks: NeuraleseRef | NeuraleseRef[]): Promise<Loss> {
    const service = activeService('objectives.klPrior');
    const remote = new HttpNeuraleseStore(service.endpoint, service.headers);
    const entries = [];
    for (const ref of Array.isArray(blocks) ? blocks : [blocks]) {
      const id = ref.$neuralese.id;
      const producer = ((await remote.meta(id)) as (NeuraleseBlockMeta & { producer?: Json }) | undefined)?.producer;
      const distribution = distributionOf(id);
      if (producer?.mean) entries.push({ mean: producer.mean, log_sigma: producer.log_sigma ?? null });
      else if (distribution) entries.push({ mean: distribution.mean, log_sigma: distribution.logScale });
      else entries.push({ mean: id, log_sigma: null });
    }
    return new Loss([{ kind: 'klPrior', blocks: entries }]);
  },
  /**
   * A law of the combinators as a consistency loss (S0 §4.2), measured through `read`: the right side is evaluated
   * to a value, and the loss is the cross-entropy of that value as the left side's readout, so gradients reach the
   * left side only. Arguments by law: mapIdentity(v, id), mapFusion(v, g, f, fAfterG), readMapCommutation(v, f),
   * combineAssociativity(a, b, c), combineIdentity(v), splitZip(a, b). The combinators come from the task's
   * `neuralese` service, or from `createLearning(service, { library })`.
   */
  async law(name: LawName, ...args: unknown[]): Promise<Loss> {
    return lawLoss(undefined, name, args);
  },
  async sum(...losses: Loss[]): Promise<Loss> { return new Loss(losses.flatMap(loss => loss.terms)); },
  async scale(loss: Loss, weight: number): Promise<Loss> {
    return new Loss(loss.terms.map(term => ({ ...term, weight: (term.weight ?? 1) * weight })));
  },
};

export const LAW_NAMES = ['mapIdentity', 'mapFusion', 'readMapCommutation', 'combineAssociativity', 'combineIdentity', 'splitZip'] as const;
export type LawName = typeof LAW_NAMES[number];

async function lawLoss(explicit: StandardLibrary | undefined, name: LawName, args: unknown[]): Promise<Loss> {
  const rec = recorder(`objectives.law(${name})`);
  const library = explicit ?? (currentFrame()?.task.services as Json | undefined)?.neuralese as StandardLibrary | undefined;
  if (!library?.bodies) throw new LearningError('learning-law-unavailable', 'law objectives need the neuralese standard library');
  const lib = createNeuraleseLibrary(library);
  const ref = (value: unknown, what: string): NeuraleseRef => {
    if (!isNeuraleseRef(value)) throw new LearningError('learning-law-arguments', `${name}: ${what} must be a Neuralese value`);
    return value;
  };
  const read = async (value: unknown) => lib.read(ref(await value, 'each side'));
  const fn = (value: unknown, what: string) => {
    if (typeof value !== 'function' && !isNeuraleseRef(value)) throw new LearningError('learning-law-arguments', `${name}: ${what} must be a function`);
    return value;
  };
  const [a, b, c, d] = args;
  const pairs: Array<[() => Promise<unknown>, () => Promise<unknown>]> = [];
  switch (name) {
    case 'mapIdentity': pairs.push([() => read(lib.map(ref(a, 'v'), fn(b, 'id'))), () => read(a)]); break;
    case 'mapFusion': pairs.push([() => read(lib.map(ref(a, 'v'), fn(b, 'g')).then(x => lib.map(ref(x, 'map(v, g)'), fn(c, 'f')))),
      () => read(lib.map(ref(a, 'v'), fn(d, 'fAfterG')))]); break;
    case 'readMapCommutation': pairs.push([() => read(lib.map(ref(a, 'v'), fn(b, 'f'))),
      async () => (b as (x: unknown) => unknown)(await read(a))]); break;
    case 'combineAssociativity': pairs.push([
      () => read(lib.combine(ref(a, 'a'), ref(b, 'b')).then(ab => lib.combine(ref(ab, 'combine(a, b)'), ref(c, 'c')))),
      () => read(lib.combine(ref(b, 'b'), ref(c, 'c')).then(bc => lib.combine(ref(a, 'a'), ref(bc, 'combine(b, c)'))))]); break;
    case 'combineIdentity': pairs.push([() => read(lib.combine(ref(a, 'v'), lib.empty())), () => read(a)]); break;
    case 'splitZip': {
      const parts = async () => {
        const split = await lib.split(ref(await lib.zip(ref(a, 'a'), ref(b, 'b')), 'zip(a, b)'));
        if (!Array.isArray(split) || split.length !== 2) throw new LearningError('learning-law-arguments', 'splitZip: split did not return a pair');
        return split;
      };
      pairs.push([async () => read((await parts())[0]), () => read(a)], [async () => read((await parts())[1]), () => read(b)]);
      break;
    }
    default: throw new LearningError('learning-law-unknown', `unknown law ${String(name)}; expected one of ${LAW_NAMES.join(', ')}`);
  }
  const terms: Term[] = [];
  for (const [left, right] of pairs) {
    let expected: unknown;
    try { expected = await right(); } catch (error) {
      throw new LearningError('learning-law-reference', `${name}: the right side failed: ${(error as Error).message}`);
    } finally { rec.claim(); }  // the right side is the reference, not a term
    let failure: unknown;
    try { await left(); } catch (error) { failure = error; }
    const turns = rec.claim();
    if (!turns.length && failure)
      throw new LearningError('learning-law-left', `${name}: the left side failed before any model turn: ${(failure as Error).message}`);
    const turn = lastTurn(turns, `law ${name}`);
    terms.push({ kind: 'crossEntropy', law: name, messages: turn.messages, tools: turn.tools, target: returnTarget(expected), ...adapted(turn) });
  }
  return new Loss(terms);
}

// Adapters -------------------------------------------------------------------------------------------
type NativeAdapterUse = NativeAdapterRef | { adapter: NativeAdapterRef; scale?: number };

const nativeAdapterScope = new AsyncLocalStorage<readonly NativeAdapterBinding[]>();
setNativeAdapterSource(() => nativeAdapterScope.getStore());
const nativeStateScope = new AsyncLocalStorage<readonly NativeStateBinding[]>();
setNativeStateSource(() => nativeStateScope.getStore());

/**
 * Run `fn` with weight adapters active: every Neuralese-server turn inside (generation and decision readouts) runs
 * the adapted model, and turns recorded for `grad` replay with the same adapters. Adapters are values: inside
 * `valueAndGrad(f, adapter)` the adapter's coefficients get gradients like any soft value. Nested scopes add up.
 */
export async function withNativeAdapters<T>(adapters: NativeAdapterUse | readonly NativeAdapterUse[], fn: () => Promise<T> | T): Promise<T> {
  const bindings = (Array.isArray(adapters) ? adapters : [adapters]).map((item: NativeAdapterUse): NativeAdapterBinding => {
    const ref = isNativeAdapterRef(item) ? item : item.adapter;
    if (!isNativeAdapterRef(ref)) throw new LearningError('learning-native-adapter', 'withNativeAdapters takes named native adapter references');
    const scale = isNativeAdapterRef(item) ? 1 : item.scale ?? 1;
    if (!Number.isFinite(scale)) throw new LearningError('learning-native-adapter', 'native adapter scales must be finite');
    return { id: ref.$nativeAdapter.id, scale };
  });
  const active = activeService('withNativeAdapters');
  if (!active.store?.getNativeTensorArtifact)
    throw new LearningError('learning-native-archive-unavailable', 'withNativeAdapters requires a named native tensor archive');
  for (const binding of bindings) {
    const artifact = await active.store.getNativeTensorArtifact(binding.id);
    if (!artifact || artifact.metadata.kind !== 'adapter-coefficients')
      throw new LearningError('learning-native-adapter', `${binding.id} is not an archived native adapter coefficient artifact`);
  }
  return nativeAdapterScope.run([...(nativeAdapterScope.getStore() ?? []), ...bindings], fn);
}

/**
 * Run `fn` with stored positional KV/hybrid state included in every model
 * call. The state is call context; it does not change model parameters. Use
 * `nativeGraphGradient` with a `native-state` target and
 * `nativeStateOptimizerStep` to learn an updated state artifact.
 */
export async function withNativeStates<T>(states: string | readonly string[], fn: () => Promise<T> | T): Promise<T> {
  const bindings = (Array.isArray(states) ? states : [states]).map((id): NativeStateBinding => {
    if (typeof id !== 'string' || !isNativeStateId(id))
      throw new LearningError('learning-native-state', 'withNativeStates takes stored native-state IDs');
    return { id };
  });
  if (!bindings.length) throw new LearningError('learning-native-state', 'withNativeStates needs at least one stored native-state ID');
  return await nativeStateScope.run(bindings, fn);
}

// System prompts -------------------------------------------------------------------------------------
const promptScope = new AsyncLocalStorage<SystemPromptBank>();
setSystemPromptSource(() => promptScope.getStore());

/**
 * Run `fn` with soft forms of the runtime's prompt pieces (DECISIONS.md 40): `prompts` maps piece IDs
 * (`promptPieces()`) to `Neuralese<SystemPrompt>` values. Under a Neuralese driver every call inside uses them in place
 * of the pieces' text, over the runtime's own bank. They are values: inside `valueAndGrad(p => …withSystemPrompts(p,
 * …)…, prompts)` each piece gets a gradient from every call that showed it, so self-improvement can adapt the system
 * prompt like a soft skill. Nested scopes override piece by piece.
 */
export async function withSystemPrompts<T>(prompts: Readonly<Record<string, NeuraleseRef>>, fn: () => Promise<T> | T): Promise<T> {
  const bank = systemPromptBank(prompts);
  const outer = promptScope.getStore();
  return promptScope.run(outer?.size ? new Map([...outer, ...bank]) : bank, fn);
}

// Arguments ------------------------------------------------------------------------------------------
const stopped = new WeakSet<object>();

/** `v` as a constant: its Neuralese values receive no gradient. */
export function stopGradient<T>(v: T): T {
  const copy = (value: unknown): unknown => {
    if (isNeuraleseRef(value)) { const ref = neuraleseRef(value.$neuralese.type, value.$neuralese.id,true,value.$neuralese.source_occurrence); stopped.add(ref); return ref; }
    if (Array.isArray(value)) return value.map(copy);
    if (value && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype)
      return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, copy(item)]));
    return value;
  };
  return copy(v) as T;
}

/** The Neuralese values in `a`, in traversal order, with the ones that are differentiable. */
function softLeaves(a: unknown): NeuraleseRef[] {
  const out: NeuraleseRef[] = [];
  const visit = (value: unknown) => {
    if (isNeuraleseRef(value)) {
      if (!isNativeStateId(value.$neuralese.id)) out.push(value);
      return;
    }
    if (Array.isArray(value)) value.forEach(visit);
    else if (value && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype) Object.values(value).forEach(visit);
  };
  visit(a);
  return out;
}

/** `a` with each Neuralese value replaced by `replace(ref)`. */
function mapLeaves(a: unknown, replace: (ref: NeuraleseRef) => unknown): unknown {
  if (isNeuraleseRef(a)) return replace(a);
  if (Array.isArray(a)) return a.map(item => mapLeaves(item, replace));
  if (a && typeof a === 'object' && Object.getPrototypeOf(a) === Object.prototype)
    return Object.fromEntries(Object.entries(a).map(([key, item]) => [key, mapLeaves(item, replace)]));
  return a;
}

type NativeLearningRef = NativeAdapterRef | NeuraleseRef;
function nativeLearningLeaves(value: unknown): NativeLearningRef[] {
  const out: NativeLearningRef[] = [];
  const visit = (item: unknown) => {
    if (isNativeAdapterRef(item)) { out.push(item); return; }
    if (isNeuraleseRef(item) && isNativeStateId(item.$neuralese.id)) {
      if (!stopped.has(item) && !item.$neuralese.stop_gradient) out.push(item);
      return;
    }
    if (Array.isArray(item)) item.forEach(visit);
    else if (item && typeof item === 'object' && Object.getPrototypeOf(item) === Object.prototype)
      Object.values(item).forEach(visit);
  };
  visit(value); return out;
}

function nativeLearningId(value: NativeLearningRef): string {
  return isNativeAdapterRef(value) ? value.$nativeAdapter.id : value.$neuralese.id;
}

function recordedNativeStateModes(turns: readonly RecordedTurn[]): Record<string, 'same-base-adapter-state'> | undefined {
  const modes: Record<string, 'same-base-adapter-state'> = {};
  for (const turn of turns) for (const [id, mode] of Object.entries(turn.native_state_modes ?? {})) {
    if (modes[id] && modes[id] !== mode)
      throw new LearningError('learning-native-state-mode', `recorded turns disagree on native state mode for ${id}`);
    modes[id] = mode;
  }
  return Object.keys(modes).length ? modes : undefined;
}

function mapNativeLearningLeaves(value: unknown, replace: (ref: NativeLearningRef) => unknown): unknown {
  if (isNativeAdapterRef(value)) return replace(value);
  if (isNeuraleseRef(value) && isNativeStateId(value.$neuralese.id)) return replace(value);
  if (Array.isArray(value)) return value.map(item => mapNativeLearningLeaves(item, replace));
  if (value && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype)
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, mapNativeLearningLeaves(item, replace)]));
  return value;
}

async function nativeLearningTargets(service: LearningService, refs: readonly NativeLearningRef[]): Promise<{
  targets: NativeRecordedGradientTarget[]; identity: NativeModelIdentity; byId: Map<string, NativeRecordedGradientTarget>;
}> {
  if (!service.store?.getNativeState || !service.store.getNativeTensorArtifact)
    throw new LearningError('learning-native-archive-unavailable', 'native gradients require named nnt1/nzs1 artifacts');
  const remote = new HttpNeuraleseStore(service.endpoint, service.headers);
  const unique = new Map<string, NativeLearningRef>();
  for (const ref of refs) unique.set(nativeLearningId(ref), ref);
  if (!unique.size) throw new LearningError('learning-native-target', 'native gradients need at least one differentiable target');
  const targets: NativeRecordedGradientTarget[] = [];
  let identity: NativeModelIdentity | undefined;
  for (const [index, [id, ref]] of [...unique].entries()) {
    let current: NativeModelIdentity;
    let target: NativeRecordedGradientTarget;
    if (isNativeAdapterRef(ref)) {
      const artifact = await service.store.getNativeTensorArtifact(id) ?? await remote.getNativeTensorArtifact(id);
      if (!artifact || artifact.metadata.kind !== 'adapter-coefficients')
        throw new LearningError('learning-native-target', `${id} is not an archived native adapter coefficient artifact`);
      await archiveNativeArtifactFrom(remote, service.store, id);
      current = artifact.metadata.native_identity;
      target = { root: `native-target-${index}`, kind: 'adapter', id, names: Object.keys(artifact.tensors).sort() };
    } else {
      const artifact = await service.store.getNativeState(id) ?? await remote.getNativeState(id);
      if (!artifact) throw new LearningError('learning-native-target', `${id} is not an archived native state artifact`);
      await archiveNativeArtifactFrom(remote, service.store, id);
      current = artifact.metadata.adapter_binding?.base_native_identity ?? {
        architecture: artifact.metadata.architecture, checkpoint_sha256: artifact.metadata.checkpoint_sha256,
        tokenizer_fingerprint: artifact.metadata.tokenizer_fingerprint, embedding_weight_sha256: artifact.metadata.embedding_weight_sha256 };
      target = { root: `native-target-${index}`, kind: 'native-state', id };
    }
    if (identity && !nativeModelIdentityEqual(identity, current))
      throw new LearningError('learning-native-target', 'native gradient roots must share one exact base checkpoint identity');
    identity = current;
    targets.push(target);
  }
  return { targets, identity: identity!, byId: new Map(targets.map(target => [target.id, target])) };
}

type NativeDerivativeContext = {
  roots: Map<string, NativeRecordedGradientTarget>;
  rootById: Map<string, string>;
  valueById: Map<string, string>;
  currentByRoot: Map<string, string>;
  gradientById: Map<string, string>;
  next: number;
};
type DerivationTrace = { nodes: Derivation[]; native?: NativeDerivativeContext };

function nativeDerivativeContext(targets: readonly NativeRecordedGradientTarget[]): NativeDerivativeContext {
  const context: NativeDerivativeContext = { roots: new Map(), rootById: new Map(), valueById: new Map(),
    currentByRoot: new Map(), gradientById: new Map(), next: 0 };
  for (const target of targets) {
    context.roots.set(target.root, target); context.rootById.set(target.id, target.root);
    context.valueById.set(target.id, target.root); context.currentByRoot.set(target.root, target.root);
  }
  return context;
}

function registerNativeGradientDerivation(context: NativeDerivativeContext, targetInfo: {
    targets: NativeRecordedGradientTarget[]; byId: Map<string, NativeRecordedGradientTarget> },
    objective: { turns: readonly RecordedTurn[]; events: readonly Record<string, unknown>[]; terms: readonly Record<string, unknown>[];
      nativeStateModes?: Record<string, 'same-base-adapter-state'> },
    gradients: Record<string, string>): Derivation {
  const bindings = Object.fromEntries(context.currentByRoot);
  const nodeTargets: { name: string; value: string }[] = [];
  const requestTargets: NativeRecordedGradientTarget[] = [];
  for (const target of targetInfo.targets) {
    let root = context.rootById.get(target.id);
    if (!root) {
      root = `native-root-${context.next++}`;
      const descriptor = { ...target, root };
      context.roots.set(root, descriptor); context.rootById.set(target.id, root);
      context.valueById.set(target.id, root); context.currentByRoot.set(root, root);
      bindings[root] = root;
    }
    const value = context.valueById.get(target.id);
    const descriptor = context.roots.get(root);
    if (!value || !descriptor) throw new LearningError('learning-native-order2', `native derivative lost exact root ${target.id}`);
    requestTargets.push(descriptor);
    const gradientId = gradients[target.id];
    if (!gradientId) throw new LearningError('learning-native-order2', `native derivative lost gradient for ${target.id}`);
    const alias = `native-gradient-${context.next++}`;
    nodeTargets.push({ name: alias, value }); context.gradientById.set(gradientId, alias);
  }
  const name = `native-grad-node-${context.next++}`;
  return { kind: 'gradient', name, request: { schema: 'natlang.native-recorded-objective-request/1',
    turns: objective.turns, events: objective.events, terms: objective.terms, targets: requestTargets,
    ...(objective.nativeStateModes ? { native_state_modes: objective.nativeStateModes } : {}) },
    bindings, targets: nodeTargets };
}

/** An opaque gradient: the shape of `A` with gradient entries where `A` has differentiable Neuralese values. */
export type Gradient<A> = { readonly $gradient: unknown; readonly __of?: A };
type GradientEntry = { $gradientBlock: { id: string; of: string; type: string } } | null;

// Service --------------------------------------------------------------------------------------------
function activeService(where: string, explicit?: LearningService): LearningService {
  const found = explicit ?? boundService.getStore() ?? (currentFrame()?.task.services as Json | undefined)?.learning;
  if (!isLearningService(found))
    throw new LearningError('learning-unavailable', `${where} needs the natlang:learning service, which this caller was not given`);
  return found;
}
const boundService = new AsyncLocalStorage<LearningService>();

/**
 * Second order: the inner computations an order-2 loss runs (gradient sessions, optimiser steps), in order. The server
 * recomputes them differentiably before the outer terms, so the outer gradient passes through the inner updates.
 */
type Derivation = Record<string, unknown> & { kind: 'grad' | 'optim' | 'gradient' | 'optimizer' };
const derivations = new AsyncLocalStorage<DerivationTrace | undefined>();

async function evaluate<A>(service: LearningService, f: (a: A) => Promise<Loss>, a: A, order?: number):
    Promise<{ loss: Loss; grad: Gradient<A> }> {
  if (order !== undefined && order !== 1 && order !== 2)
    throw new LearningError('neuralese-grad-unavailable', 'gradients of order 1 or 2');
  const leaves = softLeaves(a).filter(ref => !stopped.has(ref)&&!ref.$neuralese.stop_gradient);
  const nativeLeaves = nativeLearningLeaves(a);
  if (leaves.length)
    throw new LearningError('learning-native-target', 'projected nz1 values are not differentiable targets; pass archived nzs1 state or nnt1 coefficient artifacts');
  if (!nativeLeaves.length)
    throw new LearningError('learning-native-target', 'recorded gradients need at least one archived native state or coefficient target');
  const targetInfo = nativeLeaves.length ? await nativeLearningTargets(service, nativeLeaves) : undefined;
  const rec = new Recorder();
  // An inner first-order gradient's own inner computations are constants to it: they are not recorded.
  const derived: DerivationTrace | undefined = order === 2 ? { nodes: [],
    ...(targetInfo ? { native: nativeDerivativeContext(targetInfo.targets) } : {}) } : undefined;
  const outer = derivations.getStore();
  if (order === 2 && outer) throw new LearningError('learning-grad-order', 'native learning supports derivatives through order 2, not nested order-3 programs');
  const loss = await derivations.run(derived, () => recording.run(rec, () => boundService.run(service, () => f(a))));
  if (!(loss instanceof Loss)) throw new LearningError('learning-not-a-loss', 'the function passed to grad must return a Loss from objectives');
  // A loss without terms (a policy gradient whose samples tied) is zero, with no gradient anywhere.
  if (!loss.terms.length) {
    loss.value = 0;
    const soft = mapLeaves(a, ref => isNativeStateId(ref.$neuralese.id) ? ref : null);
    const native = mapNativeLearningLeaves(soft, (): GradientEntry => null);
    return { loss, grad: { $gradient: native } as Gradient<A> };
  }
  if (nativeLeaves.length) {
    if (leaves.length)
      throw new LearningError('learning-native-target', 'native checkpoint gradients accept native nnt1/nzs1 roots; projected nz1 values have no native checkpoint-coordinate derivative');
    if (!targetInfo) throw new LearningError('learning-native-target', 'native derivative target inventory is missing');
    const turns = [...rec.turns, ...rec.nested];
    const nativeTurns = rec.nativeTurns();
    const byTurnId = new Map(nativeTurns.map(turn => [turn.native!.id, turn.native!]));
    const exactTurns = turns.filter(turn => turn.native).map(turn => ({ ...structuredClone(turn), native: byTurnId.get(turn.native!.id)! }));
    const events = structuredClone([...rec.graphNodes.values()]);
    const requestTargets = targetInfo.targets.map(target => {
      const root = derived?.native?.rootById.get(target.id) ?? target.root;
      return { ...target, root };
    });
    const nativeStateModes = recordedNativeStateModes(exactTurns);
    const objective = { turns: exactTurns, events, terms: loss.terms, ...(nativeStateModes ? { nativeStateModes } : {}) };
    let gradientArtifacts: Record<string, NativeTensorArtifact>, derivativeExecution: string | undefined;
    if (order === 2) {
      const context = derived?.native;
      if (!context) throw new LearningError('learning-native-order2', 'native derivative program roots were not declared');
      const program = { schema: 'natlang.native-derivative-program/1' as const,
        inputs: Object.fromEntries([...context.roots].map(([root, target]) => [root, {
          kind: target.kind, id: target.id, ...(target.names ? { names: target.names } : {}) }])),
        nodes: derived.nodes, outer: { request: { schema: 'natlang.native-recorded-objective-request/1',
          turns: objective.turns, events: objective.events, terms: objective.terms, targets: requestTargets,
          ...(nativeStateModes ? { native_state_modes: nativeStateModes } : {}) },
          bindings: Object.fromEntries(context.currentByRoot) },
        wrt: requestTargets.map(target => target.root),
        arithmetic: { schema: 'natlang.native-reference-arithmetic/1' as const, query_chunk_tokens: 16 } };
      const result = await nativeDerivativeProgramGradient(service.endpoint, service.store!, program, service.headers);
      loss.value = Number(result.loss); gradientArtifacts = result.gradients; derivativeExecution = result.execution_id;
    } else {
      const result = await nativeRecordedGraphGradients(service.endpoint, service.store!, { ...objective,
        nativeStateModes, targets: requestTargets, nativeIdentity: targetInfo.identity, headers: service.headers });
      loss.value = Number(result.loss); gradientArtifacts = result.gradients;
      const gradients = Object.fromEntries([...targetInfo.byId].map(([id, target]) => [id, result.gradients[target.root]!.id]));
      if (outer?.native) outer.nodes.push(registerNativeGradientDerivation(outer.native, targetInfo,
        objective, gradients));
    }
    const gradients = Object.fromEntries([...targetInfo.byId].map(([id, target]) => [id, gradientArtifacts[target.root]!.id]));
    await service.replayRecords?.record({ arguments: Object.keys(gradients), terms: loss.terms, order: order ?? 1,
      native_turns: exactTurns, native_graph_nodes: events, native_gradients: gradients,
      native_targets: requestTargets, ...(derivativeExecution ? { derivative_execution_id: derivativeExecution } : {}), loss: loss.value });
    if (outer && !outer.native) outer.nodes.push({ kind: 'grad', arguments: Object.keys(gradients), terms: loss.terms,
      native_turns: exactTurns, native_graph_nodes: events, native_gradients: gradients });
    const grad = mapNativeLearningLeaves(a, ref => {
      if (isNativeAdapterRef(ref)) {
        const artifact = gradientArtifacts[targetInfo.byId.get(ref.$nativeAdapter.id)!.root];
        return artifact ? { $gradientBlock: { id: artifact.id, of: ref.$nativeAdapter.id, type: 'NativeAdapter' } } : null;
      }
      if (stopped.has(ref) || ref.$neuralese.stop_gradient) return null;
      const id = nativeLearningId(ref), target = targetInfo.byId.get(id), artifact = target && gradientArtifacts[target.root];
      return artifact ? { $gradientBlock: { id: artifact.id, of: id,
        type: 'NativeState' } } : null;
    });
    return { loss, grad: { $gradient: grad } as Gradient<A> };
  }
  throw new LearningError('learning-native-target', 'native gradient dispatch requires exact native target references');
}

export type Optimizer = {
  init<A>(a: A): OptimizerState<A>;
  step<A>(state: { value: A; opt: OptimizerState<A> }, grad: Gradient<A>): Promise<{ value: A; opt: OptimizerState<A> }>;
};
export type OptimizerState<A> = { readonly $optimizer: { name: string; hyper: Json; state: Json | null }; readonly __of?: A };

function optimizer(service: () => LearningService, name: 'sgd' | 'adam', hyper: Json): Optimizer {
  return {
    init: a => { void a; return { $optimizer: { name, hyper, state: null } }; },
    async step(current, grad) {
      const nativeEntries = nativeLearningLeaves(current.value);
      if (nativeEntries.length) {
        const gradients = new Map<string, Exclude<GradientEntry, null>>();
        const collect = (value: unknown) => {
          if (value && typeof value === 'object' && '$gradientBlock' in value) {
            const entry = value as Exclude<GradientEntry, null>;
            gradients.set(entry.$gradientBlock.of, entry); return;
          }
          if (Array.isArray(value)) value.forEach(collect);
          else if (value && typeof value === 'object') Object.values(value).forEach(collect);
        };
        collect(grad.$gradient);
        if (softLeaves(current.value).length || !service().store?.getNativeTensorArtifact)
          throw new LearningError('learning-native-target', 'native optimization requires native roots and a named tensor archive');
        const active = service(), store = active.store!;
        const betas = hyper.betas as [number, number] | undefined;
        const spec: NativeOptimizerSpec = { name, ...(typeof hyper.lr === 'number' ? { learning_rate: hyper.lr } : {}),
          ...(typeof hyper.weightDecay === 'number' ? { weight_decay: hyper.weightDecay } : {}),
          ...(betas ? { beta1: betas[0], beta2: betas[1] } : {}) };
        const histories = (current.opt.$optimizer.state?.native_optimizer_state_ids ?? {}) as Record<string, string>;
        const updatedIds = new Map<string, string>(), nextHistories: Record<string, string> = {};
        let latestStep = 0;
        for (const ref of nativeEntries) {
          const oldId = nativeLearningId(ref);
          if (updatedIds.has(oldId)) continue;
          const entry = gradients.get(oldId);
          if (!entry) throw new LearningError('learning-gradient-shape', `native optimizer gradient is missing for ${oldId}`);
          const gradient = await store.getNativeTensorArtifact!(entry.$gradientBlock.id);
          const expectedKind = isNativeAdapterRef(ref) ? 'adapter-gradient' : 'state-gradient';
          if (!gradient || gradient.metadata.kind !== expectedKind || gradient.metadata.source_artifact_id !== oldId)
            throw new LearningError('learning-native-target', `native optimizer needs an archived ${expectedKind} bound to ${oldId}`);
          const previousState = histories[oldId];
          let newId: string, optimizerStateId: string, step: number;
          if (isNativeAdapterRef(ref)) {
            const updated = await nativeAdapterOptimizerStep(active.endpoint, store, { gradient,
              nativeIdentity: gradient.metadata.native_identity, optimizer: spec,
              ...(previousState ? { optimizerStateId: previousState } : {}), headers: active.headers });
            newId = updated.adapter.id; optimizerStateId = updated.optimizer_state_id; step = updated.step;
          } else {
            const updated = await nativeStateOptimizerStep(active.endpoint, store, { gradient,
              nativeIdentity: gradient.metadata.native_identity, optimizer: spec,
              ...(previousState ? { optimizerStateId: previousState } : {}), headers: active.headers });
            newId = updated.native_state_id; optimizerStateId = updated.optimizer_state_id; step = updated.step;
          }
          updatedIds.set(oldId, newId);
          nextHistories[newId] = optimizerStateId;
          latestStep = Math.max(latestStep, step);
          const trace = derivations.getStore();
          if (trace?.native) {
            const value = trace.native.valueById.get(oldId), gradientName = trace.native.gradientById.get(gradient.id);
            const root = trace.native.rootById.get(oldId);
            if (!value || !gradientName || !root)
              throw new LearningError('learning-native-order2', `native optimizer cannot bind update ancestry for ${oldId}`);
            const node = `native-optim-node-${trace.native.next++}`;
            trace.nodes.push({ kind: 'optimizer', name: node, value, gradient: gradientName,
              optimizer: { name, ...(typeof hyper.lr === 'number' ? { learning_rate: hyper.lr } : {}),
                ...(typeof hyper.weightDecay === 'number' ? { weight_decay: hyper.weightDecay } : {}),
                ...(betas ? { beta1: betas[0], beta2: betas[1] } : {}) }, optimizer_state: previousState ?? null });
            trace.native.valueById.set(newId, node); trace.native.rootById.set(newId, root); trace.native.currentByRoot.set(root, node);
          } else if (trace) {
            throw new LearningError('learning-native-order2', 'native updates inside order-2 learning need native checkpoint roots');
          }
        }
        const value = mapNativeLearningLeaves(current.value, ref => {
          const next = updatedIds.get(nativeLearningId(ref));
          return next ? isNativeAdapterRef(ref) ? nativeAdapterRef(next) : neuraleseRef(ref.$neuralese.type, next) : ref;
        }) as typeof current.value;
        const state = { native_optimizer_state_ids: nextHistories, step: latestStep } as Json;
        return { value, opt: { $optimizer: { name, hyper, state } } };
      }
      throw new LearningError('learning-native-target', 'optimizers require archived native state or named coefficient targets');
    },
  };
}

function f32Values(value: NativeTensorValue): Float32Array {
  if (value.dtype !== 'F32' || value.data.byteLength % 4) throw new LearningError('learning-native-delta', 'native adapter deltas currently require FP32 coefficient tensors');
  return new Float32Array(value.data.buffer.slice(value.data.byteOffset, value.data.byteOffset + value.data.byteLength));
}
function f32Bytes(values: Float32Array): Uint8Array {
  return new Uint8Array(values.buffer.slice(values.byteOffset, values.byteOffset + values.byteLength));
}

async function nativeAdapterArtifacts(active: LearningService, refs: readonly NativeAdapterRef[]) {
  const store = active.store;
  if (!store?.getNativeTensorArtifact || !store.putNativeTensorArtifact)
    throw new LearningError('learning-native-archive-unavailable', 'native coefficient deltas require an archived tensor store');
  const artifacts = await Promise.all(refs.map(ref => store.getNativeTensorArtifact!(ref.$nativeAdapter.id)));
  if (artifacts.some(artifact => !artifact || artifact.metadata.kind !== 'adapter-coefficients'))
    throw new LearningError('learning-native-delta', 'native deltas require named adapter coefficient artifacts');
  const valid = artifacts as NativeTensorArtifact[];
  const first = valid[0]!;
  if (valid.some(artifact => artifact.metadata.source_artifact_id !== first.metadata.source_artifact_id ||
      !nativeModelIdentityEqual(artifact.metadata.native_identity, first.metadata.native_identity) ||
      JSON.stringify(artifact.metadata.tensors) !== JSON.stringify(first.metadata.tensors)))
    throw new LearningError('learning-native-delta', 'native coefficient deltas need the same exact checkpoint, basis and named tensor geometry');
  return valid;
}

async function nativeAdapterArithmetic(active: LearningService, refs: readonly { ref: NativeAdapterRef; scale: number }[],
    operation: string): Promise<NativeAdapterRef> {
  if (!refs.length || refs.some(item => !Number.isFinite(item.scale)))
    throw new LearningError('learning-native-delta', 'native delta arithmetic needs finite scales and at least one coefficient artifact');
  const artifacts = await nativeAdapterArtifacts(active, refs.map(item => item.ref));
  const first = artifacts[0]!;
  const tensors: Record<string, NativeTensorValue> = {};
  for (const name of Object.keys(first.tensors).sort()) {
    const values = new Float32Array(first.tensors[name]!.shape.reduce((count, size) => count * size, 1));
    artifacts.forEach((artifact, index) => {
      const part = f32Values(artifact.tensors[name]!);
      for (let at = 0; at < values.length; at++) values[at]! += refs[index]!.scale * part[at]!;
    });
    tensors[name] = { dtype: 'F32', shape: [...first.tensors[name]!.shape], data: f32Bytes(values) };
  }
  const provenance = { ...first.metadata.provenance, native_delta_operation: operation };
  const result = await active.store!.putNativeTensorArtifact!({ metadata: { ...first.metadata, provenance }, tensors });
  return nativeAdapterRef(result.id);
}

async function evaluateRecordedNative(service: LearningService, f: () => Promise<Loss>, target: NativeGradientTarget,
    nativeIdentity: NativeModelIdentity, options: { nativeStateModes?: Record<string, 'same-base-adapter-state'> } = {}) {
  if (!service.store) throw new LearningError('learning-native-state-store', 'native recorded learning requires an archived artifact store');
  const rec = new Recorder();
  const loss = await recording.run(rec, () => boundService.run(service, f));
  if (!(loss instanceof Loss)) throw new LearningError('learning-not-a-loss', 'native graph learning must return a Loss from objectives');
  if (!loss.terms.length) return { loss: Object.assign(loss, { value: 0 }), gradient: null };
  const nativeTurns = rec.nativeTurns();
  const byId = new Map(nativeTurns.map(turn => [turn.native!.id, turn.native!]));
  const turns = [...rec.turns, ...rec.nested].filter(turn => turn.native)
    .map(turn => ({ ...structuredClone(turn), native: byId.get(turn.native!.id)! }));
  if (!turns.length) throw new LearningError('learning-no-turns', 'native graph objective made no recorded native model turns');
  if (loss.terms.some(term => typeof term.native_turn_id !== 'string'))
    throw new LearningError('learning-native-objective', 'native graph objectives must score exact recorded native turns');
  const result = await nativeRecordedGraphGradient(service.endpoint, service.store, { turns,
    events: structuredClone([...rec.graphNodes.values()]), terms: loss.terms, target, nativeIdentity,
    nativeStateModes: options.nativeStateModes, headers: service.headers });
  loss.value = Number(result.loss);
  await service.replayRecords?.record({ arguments: [], terms: loss.terms, order: 1, native_turns: turns,
    native_graph_nodes: [...rec.graphNodes.values()], target, native_gradient: result.gradient.id, loss: loss.value });
  return { loss, gradient: result.gradient, stats: result.stats, objective: result.objective };
}

/** The learning surface bound to a service (host code), or to the current task's service (natlang code). */
export function createLearning(explicit?: LearningService, options: { library?: StandardLibrary } = {}) {
  const service = (where = 'natlang:learning') => activeService(where, explicit);
  const bound = options.library ? { ...objectives, law: (name: LawName, ...args: unknown[]) => lawLoss(options.library, name, args) } : objectives;
  return {
    grad: async <A>(f: (a: A) => Promise<Loss>, a: A, options: { order?: 1 | 2 } = {}) =>
      (await evaluate(service('grad'), f, a, options.order)).grad,
    valueAndGrad: async <A>(f: (a: A) => Promise<Loss>, a: A, options: { order?: 1 | 2 } = {}) =>
      evaluate(service('valueAndGrad'), f, a, options.order),
    stopGradient,
    objectives: bound,
    withNativeAdapters: <T>(adapters: NativeAdapterUse | readonly NativeAdapterUse[], fn: () => Promise<T> | T) =>
      boundService.run(service('withNativeAdapters'), () => withNativeAdapters(adapters, fn)),
    withNativeStates,
    withSystemPrompts,
    /** Native positional state learning. Updates return a new state artifact with immutable optimizer ancestry. */
    nativeStates: {
      gradient: async (graph: NativeGraphRow, id: string, nativeIdentity: NativeModelIdentity,
          options: { nativeStateModes?: Record<string, 'same-base-adapter-state'>; objective?: 'graph-ce' } = {}) => {
        const current = service('nativeStates.gradient');
        if (!current.store) throw new LearningError('learning-native-state-store', 'native state learning requires an archived state store');
        return nativeGraphGradient(current.endpoint, current.store, { graph, target: { kind: 'native-state', id }, nativeIdentity,
          nativeStateModes: options.nativeStateModes, objective: options.objective, headers: current.headers });
      },
      optimize: async (gradient: NativeTensorArtifact, nativeIdentity: NativeModelIdentity, optimizer: NativeOptimizerSpec,
          optimizerStateId?: string) => {
        const current = service('nativeStates.optimize');
        if (!current.store) throw new LearningError('learning-native-state-store', 'native state learning requires an archived state store');
        return nativeStateOptimizerStep(current.endpoint, current.store, { gradient, nativeIdentity, optimizer,
          optimizerStateId, headers: current.headers });
      },
    },
    /** Native gradients through the exact recorded model and producer/consumer execution graph. */
    nativeGraph: {
      gradient: (f: () => Promise<Loss>, target: NativeGradientTarget, nativeIdentity: NativeModelIdentity,
          options: { nativeStateModes?: Record<string, 'same-base-adapter-state'> } = {}) =>
        evaluateRecordedNative(service('nativeGraph.gradient'), f, target, nativeIdentity, options),
    },
    /** Explicit constructors and optimizer updates for exact-base named native coefficient artifacts. */
    nativeAdapters: {
      create: async (nativeIdentity: NativeModelIdentity, options: { parameterNames: string[]; kind?: 'xs';
          rank?: number; seed?: number }) => {
        const active = service('nativeAdapters.create');
        if (!active.store) throw new LearningError('learning-native-state-store', 'native adapter creation requires an archived tensor store');
        const artifact = await nativeAdapterCreate(active.endpoint, active.store, { nativeIdentity, ...options, headers: active.headers });
        return nativeAdapterRef(artifact.id);
      },
      ref: (id: string) => nativeAdapterRef(id),
      optimize: async (gradient: NativeTensorArtifact, nativeIdentity: NativeModelIdentity, optimizer: NativeOptimizerSpec,
          optimizerStateId?: string) => {
        const active = service('nativeAdapters.optimize');
        if (!active.store) throw new LearningError('learning-native-state-store', 'native adapter updates require an archived tensor store');
        return nativeAdapterOptimizerStep(active.endpoint, active.store, { gradient, nativeIdentity, optimizer,
          optimizerStateId, headers: active.headers });
      },
    },
    /** Named native coefficient deltas plus ordinary file patches. */
    deltas: {
      diff: async (after: NativeAdapterRef, base: NativeAdapterRef) =>
        nativeAdapterArithmetic(service('deltas.diff'), [{ ref: after, scale: 1 }, { ref: base, scale: -1 }], 'diff'),
      apply: async (base: NativeAdapterRef, delta: NativeAdapterRef, scale = 1) =>
        nativeAdapterArithmetic(service('deltas.apply'), [{ ref: base, scale: 1 }, { ref: delta, scale }], 'apply'),
      compose: async (deltas: readonly (NativeAdapterRef | { delta: NativeAdapterRef; scale?: number })[]) => {
        if (!deltas.length) throw new LearningError('learning-native-delta', 'compose needs at least one native adapter delta');
        const items = deltas.map(item => isNativeAdapterRef(item) ? { ref: item, scale: 1 } : { ref: item.delta, scale: item.scale ?? 1 });
        if (items.some(item => !isNativeAdapterRef(item.ref)))
          throw new LearningError('learning-native-delta', 'coefficient composition accepts only exact nnt1 artifacts');
        return nativeAdapterArithmetic(service('deltas.compose'), items, 'compose');
      },
      diffFiles: deltaOps.diffFiles, applyPatch: deltaOps.applyPatch, composePatches: deltaOps.composePatches,
      /**
       * Interference of deltas trained separately from one base: each delta's gain alone, the gain of their sum, and
       * Σ gains alone − gain of the sum (positive: the updates get in each other's way). `measure` is a loss (lower
       * is better) of a value, e.g. a held-out evaluation.
       */
      async interference(base: NativeAdapterRef, deltas: readonly NativeAdapterRef[], measure: (value: NativeAdapterRef) => Promise<number>) {
        if (!isNativeAdapterRef(base) || deltas.some(delta => !isNativeAdapterRef(delta)))
          throw new LearningError('learning-native-delta', 'interference requires named native coefficient artifacts');
        const active = service('deltas.interference');
        const reference = await measure(base);
        const alone = [];
        for (const delta of deltas) alone.push(reference - await measure(await nativeAdapterArithmetic(active,
          [{ ref: base, scale: 1 }, { ref: delta, scale: 1 }], 'interference-alone')));
        const jointValue = await nativeAdapterArithmetic(active, [{ ref: base, scale: 1 }, ...deltas.map(ref => ({ ref, scale: 1 }))], 'interference-joint');
        const joint = reference - await measure(jointValue);
        return { base: reference, alone, joint, interference: alone.reduce((sum, gain) => sum + gain, 0) - joint };
      },
      /**
       * Learned merge coefficients (LoRAHub-style, first order): `base + Σ cᵢ·deltaᵢ` with the cᵢ trained by Adam on
       * `loss`. The merge is linear, so ∂L/∂cᵢ = ⟨∇L(merged), deltaᵢ⟩: one gradient session per step.
       */
      async learnMerge(base: NativeAdapterRef, deltas: readonly NativeAdapterRef[], loss: (value: NativeAdapterRef) => Promise<Loss>,
          options: { steps?: number; lr?: number; init?: number | number[] } = {}) {
        if (!isNativeAdapterRef(base) || deltas.some(delta => !isNativeAdapterRef(delta)))
          throw new LearningError('learning-native-delta', 'learnMerge requires named native coefficient artifacts');
        if (!deltas.length) throw new LearningError('learning-native-delta', 'learnMerge needs at least one native adapter delta');
        const active = service('deltas.learnMerge');
        const parts = await nativeAdapterArtifacts(active, deltas);
        const c = deltas.map((_, i) => Array.isArray(options.init) ? options.init[i] ?? 1 : options.init ?? 1);
        const m = c.map(() => 0), v = c.map(() => 0), lr = options.lr ?? 0.1, trace: { loss: number; coefficients: number[] }[] = [];
        const merge = async () => nativeAdapterArithmetic(active, [{ ref: base, scale: 1 }, ...deltas.map((ref, i) => ({ ref, scale: c[i]! }))], 'learn-merge');
        for (let step = 1; step <= (options.steps ?? 8); step++) {
          const merged = await merge();
          const result = await evaluate(active, value => withNativeAdapters(value, () => loss(value)), merged);
          trace.push({ loss: +result.loss, coefficients: [...c] });
          const entry = (result.grad as { $gradient: GradientEntry }).$gradient;
          if (!entry) break;
          const gradient = await active.store?.getNativeTensorArtifact?.(entry.$gradientBlock.id);
          if (!gradient || gradient.metadata.kind !== 'adapter-gradient' || gradient.metadata.source_artifact_id !== merged.$nativeAdapter.id)
            throw new LearningError('learning-native-delta', 'learnMerge needs the actual named coefficient gradient for its merged actor');
          parts.forEach((part, i) => {
            let dot = 0;
            for (const [name, tensor] of Object.entries(part.tensors)) {
              const g = f32Values(gradient.tensors[name]!); const d = f32Values(tensor);
              for (let k = 0; k < d.length; k++) dot += g[k]! * d[k]!;
            }
            m[i] = 0.9 * m[i]! + 0.1 * dot; v[i] = 0.999 * v[i]! + 0.001 * dot * dot;
            c[i]! -= lr * (m[i]! / (1 - 0.9 ** step)) / (Math.sqrt(v[i]! / (1 - 0.999 ** step)) + 1e-8);
          });
        }
        return { coefficients: [...c], value: await merge(), trace };
      },
    },
    /** Record the model turns of one execution as a trajectory (inside grad). */
    async trajectory<T>(run: Promise<T> | (() => Promise<T>)): Promise<Trajectory> {
      const rec = recorder('trajectory');
      const failure = await settle(run);
      const claimed = rec.claim();
      // Preserve exact native execution-graph bindings alongside sampling
      // captures. Objectives consume this trajectory after the recorder closes.
      const native = new Map(rec.nativeTurns().map(turn => [turn.native!.id, turn]));
      const turns = claimed.map(turn => turn.native ? native.get(turn.native.id) ?? turn : turn);
      if (failure !== undefined && !turns.length)
        throw new LearningError('learning-trajectory-failed',
          `trajectory produced no recorded turns before failing: ${failureNote(failure).replace(/^; the output failed: /, '')}`);
      return { id: `trajectory-${++trajectories}`, turns, ...(failure === undefined ? {} : {
        failure: { ...(failure instanceof Error ? { name: failure.name,
          ...('code' in failure && typeof failure.code === 'string' ? { code: failure.code } : {}) } : {}),
          message: failure instanceof Error ? failure.message : String(failure) },
      }) };
    },
    optimizers: {
      sgd: (options: { lr: number; momentum?: number; weightDecay?: number }) => optimizer(() => service('optimizers.sgd'), 'sgd', options),
      adam: (options: { lr: number; betas?: [number, number]; weightDecay?: number }) =>
        optimizer(() => service('optimizers.adam'), 'adam', options),
    },
    /** Save archived native states and named coefficient artifacts to a `.nz` bundle. */
    async save(path: string, exports: Record<string, unknown | NzSaveExport>): Promise<{ path: string; exports: string[] }> {
      const active = service('save');
      const remote = new HttpNeuraleseStore(active.endpoint, active.headers);
      if (!active.store) throw new LearningError('neuralese-native-archive-unavailable', 'save requires a runtime artifact store');
      const entries: Record<string, NzSaveExport> = {};
      for (const [name, value] of Object.entries(exports)) {
        if (value && typeof value === 'object' && 'type' in value && 'value' in value) entries[name] = value as NzSaveExport;
        else if (isNativeAdapterRef(value)) entries[name] = { type: 'NativeAdapter', value };
        else if (isNeuraleseRef(value)) entries[name] = { type: value.$neuralese.type, value };
        else throw new LearningError('learning-save-type', `export ${name} needs a declared type ({ type, value })`);
      }
      const visit = async (value: unknown): Promise<void> => {
        if (isNeuraleseRef(value) || isNativeAdapterRef(value)) {
          const id = isNativeAdapterRef(value) ? value.$nativeAdapter.id : value.$neuralese.id;
          if (!id.startsWith('nzs1_') && !id.startsWith('nnt1_'))
            throw new LearningError('neuralese-projected-api-retired', `${id} is a flat nz1 value; save only native state or named tensor artifacts`);
          await archiveNativeArtifactFrom(remote, active.store!, id);
        } else if (Array.isArray(value)) await Promise.all(value.map(visit));
        else if (value && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype)
          await Promise.all(Object.values(value).map(visit));
      };
      for (const entry of Object.values(entries)) await visit(entry.value);
      writeFileSync(path, await saveNz(entries, { store: active.store, dialect: active.dialect ?? 'native-state/unknown' }));
      return { path, exports: Object.keys(entries) };
    },
    /** Load a native `.nz` bundle into this task's typed artifact archive. */
    async load(path: string): Promise<{ path: string; header: unknown; exports: Record<string, unknown>; types: Record<string, string> }> {
      const active = service('load');
      if (!active.store) throw new LearningError('neuralese-native-archive-unavailable', 'load requires a runtime artifact store');
      const loaded = await loadNz(new Uint8Array(readFileSync(path)));
      if (loaded.blocks.size)
        throw new LearningError('neuralese-projected-api-retired', 'this bundle contains flat nz1 values; convert them to native states before loading');
      if ((loaded.nativeStates.size || loaded.nativeTensors.size) && (!active.store.putNativeState || !active.store.putNativeTensorArtifact))
        throw new LearningError('neuralese-native-archive-unavailable', 'native bundle import needs a state and tensor capable archive');
      await importNativeArtifactBundle(active.store, loaded.nativeStates, loaded.nativeTensors);
      return { path, header: loaded.header, exports: loaded.exports, types: loaded.types };
    },
  };
}

/** The `natlang:learning` module for natlang code: bound to the current task's `learning` service. */
export const learningModule = { __esModule: true, ...createLearning() };
