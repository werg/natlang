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
import { writeFileSync } from 'node:fs';
import { currentFrame } from '../runtime/context.js';
import { isNeuraleseRef, neuraleseRef, type NeuraleseRef } from '../native/neuralese.js';
import { constantBlock, type NeuraleseBlock, type NeuraleseBlockMeta, type NeuraleseStore } from '../native/neuralese-store.js';
import { distributionOf, saveNz, type NzSaveExport } from '../native/nz-file.js';
import { fetchModel } from '../model/chat-completion.js';
import { HttpNeuraleseStore, restoreBlocks } from '../model/neuralese-server.js';
import * as deltaOps from './deltas.js';
import { setSystemPromptSource, systemPromptBank, type SystemPromptBank } from '../native/system-prompts.js';
import { setAdapterSource, setRecorderSource, type AdapterBinding, type RecordedTurn, type TurnRecorder } from './recording.js';
import { createNeuraleseLibrary, type StandardLibrary } from './combinators.js';

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
class Recorder implements TurnRecorder {
  readonly turns: RecordedTurn[] = [];
  private claimed = 0;
  /** Objectives awaiting a promise output right now, and whether two of them overlapped (see `capture`). */
  active = 0;
  overlapped = false;
  record(turn: RecordedTurn): void { this.turns.push(turn); }
  /** Turns recorded since the previous claim. */
  claim(): RecordedTurn[] { const out = this.turns.slice(this.claimed); this.claimed = this.turns.length; return out; }
}
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
export type Trajectory = { readonly id: string; readonly turns: readonly RecordedTurn[] };

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
const adapted = (turn: RecordedTurn): Json => turn.adapters?.length ? { adapters: turn.adapters } : {};

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
    return new Loss([{ kind: 'decision', messages: turn.messages, options: [...options], rule,
      target: { probabilities: probabilities.map(value => value / total) }, ...adapted(turn) }]);
  },
  /** KL from the same model given the full source to the output's final turn. */
  async selfDistill(output: Promise<unknown> | (() => Promise<unknown>), withFullSource: () => Promise<unknown>): Promise<Loss> {
    const rec = recorder('objectives.selfDistill');
    const student = lastTurn(await capture(rec, output, 'selfDistill'), 'selfDistill');
    const teacher = lastTurn(await capture(rec, withFullSource, 'selfDistill'), 'selfDistill (full source)');
    return new Loss([{ kind: 'selfDistill', messages: student.messages, tools: student.tools, target: student.reply,
      teacher_messages: teacher.messages, ...adapted(student),
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
      ...(privileged.adapters?.length ? { adapters: privileged.adapters } : {}) };
    if (own.decision) {
      if (!privileged.decision || JSON.stringify(privileged.decision.options) !== JSON.stringify(own.decision.options))
        throw new LearningError('learning-teacher-mismatch', 'conditionedDistill: the teacher must make the same decision readout (same options) as the student');
      return new Loss([{ kind: 'decision', messages: own.messages, options: [...own.decision.options], rule: options.rule ?? 'logLoss',
        teacher: teacherSide, ...adapted(own) }]);
    }
    if (privileged.decision) throw new LearningError('learning-teacher-mismatch', 'conditionedDistill: the student generated a reply but the teacher made a decision readout');
    return new Loss([{ kind: 'selfDistill', messages: own.messages, tools: own.tools, target: own.reply,
      teacher_messages: privileged.messages, ...adapted(own),
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
    return new Loss([{ kind: 'decision', messages: turn.messages, options: [...options], rule: 'expectedReward',
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
        terms.push({ kind: 'logLikelihood', messages: turn.messages, tools: turn.tools, target: turn.reply, weight, ...adapted(turn) });
    }
    return new Loss(terms);
  },
  /** Negative log-probability of a recorded trajectory: text, stop decisions, sampled payloads. */
  async logLikelihood(trajectory: Trajectory, weight = 1): Promise<Loss> {
    return new Loss(trajectory.turns.map(turn => ({ kind: 'logLikelihood', messages: turn.messages, tools: turn.tools,
      target: turn.reply, weight, ...adapted(turn) })));
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
/** The type of an adapter value: `Adapter` (spec/SPEC.md; its structure is in the block's dialect). */
export const ADAPTER_TYPE = 'Adapter';
export type AdapterSpecOptions = { kind?: 'xs' | 'tiny'; rank?: number; u?: number; layers?: number[];
  targets?: Array<'out' | 'ffn_down' | 'ffn_up'>; seed?: number };
type AdapterUse = NeuraleseRef | { adapter: NeuraleseRef; scale?: number };

const adapterScope = new AsyncLocalStorage<readonly AdapterBinding[]>();
setAdapterSource(() => adapterScope.getStore());

/**
 * Run `fn` with weight adapters active: every Neuralese-server turn inside (generation and decision readouts) runs
 * the adapted model, and turns recorded for `grad` replay with the same adapters. Adapters are values: inside
 * `valueAndGrad(f, adapter)` the adapter's coefficients get gradients like any soft value. Nested scopes add up.
 */
export async function withAdapters<T>(adapters: AdapterUse | readonly AdapterUse[], fn: () => Promise<T> | T): Promise<T> {
  const bindings = (Array.isArray(adapters) ? adapters : [adapters]).map((item: AdapterUse): AdapterBinding => {
    const ref = isNeuraleseRef(item) ? item : item.adapter;
    if (!isNeuraleseRef(ref)) throw new LearningError('learning-adapter', 'withAdapters takes adapter values');
    return { id: ref.$neuralese.id, scale: isNeuraleseRef(item) ? 1 : item.scale ?? 1 };
  });
  return adapterScope.run([...(adapterScope.getStore() ?? []), ...bindings], fn);
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
    if (isNeuraleseRef(value)) { const ref = neuraleseRef(value.$neuralese.type, value.$neuralese.id); stopped.add(ref); return ref; }
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
    if (isNeuraleseRef(value)) { out.push(value); return; }
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

async function post(service: LearningService, path: string, body: unknown): Promise<Json> {
  const response = await fetchModel(service.endpoint.replace(/\/$/, '') + path, { method: 'POST',
    headers: { 'content-type': 'application/json', ...(service.headers ?? {}) }, body: JSON.stringify(body) });
  const text = await response.text();
  if (!response.ok) {
    let code = 'learning-server', message = text.slice(0, 1000);
    try { const error = (JSON.parse(text) as { error?: { code?: string; message?: string } }).error; code = error?.code ?? code; message = error?.message ?? message; } catch { /* plain text */ }
    throw new LearningError(code, message);
  }
  return JSON.parse(text) as Json;
}

async function ensureOnServer(service: LearningService, ids: readonly string[]): Promise<void> {
  const missing = await restoreBlocks(new HttpNeuraleseStore(service.endpoint, service.headers), service.store, ids);
  if (missing.length) throw new LearningError('neuralese-unknown-block', `${missing[0]} is neither on the server nor in the runtime's store`);
}

/**
 * Second order: the inner computations an order-2 loss runs (gradient sessions, optimiser steps), in order. The server
 * recomputes them differentiably before the outer terms, so the outer gradient passes through the inner updates.
 */
type Derivation = Record<string, unknown> & { kind: 'grad' | 'optim' };
const derivations = new AsyncLocalStorage<Derivation[] | undefined>();

async function evaluate<A>(service: LearningService, f: (a: A) => Promise<Loss>, a: A, order?: number):
    Promise<{ loss: Loss; grad: Gradient<A> }> {
  if (order !== undefined && order !== 1 && order !== 2)
    throw new LearningError('neuralese-grad-unavailable', 'gradients of order 1 or 2');
  const leaves = softLeaves(a).filter(ref => !stopped.has(ref));
  const arguments_ = [...new Set(leaves.map(ref => ref.$neuralese.id))];
  const rec = new Recorder();
  // An inner first-order gradient's own inner computations are constants to it: they are not recorded.
  const derived: Derivation[] | undefined = order === 2 ? [] : undefined;
  const outer = derivations.getStore();
  const loss = await derivations.run(derived, () => recording.run(rec, () => boundService.run(service, () => f(a))));
  if (!(loss instanceof Loss)) throw new LearningError('learning-not-a-loss', 'the function passed to grad must return a Loss from objectives');
  // A loss without terms (a policy gradient whose samples tied) is zero, with no gradient anywhere.
  if (!loss.terms.length) {
    loss.value = 0;
    return { loss, grad: { $gradient: mapLeaves(a, (): GradientEntry => null) } as Gradient<A> };
  }
  await ensureOnServer(service, arguments_);
  // Every recorded turn that wrote blocks is a producer: a term that reads one of its blocks (a soft call's result,
  // spliced into its caller's turns) replays the write from that turn with gradient, so the loss reaches the
  // arguments in the producing call's context (spec/NEURALESE_GRAPH.md, "Replay", step 4).
  const producers = rec.turns.filter(turn => !turn.decision && turn.blocks.length)
    .map(turn => ({ messages: turn.messages, ...(turn.tools ? { tools: turn.tools } : {}), reply: turn.reply, ...adapted(turn) }));
  const result = await post(service, '/v1/neuralese/grad', { arguments: arguments_, terms: loss.terms,
    ...(producers.length ? { producers } : {}), ...(derived ? { order: 2, derived } : {}) });
  loss.value = Number(result.loss);
  const gradients = (result.gradients ?? {}) as Record<string, string>;
  outer?.push({ kind: 'grad', arguments: arguments_, terms: loss.terms, ...(producers.length ? { producers } : {}), gradients });
  const grad = mapLeaves(a, (ref): GradientEntry => stopped.has(ref) || !gradients[ref.$neuralese.id] ? null :
    { $gradientBlock: { id: gradients[ref.$neuralese.id]!, of: ref.$neuralese.id, type: ref.$neuralese.type } });
  return { loss, grad: { $gradient: grad } as Gradient<A> };
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
      const params: string[] = [], grads: string[] = [];
      const entries = softLeaves(current.value);
      const gradEntries: GradientEntry[] = [];
      const visit = (value: unknown) => {
        if (value && typeof value === 'object' && '$gradientBlock' in value) { gradEntries.push(value as GradientEntry); return; }
        if (value === null) { gradEntries.push(null); return; }
        if (Array.isArray(value)) value.forEach(visit);
        else if (value && typeof value === 'object') Object.values(value).forEach(visit);
      };
      visit(grad.$gradient);
      if (gradEntries.length !== entries.length)
        throw new LearningError('learning-gradient-shape', 'the gradient does not have the shape of the value');
      entries.forEach((ref, index) => {
        const entry = gradEntries[index];
        if (entry) { params.push(ref.$neuralese.id); grads.push(entry.$gradientBlock.id); }
      });
      const result = await post(service(), '/v1/neuralese/optim', { optimizer: name, hyper, params, grads,
        state: current.opt.$optimizer.state });
      derivations.getStore()?.push({ kind: 'optim', optimizer: name, hyper, params, grads, state: current.opt.$optimizer.state,
        results: { params: result.params, state: result.state } });
      const updated = new Map(params.map((id, index) => [id, (result.params as string[])[index]!]));
      const value = mapLeaves(current.value, ref => updated.has(ref.$neuralese.id) ?
        neuraleseRef(ref.$neuralese.type, updated.get(ref.$neuralese.id)!) : ref) as typeof current.value;
      return { value, opt: { $optimizer: { name, hyper, state: result.state as Json } } };
    },
  };
}

/** Delta blocks are read from the runtime's store or the server and written to the runtime's store (uploaded to the
 * server when a call or a gradient session needs them). */
function deltaStores(service: LearningService): deltaOps.DeltaStores {
  const remote = new HttpNeuraleseStore(service.endpoint, service.headers);
  return { read: async id => (await service.store?.get(id)) ?? (await remote.get(id)), write: service.store ?? remote };
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
    withAdapters,
    withSystemPrompts,
    /** Residual updates (LEARNING_CONTINUUM §5): soft values and adapters by block arithmetic, crisp files by patch. */
    deltas: {
      diff: (after: NeuraleseRef, base: NeuraleseRef) => deltaOps.diff(deltaStores(service('deltas.diff')), after, base),
      apply: (base: NeuraleseRef, delta: NeuraleseRef, scale = 1) => deltaOps.apply(deltaStores(service('deltas.apply')), base, delta, scale),
      compose: (deltas: Parameters<typeof deltaOps.compose>[1]) => deltaOps.compose(deltaStores(service('deltas.compose')), deltas),
      diffFiles: deltaOps.diffFiles, applyPatch: deltaOps.applyPatch, composePatches: deltaOps.composePatches,
      /**
       * Interference of deltas trained separately from one base: each delta's gain alone, the gain of their sum, and
       * Σ gains alone − gain of the sum (positive: the updates get in each other's way). `measure` is a loss (lower
       * is better) of a value, e.g. a held-out evaluation.
       */
      async interference(base: NeuraleseRef, deltas: readonly NeuraleseRef[], measure: (value: NeuraleseRef) => Promise<number>) {
        const stores = deltaStores(service('deltas.interference'));
        const reference = await measure(base);
        const alone = [];
        for (const delta of deltas) alone.push(reference - await measure(await deltaOps.apply(stores, base, delta)));
        const joint = reference - await measure(await deltaOps.apply(stores, base, await deltaOps.compose(stores, deltas)));
        return { base: reference, alone, joint, interference: alone.reduce((sum, gain) => sum + gain, 0) - joint };
      },
      /**
       * Learned merge coefficients (LoRAHub-style, first order): `base + Σ cᵢ·deltaᵢ` with the cᵢ trained by Adam on
       * `loss`. The merge is linear, so ∂L/∂cᵢ = ⟨∇L(merged), deltaᵢ⟩: one gradient session per step.
       */
      async learnMerge(base: NeuraleseRef, deltas: readonly NeuraleseRef[], loss: (value: NeuraleseRef) => Promise<Loss>,
          options: { steps?: number; lr?: number; init?: number | number[] } = {}) {
        const active = service('deltas.learnMerge'), stores = deltaStores(active);
        const parts = await Promise.all(deltas.map(async delta => {
          const block = await stores.read(delta.$neuralese.id);
          if (!block) throw new LearningError('neuralese-unknown-block', `delta ${delta.$neuralese.id} not found`);
          return deltaOps.blockFloats(block);
        }));
        const c = deltas.map((_, i) => Array.isArray(options.init) ? options.init[i] ?? 1 : options.init ?? 1);
        const m = c.map(() => 0), v = c.map(() => 0), lr = options.lr ?? 0.1, trace: { loss: number; coefficients: number[] }[] = [];
        const merge = async () => deltaOps.apply(stores, base, await deltaOps.compose(stores, deltas.map((delta, i) => ({ delta, scale: c[i]! }))));
        for (let step = 1; step <= (options.steps ?? 8); step++) {
          const merged = await merge();
          const result = await evaluate(active, loss, merged);
          trace.push({ loss: +result.loss, coefficients: [...c] });
          const entry = (result.grad as { $gradient: GradientEntry }).$gradient;
          if (!entry) break;
          const block = await stores.read(entry.$gradientBlock.id);
          if (!block) throw new LearningError('neuralese-unknown-block', `gradient ${entry.$gradientBlock.id} not found`);
          const g = deltaOps.blockFloats(block);
          parts.forEach((part, i) => {
            let dot = 0;
            for (let k = 0; k < part.length; k++) dot += g[k]! * part[k]!;
            m[i] = 0.9 * m[i]! + 0.1 * dot; v[i] = 0.999 * v[i]! + 0.001 * dot * dot;
            c[i]! -= lr * (m[i]! / (1 - 0.9 ** step)) / (Math.sqrt(v[i]! / (1 - 0.999 ** step)) + 1e-8);
          });
        }
        return { coefficients: [...c], value: await merge(), trace };
      },
    },
    adapters: {
      /** A zero adapter for the server's backbone (the identity until trained); see model/tiny_adapters.py. */
      async create(options: AdapterSpecOptions = {}): Promise<NeuraleseRef> {
        const meta = await post(service('adapters.create'), '/v1/neuralese/adapters', { ...options, type: ADAPTER_TYPE });
        return neuraleseRef(ADAPTER_TYPE, String(meta.id));
      },
    },
    /** Record the model turns of one execution as a trajectory (inside grad). */
    async trajectory<T>(run: Promise<T> | (() => Promise<T>)): Promise<Trajectory> {
      const rec = recorder('trajectory');
      await settle(run);
      return { id: `trajectory-${++trajectories}`, turns: rec.claim() };
    },
    optimizers: {
      sgd: (options: { lr: number; momentum?: number; weightDecay?: number }) => optimizer(() => service('optimizers.sgd'), 'sgd', options),
      adam: (options: { lr: number; betas?: [number, number]; weightDecay?: number }) =>
        optimizer(() => service('optimizers.adam'), 'adam', options),
    },
    /** Write values to a `.nz` file. Blocks are fetched from the runtime's store or the server. */
    async save(path: string, exports: Record<string, unknown | NzSaveExport>): Promise<{ path: string; exports: string[] }> {
      const active = service('save');
      const remote = new HttpNeuraleseStore(active.endpoint, active.headers);
      const store = { async get(id: string): Promise<NeuraleseBlock | undefined> {
        return (await active.store?.get(id)) ?? (await remote.get(id)); } } as NeuraleseStore;
      const entries: Record<string, NzSaveExport> = {};
      for (const [name, value] of Object.entries(exports)) {
        if (value && typeof value === 'object' && 'type' in value && 'value' in value) entries[name] = value as NzSaveExport;
        else if (isNeuraleseRef(value)) entries[name] = { type: value.$neuralese.type, value };
        else throw new LearningError('learning-save-type', `export ${name} needs a declared type ({ type, value })`);
      }
      writeFileSync(path, await saveNz(entries, { store, dialect: active.dialect ?? 'nd:natlang@1' }));
      return { path, exports: Object.keys(entries) };
    },
  };
}

/** The `natlang:learning` module for natlang code: bound to the current task's `learning` service. */
export const learningModule = { __esModule: true, ...createLearning() };
