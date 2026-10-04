/**
 * `natlang:learning` (spec/SPEC.md, Neuralese chapter "Learning"; S0 §9): `grad`, `valueAndGrad`, `stopGradient`,
 * objectives, immutable optimisers and `save`, backed by a Neuralese server's gradient replay sessions.
 *
 * `grad(f, a)` runs `f(a)` while recording every model turn the Neuralese driver makes (`recording.ts`). The loss
 * that `f` returns is a list of terms over those recorded turns; the server replays them with discrete choices held
 * fixed and returns gradients for the Neuralese values in `a` (spec/NEURALESE_GRAPH.md, "Replay"). Values wrapped in
 * `stopGradient`, and gradients computed by an inner `grad` (first-order nested differentiation), are constants.
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
import { HttpNeuraleseStore } from '../model/neuralese-server.js';
import { setRecorderSource, type RecordedTurn, type TurnRecorder } from './recording.js';
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

const settle = async (output: unknown): Promise<void> => {
  try { await (typeof output === 'function' ? (output as () => unknown)() : output); } catch { /* the recorded turns still count */ }
};

function lastTurn(turns: readonly RecordedTurn[], where: string): RecordedTurn {
  const turn = turns.filter(item => !item.decision).at(-1);
  if (!turn) throw new LearningError('learning-no-turns', `${where}: the output made no recorded model turn (is the model a Neuralese server?)`);
  return turn;
}

function returnTarget(expected: unknown): Json {
  return { role: 'assistant', content: null, tool_calls: [{ id: 'expected', type: 'function', function: {
    name: 'return_result', arguments: JSON.stringify({ status: 'success', value: expected }) } }] };
}

let trajectories = 0;

export const objectives = {
  /** Cross-entropy of the expected result as the output call's final answer. */
  async crossEntropy(output: Promise<unknown>, expected: unknown): Promise<Loss> {
    const rec = recorder('objectives.crossEntropy');
    await settle(output);
    const turn = lastTurn(rec.claim(), 'crossEntropy');
    return new Loss([{ kind: 'crossEntropy', messages: turn.messages, tools: turn.tools, target: returnTarget(expected) }]);
  },
  /**
   * A proper scoring rule on a decision readout (a call with `readout: decision`): the output's scored options,
   * normalised, against `expected` probabilities by option value (`{ "yes": 0.8, "no": 0.2 }`) or one value (all
   * mass on it). `rule`: `logLoss` (default), `brier`, or `rps` for ordered options (declaration order).
   */
  async decision(output: Promise<unknown>, expected: unknown, rule: 'logLoss' | 'brier' | 'rps' = 'logLoss'): Promise<Loss> {
    const rec = recorder('objectives.decision');
    await settle(output);
    const turn = rec.claim().filter(item => item.decision).at(-1);
    if (!turn?.decision) throw new LearningError('learning-no-decision', 'objectives.decision: the output made no decision readout (does its function declare readout: decision?)');
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
      target: { probabilities: probabilities.map(value => value / total) } }]);
  },
  /** KL from the same model given the full source to the output's final turn. */
  async selfDistill(output: Promise<unknown>, withFullSource: () => Promise<unknown>): Promise<Loss> {
    const rec = recorder('objectives.selfDistill');
    await settle(output);
    const student = lastTurn(rec.claim(), 'selfDistill');
    await settle(withFullSource);
    const teacher = lastTurn(rec.claim(), 'selfDistill (full source)');
    return new Loss([{ kind: 'selfDistill', messages: student.messages, tools: student.tools, target: student.reply,
      teacher_messages: teacher.messages }]);
  },
  /** Negative log-probability of a recorded trajectory: text, stop decisions, sampled payloads. */
  async logLikelihood(trajectory: Trajectory, weight = 1): Promise<Loss> {
    return new Loss(trajectory.turns.map(turn => ({ kind: 'logLikelihood', messages: turn.messages, tools: turn.tools,
      target: turn.reply, weight })));
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
    terms.push({ kind: 'crossEntropy', law: name, messages: turn.messages, tools: turn.tools, target: returnTarget(expected) });
  }
  return new Loss(terms);
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
  const remote = new HttpNeuraleseStore(service.endpoint, service.headers);
  for (const id of ids) {
    if (await remote.has(id)) continue;
    const block = (service.store && await service.store.get(id)) ?? constantBlock(id);
    if (!block) throw new LearningError('neuralese-unknown-block', `${id} is neither on the server nor in the runtime's store`);
    const { id: _, ...rest } = block.meta;
    await remote.put({ ...rest, data: block.data });
  }
}

async function evaluate<A>(service: LearningService, f: (a: A) => Promise<Loss>, a: A, order?: number):
    Promise<{ loss: Loss; grad: Gradient<A> }> {
  if (order !== undefined && order !== 1)
    throw new LearningError('neuralese-grad-unavailable', 'only first-order gradients are supported');
  const leaves = softLeaves(a).filter(ref => !stopped.has(ref));
  const arguments_ = [...new Set(leaves.map(ref => ref.$neuralese.id))];
  const rec = new Recorder();
  const loss = await recording.run(rec, () => boundService.run(service, () => f(a)));
  if (!(loss instanceof Loss)) throw new LearningError('learning-not-a-loss', 'the function passed to grad must return a Loss from objectives');
  await ensureOnServer(service, arguments_);
  const result = await post(service, '/v1/neuralese/grad', { arguments: arguments_, terms: loss.terms });
  loss.value = Number(result.loss);
  const gradients = (result.gradients ?? {}) as Record<string, string>;
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
      const updated = new Map(params.map((id, index) => [id, (result.params as string[])[index]!]));
      const value = mapLeaves(current.value, ref => updated.has(ref.$neuralese.id) ?
        neuraleseRef(ref.$neuralese.type, updated.get(ref.$neuralese.id)!) : ref) as typeof current.value;
      return { value, opt: { $optimizer: { name, hyper, state: result.state as Json } } };
    },
  };
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
